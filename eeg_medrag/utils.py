"""Tokenization, PDF input, content identities, atomic output and run logging."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

import hashlib
import inspect
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from time import perf_counter
from dataclasses import asdict, is_dataclass
from functools import partial
from pathlib import Path

from ._vendor.hypergraphrag import operate as native_operate

if TYPE_CHECKING:
    from .base import BuildConfig, Document
    from .storage import Storage


class IntegrityError(ValueError):
    pass


class IncompleteStage(RuntimeError):
    pass


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def identity(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def component_descriptor(component: Any) -> dict[str, str]:
    """Describe a callable and bound arguments for request/cache identities."""
    if inspect.ismethod(component):
        return component_descriptor(partial(component.__func__, component.__self__))
    if isinstance(component, partial):

        def encode(value: Any) -> Any:
            if inspect.isclass(value):
                return component_descriptor(value)
            if isinstance(value, Path):
                return value.as_posix()
            if is_dataclass(value) and not isinstance(value, type):
                return asdict(value)
            if hasattr(value, "signature"):
                return value.signature
            if callable(value):
                return component_descriptor(value)
            raise TypeError("A bound component argument needs a stable signature")

        bindings = json.dumps(
            [component.args, component.keywords or {}],
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            default=encode,
        )
        return {
            **component_descriptor(component.func),
            "bound_arguments_sha256": hashlib.sha256(
                bindings.encode("utf-8")
            ).hexdigest(),
        }
    instance_signature = None
    if not inspect.isclass(component) and not inspect.isroutine(component):
        instance_signature = getattr(component, "signature", None)
        if instance_signature is None:
            raise TypeError("A callable instance needs a stable signature for cache reuse")
        component = component.__class__
    result = {
        "qualified_name": component.__module__ + "." + component.__qualname__,
    }
    if instance_signature is not None:
        result["instance_signature_sha256"] = identity(instance_signature)
    return result


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: object) -> None:
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Refuse symlink output")
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(serialized)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


class _LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        for name in ("OPENAI_API_KEY", "APIYI_API_KEY"):
            secret = os.environ.get(name)
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text


@contextmanager
def log_run(log_file: str | Path, stage: str) -> Iterator[logging.Logger]:
    """Configure CLI logs for one run and restore the caller's logging state."""
    path = Path(log_file)
    formatter = _LogFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    file_error = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(
            path, mode="a", encoding="utf-8", errors="backslashreplace",
        ))
    except OSError as error:
        file_error = error
    for handler in handlers:
        handler.setFormatter(formatter)
    targets = [logging.getLogger("eeg_medrag"), logging.getLogger("hypergraphrag")]
    previous = [
        (target, target.level, target.propagate, target.handlers[:])
        for target in targets
    ]
    for target in targets:
        target.handlers = handlers[:]
        target.setLevel(logging.INFO)
        target.propagate = False
    logger = logging.getLogger("eeg_medrag.run")
    started = perf_counter()
    try:
        logger.info("Run started stage=%s pid=%d log_file=%s", stage, os.getpid(), path)
        if file_error is not None:
            logger.warning(
                "Log file unavailable path=%s error=%s; using stderr",
                path, file_error,
            )
        yield logger
    except BaseException:
        logger.exception("Run failed stage=%s elapsed_seconds=%.3f", stage, perf_counter() - started)
        raise
    else:
        logger.info("Run completed stage=%s elapsed_seconds=%.3f", stage, perf_counter() - started)
    finally:
        for target, level, propagate, existing in previous:
            target.handlers = existing
            target.setLevel(level)
            target.propagate = propagate
        for handler in handlers:
            with suppress(OSError):
                handler.close()


class HyperTokenizer:
    """Token counts, chunks and source offsets using HyperGraphRAG tokenization."""

    def __init__(self, model: str | None = None) -> None:
        from .base import BuildConfig

        self.model = BuildConfig.tokenizer_model if model is None else model
        self.operate = native_operate

    def chunks(self, text: str, size: int, overlap: int) -> list[dict]:
        return self.operate.chunking_by_token_size(
            text,
            overlap_token_size=overlap,
            max_token_size=size,
            tiktoken_model=self.model,
        )

    def count(self, text: str) -> int:
        return len(self.operate.encode_string_by_tiktoken(text, model_name=self.model))

    def chunk_offsets(self, text: str, size: int, overlap: int) -> list[tuple[int, int]]:
        """Return token-window byte spans for PDF page provenance."""
        from ._vendor.hypergraphrag import utils as native_utils

        encoder = native_utils._get_tiktoken_encoder(self.model)
        tokens = encoder.encode(text)
        ends = [0]
        for token in tokens:
            ends.append(ends[-1] + len(encoder.decode_single_token_bytes(token)))
        return [
            (ends[start], ends[min(start + size, len(tokens))])
            for start in range(0, len(tokens), size - overlap)
        ]


logger = logging.getLogger(__name__)


class PDFCorpus:
    """Discover PDFs and cache parsed documents, chunks and page provenance."""

    def __init__(self, store: Storage, tokenizer: HyperTokenizer, config: BuildConfig) -> None:
        self.store = store
        self.tokenizer, self.config = tokenizer, config

    def _read(self, path: Path) -> dict:
        from pypdf import PdfReader

        logger.info("Reading new knowledge PDF: %s", path.name)
        pages = [
            (number, (page.extract_text() or "").strip())
            for number, page in enumerate(PdfReader(path).pages, 1)
        ]
        pages = [(number, text) for number, text in pages if text]
        if not pages:
            raise ValueError(
                "No readable text in " + path.name
            )
        full_text = "\n\n".join(text for _, text in pages).strip()
        full_doc_id = self.tokenizer.operate.compute_mdhash_id(full_text, prefix="doc-")
        full_doc = self.store.get("full_docs", full_doc_id)
        if full_doc is None:
            chunks = self.tokenizer.chunks(
                full_text, self.config.chunk_tokens, self.config.overlap_tokens
            )
            offsets = self.tokenizer.chunk_offsets(
                full_text, self.config.chunk_tokens, self.config.overlap_tokens
            )
            windows = []
            for chunk, (start, end) in zip(chunks, offsets):
                if not chunk["content"].strip():
                    continue
                chunk_id = self.tokenizer.operate.compute_mdhash_id(
                    chunk["content"], prefix="chunk-"
                )
                windows.append(
                    {
                        "id": chunk_id,
                        "start": start,
                        "end": end,
                        "chunk_order_index": chunk["chunk_order_index"],
                    }
                )
                if self.store.get("text_chunks", chunk_id) is None:
                    self.store.put(
                        "text_chunks",
                        chunk_id,
                        {**chunk, "full_doc_id": full_doc_id, "source": {"origins": []}},
                    )
            full_doc = {"content": full_text, "origins": [], "windows": windows}
            self.store.put("full_docs", full_doc_id, full_doc)

        boundaries = []
        position = 0
        for page_no, text in pages:
            end = position + len(text.encode("utf-8"))
            boundaries.append((position, end, page_no))
            position = end + 2
        return {
            "full_doc_id": full_doc_id,
            "paths": [],
            "chunks": [
                {
                    "id": window["id"],
                    "chunk_order_index": window["chunk_order_index"],
                    "pages": [
                        number
                        for left, right, number in boundaries
                        if left < window["end"] and right > window["start"]
                    ],
                }
                for window in full_doc["windows"]
            ],
        }

    def _attach(self, record: dict, relative_path: str, sha256: str) -> None:
        if relative_path in record["paths"]:
            return
        record["paths"] = sorted([*record["paths"], relative_path])
        full_doc_id = record["full_doc_id"]
        full_doc = self.store.get("full_docs", full_doc_id)
        origin = {"document": relative_path, "sha256": sha256}
        if origin not in full_doc["origins"]:
            full_doc["origins"].append(origin)
            self.store.put("full_docs", full_doc_id, full_doc)
        for chunk in record["chunks"]:
            row = self.store.get("text_chunks", chunk["id"])
            source = {
                **origin,
                "full_doc_id": full_doc_id,
                "pages": chunk["pages"],
                "chunk_order_index": chunk["chunk_order_index"],
            }
            if source not in row["source"]["origins"]:
                row["source"]["origins"].append(source)
                self.store.put("text_chunks", chunk["id"], row)

    def load(self, contexts_root: str | Path) -> list[Document]:
        from .base import Document

        root = Path(contexts_root).resolve()
        work = self.store.directory
        if work.is_relative_to(root):
            raise ValueError("Keep working_dir outside contexts")
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        if not (paths or self.store.count("text_chunks")):
            raise ValueError("No source PDFs")

        chunk_ids = {}
        for path in paths:
            sha256 = file_hash(path)
            relative_path = path.relative_to(root).as_posix()
            # File identity, chunks and provenance form one transaction.
            with self.store.transaction():
                record = self.store.get("pdf_files", sha256)
                if record is None:
                    record = self._read(path)
                else:
                    logger.info("Reusing cached knowledge PDF: %s", relative_path)
                self._attach(record, relative_path, sha256)
                self.store.put("pdf_files", sha256, record)
            chunk_ids.update(dict.fromkeys(chunk["id"] for chunk in record["chunks"]))

        documents = []
        for chunk_id in chunk_ids:
            row = self.store.get("text_chunks", chunk_id)
            source = row["source"]
            if row.get("registered_source") != identity(source):
                documents.append(Document(chunk_id, row["content"], source))
        return documents
