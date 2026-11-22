"""Bundled HyperGraphRAG storage components and durable checkpoints."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, cast

import asyncio
import copy
import hashlib
import logging
from time import perf_counter
import os
import shutil
import uuid
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import networkx as nx
from numpy.typing import ArrayLike

from ._vendor.hypergraphrag.storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from ._vendor.hypergraphrag.utils import EmbeddingFunc
from ._vendor.hypergraphrag.prompt import GRAPH_FIELD_SEP
from .base import BuildConfig, LLMConfig, ModelProvider
from .response_cache import RESPONSE_NAMESPACES, SQLiteResponseCache
from .request_usage import record_request, request_context
from .eeg_metadata import EEGMetadataStorage
from .utils import (
    IntegrityError,
    atomic_json,
    canonical,
    file_hash,
    identity,
    read_json,
)


logger = logging.getLogger(__name__)


class CheckpointStorage(JsonKVStorage):
    """JSON checkpoints with defensive reads and atomic single-file writes."""

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return copy.deepcopy(value)

    async def index_done_callback(self) -> None:
        atomic_json(self._file_name, self._data)


class VectorStorage(NanoVectorDBStorage):
    """Vector index with matrix reads for retrieval and graph links."""

    def rows(
        self, vector_id: str | None = None, *, copy_vectors: bool = True,
    ) -> Iterator[tuple[str, np.ndarray, dict]]:
        # NanoVectorDB.get returns metadata; the stored matrix supplies vectors.
        data = self._client._NanoVectorDB__storage
        records = data["data"]
        if vector_id is None:
            positions = range(len(records))
        else:
            # Native updates retain IDs/positions; inserts grow the list.
            # Deletion and transaction rollback replace the metadata list.
            if getattr(self, "_indexed_records", None) is not records or self._indexed_count != len(records):
                self._row_positions = {row["__id__"]: i for i, row in enumerate(records)}
                self._indexed_records, self._indexed_count = records, len(records)
            position = self._row_positions.get(vector_id)
            positions = () if position is None else (position,)
        for position in positions:
            row = records[position]
            vector = np.asarray(data["matrix"][position], dtype=np.float32)
            if copy_vectors:
                vector = vector.copy()
            else:
                vector = vector.view()
                vector.flags.writeable = False
            yield (
                row["__id__"],
                vector,
                {key: value for key, value in row.items() if key != "__id__"},
            )


class Storage:
    """Compose response checkpoints and native artifact stores under one root.

    A small redo journal publishes complete file batches. Validated model calls
    are checkpointed separately before any graph/index publication starts.
    """

    kv_storage_cls = CheckpointStorage
    vector_storage_cls = VectorStorage
    graph_storage_cls = NetworkXStorage
    case_batch_size = 64

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._kv: dict[str, CheckpointStorage | SQLiteResponseCache | EEGMetadataStorage] = {}
        self._vdb: dict[str, VectorStorage] = {}
        self._graphs: dict[str, NetworkXStorage] = {}
        self._frames: list[dict] = []
        self._dirty: set[tuple[str, str]] = set()
        self._eeg: dict[str, EEGStorage] = {}
        self._request_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()
        self._published_batches = 0
        self._staged_files: dict[str, str] = {}
        self._eeg_updates: dict[str, dict] = {}
        self.model: CachedModel | None = None
        self.dimension = None
        self.batch_size = LLMConfig.embedding_batch_size
        self.threshold = BuildConfig.cosine_better_than_threshold
        recovering = (self.directory / ".pending_storage").exists()
        if recovering:
            logger.warning("Recovering interrupted storage publication directory=%s", self.directory)
        self._recover()
        if recovering:
            logger.info("Storage recovery completed directory=%s", self.directory)

    @staticmethod
    def namespace(name: str) -> str:
        return name.replace(":", "_")

    def kv(self, space: str) -> CheckpointStorage | SQLiteResponseCache | EEGMetadataStorage:
        if space not in self._kv:
            if space in RESPONSE_NAMESPACES:
                self._kv[space] = SQLiteResponseCache(self.directory, space)
            elif space.startswith("eeg:"):
                self._kv[space] = EEGMetadataStorage(self.directory, space)
            else:
                self._kv[space] = self.kv_storage_cls(
                    namespace=self.namespace(space),
                    global_config={"working_dir": str(self.directory)},
                    embedding_func=None,  # type: ignore[arg-type]  # Native JSON KV does not use embeddings.
                )
        return self._kv[space]

    def bind_model(
        self,
        model: CachedModel,
        dimension: int | None,
        batch_size: int = LLMConfig.embedding_batch_size,
        threshold: float = BuildConfig.cosine_better_than_threshold,
    ) -> None:
        self.model, self.dimension = model, dimension
        self.batch_size, self.threshold = batch_size, threshold

    def vdb(self, space: str) -> VectorStorage:
        if space not in self._vdb:
            catalog = self.get("vector_indexes", space)
            dimension = (
                self.dimension
                if self.dimension is not None
                else catalog and catalog["dimension"]
            )

            async def embed(texts: list[str]) -> np.ndarray:
                return await cast(CachedModel, self.model).embed(texts)

            config = getattr(getattr(self.model, "provider", None), "config", None)
            self._vdb[space] = self.vector_storage_cls(
                namespace=self.namespace(space),
                global_config={
                    "working_dir": str(self.directory),
                    "embedding_batch_num": self.batch_size,
                    "cosine_better_than_threshold": self.threshold,
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=dimension,
                    max_token_size=getattr(
                        config, "embedding_max_token_size", LLMConfig.embedding_max_token_size
                    ),
                    func=embed,
                ),
                meta_fields={
                    "content",
                    "kind",
                    "graph_id",
                    "score",
                    "entity_name",
                    "hyperedge_name",
                    "domain",
                    "subject_id",
                },
            )
        return self._vdb[space]

    async def upsert_vectors(self, space: str, data: dict) -> None:
        if not data:
            return
        index = self.vdb(space)
        with self.transaction():
            self._touch("vector", space)
            await index.upsert(data)
            self.put(
                "vector_indexes",
                space,
                {"dimension": index.embedding_func.embedding_dim},
            )

    def graph(self, name: str) -> NetworkXStorage:
        if name not in self._graphs:
            self._graphs[name] = self.graph_storage_cls(
                namespace=self.namespace(name),
                global_config={"working_dir": str(self.directory)},
            )
        return self._graphs[name]

    def set_graph(self, name: str, graph: nx.Graph) -> None:
        with (nullcontext() if self._frames else self.transaction()):
            self._touch("graph", name)
            self.graph(name)._graph = graph.copy()

    @staticmethod
    def _location(space: str, key: str) -> tuple[str, str]:
        if space in {"pdf_files", "stages", "vector_indexes"}:
            return "state", space + ":" + key
        return space, key

    def get(self, space: str, key: str, default: Any = None) -> Any:
        physical, key = self._location(space, key)
        return self.kv(physical).get(key, default)

    def put(self, space: str, key: str, value: Any) -> None:
        if space in RESPONSE_NAMESPACES:
            raise ValueError("Model responses must be committed with cache_put")
        if space.startswith("eeg:"):
            raise ValueError("EEG locations must be published with their NPY shard")
        # Standalone writes open a transaction; scoped writes share its snapshot.
        with (nullcontext() if self._frames else self.transaction()):
            physical, key = self._location(space, key)
            self._touch("kv", physical)
            cast(CheckpointStorage, self.kv(physical))._data[key] = copy.deepcopy(value)

    def request_lock(self, key: str) -> asyncio.Lock:
        """Share in-flight request locks across model adapters using this store."""
        locks = self._request_locks.setdefault(asyncio.get_running_loop(), {})
        return locks.setdefault(key, asyncio.Lock())

    async def cache_put(self, space: str, data: dict) -> None:
        """Commit validated responses independently of artifact transactions."""
        if space not in RESPONSE_NAMESPACES:
            raise ValueError("Unknown model response namespace: " + space)
        started = perf_counter()
        await self.kv(space).upsert(data)
        logger.info(
            "Response cache committed namespace=%s entries=%d elapsed_seconds=%.3f",
            space, len(data), perf_counter() - started,
        )

    def items(self, space: str) -> Iterator[tuple[str, Any]]:
        physical, prefix = self._location(space, "")
        storage = self.kv(physical)
        if isinstance(storage, (SQLiteResponseCache, EEGMetadataStorage)):
            yield from storage.items()
            return
        for key in sorted(storage._data):
            if key.startswith(prefix):
                yield key[len(prefix):], storage.get(key)

    def count(self, space: str) -> int:
        if space in RESPONSE_NAMESPACES or space.startswith("eeg:"):
            return cast(SQLiteResponseCache | EEGMetadataStorage, self.kv(space)).count()
        return sum(1 for _ in self.items(space))

    def stage(self, name: str) -> dict | None:
        return self.get("stages", name)

    def set_stage(self, name: str, revision: str, complete: bool, **values) -> dict:
        state = {
            "revision": revision,
            "complete": complete,
            **values,
        }
        self.put("stages", name, state)
        return state

    def _touch(self, kind: str, name: str) -> None:
        if kind == "kv" and (name in RESPONSE_NAMESPACES or name.startswith("eeg:")):
            raise ValueError("Incremental stores do not use full artifact snapshots")
        marker = kind, name
        for frame in self._frames:
            if marker not in frame["backup"]:
                if kind == "kv":
                    value = self.kv(name)._data
                elif kind == "vector":
                    value = self.vdb(name)._client
                else:
                    value = self.graph(name)._graph
                frame["backup"][marker] = copy.deepcopy(value)
        self._dirty.add(marker)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        frame: dict = {"backup": {}, "dirty": set(self._dirty)}
        self._frames.append(frame)
        try:
            yield
            if len(self._frames) == 1:
                self._publish()
        except BaseException:
            for (kind, name), value in frame["backup"].items():
                if kind == "kv":
                    self.kv(name)._data = value
                elif kind == "vector":
                    self.vdb(name)._client = value
                else:
                    self.graph(name)._graph = value
            self._dirty = frame["dirty"]
            raise
        finally:
            self._frames.pop()

    def _publish(self) -> None:
        if not self._dirty and not self._staged_files and not self._eeg_updates:
            return
        pending = self.directory / ".pending_storage"
        if pending.exists():
            if not self._staged_files or (pending / "manifest.json").exists():
                raise RuntimeError("Recover pending storage before writing")
        else:
            pending.mkdir()
        entries = []
        try:
            for staged_name, target_name in self._staged_files.items():
                entries.append({
                    "staged": staged_name, "target": target_name,
                    "sha256": file_hash(pending / staged_name),
                })
            for number, (kind, name) in enumerate(sorted(self._dirty)):
                staged = pending / (str(number) + ".part")
                if kind == "kv":
                    kv = self.kv(name)
                    target = Path(kv._file_name)
                    atomic_json(staged, kv._data)
                elif kind == "vector":
                    index = self.vdb(name)
                    target = Path(index._client_file_name)
                    previous = index._client.storage_file
                    try:
                        index._client.storage_file = str(staged)
                        index._client.save()
                    finally:
                        index._client.storage_file = previous
                else:
                    graph = self.graph(name)
                    target = Path(graph._graphml_xml_file)
                    NetworkXStorage.write_nx_graph(graph._graph, str(staged))
                with staged.open("r+b") as handle:
                    os.fsync(handle.fileno())
                entries.append(
                    {
                        "staged": staged.name,
                        "target": target.relative_to(self.directory).as_posix(),
                        "sha256": file_hash(staged),
                    }
                )
            for number, (space, rows) in enumerate(self._eeg_updates.items()):
                staged = pending / ("eeg-" + str(number) + ".part")
                atomic_json(staged, {"space": space, "rows": rows})
                entries.append({
                    "kind": "eeg_index", "staged": staged.name, "sha256": file_hash(staged),
                })
            atomic_json(pending / "manifest.json", entries)
            self._published_batches += 1
        except BaseException:
            if not (
                pending.resolve().parent == self.directory
                and (not pending.is_symlink())
            ):
                raise IntegrityError("Unsafe storage journal")
            if not (pending / "manifest.json").exists():
                shutil.rmtree(pending)
            raise
        # A durable manifest commits the batch to recovery and publication.
        self._recover()
        self._dirty.clear()

    def _recover(self) -> None:
        pending = self.directory / ".pending_storage"
        if not pending.exists():
            return
        if not (
            pending.resolve().parent == self.directory and (not pending.is_symlink())
        ):
            raise IntegrityError("Unsafe storage journal")
        manifest = pending / "manifest.json"
        if manifest.exists():
            for entry in read_json(manifest):
                if entry.get("kind") == "eeg_index":
                    staged = (pending / entry["staged"]).resolve()
                    if staged.parent != pending or not staged.is_file():
                        raise IntegrityError("Missing or unsafe EEG metadata delta")
                    if file_hash(staged) != entry["sha256"]:
                        raise IntegrityError("Corrupt EEG metadata delta")
                    delta = read_json(staged)
                    if not delta["space"].startswith("eeg:"):
                        raise IntegrityError("Invalid EEG metadata namespace")
                    cast(EEGMetadataStorage, self.kv(delta["space"])).apply(delta["rows"])
                    continue
                target = (self.directory / entry["target"]).resolve()
                staged = pending / entry["staged"]
                if not (
                    target.is_relative_to(self.directory) and staged.parent == pending
                ):
                    raise IntegrityError("Unsafe journal entry")
                if staged.exists():
                    if file_hash(staged) != entry["sha256"]:
                        raise IntegrityError("Corrupt staged storage")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged, target)
                else:
                    if not (target.exists() and file_hash(target) == entry["sha256"]):
                        raise IntegrityError("Incomplete storage publication")
            # Remove the commit marker before deleting replayable metadata deltas.
            manifest.unlink()
        shutil.rmtree(pending)

    @staticmethod
    def vector_namespaces(space: str, key: str | None = None) -> tuple[str, ...]:
        """Resolve a logical collection to vector-index namespaces."""
        if space == "knowledge":
            if key is None:
                return ("entities", "hyperedges")
            return ("entities" if key.startswith("ent-") else "hyperedges",)
        if space == "knowledge_chunks":
            return ("chunks",)
        return (space,)

    def vector(self, space: str, key: str) -> tuple[np.ndarray, dict] | None:
        if space.startswith("eeg:"):
            return self.eeg(space).get(key)
        for collection in self.vector_namespaces(space, key):
            row = next(self.vdb(collection).rows(vector_id=key), None)
            if row is not None:
                return row[1], row[2]
        return None

    def vectors(
        self, space: str, *, copy_vectors: bool = True,
    ) -> Iterator[tuple[str, np.ndarray, dict]]:
        if space.startswith("eeg:"):
            yield from self.eeg(space).rows()
            return
        spaces = self.vector_namespaces(space)
        rows = [row for name in spaces for row in self.vdb(name).rows(copy_vectors=copy_vectors)]
        yield from sorted(rows, key=lambda row: row[0])

    def eeg(self, space: str) -> EEGStorage:
        if space not in self._eeg:
            self._eeg[space] = EEGStorage(self, space)
        return self._eeg[space]

    def digest(
        self, kv_spaces: Iterable[str] = (), vector_spaces: Iterable[str] = (),
        graph_spaces: Iterable[str] = (), keys: Iterable[str] | None = None,
    ) -> str:
        selected = None if keys is None else set(keys)
        result = hashlib.sha256()
        for space in sorted(kv_spaces):
            for key, value in self.items(space):
                if selected is None or key in selected:
                    result.update(canonical(["kv", space, key, value]).encode())

        def attrs(data: Mapping) -> dict:
            data = dict(data)
            if "source_id" in data:
                data["source_id"] = GRAPH_FIELD_SEP.join(
                    sorted(set(data["source_id"].split(GRAPH_FIELD_SEP)))
                )
            return data

        for space in sorted(graph_spaces):
            graph = self.graph(space)._graph
            for key, data in sorted(graph.nodes(data=True)):
                result.update(canonical(["node", space, key, attrs(data)]).encode())
            edges = [
                (min(left, right), max(left, right), data)
                for left, right, data in graph.edges(data=True)
            ]
            for left, right, data in sorted(edges, key=lambda row: (row[0], row[1])):
                result.update(
                    canonical(["edge", space, left, right, attrs(data)]).encode()
                )
        for space in sorted(vector_spaces):
            for key, vector, meta in self.vectors(space):
                if selected is None or key in selected:
                    result.update(
                        canonical(["vector", space, key, len(vector), meta]).encode()
                    )
                    result.update(np.asarray(vector, dtype="<f4").tobytes())
        return result.hexdigest()


class EEGStorage:
    """PAA batches bounded by both window count and numeric payload size."""

    def __init__(
        self, storage: Storage, space: str, batch_size: int = 16384,
        max_batch_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.storage, self.space, self.batch_size = storage, space, batch_size
        self.max_batch_bytes = max_batch_bytes
        self.pending: dict[str, tuple[np.ndarray, dict]] = {}
        self._pending_bytes = 0
        self._known_ids: set[str] | None = None
        cast(EEGMetadataStorage, self.storage.kv(space))._path()

    def put(self, key: str, vector: ArrayLike, meta: dict) -> None:
        value = np.asarray(vector, dtype="<f4").copy()
        previous_bytes = self.pending[key][0].nbytes if key in self.pending else 0
        if self.pending and self._pending_bytes - previous_bytes + value.nbytes > self.max_batch_bytes:
            self.flush()
            previous_bytes = 0
        self.pending[key] = (value, copy.deepcopy(meta))
        self._pending_bytes += value.nbytes - previous_bytes
        if len(self.pending) >= self.batch_size or self._pending_bytes >= self.max_batch_bytes:
            self.flush()

    def contains(self, key: str) -> bool:
        if key in self.pending:
            return True
        if self._known_ids is None:
            self._known_ids = cast(EEGMetadataStorage, self.storage.kv(self.space)).keys()
        return key in self._known_ids

    def get(self, key: str) -> tuple[np.ndarray, dict] | None:
        if key in self.pending:
            vector, meta = self.pending[key]
            return vector.copy(), copy.deepcopy(meta)
        row = self.storage.get(self.space, key)
        if row is None:
            return None
        array = np.load(self.storage.directory / row["file"], mmap_mode="r", allow_pickle=False)
        try:
            return np.array(array[row["row"]], dtype=np.float32, copy=True), row["meta"]
        finally:
            array._mmap.close()

    def rows(self) -> Iterator[tuple[str, np.ndarray, dict]]:
        from collections import OrderedDict

        arrays = OrderedDict()
        try:
            for key, row in self.storage.items(self.space):
                if key in self.pending:
                    continue
                name = row["file"]
                if name not in arrays:
                    arrays[name] = np.load(self.storage.directory / name, mmap_mode="r", allow_pickle=False)
                arrays.move_to_end(name)
                while len(arrays) > 4:
                    _, old = arrays.popitem(last=False)
                    old._mmap.close()
                yield key, np.array(arrays[name][row["row"]], dtype=np.float32, copy=True), row["meta"]
        finally:
            for array in arrays.values():
                array._mmap.close()
        for key, (vector, meta) in sorted(self.pending.items()):
            yield key, vector.copy(), copy.deepcopy(meta)

    def flush(self) -> None:
        if not self.pending:
            return
        journal = self.storage.directory / ".pending_storage"
        if journal.exists():
            raise RuntimeError("Reopen storage to recover interrupted publication")
        publication = self.storage._published_batches
        keys = sorted(self.pending)
        filename = self.storage.namespace(self.space) + "_" + uuid.uuid4().hex + ".npy"
        relative = "eeg/" + filename
        staged_name = filename + ".part"
        try:
            journal.mkdir()
            self.storage._staged_files[staged_name] = relative
            with (journal / staged_name).open("wb") as handle:
                np.save(handle, np.stack([self.pending[key][0] for key in keys]), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            with self.storage.transaction():
                self.storage._eeg_updates[self.space] = {
                    key: {
                        "file": relative, "row": number, "meta": self.pending[key][1],
                    } for number, key in enumerate(keys)
                }
        except BaseException:
            if self.storage._published_batches != publication or (journal / "manifest.json").exists():
                self.pending.clear()
                self._pending_bytes = 0
                self._known_ids = None
            else:
                if journal.exists():
                    shutil.rmtree(journal)
            raise
        finally:
            self.storage._staged_files.clear()
            self.storage._eeg_updates.clear()
        logger.info(
            "EEG batch checkpointed domain=%s windows=%d file=%s",
            self.space, len(keys), relative,
        )
        self.pending.clear()
        self._pending_bytes = 0
        if self._known_ids is not None:
            self._known_ids.update(keys)


class CachedModel:
    def __init__(self, provider: ModelProvider, store: Storage) -> None:
        self.provider = provider
        self.store = store
        self.signature = provider.signature
        self.chat_signature = getattr(provider, "chat_signature", self.signature)
        self.embedding_signature = getattr(provider, "embedding_signature", self.signature)
        self.descriptor = getattr(provider, "descriptor", {"signature": self.signature})
        self.chat_descriptor = {
            key: value for key, value in self.descriptor.items()
            if not key.startswith("embedding_")
        }
        if "signature" in self.chat_descriptor:
            self.chat_descriptor["signature"] = self.chat_signature


    async def text(
        self,
        messages: list[dict],
        cache_namespace: str,
        validator: Callable[[str], object] | None = None,
        **parameters,
    ) -> str:
        """Cache successful text requests by model, messages, namespace and options.

        cache_namespace partitions operations in the request fingerprint.
        validator checks new responses before publication. Successful cached
        responses are reused directly.
        """
        key = identity(
            [
                self.chat_signature,
                messages,
                cache_namespace,
                parameters,
            ]
        )

        def validate(value: str) -> str:
            if not (type(value) is str and value.strip()):
                raise IntegrityError("Empty text output")
            if validator is not None:
                validator(value)
            return value

        async with self.store.request_lock(key):
            cached = await self.store.kv("llm").get_by_id(key)
            if cached is not None:
                with request_context(ledger=self.store.directory / "model_requests.jsonl",
                                     request_key=key, namespace=cache_namespace):
                    record_request(event="cache_hit", operation="chat")
                logger.info("Completion cache hit namespace=%s request=%s", cache_namespace, key[:12])
                return cached
            started = perf_counter()
            logger.info("Completion requested namespace=%s request=%s", cache_namespace, key[:12])
            try:
                with request_context(ledger=self.store.directory / "model_requests.jsonl",
                                     request_key=key, namespace=cache_namespace):
                    record_request(event="request_started", operation="chat", parameters=parameters)
                    value = validate(
                        await self.provider.complete(messages, validate, **parameters)
                    )
                await self.store.cache_put("llm", {key: value})
            except Exception as error:
                logger.error(
                    "Completion failed namespace=%s request=%s error=%s http_status=%s request_id=%s",
                    cache_namespace, key[:12], type(error).__name__,
                    getattr(error, "status_code", None), getattr(error, "request_id", None),
                )
                raise
            logger.info(
                "Completion checkpointed namespace=%s request=%s elapsed_seconds=%.3f",
                cache_namespace, key[:12], perf_counter() - started,
            )
            return value

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [
            identity(
                [
                    self.embedding_signature,
                    "embedding",
                    text,
                ]
            )
            for text in texts
        ]
        inputs = dict(zip(keys, texts))
        locks = []
        vectors = {}
        try:
            for key in sorted(inputs):
                lock = self.store.request_lock(key)
                await lock.acquire()
                locks.append(lock)
            missing = []
            cached = cast(SQLiteResponseCache, self.store.kv("embedding_responses")).get_many(inputs)
            for key, text in inputs.items():
                row = cached.pop(key, None)
                if row is None:
                    missing.append((key, text))
                else:
                    vectors[key] = np.asarray(row["vector"], dtype=np.float32)
            if vectors:
                with request_context(ledger=self.store.directory / "model_requests.jsonl", namespace="embedding"):
                    record_request(event="cache_hit", operation="embedding", count=len(vectors))
            size = getattr(
                getattr(self.provider, "config", None), "embedding_batch_size", LLMConfig.embedding_batch_size
            )
            logger.info(
                "Embedding cache inputs=%d unique=%d cached=%d pending=%d",
                len(texts), len(inputs), len(inputs) - len(missing), len(missing),
            )
            for start in range(0, len(missing), size):
                batch = missing[start : start + size]
                started = perf_counter()
                logger.info(
                    "Embedding batch started batch=%d inputs=%d pending_total=%d",
                    start // size + 1, len(batch), len(missing),
                )
                try:
                    with request_context(ledger=self.store.directory / "model_requests.jsonl",
                                         request_key=identity([key for key, _ in batch]), namespace="embedding"):
                        record_request(event="request_started", operation="embedding", inputs=len(batch))
                        result = await self.provider.embed([text for _, text in batch])
                except Exception as error:
                    logger.error(
                        "Embedding batch failed batch=%d request=%s error=%s http_status=%s "
                        "request_id=%s",
                        start // size + 1, batch[0][0][:12], type(error).__name__,
                        getattr(error, "status_code", None), getattr(error, "request_id", None),
                    )
                    raise
                result = np.asarray(result, dtype=np.float32)
                if not (
                    result.ndim == 2
                    and result.shape[0] == len(batch)
                    and (result.shape[1] > 0)
                    and np.isfinite(result).all()
                    and (np.linalg.norm(result.astype(np.float64), axis=1) > 0).all()
                ):
                    raise IntegrityError("Embedding provider batch mismatch")
                await self.store.cache_put(
                    "embedding_responses",
                    {
                        key: {"vector": vector}
                        for (key, _), vector in zip(batch, result)
                    },
                )
                vectors.update({key: vector for (key, _), vector in zip(batch, result)})
                logger.info(
                    "Embedding batch checkpointed batch=%d completed=%d/%d elapsed_seconds=%.3f",
                    start // size + 1, start + len(batch), len(missing), perf_counter() - started,
                )
        finally:
            for lock in reversed(locks):
                lock.release()
        a = np.asarray([vectors[key] for key in keys], dtype=np.float32)
        normalized = a.astype(np.float64)
        normalized /= np.linalg.norm(normalized, axis=1, keepdims=True)
        return normalized.astype(np.float32)
