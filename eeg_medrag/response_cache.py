"""SQLite checkpoints for validated text and embedding responses."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from .utils import IntegrityError


CACHE_FILE = "model_responses.sqlite"
RESPONSE_NAMESPACES = frozenset(("llm", "embedding_responses"))


@contextmanager
def _connection(path: Path, *, readonly: bool = False):
    mode = "ro" if readonly else "rw"
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=" + mode, uri=True)) as db:
        if not readonly:
            db.execute("PRAGMA synchronous=FULL")
        with db:
            yield db


def _initialize(path: Path) -> None:
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("""
                CREATE TABLE responses (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    text_value TEXT,
                    vector BLOB,
                    dimension INTEGER,
                    PRIMARY KEY (namespace, key),
                    CHECK (
                        (namespace = 'llm' AND text_value IS NOT NULL
                         AND vector IS NULL AND dimension IS NULL)
                        OR
                        (namespace = 'embedding_responses' AND text_value IS NULL
                         AND vector IS NOT NULL AND dimension > 0
                         AND length(vector) = dimension * 4)
                    )
                ) WITHOUT ROWID
            """)
            db.execute("PRAGMA user_version=1")


def _encode(space: str, value: Any) -> tuple:
    if space == "llm":
        if not isinstance(value, str) or not value.strip():
            raise IntegrityError("Invalid cached text response")
        return value, None, None
    if not isinstance(value, dict) or set(value) != {"vector"}:
        raise IntegrityError("Invalid cached embedding record")
    vector = np.asarray(value["vector"], dtype="<f4")
    if not (
        vector.ndim == 1 and vector.size > 0 and np.isfinite(vector).all()
        and np.linalg.norm(vector.astype(np.float64)) > 0
    ):
        raise IntegrityError("Invalid cached embedding vector")
    return None, vector.tobytes(), int(vector.size)


def _decode(row: tuple) -> Any:
    text, vector, dimension = row
    if vector is None:
        return text
    if len(vector) != dimension * 4:
        raise IntegrityError("Invalid cached embedding length")
    return {"vector": np.frombuffer(vector, dtype="<f4").tolist()}


class SQLiteResponseCache:
    """Insert-only responses; each upsert commits independently of artifact writes."""

    def __init__(self, directory: str | Path, namespace: str) -> None:
        if namespace not in RESPONSE_NAMESPACES:
            raise ValueError("Unknown model response namespace: " + namespace)
        self.directory = Path(directory)
        self.namespace = namespace
        self._file_name = str(self.directory / CACHE_FILE)
        self._ready = False

    def _prepare(self) -> Path:
        path = Path(self._file_name)
        if not self._ready:
            if not path.exists():
                self.directory.mkdir(parents=True, exist_ok=True)
                _initialize(path)
            with _connection(path) as db:
                if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise IntegrityError("Unsupported model response cache schema")
            self._ready = True
        return path

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(dict.fromkeys(keys))
        result = {}
        with _connection(self._prepare()) as db:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = db.execute(
                    "SELECT key, text_value, vector, dimension FROM responses "
                    f"WHERE namespace = ? AND key IN ({placeholders})",
                    [self.namespace, *batch],
                )
                for key, *value in rows:
                    result[key] = _decode(tuple(value))
        return result

    def get(self, key: str, default: Any = None) -> Any:
        return self.get_many([key]).get(key, default)

    async def get_by_id(self, key: str) -> Any:
        return self.get(key)

    async def get_by_ids(self, keys: list[str], fields=None) -> list:
        rows = self.get_many(keys)
        return [
            rows.get(key) if fields is None or key not in rows
            else {field: value for field, value in rows[key].items() if field in fields}
            for key in keys
        ]

    def items(self) -> Iterator[tuple[str, Any]]:
        with _connection(self._prepare()) as db:
            for key, *value in db.execute(
                "SELECT key, text_value, vector, dimension FROM responses "
                "WHERE namespace = ? ORDER BY key", (self.namespace,),
            ):
                yield key, _decode(tuple(value))

    def count(self) -> int:
        with _connection(self._prepare()) as db:
            return db.execute(
                "SELECT count(*) FROM responses WHERE namespace = ?", (self.namespace,),
            ).fetchone()[0]

    async def all_keys(self) -> list[str]:
        with _connection(self._prepare()) as db:
            return [row[0] for row in db.execute(
                "SELECT key FROM responses WHERE namespace = ? ORDER BY key", (self.namespace,),
            )]

    async def filter_keys(self, keys: list[str]) -> set[str]:
        return set(keys) - self.get_many(keys).keys()

    async def upsert(self, data: dict) -> dict:
        encoded = [(self.namespace, key, *_encode(self.namespace, value)) for key, value in data.items()]
        inserted = {}
        with _connection(self._prepare()) as db:
            for row in encoded:
                cursor = db.execute(
                    "INSERT INTO responses VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(namespace, key) DO NOTHING", row,
                )
                if cursor.rowcount:
                    inserted[row[1]] = data[row[1]]
        return inserted

    async def index_done_callback(self) -> None:
        # Every successful upsert has already committed and closed its connection.
        pass
