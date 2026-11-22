"""Incremental EEG shard locations; numeric arrays remain immutable NPY files."""

from collections.abc import Iterator
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .utils import IntegrityError, canonical


EEG_INDEX_FILE = "eeg_index.sqlite"


@contextmanager
def _connection(path: Path, *, readonly: bool = True):
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
            db.execute("""CREATE TABLE windows (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            ) WITHOUT ROWID""")
            db.execute("PRAGMA user_version=1")


class EEGMetadataStorage:
    """Read shard locations and replay a durable publication's metadata delta."""

    def __init__(self, directory: str | Path, namespace: str) -> None:
        if not namespace.startswith("eeg:"):
            raise ValueError("Not an EEG index namespace")
        self.directory = Path(directory)
        self.namespace = namespace
        self._file_name = str(self.directory / EEG_INDEX_FILE)
        self._ready = False

    def _path(self, *, write: bool = False) -> Path | None:
        path = Path(self._file_name)
        if not path.exists():
            if not write:
                return None
            fd, name = tempfile.mkstemp(prefix=EEG_INDEX_FILE + ".", suffix=".tmp", dir=self.directory)
            os.close(fd)
            staged = Path(name)
            try:
                _initialize(staged)
                os.replace(staged, path)
            finally:
                if staged.exists():
                    staged.unlink()
        if not self._ready:
            with _connection(path) as db:
                if db.execute("PRAGMA user_version").fetchone()[0] != 1:
                    raise IntegrityError("Unsupported EEG index schema")
            self._ready = True
        return path

    def get(self, key: str, default=None):
        path = self._path()
        if path is None:
            return default
        with _connection(path) as db:
            row = db.execute("SELECT value FROM windows WHERE namespace=? AND key=?",
                             (self.namespace, key)).fetchone()
            return json.loads(row[0]) if row is not None else default

    def items(self) -> Iterator[tuple[str, dict]]:
        path = self._path()
        if path is None:
            return
        with _connection(path) as db:
            for key, value in db.execute(
                "SELECT key,value FROM windows WHERE namespace=? ORDER BY key", (self.namespace,),
            ):
                yield key, json.loads(value)

    def keys(self) -> set[str]:
        path = self._path()
        if path is None:
            return set()
        with _connection(path) as db:
            return {row[0] for row in db.execute(
                "SELECT key FROM windows WHERE namespace=?", (self.namespace,),
            )}

    def count(self) -> int:
        path = self._path()
        if path is None:
            return 0
        with _connection(path) as db:
            return db.execute("SELECT count(*) FROM windows WHERE namespace=?",
                              (self.namespace,)).fetchone()[0]

    def apply(self, rows: dict[str, dict]) -> None:
        path = self._path(write=True)
        with _connection(path, readonly=False) as db:
            db.executemany(
                "INSERT INTO windows VALUES (?,?,?) "
                "ON CONFLICT(namespace,key) DO UPDATE SET value=excluded.value",
                ((self.namespace, key, canonical(value)) for key, value in rows.items()),
            )
