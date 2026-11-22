"""SQLite checkpoints for ordered retrieval and prediction results."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from .utils import IntegrityError, canonical


FINAL_PREDICTION_STATES = ("success", "content_filter")
_QUESTIONS_TABLE = """CREATE TABLE questions (
    id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE,
    status TEXT CHECK(status IN ('success', 'error', 'content_filter')), payload TEXT,
    CHECK((status IS NULL) = (payload IS NULL)))"""


def validate_result_output(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix != ".sqlite":
        raise ValueError("Result checkpoints require a .sqlite output")
    if path.is_symlink():
        raise ValueError("Refuse symlink result output")
    return path


def _metadata(value: dict) -> tuple[str, dict]:
    kinds = [key for key in ("retrievals", "predictions") if key in value]
    if len(kinds) != 1:
        raise IntegrityError("Expected one retrieval or prediction result collection")
    kind = kinds[0]
    metadata = {k: v for k, v in value.items() if k not in (kind, "status")}
    ids = metadata.get("question_ids")
    if (not isinstance(ids, list) or any(not isinstance(q, str) or not q for q in ids)
            or len(set(ids)) != len(ids) or not isinstance(metadata.get("run_id"), str)
            or not metadata["run_id"]):
        raise IntegrityError("Invalid result run identity or question IDs")
    return kind, metadata


def _validate_row(row: dict, kind: str) -> None:
    if isinstance(row, dict) and kind == "predictions" and row.get("status") == "content_filter":
        if row.get("answer") != "":
            raise IntegrityError("A filtered prediction must have an empty answer")
        return
    if not isinstance(row, dict) or row.get("status") not in ("success", "error"):
        raise IntegrityError("Invalid result checkpoint status")
    if row["status"] == "success":
        field = "knowledge" if kind == "retrievals" else "answer"
        if not isinstance(row.get(field), str) or (field == "answer" and not row[field].strip()):
            raise IntegrityError("Successful result checkpoint has no " + field)


def _header(db: sqlite3.Connection) -> tuple[str, dict]:
    if db.execute("PRAGMA user_version").fetchone()[0] not in (1, 2):
        raise IntegrityError("Unsupported result checkpoint schema")
    kind, payload = db.execute("SELECT kind, payload FROM run WHERE singleton=1").fetchone()
    metadata = json.loads(payload)
    _metadata({**metadata, kind: []})
    ids = [r[0] for r in db.execute("SELECT id FROM questions ORDER BY ordinal")]
    if ids != metadata["question_ids"]:
        raise IntegrityError("Result checkpoint question scope mismatch")
    return kind, metadata


def _rows(db: sqlite3.Connection, kind: str) -> list[dict]:
    rows = []
    for qid, status, payload in db.execute(
        "SELECT id, status, payload FROM questions WHERE payload IS NOT NULL ORDER BY ordinal"
    ):
        row = json.loads(payload)
        _validate_row(row, kind)
        if row.get("id") != qid or row["status"] != status:
            raise IntegrityError("Result checkpoint row identity mismatch")
        rows.append(row)
    return rows


def _document(metadata: dict, kind: str, rows: list[dict]) -> dict:
    final_states = FINAL_PREDICTION_STATES if kind == "predictions" else ("success",)
    status = "complete" if len(rows) == len(metadata["question_ids"]) and all(
        row["status"] in final_states for row in rows
    ) else "partial"
    return {**metadata, "status": status, kind: rows}


def read_results(path: str | Path) -> dict:
    """Read the ordered result document from a SQLite checkpoint."""
    path = Path(path)
    if path.suffix != ".sqlite":
        raise ValueError("Expected a .sqlite result checkpoint")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        kind, metadata = _header(db)
        return _document(metadata, kind, _rows(db, kind))


class ResultStore:
    """One writer per run; each put commits only one complete question payload."""

    def __init__(self, path: str | Path, initial: dict):
        self.path = validate_result_output(path)
        self.kind, self.metadata = _metadata(initial)
        self.ids = set(self.metadata["question_ids"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        self.db = sqlite3.connect(self.path, timeout=1.0)
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            with self.db:
                if not exists:
                    self.db.execute("BEGIN IMMEDIATE")
                    self.db.execute("CREATE TABLE run (singleton INTEGER PRIMARY KEY CHECK(singleton=1), kind TEXT NOT NULL, payload TEXT NOT NULL)")
                    self.db.execute(_QUESTIONS_TABLE)
                    self.db.execute("INSERT INTO run VALUES (1, ?, ?)", (self.kind, canonical(self.metadata)))
                    self.db.executemany("INSERT INTO questions(id, ordinal) VALUES (?, ?)",
                                        ((qid, n) for n, qid in enumerate(self.metadata["question_ids"])))
                    self.db.execute("PRAGMA user_version=2")
                else:
                    kind, metadata = _header(self.db)
                    if (kind != self.kind or metadata["run_id"] != self.metadata["run_id"]
                            or metadata["question_ids"] != self.metadata["question_ids"]):
                        raise ValueError("Result inputs or policy changed; choose a new output path")
                    # Preserve identity checks while refreshing descriptive metadata.
                    if canonical(metadata) != canonical(self.metadata):
                        self.db.execute("UPDATE run SET payload=? WHERE singleton=1", (canonical(self.metadata),))
        except BaseException:
            self.db.close()
            raise

    def rows(self) -> dict[str, dict]:
        return {row["id"]: row for row in _rows(self.db, self.kind)}

    def _write(self, row: dict, payload: str) -> None:
        # Upgrade the old CHECK constraint only when a filtered row is first needed.
        if row["status"] == "content_filter" and self.db.execute("PRAGMA user_version").fetchone()[0] == 1:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("ALTER TABLE questions RENAME TO questions_v1")
            self.db.execute(_QUESTIONS_TABLE)
            self.db.execute("INSERT INTO questions SELECT id, ordinal, status, payload FROM questions_v1")
            self.db.execute("DROP TABLE questions_v1")
            self.db.execute("PRAGMA user_version=2")
        self.db.execute("UPDATE questions SET status=?, payload=? WHERE id=?",
                        (row["status"], payload, row["id"]))

    def put(self, row: dict) -> None:
        _validate_row(row, self.kind)
        if row.get("id") not in self.ids:
            raise IntegrityError("Unknown result checkpoint question ID")
        payload = canonical(row)
        for attempt in range(3):
            try:
                with self.db:
                    self._write(row, payload)
                return
            except sqlite3.OperationalError as error:
                code = getattr(error, "sqlite_errorcode", 0) & 255
                if code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
