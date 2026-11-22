"""Read published cases, EEG windows, events and benchmark associations."""

from __future__ import annotations
from collections.abc import Iterator
from typing import Self, cast

import json
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .base import (
    Case, ClinicalCaseView, Event, Recording, Segment, SegmentEvidence, Query,
)
from .clinical import (
    clinical_content,
    render_eeg_evidence,
)
from .utils import identity, read_json


class Dataset:
    """Read-only indexed cases, events and EEG with dataset-relative file links."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._arrays: OrderedDict[str, np.memmap] = OrderedDict()
        index = self.root / "index.sqlite"
        self.db = sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.metadata = {
            row["key"]: json.loads(row["value"])
            for row in self.db.execute("SELECT key,value FROM metadata")
        }
        self.spaces = self.metadata["spaces"]
        self.domains = self.metadata["domains"]

    def case(self, case_id: str) -> Case:
        row = self.db.execute(
            "SELECT * FROM cases WHERE case_id=?", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        with (self.root / row["case_file"]).open("rb") as stream:
            stream.seek(row["byte_offset"])
            data = stream.read(row["byte_length"])
        case = json.loads(data)
        return cast(Case, case)

    def case_subject(self, case_id: str) -> str:
        """Resolve the patient identity from the catalog without reading case text."""
        row = self.db.execute(
            "SELECT subject_id FROM cases WHERE case_id=?", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return row["subject_id"]

    def evidence(self, segment_id: str) -> SegmentEvidence:
        segment = self.segment(segment_id)
        case = self.case(segment["case_id"])
        events = self.events_for_segment(segment_id)
        return {
            "case_id": case["case_id"],
            "subject_id": case["subject_id"],
            "domain_id": case["domain_id"],
            "record_id": segment["record_id"],
            "segment_id": segment_id,
            "eeg_space_id": segment["eeg_space_id"],
            "binding_level": "event" if events else "record",
            "event_ids": [event["event_id"] for event in events],
            "text": render_eeg_evidence(case, segment, events),
        }

    def signal(self, segment_id: str) -> np.ndarray:
        segment = self.segment(segment_id)
        record = self._record(segment["record_id"])
        key = record["record_id"]
        if key not in self._arrays:
            self._arrays[key] = np.load(
                self.root / record["signal_path"], mmap_mode="r", allow_pickle=False
            )
        self._arrays.move_to_end(key)
        while len(self._arrays) > 4:
            _, array = self._arrays.popitem(last=False)
            array._mmap.close()  # type: ignore[attr-defined]  # NumPy stubs omit the backing mmap handle.
        return np.array(self._arrays[key][segment["array_row"]], copy=True)

    def close(self) -> None:
        for array in self._arrays.values():
            array._mmap.close()  # type: ignore[attr-defined]  # NumPy stubs omit the backing mmap handle.
        self._arrays.clear()
        self.db.close()

    def _record(self, record_id: str) -> Recording:
        r = self.db.execute(
            "SELECT r.*,c.case_id,c.subject_id,c.domain_id FROM records r JOIN cases c ON c.id=r.case_fk WHERE r.record_id=?",
            (record_id,),
        ).fetchone()
        if r is None:
            raise KeyError(record_id)
        return cast(Recording, dict(r))

    def clinical_case(self, case_id: str) -> ClinicalCaseView:
        """Return medical attributes/conditions for a reference-case graph.

        The caller binds case_id outside the model payload. Source diagnoses
        remain available here; this method is not a diagnostic-input filter.
        """
        return clinical_content(self.case(case_id))

    def iter_cases(self) -> Iterator[Case]:
        for row in self.db.execute("SELECT case_id FROM cases ORDER BY id"):
            yield self.case(row[0])

    def iter_segments(self, domain_id: str) -> Iterator[Segment]:
        sql = "SELECT s.*,r.record_id,r.eeg_space_id,c.case_id,c.subject_id,c.domain_id FROM segments s JOIN records r ON r.id=s.record_fk JOIN cases c ON c.id=r.case_fk WHERE c.domain_id=? ORDER BY s.record_fk,s.array_row"
        for row in self.db.execute(sql, (domain_id,)):
            yield self._segment_dict(row)

    @staticmethod
    def _segment_dict(row: sqlite3.Row) -> Segment:
        r = dict(row)
        r["segment_id"] = r["record_id"] + f"__seg-{r['array_row']:06d}"
        if r["has_annotated_seizure"] is not None:
            r["has_annotated_seizure"] = bool(r["has_annotated_seizure"])
        return cast(Segment, r)

    def segment(self, segment_id: str) -> Segment:
        record_id, number = segment_id.rsplit("__seg-", 1)
        row = int(number)
        r = self.db.execute(
            "SELECT s.*,r.record_id,r.eeg_space_id,c.case_id,c.subject_id,c.domain_id FROM segments s JOIN records r ON r.id=s.record_fk JOIN cases c ON c.id=r.case_fk WHERE r.record_id=? AND s.array_row=?",
            (record_id, row),
        ).fetchone()
        if r is None:
            raise KeyError(segment_id)
        return self._segment_dict(r)

    def events_for_segment(self, segment_id: str) -> list[Event]:
        s = self.segment(segment_id)
        start = s["onset_seconds"]
        end = start + s["duration_seconds"]
        rows = self.db.execute(
            "SELECT * FROM events WHERE record_fk=? AND onset_seconds<? AND ((duration_seconds>0 AND onset_seconds+duration_seconds>?) OR ((duration_seconds IS NULL OR duration_seconds=0) AND onset_seconds>=?)) ORDER BY onset_seconds,source_event_no",
            (s["record_fk"], end, start, start),
        )
        result = []
        for row in rows:
            e = dict(row)
            e["event_id"] = s["record_id"] + f"__event-{e['source_event_no']:06d}"
            e["details"] = json.loads(e.pop("details_json"))
            result.append(cast(Event, e))
        return result


class DatasetCorpus:
    def __init__(self, dataset: Dataset, benchmark: dict) -> None:
        self.dataset = dataset
        self.benchmark = benchmark
        split = benchmark["patient_split"]
        self.reference = frozenset(split["reference"])
        self.test = frozenset(split["test"])
        self.queries = {row["id"]: Query.from_row(row) for row in benchmark["questions"]}
        self.signature = identity(split)

    @classmethod
    def open(cls, root: str | Path) -> Self:
        root = Path(root)
        benchmark = read_json(root / "questions.json")
        return cls(Dataset(root), benchmark)

    def reference_cases(self) -> Iterator[Case]:
        for case in self.dataset.iter_cases():
            if case["subject_id"] in self.reference:
                yield case

    def reference_segments(self, domain: str) -> Iterator[Segment]:
        for segment in self.dataset.iter_segments(domain):
            if segment["subject_id"] in self.reference:
                yield segment

    def query_subject(self, query: Query) -> str | None:
        return (
            self.dataset.case_subject(query.case_id)
            if query.case_id
            else None
        )

    def close(self) -> None:
        self.dataset.close()


