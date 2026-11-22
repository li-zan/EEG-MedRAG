"""Shared evaluation scope, context rendering and checkpoint contracts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import random
import re

from ..generation import effective_context
from ..utils import atomic_json, identity, read_json


PROTOCOL = "eeg-medrag-evaluation-v1"
KNOWLEDGE_TYPES = {"foundational_knowledge", "knowledge_application"}
GROUPS = ("domain", "question_type", "role", "task")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    judge_model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 8192
    relevance_sample_size: int = 24
    relevance_seed: int = 20261022
    bootstrap_repetitions: int = 2000
    seed: int = 20261022
    eeg_k: int = 5


def indexed(rows: list[dict], *, name: str = "rows") -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Duplicate or invalid ID in {name}: {key!r}")
        result[key] = row
    return result


def questions(benchmark: dict, ids: list[str] | None = None) -> list[dict]:
    by_id = indexed(benchmark["questions"], name="benchmark")
    ids = list(by_id) if ids is None else list(ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) - by_id.keys():
        raise ValueError("Evaluation scope must contain unique, known question IDs")
    for qid in ids:
        gold = by_id[qid]["golden_answers"]
        if not gold or not all(isinstance(a, str) and a.strip() for a in gold):
            raise ValueError(f"Missing reference answer: {qid}")
    return [by_id[qid] for qid in ids]


def scoped_rows(benchmark: dict, artifact: dict, field: str) -> tuple[list[dict], dict]:
    qs = questions(benchmark, artifact["question_ids"])
    if set(artifact["question_ids"]) != {q["id"] for q in benchmark["questions"]}:
        raise ValueError("Evaluation requires the full benchmark question_ids")
    rows = indexed(artifact[field], name=field)
    if set(rows) - set(artifact["question_ids"]):
        raise ValueError(f"{field} contains rows outside declared scope")
    return qs, rows


def question_view(q: dict, *, gold: bool = False) -> dict:
    # IDs, disease routing, hidden case records and answer provenance stay out.
    value = {k: q[k] for k in ("question", "role", "masked_case")}
    if gold:
        value["reference_answers"] = q["golden_answers"]
    return value


def metadata(q: dict) -> dict:
    return {k: q[k] for k in ("id", "family_id", "case_id", *GROUPS)}


def context_units(text: str) -> list[dict]:
    """Deterministic line/sentence spans; do not split decimals or rewrite text."""
    units = []
    for line in re.finditer(r"[^\r\n]+", text):
        value = line.group()
        if re.fullmatch(r"\s*(?:---.*---|Reference case \d+:|EEG window \d+:)\s*", value):
            continue
        starts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", value)]
        for start, end in zip(starts, starts[1:] + [len(value)]):
            part = value[start:end]
            stripped = part.strip()
            if not stripped:
                continue
            begin = line.start() + start + len(part) - len(part.lstrip())
            units.append({"id": f"u{len(units) + 1:04d}", "text": stripped,
                          "start": begin, "end": begin + len(stripped)})
    return units


def sample_relevance_units(units: list[dict], question_id: str, config: EvaluationConfig) -> list[dict]:
    """Draw units uniformly without replacement and present them in source order."""
    if len(units) <= config.relevance_sample_size:
        return units
    rng = random.Random(f"{config.relevance_seed}:{question_id}")
    positions = sorted(rng.sample(range(len(units)), config.relevance_sample_size))
    return [units[i] for i in positions]


def aggregate(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    value = {"n": len(rows)}
    for metric in metrics:
        scores = [r[metric] for r in rows if r.get(metric) is not None]
        value[metric] = sum(scores) / len(scores) if scores else None
        value[metric + "_n"] = len(scores)
    return value


def summarize(rows: list[dict], metrics: tuple[str, ...]) -> dict:
    from collections import defaultdict
    output = {"overall": aggregate(rows, metrics)}
    for field in GROUPS:
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        output[field] = {key: aggregate(values, metrics) for key, values in sorted(groups.items())}
        output[field + "_macro"] = {
            metric: aggregate(list(output[field].values()), (metric,))[metric]
            for metric in metrics
        }
    return output


class Checkpoint:
    """One immutable input identity per output path; successful rows are resumable."""

    def __init__(self, path: str | Path, stage: str, qs: list[dict], inputs: object, **extra):
        self.path, self.qs = Path(path), qs
        self.header = {"protocol": PROTOCOL, "stage": stage,
                       "run_id": identity([PROTOCOL, stage, qs, inputs]),
                       "question_ids": [q["id"] for q in qs], **extra}
        previous = read_json(self.path) if self.path.exists() else {}
        if previous and previous.get("run_id") != self.header["run_id"]:
            raise ValueError("Evaluation inputs changed; choose a new output path")
        self.rows = indexed(previous.get("per_question", []), name="checkpoint")
        if set(self.rows) - set(self.header["question_ids"]):
            raise ValueError("Checkpoint contains out-of-scope rows")

    def done(self, qid: str) -> bool:
        return self.rows.get(qid, {}).get("status") == "success"

    def save(self, q: dict, **result) -> None:
        self.rows[q["id"]] = {**metadata(q), "status": "success", **result}
        self.export()
        logger.info("Evaluation checkpoint stage=%s question=%s completed=%d/%d",
                    self.header["stage"], q["id"], sum(self.done(k) for k in self.rows), len(self.qs))

    def failed(self, q: dict, error: Exception) -> None:
        self.rows[q["id"]] = {**metadata(q), "status": "error", "error_type": type(error).__name__}
        self.export()

    def export(self, **extra) -> dict:
        complete = all(self.done(q["id"]) for q in self.qs)
        result = {**self.header, "status": "complete" if complete else "partial",
                  "per_question": [self.rows[q["id"]] for q in self.qs if q["id"] in self.rows], **extra}
        atomic_json(self.path, result)
        return result


def rubric_signature(rubric: dict) -> str:
    return identity({k: rubric[k] for k in ("protocol", "benchmark_signature", "per_question")})


def validate_rubric(rubric: dict, benchmark: dict) -> dict:
    if (rubric.get("protocol") != PROTOCOL or rubric.get("status") != "complete"
            or rubric.get("benchmark_signature") != identity(benchmark)
            or rubric.get("rubric_signature") != rubric_signature(rubric)):
        raise ValueError("Rubric is incomplete, changed, or belongs to another benchmark")
    rows = indexed(rubric["per_question"], name="rubric")
    eligible = {q["id"] for q in benchmark["questions"] if q["question_type"] in KNOWLEDGE_TYPES}
    if set(rows) != eligible or any(r.get("status") != "success" for r in rows.values()):
        raise ValueError("Rubric must cover all knowledge questions before method evaluation")
    return rows
