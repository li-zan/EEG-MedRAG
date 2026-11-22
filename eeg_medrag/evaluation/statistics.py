"""Patient/family-clustered intervals and paired method differences."""

from collections import defaultdict

import numpy as np

from ..utils import identity
from ..result_store import FINAL_PREDICTION_STATES
from .protocol import EvaluationConfig, PROTOCOL, indexed, questions


def cluster_map(benchmark: dict, dataset) -> dict:
    result = {}
    for q in questions(benchmark):
        subject = dataset.case_subject(q["case_id"]) if q["case_id"] else None
        result[q["id"]] = {
            "stratum": (q["domain"], subject is not None),
            "cluster": ("subject", subject) if subject else ("family", q["family_id"]),
        }
    return result


def bootstrap(rows: list[dict], clusters: dict, metrics: tuple[str, ...],
              config: EvaluationConfig | None = None) -> dict:
    config = config or EvaluationConfig()
    output = {"repetitions": config.bootstrap_repetitions, "seed": config.seed,
              "confidence": 0.95, "unit": "subject or family; domain/patient-linkage strata"}
    for metric in metrics:
        eligible = [r for r in rows if r.get(metric) is not None]
        if not eligible:
            output[metric + "_95"] = None
            continue
        strata = defaultdict(lambda: defaultdict(list))
        for row in eligible:
            c = clusters[row["id"]]
            strata[tuple(c["stratum"])][tuple(c["cluster"])].append(float(row[metric]))
        rng = np.random.default_rng(config.seed)
        samples = np.zeros(config.bootstrap_repetitions)
        nclusters = 0
        for groups in strata.values():
            blocks = list(groups.values())
            nclusters += len(blocks)
            sums = np.array([sum(b) for b in blocks])
            sizes = np.array([len(b) for b in blocks])
            indices = rng.integers(0, len(blocks), size=(config.bootstrap_repetitions, len(blocks)))
            samples += (sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
                        * sizes.sum() / len(eligible))
        output[metric + "_95"] = np.quantile(samples, [0.025, 0.975]).tolist()
        output[metric + "_clusters"] = nclusters
    return output


def validate_report(report: dict) -> dict:
    if report.get("protocol") != PROTOCOL or report.get("status") != "complete":
        raise ValueError("Use a complete semantic evaluation report")
    rows = indexed(report["per_question"], name="report")
    if set(rows) != set(report["question_ids"]) or any(r.get("status") != "success" for r in rows.values()):
        raise ValueError("Report has incomplete judgments")
    if report["stage"] == "answers" and any(
        r.get("prediction_status") not in FINAL_PREDICTION_STATES
        or (r["prediction_status"] == "success" and r["empty_answer"]) for r in rows.values()
    ):
        raise ValueError("Answer report includes unfinished predictions; resume generation first")
    return rows


def paired_comparison(left: dict, right: dict, benchmark: dict, dataset,
                      config: EvaluationConfig | None = None) -> dict:
    a, b = validate_report(left), validate_report(right)
    if left["stage"] not in ("answers", "retrieval"):
        raise ValueError("Paired comparison expects answer or retrieval reports")
    for key in ("benchmark_signature", "stage", "judge", "rubric_signature", "generation_signature",
                "generation_policy", "answer_model"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Paired evaluation requires matching {key}")
    if (set(a) != set(b) or set(a) != {q["id"] for q in benchmark["questions"]}
            or left["benchmark_signature"] != identity(benchmark)):
        raise ValueError("Paired evaluation requires identical question scope and benchmark")
    metrics = ("ac", "f1") if left["stage"] == "answers" else ("ec", "crel")
    rows = []
    for q in questions(benchmark, list(a)):
        row = {"id": q["id"]}
        for metric in metrics:
            av, bv = a[q["id"]].get(metric), b[q["id"]].get(metric)
            if (av is None) != (bv is None):
                raise ValueError("Metric applicability differs between methods")
            row[metric] = None if av is None else av - bv
        rows.append(row)
    return {"protocol": PROTOCOL, "direction": "left minus right", "stage": left["stage"],
            "left_run_id": left["run_id"], "right_run_id": right["run_id"],
            "n": len(rows),
            "difference": {m: float(np.mean([r[m] for r in rows if r[m] is not None]))
                           if any(r[m] is not None for r in rows) else None for m in metrics},
            "intervals": bootstrap(rows, cluster_map(benchmark, dataset), metrics, config)}
