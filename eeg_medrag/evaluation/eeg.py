"""Reference-only EEG label agreement without loading or recomputing waveforms."""

from collections import Counter, defaultdict

from ..utils import identity
from .protocol import EvaluationConfig, PROTOCOL, aggregate, scoped_rows
from .statistics import bootstrap


SOURCE_LABELS = {
    "alzheimer": {"Alzheimer disease", "frontotemporal dementia", "cognitively normal"},
    "depression": {"current_MDD", "past_MDD", "no_current_or_past_MDD"},
    "mild_tbi": {"mild traumatic brain injury", "control"},
    "parkinson": {"Parkinson disease", "control"},
    "psychiatric": {"first episode psychosis", "control"},
}


def window_label(segment: dict, case: dict):
    domain = segment["domain_id"]
    if domain == "epilepsy":
        value = segment.get("has_annotated_seizure")
        return ("seizure_overlap" if value else "no_seizure_overlap") if value is not None else None
    if domain == "sleep_deprivation":
        value = case["recording_context"].get("sleep_state")
        return value if value in ("normal_sleep", "sleep_deprivation") else None
    if domain not in SOURCE_LABELS:
        raise ValueError(f"No evaluation label definition for {domain}")
    value = case["medical_attributes"].get("source_group")
    return value if value in SOURCE_LABELS[domain] else None


def evaluate_eeg(benchmark: dict, retrieval: dict, dataset,
                 config: EvaluationConfig | None = None) -> dict:
    config = config or EvaluationConfig()
    qs, records = scoped_rows(benchmark, retrieval, "retrievals")
    reference, test = map(set, (benchmark["patient_split"]["reference"], benchmark["patient_split"]["test"]))
    if reference & test:
        raise ValueError("Reference/test subjects overlap")
    queries = {}
    for q in qs:
        if not q["EEG"]:
            continue
        if len(q["EEG"]) != 1:
            raise ValueError("This EEG protocol evaluates single-window queries")
        record = records.get(q["id"])
        if not record or record.get("status", "success") != "success":
            raise ValueError("Finish EEG retrieval before scoring")
        order = record.get("trace", {}).get("candidate_order", {}).get("eeg")
        if not isinstance(order, list):
            raise ValueError("Missing pre-budget EEG candidate_order; do not infer it from displayed evidence")
        if len(order) != len(set(order)):
            raise ValueError("Duplicated EEG neighbor IDs")
        sid = q["EEG"][0]
        previous = queries.get(sid)
        if previous and previous["neighbors"] != order[:config.eeg_k]:
            raise ValueError("The same query window has different neighbors across role variants")
        if previous:
            previous["question_ids"].append(q["id"])
        else:
            queries[sid] = {"question": q, "question_ids": [q["id"]], "neighbors": order[:config.eeg_k]}
    if not queries:
        raise ValueError("No EEG query windows in this scope")
    cases = {}

    def case(cid):
        if cid not in cases:
            cases[cid] = dataset.case(cid)
        return cases[cid]

    domains = {v["question"]["domain"] for v in queries.values()}
    counts = {domain: Counter() for domain in domains}
    totals = Counter()
    for domain in sorted(domains):
        for segment in dataset.iter_segments(domain):
            if segment["subject_id"] not in reference:
                continue
            totals[domain] += 1
            counts[domain][window_label(segment, case(segment["case_id"]))] += 1
        if totals[domain] < config.eeg_k:
            raise ValueError("Reference domain has fewer windows than k")
    rows, clusters = [], {}
    for sid, info in sorted(queries.items()):
        q = info["question"]
        query = dataset.segment(sid)
        if (query["subject_id"] not in test or query["case_id"] != q["case_id"]
                or query["domain_id"] != q["domain"]):
            raise ValueError("Invalid query case, domain or test membership")
        label = window_label(query, case(query["case_id"]))
        neighbors = []
        for neighbor in info["neighbors"]:
            segment = dataset.segment(neighbor)
            if (segment["subject_id"] not in reference or segment["domain_id"] != q["domain"]
                    or segment["eeg_space_id"] != query["eeg_space_id"]):
                raise ValueError("EEG neighbor violates reference/domain/space isolation")
            neighbors.append({"segment_id": neighbor, "subject_id": segment["subject_id"],
                              "case_id": segment["case_id"],
                              "label": window_label(segment, case(segment["case_id"]))})
        hits = sum(label is not None and n["label"] == label for n in neighbors)
        random = counts[q["domain"]][label] / totals[q["domain"]] if label is not None else None
        score = hits / config.eeg_k if label is not None else None
        rows.append({"id": sid, "domain": q["domain"], "query_label": label,
                     "question_ids": info["question_ids"], "neighbors": neighbors,
                     "la5": score, "random_la5": random,
                     "gain_over_random": score - random if score is not None else None,
                     "returned_neighbors": len(neighbors),
                     "known_neighbor_fraction": sum(n["label"] is not None for n in neighbors) / config.eeg_k,
                     "unique_neighbor_subjects": len({n["subject_id"] for n in neighbors}),
                     "unique_neighbor_cases": len({n["case_id"] for n in neighbors})})
        clusters[sid] = {"stratum": (q["domain"], True), "cluster": ("subject", query["subject_id"])}
    groups = defaultdict(list)
    label_groups = defaultdict(list)
    for row in rows:
        groups[row["domain"]].append(row)
        label_groups[row["domain"], row["query_label"]].append(row)
    metrics = ("la5", "random_la5", "gain_over_random", "known_neighbor_fraction")
    return {"protocol": PROTOCOL, "stage": "eeg", "status": "complete", "k": config.eeg_k,
            "benchmark_signature": identity(benchmark), "retrieval_signature": identity(retrieval),
            "overall": aggregate(rows, metrics),
            "domain": {d: aggregate(v, metrics) for d, v in sorted(groups.items())},
            "domain_label": {f"{d}/{label}": aggregate(v, metrics)
                             for (d, label), v in label_groups.items()},
            "per_window": rows, "unknown_query_labels": sum(r["query_label"] is None for r in rows),
            "reference_label_counts": {d: {str(k): v for k, v in c.items()} for d, c in counts.items()},
            "intervals": bootstrap(rows, clusters, metrics[:3], config),
            "random_baseline": "exact expected agreement for uniform reference-window sampling within domain; unknown labels remain in denominator",
            "interpretation": "task-label agreement, not clinical similarity; absent/unknown neighbor slots cannot count as matches"}
