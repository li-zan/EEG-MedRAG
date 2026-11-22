"""Medical case views, attribute queries and reference EEG evidence text."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import cast

from .base import AttributeQuery, Case, ClinicalCaseView, Event, Evidence, Segment
from .utils import canonical

EMPTY = {
    "",
    "n/a",
    "na",
    "none",
    "null",
    "not_provided",
    "not_applicable",
    "not_specified_by_source",
    "unknown",
}


def provided(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in EMPTY
    return True


def display(value: object) -> str:
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, (dict, list)):
        return canonical(value)
    return str(value)


def clinical_content(case: Case) -> ClinicalCaseView:
    """Medical content for case-graph construction; no identity or file metadata.

    Includes source diagnoses/labels. This is a reference-case view, not a
    task-specific diagnostic-question input and not a benchmark/gold generator.
    The structured view supplies case attributes. Case vectors and display text
    use the published case_text. Query inputs use the benchmark masked_case.
    """
    a = case["medical_attributes"]
    result: dict = {}
    demographics = {k: v for k, v in a["demographics"].items() if provided(v)}
    if demographics:
        result["demographics"] = demographics
    for key in ["source_group", "diagnosis"]:
        value = a.get(key)
        if provided(value) and value not in ([], {}):
            result[key] = value
    notes = [n["value"] for n in a.get("source_notes", []) if provided(n.get("value"))]
    if notes:
        result["source_notes"] = notes
    assessments = []
    for item in a["assessments"]:
        if (
            not provided(item.get("value"))
            or item.get("value_status") == "not_assessed"
        ):
            continue
        assessments.append(
            {
                k: item.get(k)
                for k in [
                    "name",
                    "instrument",
                    "value",
                    "unit",
                    "description",
                    "scope",
                    "level_description",
                ]
                if k in item and provided(item.get(k))
            }
        )
    if assessments:
        result["assessments"] = assessments
    context = {
        k: v
        for k, v in case["recording_context"].items()
        if k
        in {
            "task",
            "eyes",
            "sleep_state",
            "medication_state",
            "visit_stage",
            "session_order",
        }
        and provided(v)
    }
    if context:
        result["recording_conditions"] = context
    return cast(ClinicalCaseView, result)


# Query fields refer to the medical view used by CHG construction.
_CASE_FIELDS = {
    "age_years": "/demographics/age_years",
    "sex": "/demographics/sex",
    "eyes": "/recording_conditions/eyes",
    "task": "/recording_conditions/task",
    "diagnosis": "/diagnosis",
    "documented_diagnosis": "/diagnosis",
    "presentation": "/diagnosis",
    "medication_condition": "/recording_conditions/medication_state",
    "sleep_condition": "/recording_conditions/sleep_state",
    "visit_stage": "/recording_conditions/visit_stage",
}
_CASE_ASSESSMENTS = {
    "MMSE": "MMSE", "BDI": "BDI", "STAI": "STAI",
    "SCID_result": "SCID", "diagnostic_history": "SCID", "HamD": "HamD",
    "BPRS_total_items_1_to_18": "bprs.BPRST18",
    "BPRS_total_items_1_to_19": "bprs.BPRST19",
    "SAPS_auditory_hallucinations": "saps.SAPS_Q1",
    "SANS_global_affective_flattening": "sans.SANS_Q7",
    "role_functioning_current_0_to_10": "gafgas.ROLECURR",
    "role_functioning_highest_lifetime_0_to_10": "gafgas.ROLEHIGH",
    "social_functioning_current_0_to_10": "gafgas.SOCIALCURR",
    "WASI_two_subtest_IQ_estimate": "wasi.FULL2IQ",
    "BPRS_hallucinatory_behavior": "bprs.bprs_q12",
    "PSQI_global_score": "PSQI_GlobalScore",
    "sleep_diary_quality_score": "SleepDiary_item3_NS",
    "living_with": "sfs.SFS_Q2_LIVEWHO",
}
_SLEEP_ASSESSMENTS = {
    "PVT_lapse_count": "PVT_item1", "SAI_score": "SAI",
    "SSS_score": "SSS", "PANAS_positive_affect": "PANAS_P",
    "PANAS_negative_affect": "PANAS_N", "KSS_score": "KSS",
}
_PSQI_COMPONENTS = (
    "subjective_sleep_quality", "sleep_latency", "sleep_duration",
    "sleep_efficiency", "sleep_disturbance", "sleep_medication_use",
    "daytime_dysfunction",
)
_CASE_LABELS = {
    "resting eeg": "rest",
    "seizure monitoring": "szmonitoring",
    "not applicable for the source control cohort": "not applicable control",
    "current major depressive disorder": "major depressive disorder",
}


def _attribute_value(value: object, controlled: bool = False) -> tuple:
    """Compare numeric values without conflating booleans and numbers."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        return ("number", value)
    if isinstance(value, str):
        text = value.replace("_", " ") if controlled else value
        text = " ".join(text.casefold().split())
        return ("str", _CASE_LABELS.get(text, text) if controlled else text)
    return ("json", canonical(value))


def query_attributes(masked_case: Mapping | None) -> list[AttributeQuery]:
    """Translate allowed query fields into CHG medical predicates."""
    facts: dict[tuple, AttributeQuery] = {}

    def add(
        attribute: str, value: object, source: str, *, name: str | None = None,
        qualifiers: dict[str, object] | None = None, primary: bool = True,
    ) -> None:
        qualifiers = qualifiers or {}
        if value is None and not qualifiers:
            return
        if value is not None and not provided(value):
            return
        key = (attribute, name, qualifiers.get("scope"))
        if key in facts:
            fact = facts[key]
            if fact.value is None:
                fact.value = value
            fact.qualifiers.update(qualifiers)
            fact.primary = fact.primary or primary
            fact.sources += (source,)
        else:
            facts[key] = AttributeQuery(
                attribute, value, name, qualifiers, primary, (source,),
            )

    for field, value in (masked_case or {}).items():
        if field in _CASE_FIELDS:
            path = _CASE_FIELDS[field]
            add(
                path, value, field,
                primary=path == "/diagnosis" or path.endswith(
                    ("/medication_state", "/sleep_state", "/visit_stage")
                ),
            )
        elif field in _CASE_ASSESSMENTS:
            name = _CASE_ASSESSMENTS[field]
            qualifiers = {"scope": "normal_sleep"} if name.startswith("SleepDiary_") else {}
            if isinstance(value, dict):
                for key, target in (
                    ("assessment_scope", "scope"), ("recorded_severity", "level_description"),
                ):
                    if key in value:
                        qualifiers[target] = value[key]
                value = value["score"]
            add("/assessments", value, field, name=name, qualifiers=qualifiers)
        elif field in _SLEEP_ASSESSMENTS:
            state = (masked_case or {}).get("sleep_condition")
            suffix = {"normal_sleep": "NS", "sleep_deprivation": "SD"}.get(state)
            if suffix:
                add(
                    "/assessments", value, field,
                    name=_SLEEP_ASSESSMENTS[field] + "_" + suffix,
                    qualifiers={"scope": state},
                )
        elif field == "PSQI_component_scores":
            for component, score in value.items():
                add(
                    "/assessments", score, field + "/" + component,
                    name="PSQI_item" + str(_PSQI_COMPONENTS.index(component) + 1),
                    qualifiers={"scope": "participant"},
                )
        elif field == "Trail_Making_Test":
            for key, name in (
                ("raw_time_seconds", "matrics.TMTRAW"),
                ("age_and_gender_corrected_T_score", "matrics.TMTTSCR"),
            ):
                if key in value:
                    add(
                        "/assessments", value[key], field + "/" + key, name=name,
                        qualifiers={"unit": "seconds"} if key == "raw_time_seconds" else {},
                    )
        elif field == "employment_assessment":
            qualifiers = (
                {"level_description": value["recorded_status"]}
                if "recorded_status" in value else {}
            )
            add(
                "/assessments", value["recorded_code"], field,
                name="sfs.SFS_Q14", qualifiers=qualifiers,
            )
        elif field == "recorded_employment_status":
            add(
                "/assessments", None, field, name="sfs.SFS_Q14",
                qualifiers={"level_description": value},
            )
        elif field == "sleep_diary":
            for key, name in (
                ("bedtime_24h", "SleepDiary_item1_NS"),
                ("out_of_bed_time_24h", "SleepDiary_item2_NS"),
            ):
                if key in value:
                    add(
                        "/assessments", value[key], field + "/" + key,
                        name=name, qualifiers={"scope": "normal_sleep"},
                    )
    return list(facts.values())


def attribute_matches(query: AttributeQuery, attribute: Mapping) -> bool:
    """Match a query fact against an existing CHG attribute representation."""
    if query.attribute != attribute["attribute"]:
        return False
    value = attribute["value"]
    if query.name is None:
        return _attribute_value(query.value, True) == _attribute_value(value, True)
    if not isinstance(value, dict) or value.get("name") != query.name:
        return False
    return (
        (query.value is None or _attribute_value(query.value) == _attribute_value(value["value"]))
        and all(
            _attribute_value(expected) == _attribute_value(value.get(key))
            for key, expected in query.qualifiers.items()
        )
    )


def organize_reference_evidence(
    evidence: Iterable[Evidence], case_texts: Mapping[str, str], case_order: Iterable[str],
) -> list[Evidence]:
    """Group selected hits by case rank, sharing background and ordering windows by DTW."""
    groups: dict[str, list[Evidence]] = {}
    knowledge = []
    for item in evidence:
        if item.kind in {"case", "eeg"}:
            case_id = item.id if item.kind == "case" else item.provenance["case_id"]
            groups.setdefault(case_id, []).append(item)
        else:
            knowledge.append(item)

    case_ranks = {case_id: rank for rank, case_id in enumerate(case_order)}
    result = []
    for number, case_id in enumerate(sorted(groups, key=case_ranks.__getitem__), 1):
        background = case_texts[case_id]
        items = sorted(groups[case_id], key=lambda item: (
            item.kind != "case",
            item.provenance["distance"] if item.kind == "eeg" else 0.0,
            item.id,
        ))
        window_number = 0
        for index, item in enumerate(items):
            text = (
                "" if item.kind == "case"
                else item.text.removeprefix(background).removeprefix("\n")
            )
            if item.kind == "eeg":
                window_number += 1
                text = f"EEG window {window_number}:\n{text}"
            if index == 0:
                heading = f"Reference case {number}:\n{background}"
                text = heading + ("\n\n" + text if text else "")
            result.append(replace(item, text=text))
    return [*result, *knowledge]


def event_text(event: Event, start: float, stop: float) -> str:
    onset = event["onset_seconds"]
    duration = event["duration_seconds"]
    kind = event["kind"]
    description = event.get("description")
    label = description or (
        "source-annotated seizure"
        if kind == "seizure"
        else "source-annotated background"
        if kind == "annotated_background"
        else "source task condition"
    )
    text = label + ": onset " + display(onset) + " s"
    if duration is not None:
        text += ", duration " + display(duration) + " s"
        overlap = max(0.0, min(stop, onset + duration) - max(start, onset))
        text += ", overlap with this window " + display(overlap) + " s"
    clinical_keys = {
        "lateralization",
        "localization",
        "vigilance",
        "confidence",
        "channels",
        "trial_type",
        "response_time",
    }
    details = {k: v for k, v in event["details"].items() if k in clinical_keys}
    if details:
        text += "; " + ", ".join(k + ": " + display(v) for k, v in details.items())
    return text + "."


def render_eeg_evidence(case: Case, segment: Segment, events: Iterable[Event]) -> str:
    """Combine published case_text, window conditions and overlapping source events."""
    start = segment["onset_seconds"]
    stop = start + segment["duration_seconds"]
    parts = [
        case["case_text"],
        "EEG window: "
        + display(start)
        + " to "
        + display(stop)
        + " seconds of the source recording.",
    ]
    eyes = segment.get("eyes")
    if provided(eyes):
        parts.append("Window eye condition: " + str(eyes).replace("_", " ") + ".")
    flag = segment.get("has_annotated_seizure")
    if flag is True:
        parts.append(
            "Overlap with source-annotated seizures (union): "
            + display(segment["seizure_overlap_seconds"])
            + " seconds."
        )
    elif flag is False:
        parts.append(
            "This window does not overlap the source-annotated seizure intervals."
        )
    eye_states = set()
    for event in events:
        trial = (
            str((event.get("details") or {}).get("trial_type") or "").strip().lower()
        )
        if event["kind"] == "task_event" and trial.startswith(
            ("eyes open", "eyes closed")
        ):
            eye_states.add("open" if trial.startswith("eyes open") else "closed")
        else:
            parts.append(event_text(event, start, stop))
    if eye_states:
        parts.append(
            "Source eye-condition annotations in this window: "
            + ", ".join(sorted(eye_states))
            + "."
        )
    return "\n".join(parts)
