"""Configuration and interchange schemas for EEG-MedRAG."""

from collections.abc import Callable
from typing import Any, Literal, NotRequired, Protocol, Self, TypeAlias, TypedDict

import numpy as np

from dataclasses import dataclass, field

from .utils import canonical

# Framework interchange objects

@dataclass
class Document:
    id: str
    text: str
    source: dict


@dataclass
class Query:
    """Allowlisted task input. Gold, answer_source and full cases cannot enter."""

    id: str
    domain: str
    role: str
    question: str
    masked_case: dict | None
    case_id: str | None
    eeg: tuple[str, ...] = ()

    @classmethod
    def from_row(cls, row: dict) -> Self:
        return cls(
            row["id"], row["domain"], row["role"], row["question"],
            row["masked_case"], row["case_id"], tuple(row["EEG"]),
        )

    def semantic_text(self) -> str:
        return self.question + "\n" + canonical(self.masked_case or {})


@dataclass
class AttributeQuery:
    """One medical fact and its measurement constraints for case matching."""

    attribute: str
    value: object
    name: str | None = None
    qualifiers: dict[str, object] = field(default_factory=dict)
    primary: bool = True
    sources: tuple[str, ...] = ()


@dataclass
class Evidence:
    id: str
    kind: str
    text: str
    score: float
    provenance: dict = field(default_factory=dict)
    components: dict = field(default_factory=dict)


@dataclass
class RetrievalCandidates:
    """Complete ranked lists before context selection; scores are source-specific."""

    entities: list[Evidence] = field(default_factory=list)
    hyperedges: list[Evidence] = field(default_factory=list)
    cases: list[Evidence] = field(default_factory=list)
    eeg: list[Evidence] = field(default_factory=list)
    chunks: list[Evidence] = field(default_factory=list)
    case_texts: dict[str, str] = field(default_factory=dict)
    case_order: list[str] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


class ModelProvider(Protocol):
    """Completion/embedding provider; matching implementations need no inheritance."""

    signature: str

    async def complete(
        self, messages: list[dict], validator: Callable[[str], Any] | None,
        **parameters,
    ) -> Any: ...

    async def embed(self, texts: list[str]) -> np.ndarray: ...


class TextModel(Protocol):
    """Text-only model interface consumed by the shared answer generator."""

    signature: str

    async def text(
        self, messages: list[dict], cache_namespace: str,
        validator: Callable[[str], object], **parameters,
    ) -> str: ...


class Tokenizer(Protocol):
    """Token counting/chunking interface used during answering and retrieval."""

    def count(self, text: str) -> int: ...

    def chunks(self, text: str, size: int, overlap: int) -> list[dict]: ...


# Experiment configuration

@dataclass
class BuildConfig:
    """Construction parameters."""

    chunk_tokens: int = 1200
    overlap_tokens: int = 100
    tokenizer_model: str = "gpt-4o-mini"
    concurrency: int = 16
    entity_summary_to_max_tokens: int = 500
    llm_model_max_token_size: int = 32768
    cosine_better_than_threshold: float = 0.2
    entity_extract_max_gleaning: int = 2
    softlink_k: int = 8
    paa_segments: int = 1920
    z_normalize: bool = False
    dtw_window_seconds: float = 0.25


@dataclass
class QueryConfig:
    """Retrieval parameters."""

    top_k: int = 5
    case_text_top_k: int = 5
    case_attribute_top_k: int = 5
    case_top_k: int = 5
    case_rrf_k: int = 60
    entity_top_k: int = 60
    hyperedge_top_k: int = 60
    candidate_pool: int = 240
    expansion_radius: int = 1
    mmr_lambda: float = 0.5
    entity_context_tokens: int = 4000
    hyperedge_context_tokens: int = 6000
    source_context_tokens: int = 10000
    clinical_context_tokens: int = 12000
    context_tokens: int = 32000


@dataclass(frozen=True)
class GenerationConfig:
    """Shared answer decoding policy, independent of retrieval and judging."""

    temperature: float = 0.0
    max_tokens: int = 4096

    def request_parameters(self, model_id: str | None) -> dict:
        """Return shared decoding parameters and model-specific overrides."""
        parameters = {"temperature": self.temperature, "max_tokens": self.max_tokens}
        if model_id == "gemini-2.5-flash-lite":
            parameters["reasoning_effort"] = "none"
        # DeepSeek uses the server's default thinking mode.
        # elif model_id == "deepseek-v4-flash-260425":
        #     parameters["extra_body"] = {"thinking": {"type": "disabled"}}
        return parameters


@dataclass
class LLMConfig:
    """Model and request parameters."""

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_max_token_size: int = 8191
    max_attempts: int = 3
    concurrency: int = 16
    embedding_batch_size: int = 16
    backoff_seconds: float = 4


# Published dataset structures

JsonValue: TypeAlias = (
    str | int | float | bool | None
    | list["JsonValue"] | dict[str, "JsonValue"]
)


class Demographics(TypedDict):
    age_years: float | None
    sex: str | None


class Assessment(TypedDict):
    name: str
    value: float | str | None
    unit: str | None
    description: str | None
    scope: str
    source_field: str
    instrument: NotRequired[str]
    level_description: NotRequired[str | None]
    raw_value: NotRequired[JsonValue]
    value_status: NotRequired[str]


class SourceNote(TypedDict):
    field: str
    value: JsonValue


class MedicalAttributes(TypedDict):
    demographics: Demographics
    source_group: str
    diagnosis: str | None
    diagnosis_basis: str
    source_notes: list[SourceNote]
    assessments: list[Assessment]


class RecordingContext(TypedDict):
    task: str
    eyes: NotRequired[str | None]
    sleep_state: NotRequired[str | None]
    medication_state: NotRequired[str | None]
    visit_stage: NotRequired[str | None]
    session_order: NotRequired[str | None]
    collection_site: NotRequired[str]


class EEGMetadata(TypedDict):
    signal_path: str
    eeg_space_id: str
    shape: list[int]
    unit: str
    sampling_rate_hz: float
    source_duration_seconds: float
    seizure_annotation_available: bool


class SourceReference(TypedDict):
    dataset_id: str
    source_record_id: str


class Case(TypedDict):
    """Complete published case; engineering fields are not model input."""

    schema_version: str
    case_id: str
    subject_id: str
    domain_id: str
    dataset_id: str
    record_id: str
    source_subject_id: str
    session_id: str | None
    run: str | None
    medical_attributes: MedicalAttributes
    recording_context: RecordingContext
    eeg: EEGMetadata
    source_reference: SourceReference
    case_text: str


class Recording(TypedDict):
    """Recording row joined to its case identity, as returned by Dataset._record."""

    id: int
    record_id: str
    case_fk: int
    eeg_space_id: str
    signal_path: str
    n_segments: int
    n_channels: int
    n_samples: int
    sampling_rate_hz: float
    unit: str
    dtype: str
    signal_sha256: str
    signal_bytes: int
    case_id: str
    subject_id: str
    domain_id: str


class Segment(TypedDict):
    """Joined window record; segment_id and nullable seizure flags are decoded."""

    id: int
    record_fk: int
    array_row: int
    onset_seconds: float
    duration_seconds: float
    eyes: str | None
    has_annotated_seizure: bool | None
    seizure_overlap_seconds: float | None
    record_id: str
    eeg_space_id: str
    case_id: str
    subject_id: str
    domain_id: str
    segment_id: str


class EventDetails(TypedDict, total=False):
    lateralization: str
    localization: str
    vigilance: str
    confidence: JsonValue
    channels: JsonValue
    trial_type: str
    value: JsonValue
    response_time: JsonValue
    stim_file: JsonValue


class Event(TypedDict):
    """Source event with a derived event_id and parsed details, not details_json."""

    id: int
    record_fk: int
    source_event_no: int
    onset_seconds: float
    duration_seconds: float | None
    kind: str
    event_code: str | None
    description: str | None
    event_id: str
    details: EventDetails


class SegmentEvidence(TypedDict):
    """Source-bound window text from Dataset.evidence, before retrieval scoring."""

    case_id: str
    subject_id: str
    domain_id: str
    record_id: str
    segment_id: str
    eeg_space_id: str
    binding_level: Literal["event", "record"]
    event_ids: list[str]
    text: str


# Filtered reference-case medical views

class ClinicalDemographics(TypedDict, total=False):
    age_years: float
    sex: str


class ClinicalAssessment(TypedDict):
    name: str
    value: float | str
    instrument: NotRequired[str]
    unit: NotRequired[str]
    description: NotRequired[str]
    scope: NotRequired[str]
    level_description: NotRequired[str]


class ClinicalRecordingConditions(TypedDict, total=False):
    task: str
    eyes: str
    sleep_state: str
    medication_state: str
    visit_stage: str
    session_order: str


class ClinicalCaseView(TypedDict, total=False):
    """Structured reference-case attributes; not a query mask."""

    demographics: ClinicalDemographics
    source_group: str
    diagnosis: str
    source_notes: list[JsonValue]
    assessments: list[ClinicalAssessment]
    recording_conditions: ClinicalRecordingConditions
