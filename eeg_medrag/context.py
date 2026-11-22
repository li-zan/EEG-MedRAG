"""Budgeted knowledge and case-linked EEG context construction."""

from dataclasses import replace

from .base import Evidence, QueryConfig, RetrievalCandidates, Tokenizer
from .clinical import organize_reference_evidence
from .generation import join_evidence_text
from .prompt import CONTEXT_SECTIONS


def render_context_section(
    section: str, evidence: list[Evidence], candidates: RetrievalCandidates,
) -> list[Evidence]:
    """Render a complete section without changing its candidate records."""
    if not evidence:
        return []
    if section == "clinical":
        result = organize_reference_evidence(
            evidence, candidates.case_texts, candidates.case_order,
        )
    else:
        result = []
        for number, item in enumerate(evidence, 1):
            if section == "entities":
                text = f"Entity {number}: {item.text}"
            elif section == "hyperedges":
                text = f"Fact {number}:\n{item.text}"
            else:
                origins = [
                    origin
                    for source in item.provenance.get("sources", [])
                    for origin in source.get("origins", [])
                ]
                references = []
                for origin in origins:
                    reference = origin["document"]
                    if origin.get("pages"):
                        reference += " (pages " + ", ".join(map(str, origin["pages"])) + ")"
                    references.append(reference)
                source = (
                    "Source: " + "; ".join(dict.fromkeys(references)) + "\n"
                    if references else ""
                )
                text = f"Passage {number}:\n{source}{item.text}"
            result.append(replace(item, text=text))
    result[0] = replace(
        result[0], text=CONTEXT_SECTIONS[section].format(content=result[0].text),
    )
    return result


class ContextBuilder:
    """Select whole evidence units using the same rendering as the final prompt."""

    def __init__(
        self, candidates: RetrievalCandidates, tokenizer: Tokenizer, config: QueryConfig,
    ) -> None:
        self.candidates, self.tokenizer, self.config = candidates, tokenizer, config
        self.pools = {
            "entities": candidates.entities,
            "hyperedges": candidates.hyperedges,
            "chunks": candidates.chunks,
            "clinical": [*candidates.cases, *candidates.eeg],
        }
        self.limits = {
            "entities": config.entity_context_tokens,
            "hyperedges": config.hyperedge_context_tokens,
            "chunks": config.source_context_tokens,
            "clinical": config.clinical_context_tokens,
        }
        active = [section for section, pool in self.pools.items() if pool]
        released = sum(limit for section, limit in self.limits.items() if section not in active)
        active_total = sum(self.limits[section] for section in active)
        self.limits = {
            section: limit + released * limit // active_total if section in active else 0
            for section, limit in self.limits.items()
        }
        self.selected: dict[str, list[Evidence]] = {section: [] for section in self.pools}
        self.rendered: dict[str, list[Evidence]] = {section: [] for section in self.pools}

    def _try_add(self, section: str, additions: list[Evidence]) -> bool:
        trial = [*self.selected[section], *additions]
        rendered = render_context_section(section, trial, self.candidates)
        # Charge each inter-section separator to the following section.
        prefix = "\n\n" if any(self.rendered[key] for key in self.pools if key != section) else ""
        if self.tokenizer.count(prefix + join_evidence_text(rendered)) > self.limits[section]:
            return False
        combined = [
            item for key in self.pools
            for item in (rendered if key == section else self.rendered[key])
        ]
        if self.tokenizer.count(join_evidence_text(combined)) > self.config.context_tokens:
            return False
        self.selected[section], self.rendered[section] = trial, rendered
        return True

    def build(self) -> tuple[list[Evidence], dict]:
        for section in ("entities", "hyperedges", "chunks"):
            for item in self.pools[section]:
                self._try_add(section, [item])

        backgrounds: set[str] = set()
        windows: set[str] = set()
        cases = {item.id: item for item in self.candidates.cases}
        eeg = sorted(self.candidates.eeg, key=lambda item: (item.provenance["distance"], item.id))

        def add_window(item: Evidence) -> bool:
            cid = item.provenance["case_id"]
            additions = [cases[cid]] if cid not in backgrounds and cid in cases else []
            if not self._try_add("clinical", [*additions, item]):
                return False
            backgrounds.add(cid)
            windows.add(item.id)
            return True

        for item in eeg:
            if add_window(item):
                break
        for cid in self.candidates.case_order:
            if cid in cases and cid not in backgrounds and self._try_add("clinical", [cases[cid]]):
                backgrounds.add(cid)
        for item in eeg:
            if item.id not in windows:
                add_window(item)

        evidence = [item for section in self.pools for item in self.rendered[section]]
        used = {}
        has_previous = False
        for section, items in self.rendered.items():
            text = join_evidence_text(items)
            used[section] = self.tokenizer.count(("\n\n" if has_previous else "") + text) if items else 0
            has_previous = has_previous or bool(items)
        trace = {
            "context_tokens": self.tokenizer.count(join_evidence_text(evidence)),
            "context_budget": {"limits": self.limits, "used": used},
        }
        return evidence, trace
