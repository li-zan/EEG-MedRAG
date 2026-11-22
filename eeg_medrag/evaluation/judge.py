"""Validated JSON judgments backed by the existing text request cache."""

from __future__ import annotations

from dataclasses import asdict
import json

from ..utils import IntegrityError, canonical, identity
from ..request_usage import request_context
from . import prompts
from .protocol import EvaluationConfig, PROTOCOL


def object_response(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise IntegrityError("Expected a JSON object")
    return value


def reason(row: dict) -> None:
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise IntegrityError("Missing judgment reason")


def boolean_result(row: dict, key: str) -> None:
    if type(row.get(key)) is not bool:
        raise IntegrityError(f"{key} must be a JSON boolean")
    reason(row)


def exact_rows(value: dict, field: str, ids: list[str]) -> list[dict]:
    rows = value.get(field)
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise IntegrityError(f"Missing {field} list")
    actual = [r.get("id") for r in rows]
    if len(actual) != len(ids) or set(actual) != set(ids):
        raise IntegrityError(f"Incomplete or duplicated {field} judgments")
    by_id = {r["id"]: r for r in rows}
    return [by_id[k] for k in ids]


def relevant_targets(value: dict, ids: list[str]) -> set[str]:
    reason(value)
    selected = value.get("relevant_unit_ids")
    if not isinstance(selected, list) or not all(isinstance(i, str) for i in selected):
        raise IntegrityError("relevant_unit_ids must be a list of target IDs")
    if len(set(selected)) != len(selected) or set(selected) - set(ids):
        raise IntegrityError("Select unique IDs from target_units only")
    return set(selected)


class Judge:
    def __init__(self, model, config: EvaluationConfig | None = None):
        self.model = model
        self.config = config or EvaluationConfig()
        self._model_descriptor = {
            "model": getattr(model, "chat_descriptor", {"signature": model.signature}),
            "model_signature": getattr(model, "chat_signature", model.signature),
        }

    def descriptor_for(self, stage: str) -> dict:
        instructions = {"rubric": [prompts.EXTRACT, prompts.REVIEW],
                        "answers": [prompts.ANSWER],
                        "retrieval": [prompts.COVERAGE, prompts.RELEVANCE]}[stage]
        options = {k: v for k, v in asdict(self.config).items()
                   if k in ("judge_model", "temperature", "max_tokens")
                   or stage == "retrieval" and k in ("relevance_sample_size", "relevance_seed")}
        return {**self._model_descriptor, "stage": stage, "config": options,
                "prompts": identity([prompts.COMMON, *instructions])}

    def signature_for(self, stage: str) -> str:
        return identity(self.descriptor_for(stage))

    async def ask(self, task: str, instruction: str, payload: dict, validate) -> dict:
        def parse(raw):
            value = object_response(raw)
            validate(value)
            return value
        messages = [{"role": "system", "content": prompts.COMMON + instruction},
                    {"role": "user", "content": canonical(payload)}]
        with request_context(stage="evaluation", task=task):
            raw = await self.model.text(
                messages, cache_namespace=f"{PROTOCOL}:{task}", validator=parse,
                temperature=self.config.temperature, max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
            )
        return parse(raw)
