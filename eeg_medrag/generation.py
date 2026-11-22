"""Shared answering policy for all RAG methods, independent of retrieval internals."""

from __future__ import annotations
from collections.abc import Iterable

import re
import logging

from .base import Evidence, GenerationConfig, QueryConfig, Query, TextModel, Tokenizer
from .prompt import GENERATION, ROLE_INSTRUCTIONS
from .utils import IntegrityError, canonical, component_descriptor, identity


def join_evidence_text(evidence: Iterable[Evidence]) -> str:
    """Join model-readable evidence in retrieval order."""
    return "\n\n".join(item.text for item in evidence)


logger = logging.getLogger(__name__)


def generation_inputs(benchmark: dict, retrieval: dict) -> tuple[list[Query], dict]:
    """Validate IDs and allowed input modes before requesting any answers."""
    def index(rows):
        result = {}
        for row in rows:
            qid = row["id"]
            if not isinstance(qid, str) or not qid or qid in result:
                raise ValueError("Duplicate or invalid question ID")
            result[qid] = row
        return result

    by_id = index(benchmark["questions"])
    ids = retrieval["question_ids"]
    if (not isinstance(ids, list) or not ids or not all(isinstance(qid, str) for qid in ids)
            or len(set(ids)) != len(ids) or set(ids) - by_id.keys()):
        raise ValueError("Expected unique, known question_ids")
    records = index(retrieval["retrievals"])
    if set(records) != set(ids) or any(r.get("status", "success") != "success" for r in records.values()):
        raise ValueError("Finish retrieval for every declared question before generation")
    if any(not isinstance(r.get("knowledge"), str) for r in records.values()):
        raise ValueError("Retrieval knowledge must be text")
    return [Query.from_row(by_id[qid]) for qid in ids], records


def effective_context(text: str, tokenizer: Tokenizer, budget: int | None = None) -> str:
    """Use exactly the same context truncation for generation and evaluation."""
    if not isinstance(text, str):
        raise ValueError("Retrieval knowledge must be text")
    budget = QueryConfig().context_tokens if budget is None else budget
    return tokenizer.chunks(text, budget, 0)[0]["content"] if tokenizer.count(text) > budget else text


class AnswerGenerator:
    def __init__(self, model: TextModel, tokenizer: Tokenizer,
                 config: GenerationConfig | None = None) -> None:
        self.model, self.tokenizer = model, tokenizer
        self.config = config or GenerationConfig()
        self.model_signature = getattr(model, "chat_signature", model.signature)
        self.model_descriptor = getattr(
            model, "chat_descriptor",
            getattr(model, "descriptor", {"signature": self.model_signature}),
        )
        model_id = self.model_descriptor.get("requested_model") or self.model_descriptor.get("chat_model")
        self.request_parameters = self.config.request_parameters(model_id)
        self.descriptor = {
            "prompt": identity([GENERATION, ROLE_INSTRUCTIONS]),
            "decoding": self.request_parameters,
            "tokenizer": getattr(tokenizer, "model", component_descriptor(type(tokenizer))),
            "context_tokens": QueryConfig().context_tokens,
            "implementation": component_descriptor(type(self)),
        }
        self.signature = identity([self.model_signature, self.descriptor])

    def policy(self, config: QueryConfig | None = None) -> dict:
        config = config or QueryConfig()
        return {**self.descriptor, "context_tokens": config.context_tokens,
                "input_mode": "question_masked_case"}

    def messages(
        self,
        query: Query,
        knowledge: str,
        config: QueryConfig | None = None,
    ) -> list[dict[str, str]]:
        config = config or QueryConfig()
        input_tokens = self.tokenizer.count(knowledge)
        if input_tokens > config.context_tokens:
            logger.info(
                "Generation context truncated query_id=%s tokens=%d budget=%d",
                query.id, input_tokens, config.context_tokens,
            )
        knowledge = effective_context(knowledge, self.tokenizer, config.context_tokens)
        prompt = GENERATION.format(
            role=ROLE_INSTRUCTIONS[query.role],
            case=canonical(query.masked_case),
            knowledge=knowledge,
            question=query.question,
        )
        return [{"role": "user", "content": prompt}]

    async def generate(
        self,
        query: Query,
        knowledge: str,
        config: QueryConfig | None = None,
    ) -> dict[str, str]:
        def parse(text: str) -> dict[str, str]:
            answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
            if len(answers) != 1 or not answers[0].strip():
                raise IntegrityError("Missing or ambiguous answer tags")
            reasoning = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
            return {
                "answer": answers[0].strip(),
                "rationale": reasoning.group(1).strip() if reasoning else "",
            }

        raw = await self.model.text(
            self.messages(query, knowledge, config),
            cache_namespace="answer_generation",
            validator=parse,
            **self.request_parameters,
        )
        return parse(raw)
