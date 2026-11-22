"""Prepare shared rubrics and evaluate saved answers and retrieved contexts."""

from __future__ import annotations

from ..utils import IntegrityError, identity
from ..result_store import FINAL_PREDICTION_STATES
from . import prompts
from .judge import boolean_result, exact_rows, reason, relevant_targets
from .lexical import answer_scores
from .protocol import (
    Checkpoint, KNOWLEDGE_TYPES, context_units, effective_context, metadata,
    question_view, questions, rubric_signature, sample_relevance_units, scoped_rows, summarize, validate_rubric,
)


async def prepare_rubric(benchmark: dict, judge, output) -> dict:
    qs = [q for q in questions(benchmark) if q["question_type"] in KNOWLEDGE_TYPES]
    checkpoint = Checkpoint(output, "rubric", qs, [benchmark, judge.signature_for("rubric")],
                            benchmark_signature=identity(benchmark), judge=judge.descriptor_for("rubric"))
    for q in qs:
        if checkpoint.done(q["id"]):
            continue
        payload = question_view(q, gold=True)

        def validate(value):
            reason(value)
            claims = value.get("claims")
            if not isinstance(claims, list):
                raise IntegrityError("Expected a claims list")
            seen = set()
            for claim in claims:
                if not isinstance(claim, dict):
                    raise IntegrityError("Each claim must be an object")
                text, quote = claim.get("text"), claim.get("answer_quote")
                if (not isinstance(text, str) or not text.strip() or text in seen
                        or not isinstance(quote, str) or not quote.strip()
                        or not any(quote in a for a in q["golden_answers"])):
                    raise IntegrityError("Unanchored, empty or duplicated external claim")
                seen.add(text)

        try:
            for attempt in range(3):
                draft = await judge.ask("extract", prompts.EXTRACT, payload, validate)
                review = await judge.ask(
                    "review", prompts.REVIEW,
                    {**question_view(q, gold=True), "draft": draft},
                    lambda value: boolean_result(value, "approved"),
                )
                if review["approved"]:
                    break
                payload = {**question_view(q, gold=True), "draft": draft,
                           "revision_feedback": review["reason"], "revision": attempt + 1}
            else:
                raise IntegrityError(f"Rubric review failed for {q['id']}: {review['reason']}")
            checkpoint.save(q, claims=[{"id": f"k{i + 1}", **claim}
                                       for i, claim in enumerate(draft["claims"])],
                            reason=draft["reason"], review=review,
                            answer_source=q.get("answer_source", []))
        except Exception as error:
            checkpoint.failed(q, error)
            raise
    result = checkpoint.export()
    return checkpoint.export(rubric_signature=rubric_signature(result))


async def evaluate_answers(benchmark: dict, predictions: dict, judge, output) -> dict:
    qs, records = scoped_rows(benchmark, predictions, "predictions")
    for q in qs:
        row = records.get(q["id"])
        state = row.get("status", "success") if row else "missing"
        if state not in FINAL_PREDICTION_STATES:
            raise ValueError(f"Unfinished prediction for {q['id']} ({state}); resume generation first")
        answer = row.get("answer")
        if state == "content_filter" and answer == "":
            continue
        if state != "success" or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Invalid {state} prediction for {q['id']}")
    checkpoint = Checkpoint(
        output, "answers", qs, [benchmark, predictions, judge.signature_for("answers")],
        benchmark_signature=identity(benchmark), judge=judge.descriptor_for("answers"),
        method=predictions.get("method"), generation_signature=predictions.get("generation_signature"),
        generation_policy=predictions.get("generation_policy", {}),
        prediction_signature=identity(predictions),
        prediction_run_id=predictions.get("run_id"),
        answer_model=predictions.get("models", {}).get("answer"),
    )
    for q in qs:
        if checkpoint.done(q["id"]):
            continue
        row = records[q["id"]]
        state = row.get("status", "success")
        answer = row["answer"]
        try:
            if state == "success":
                result = await judge.ask(
                    "answer", prompts.ANSWER,
                    {**question_view(q, gold=True), "candidate_answer": answer},
                    lambda value: boolean_result(value, "correct"),
                )
            else:
                result = {"correct": False, "reason": "The answering service returned content_filter without an answer."}
            checkpoint.save(q, ac=float(result["correct"]), **answer_scores(answer, q["golden_answers"]),
                            judgment=result, prediction_status=state,
                            empty_answer=not bool(answer.strip()))
        except Exception as error:
            checkpoint.failed(q, error)
            raise
    rows = [checkpoint.rows[q["id"]] for q in qs]
    return checkpoint.export(
        **summarize(rows, ("ac", "f1")),
        content_filtered_predictions=sum(r["prediction_status"] == "content_filter" for r in rows),
        missing_predictions=sum(r["prediction_status"] == "missing" for r in rows),
        failed_predictions=sum(r["prediction_status"] not in ("success", "missing") for r in rows),
        empty_answers=sum(r["empty_answer"] for r in rows),
    )


async def evaluate_retrieval(benchmark: dict, retrieval: dict, rubric: dict, judge,
                             tokenizer, output) -> dict:
    rubric_rows = validate_rubric(rubric, benchmark)
    if rubric.get("judge") != judge.descriptor_for("rubric"):
        raise ValueError("Rubric and evaluation must use the same judge protocol")
    qs, records = scoped_rows(benchmark, retrieval, "retrievals")
    checkpoint = Checkpoint(
        output, "retrieval", qs, [benchmark, retrieval, rubric["rubric_signature"], judge.signature_for("retrieval"),
                                  getattr(tokenizer, "model", type(tokenizer).__qualname__)],
        benchmark_signature=identity(benchmark), judge=judge.descriptor_for("retrieval"),
        method=retrieval.get("method"), retrieval_signature=identity(retrieval),
        retrieval_run_id=retrieval.get("run_id"), rubric_signature=rubric["rubric_signature"],
    )
    for q in qs:
        if checkpoint.done(q["id"]):
            continue
        record = records.get(q["id"])
        if record is None or record.get("status", "success") != "success":
            raise ValueError(f"Retrieval is unfinished for {q['id']}; finish it before judging")
        try:
            context = effective_context(record["knowledge"], tokenizer)
            units = context_units(context)
            claims = rubric_rows.get(q["id"], {}).get("claims", [])
            coverage, relevance = [], []
            if claims and units:
                def validate_coverage(value):
                    for row in exact_rows(value, "claims", [c["id"] for c in claims]):
                        boolean_result(row, "supported")

                result = await judge.ask(
                    "coverage", prompts.COVERAGE,
                    {**question_view(q), "claims": [{"id": c["id"], "text": c["text"]} for c in claims],
                     "context_units": units}, validate_coverage,
                )
                coverage = exact_rows(result, "claims", [c["id"] for c in claims])
            elif claims:
                coverage = [{"id": c["id"], "supported": False,
                             "reason": "Empty retrieved context"} for c in claims]
            if units:
                targets = sample_relevance_units(units, q["id"], judge.config)
                local_ids = [f"t{i + 1:02d}" for i in range(len(targets))]
                result = await judge.ask(
                    "relevance", prompts.RELEVANCE,
                    {**question_view(q), "context": context,
                     "target_units": [{**u, "id": uid} for u, uid in zip(targets, local_ids)]},
                    lambda value: relevant_targets(value, local_ids),
                )
                selected = relevant_targets(result, local_ids)
                relevance = [{"id": u["id"], "relevant": uid in selected}
                             for u, uid in zip(targets, local_ids)]
            ec = sum(c["supported"] for c in coverage) / len(claims) if claims else None
            crel = (sum(u["relevant"] for u in relevance) / len(relevance) if relevance else 0.0)
            checkpoint.save(
                q, ec=ec, crel=crel, context_tokens=tokenizer.count(context),
                context_signature=identity(context), units=units, coverage=coverage, relevance=relevance,
                external_claim_count=len(claims), empty_context=not bool(context.strip()),
                ec_exclusion=("outside_knowledge_types"
                              if q["question_type"] not in KNOWLEDGE_TYPES else "no_external_claims"
                              if not claims else None),
            )
        except Exception as error:
            checkpoint.failed(q, error)
            raise
    rows = [checkpoint.rows[q["id"]] for q in qs]
    return checkpoint.export(**summarize(rows, ("ec", "crel", "context_tokens")),
                             empty_contexts=sum(r["empty_context"] for r in rows))
