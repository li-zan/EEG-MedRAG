"""Step 5: apply the same answering prompt/model/budget to any RAG retrieval file."""

import argparse
from collections.abc import Sequence
import asyncio
import json
import logging
from time import perf_counter
import os
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_medrag import AnswerGenerator, LLMConfig, LLMProvider, QueryConfig, Query
from eeg_medrag.storage import CachedModel, Storage
from eeg_medrag.generation import effective_context, generation_inputs
from eeg_medrag.llm import ContentFilteredAnswer
from eeg_medrag.request_usage import request_context, summarize_requests
from eeg_medrag.utils import HyperTokenizer, identity, log_run, read_json
from eeg_medrag.result_store import FINAL_PREDICTION_STATES, ResultStore, read_results, validate_result_output


logger = logging.getLogger("eeg_medrag.generation")


async def run_questions(
    generator: AnswerGenerator, benchmark: dict, retrieval: dict,
    output: str | Path,
) -> dict:
    """Join IDs to allowed benchmark inputs; consume knowledge text only."""
    selected, records = generation_inputs(benchmark, retrieval)
    queries = {q.id: q for q in selected}
    ids = list(queries)
    config = QueryConfig()
    policy = generator.policy(config)
    generation_signature = identity([generator.model_signature, policy])
    retrieval_run_id = retrieval["run_id"] if "run_id" in retrieval else identity(retrieval)
    run_id = identity([
        [asdict(queries[qid]) for qid in ids], retrieval, generation_signature,
    ])

    output = Path(output)
    saved: dict[str, dict] = {}

    def snapshot() -> dict:
        predictions = [saved[qid] for qid in ids if qid in saved]
        value = {
            "status": "complete"
            if len(predictions) == len(ids)
            and all(row["status"] in FINAL_PREDICTION_STATES for row in predictions)
            else "partial",
            "run_id": run_id,
            "method": retrieval.get("method", "RAG"),
            "generation_signature": generation_signature,
            "generation_policy": policy,
            "models": {"answer": generator.model_descriptor},
            "answer_model_signature": generator.model_signature,
            "question_ids": ids,
            "retrieval_run_ids": [retrieval_run_id],
            "predictions": predictions,
        }
        return value

    checkpoint = ResultStore(output, snapshot())
    try:
        saved = checkpoint.rows()
    except BaseException:
        checkpoint.close()
        raise
    logger.info(
        "Generation run ready run_id=%s questions=%d resumed=%d output=%s",
        run_id[:12], len(ids), sum(row["status"] in FINAL_PREDICTION_STATES for row in saved.values()),
        output,
    )
    try:
        for number, qid in enumerate(ids, 1):
            query = queries[qid]
            old = saved.get(qid)
            if old and old["status"] in FINAL_PREDICTION_STATES:
                logger.info("Generation reused query_id=%s progress=%d/%d", qid, number, len(ids))
                continue
            row = records[qid]
            started = perf_counter()
            logger.info(
                "Generation started query_id=%s domain=%s progress=%d/%d",
                qid, query.domain, number, len(ids),
            )
            try:
                with request_context(stage="generation", run_id=run_id, question_id=qid) as events:
                    answer = await generator.generate(query, row["knowledge"], config)
                state = "success"
            except ContentFilteredAnswer:
                state = "content_filter"
                answer = {"answer": "", "rationale": ""}
                logger.warning("Generation content-filtered query_id=%s progress=%d/%d", qid, number, len(ids))
            except Exception as error:
                logger.error("Generation failed query_id=%s error=%s", qid, type(error).__name__)
                saved[qid] = {"id": qid, "status": "error", "error_type": type(error).__name__,
                              "execution": {"elapsed_seconds": perf_counter() - started,
                                            **summarize_requests(events)}}
                checkpoint.put(saved[qid])
                raise
            saved[qid] = {"id": qid, "status": state, **answer,
                          "execution": {"elapsed_seconds": perf_counter() - started,
                                        **summarize_requests(events)},
                          "input_tokens_estimate": sum(generator.tokenizer.count(m["content"])
                                                       for m in generator.messages(query, row["knowledge"], config)),
                          "context_tokens": generator.tokenizer.count(effective_context(
                              row["knowledge"], generator.tokenizer, config.context_tokens))}
            checkpoint.put(saved[qid])
            logger.info(
                "Generation checkpointed query_id=%s elapsed_seconds=%.3f",
                qid, perf_counter() - started,
            )
    finally:
        checkpoint.close()
    result = snapshot()
    logger.info(
        "Generation saved status=%s successful=%d/%d content_filtered=%d output=%s",
        result["status"], sum(row["status"] == "success" for row in result["predictions"]),
        len(ids), sum(row["status"] == "content_filter" for row in result["predictions"]), output,
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    command.add_argument("--retrieval", type=Path, required=True)
    command.add_argument(
        "--working-dir", type=Path,
        help="Response cache directory; defaults to output-parent/cache"
    )
    command.add_argument(
        "--results-dir", type=Path,
        help="Output directory; defaults to retrieval-directory/../generation/<model>"
    )
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--model",
        default=LLMConfig().chat_model,
        choices=["gpt-4o-mini", "gemini-2.5-flash", "deepseek-v4-flash"],
    )
    command.add_argument("--base_url")
    args = command.parse_args(argv)
    benchmark_path = args.data_root / "questions.json"
    output = (
        args.output
        or (args.results_dir or args.retrieval.parent.parent / "generation" / args.model)
        / (args.retrieval.stem.replace("_retrieval", "") + "_predictions.sqlite")
    ).resolve()
    work = (args.working_dir or output.parent / "cache").resolve()
    if (output in (benchmark_path.resolve(), args.retrieval.resolve())
            or any(path.is_relative_to(root.resolve()) for path in (output, work)
                   for root in (args.data_root, PROJECT_ROOT / "contexts"))):
        raise ValueError("Keep output separate from benchmark and retrieval inputs")
    with log_run(output.parent / "logs" / (output.stem + ".log"), "generation") as run_logger:
        run_logger.info(
            "Inputs data_root=%s retrieval=%s working_dir=%s model=%s output=%s",
            args.data_root, args.retrieval, work, args.model, output,
        )
        retrieval = read_results(args.retrieval)
        rows = {row["id"]: row for row in retrieval["retrievals"]}
        if not all(
            qid in rows and rows[qid].get("status", "success") == "success"
            for qid in retrieval["question_ids"]
        ):
            command.error("Finish retrieval before generation")
        benchmark = read_json(benchmark_path)
        generation_inputs(benchmark, retrieval)
        validate_result_output(output)
        os.environ["OPENAI_BASE_URL"] = (
            args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"
        )
        os.environ["OPENAI_API_KEY"] = (
            (PROJECT_ROOT / "openai_api_key.txt").read_text(encoding="utf-8").strip()
        )
        store = Storage(work)
        provider = LLMProvider(LLMConfig(chat_model=args.model))
        generator = AnswerGenerator(
            CachedModel(provider, store), HyperTokenizer()
        )
        result = asyncio.run(
            run_questions(generator, benchmark, retrieval, output)
        )
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output": str(output),
                    "questions": len(result["question_ids"]),
                    "content_filtered": sum(row["status"] == "content_filter" for row in result["predictions"]),
                }
            )
        )


if __name__ == "__main__":
    main()
