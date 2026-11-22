"""Step 4: retrieve and persist reference evidence."""

import argparse
from collections.abc import Iterable, Sequence
from typing import cast
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

from eeg_medrag import DatasetCorpus, EEGMedRAG, QueryConfig, Query
from eeg_medrag.generation import join_evidence_text
from eeg_medrag.request_usage import request_context, summarize_requests
from eeg_medrag.utils import identity, log_run
from eeg_medrag.result_store import ResultStore, validate_result_output


logger = logging.getLogger("eeg_medrag.retrieval")


async def retrieve_questions(
    rag: EEGMedRAG, queries: Iterable[Query], config: QueryConfig, output: str | Path,
) -> dict:
    queries = list(queries)
    corpus = cast(DatasetCorpus, rag.corpus)
    revisions = rag.artifact_revisions(queries)
    run_id = identity(
        [
            corpus.signature,
            rag.construction_signature,
            rag.retrieval_signature,
            rag.model.signature,
            asdict(config),
            revisions,
            [asdict(q) for q in queries],
        ]
    )

    output = Path(output)
    saved: dict[str, dict] = {}

    def snapshot() -> dict:
        rows = [saved[q.id] for q in queries if q.id in saved]
        value = {
            "status": "complete"
            if len(rows) == len(queries) and all(r["status"] == "success" for r in rows)
            else "partial",
            "run_id": run_id,
            "method": "EEG-MedRAG",
            "input_mode": "question_masked_case",
            "corpus_signature": corpus.signature,
            "construction_signature": rag.construction_signature,
            "retrieval_signature": rag.retrieval_signature,
            "retrieval_model_signature": rag.model.signature,
            "query_config": asdict(config),
            "artifact_revisions": revisions,
            "question_ids": [q.id for q in queries],
            "retrievals": rows,
        }
        return value

    checkpoint = ResultStore(output, snapshot())
    try:
        saved = checkpoint.rows()
    except BaseException:
        checkpoint.close()
        raise
    logger.info(
        "Retrieval run ready run_id=%s questions=%d resumed=%d output=%s",
        run_id[:12], len(queries),
        sum(saved.get(q.id, {}).get("status") == "success" for q in queries), output,
    )
    try:
        for number, query in enumerate(queries, 1):
            old = saved.get(query.id)
            if old and old["status"] == "success":
                logger.info(
                    "Retrieval reused query_id=%s progress=%d/%d",
                    query.id, number, len(queries),
                )
                continue
            started = perf_counter()
            logger.info(
                "Retrieval started query_id=%s domain=%s progress=%d/%d",
                query.id, query.domain, number, len(queries),
            )
            try:
                with request_context(stage="retrieval", run_id=run_id, question_id=query.id) as events:
                    evidence, trace = await rag.retrieve(query, config)
                row = {
                    "id": query.id,
                    "status": "success",
                    "knowledge": join_evidence_text(evidence),
                    "evidence": [asdict(item) for item in evidence],
                    "trace": trace,
                    "execution": {"elapsed_seconds": perf_counter() - started,
                                  **summarize_requests(events)},
                }
            except Exception as error:
                logger.error(
                    "Retrieval failed query_id=%s domain=%s error=%s",
                    query.id, query.domain, type(error).__name__,
                )
                saved[query.id] = {"id": query.id, "status": "error", "error_type": type(error).__name__,
                                   "execution": {"elapsed_seconds": perf_counter() - started,
                                                 **summarize_requests(events)}}
                checkpoint.put(saved[query.id])
                raise
            saved[query.id] = row
            checkpoint.put(row)
            logger.info(
                "Retrieval checkpointed query_id=%s evidence=%d context_tokens=%s "
                "elapsed_seconds=%.3f",
                query.id, len(evidence), trace.get("context_tokens", 0), perf_counter() - started,
            )
    finally:
        checkpoint.close()
    result = snapshot()
    logger.info(
        "Retrieval saved status=%s successful=%d/%d output=%s",
        result["status"], sum(row["status"] == "success" for row in result["retrievals"]),
        len(queries), output,
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    command.add_argument(
        "--working-dir", type=Path, default=PROJECT_ROOT / "exp/workspace"
    )
    command.add_argument(
        "--results-dir", type=Path, default=PROJECT_ROOT / "exp/run/retrieval"
    )
    command.add_argument("--output", type=Path)
    command.add_argument("--domain")
    command.add_argument("--id", action="append", dest="ids")
    command.add_argument("--base_url")
    args = command.parse_args(argv)
    corpus = DatasetCorpus.open(args.data_root)
    try:
        if args.domain is not None and args.domain not in corpus.dataset.domains:
            command.error(
                f"Unknown domain {args.domain!r}. "
                f"Available domains: {', '.join(sorted(corpus.dataset.domains))}"
            )
        queries = [
            q
            for q in corpus.queries.values()
            if (args.domain is None or q.domain == args.domain)
            and (not args.ids or q.id in args.ids)
        ]
        if not queries or (args.ids and not set(args.ids) <= {q.id for q in queries}):
            raise ValueError("No matching questions or unknown/out-of-domain IDs")
        output = (
            args.output
            or args.results_dir / ((args.domain or "all") + "_retrieval.sqlite")
        ).resolve()
        if output.is_relative_to(args.data_root.resolve()) or output.is_relative_to(
            PROJECT_ROOT / "contexts"
        ):
            raise ValueError("Keep results outside source data")
        validate_result_output(output)
        with log_run(output.parent / "logs" / (output.stem + ".log"), "retrieval") as run_logger:
            run_logger.info(
                "Inputs data_root=%s working_dir=%s domain=%s questions=%d output=%s",
                args.data_root, args.working_dir, args.domain, len(queries), output,
            )
            os.environ["OPENAI_BASE_URL"] = (
                args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"
            )
            os.environ["OPENAI_API_KEY"] = (
                (PROJECT_ROOT / "openai_api_key.txt").read_text(encoding="utf-8").strip()
            )
            rag = EEGMedRAG(args.working_dir, corpus)
            result = asyncio.run(
                retrieve_questions(
                    rag, queries, QueryConfig(), output
                )
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "output": str(output),
                        "questions": len(result["question_ids"]),
                    }
                )
            )
    finally:
        corpus.close()

if __name__ == "__main__":
    main()
