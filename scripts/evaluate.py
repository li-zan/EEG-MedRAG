"""Prepare, run and compare evaluations with the fixed experiment judge."""

import argparse
import asyncio
from pathlib import Path
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_medrag.base import LLMConfig
from eeg_medrag.dataset import Dataset
from eeg_medrag.llm import LLMProvider
from eeg_medrag.storage import CachedModel, Storage
from eeg_medrag.utils import HyperTokenizer, atomic_json, log_run, read_json
from eeg_medrag.result_store import read_results
from eeg_medrag.evaluation import EvaluationConfig
from eeg_medrag.evaluation.judge import Judge
from eeg_medrag.evaluation.runner import evaluate_answers, evaluate_retrieval, prepare_rubric
from eeg_medrag.evaluation.eeg import evaluate_eeg
from eeg_medrag.evaluation.protocol import questions, scoped_rows, validate_rubric
from eeg_medrag.evaluation.statistics import (
    bootstrap, cluster_map, paired_comparison,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    stages = command.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "answers", "retrieval", "eeg", "compare"):
        sub = stages.add_parser(name, allow_abbrev=False)
        sub.add_argument("--output", type=Path, required=name == "compare",
                         help="Explicit output file; otherwise inferred from the experiment layout")
        sub.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "datasets")
        if name in ("prepare", "answers", "retrieval"):
            sub.add_argument("--base_url")
            sub.add_argument("--working-dir", type=Path, help="Evaluation response cache; defaults to output-parent/cache")
        if name == "answers":
            sub.add_argument("--predictions", type=Path, required=True)
        if name in ("retrieval", "eeg"):
            sub.add_argument("--retrieval", type=Path, required=True)
        if name == "retrieval":
            sub.add_argument("--rubric", type=Path, required=True)
        if name in ("answers", "retrieval"):
            sub.add_argument("--bootstrap", action="store_true")
        if name == "compare":
            sub.add_argument("--left", type=Path, required=True)
            sub.add_argument("--right", type=Path, required=True)
    return command


def output_path(args) -> Path:
    if args.output is not None:
        return args.output.resolve()
    if args.stage == "prepare":
        return PROJECT_ROOT / "exp/run/evaluation/protocol/knowledge_rubric.json"
    if args.stage == "answers":
        source = args.predictions.resolve()
        # Standard layout: <run>/generation/<answer-model>/all_predictions.sqlite.
        return source.parent.parent.parent / "evaluation" / source.parent.name / "answers.json"
    source = args.retrieval.resolve()
    # Standard layout: <run>/retrieval/all_retrieval.sqlite.
    return source.parent.parent / "evaluation" / (args.stage + ".json")


def main(argv=None) -> None:
    command = parser()
    args = command.parse_args(argv)
    output = output_path(args)
    benchmark_path = args.data_root / "questions.json"
    inputs = [benchmark_path] + [getattr(args, k) for k in
                                 ("predictions", "retrieval", "rubric", "left", "right")
                                 if getattr(args, k, None) is not None]
    if (output in [p.resolve() for p in inputs]
            or output.is_relative_to(args.data_root.resolve())
            or output.is_relative_to((PROJECT_ROOT / "contexts").resolve())):
        raise ValueError("Keep evaluation output separate from inputs and source data")
    if hasattr(args, "working_dir"):
        work = (args.working_dir or output.parent / "cache").resolve()
        if work.is_relative_to(args.data_root.resolve()) or work.is_relative_to((PROJECT_ROOT / "contexts").resolve()):
            raise ValueError("Keep evaluation cache outside source data")
    benchmark = read_json(benchmark_path)
    questions(benchmark)
    config = EvaluationConfig()
    with log_run(output.parent / "logs" / (output.stem + ".log"), "evaluation:" + args.stage):
        if args.stage in ("prepare", "answers", "retrieval"):
            predictions = read_results(args.predictions) if args.stage == "answers" else None
            retrieval = read_results(args.retrieval) if args.stage == "retrieval" else None
            rubric = read_json(args.rubric) if args.stage == "retrieval" else None
            if predictions is not None:
                scoped_rows(benchmark, predictions, "predictions")
            if retrieval is not None:
                scoped_rows(benchmark, retrieval, "retrievals")
                validate_rubric(rubric, benchmark)
            os.environ["OPENAI_BASE_URL"] = (
                args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"
            )
            os.environ["OPENAI_API_KEY"] = (
                (PROJECT_ROOT / "openai_api_key.txt").read_text(encoding="utf-8").strip()
            )
            provider = LLMProvider(LLMConfig(chat_model=config.judge_model))
            judge = Judge(CachedModel(provider, Storage(work)), config)
            if args.stage == "prepare":
                report = asyncio.run(prepare_rubric(benchmark, judge, output))
            elif args.stage == "answers":
                report = asyncio.run(evaluate_answers(
                    benchmark, predictions, judge, output,
                ))
            else:
                report = asyncio.run(evaluate_retrieval(
                    benchmark, retrieval, rubric, judge, HyperTokenizer(), output,
                ))
            if getattr(args, "bootstrap", False):
                dataset = Dataset(args.data_root)
                try:
                    metrics = ("ac", "f1") if args.stage == "answers" else ("ec", "crel")
                    report["intervals"] = bootstrap(report["per_question"], cluster_map(benchmark, dataset), metrics, config)
                finally:
                    dataset.close()
                atomic_json(output, report)
        else:
            dataset = Dataset(args.data_root)
            try:
                if args.stage == "eeg":
                    report = evaluate_eeg(benchmark, read_results(args.retrieval), dataset, config)
                else:
                    report = paired_comparison(read_json(args.left), read_json(args.right), benchmark, dataset, config)
            finally:
                dataset.close()
            atomic_json(output, report)
        print(json.dumps({"output": str(output), "stage": args.stage,
                          **({"content_filtered_predictions": report["content_filtered_predictions"]}
                             if args.stage == "answers" else {}),
                          **({"overall": report["overall"]} if "overall" in report else {})}, ensure_ascii=False))


if __name__ == "__main__":
    main()
