"""Step 2: construct the shared hypergraph from all reference cases."""

import argparse
from collections.abc import Sequence
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_medrag import DatasetCorpus, EEGMedRAG
from eeg_medrag.utils import log_run


def main(argv: Sequence[str] | None = None) -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    command.add_argument(
        "--working-dir", type=Path, default=PROJECT_ROOT / "exp/workspace"
    )
    command.add_argument("--base_url")
    args = command.parse_args(argv)
    with log_run(args.working_dir.parent / "logs/build_cases.log", "build_cases") as logger:
        logger.info("Inputs data_root=%s working_dir=%s", args.data_root, args.working_dir)
        os.environ["OPENAI_BASE_URL"] = (
            args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"
        )
        os.environ["OPENAI_API_KEY"] = (
            (PROJECT_ROOT / "openai_api_key.txt").read_text(encoding="utf-8").strip()
        )
        corpus = DatasetCorpus.open(args.data_root)
        try:
            rag = EEGMedRAG(args.working_dir, corpus)
            print(
                json.dumps(asyncio.run(rag.build_cases()), indent=2)
            )
        finally:
            corpus.close()


if __name__ == "__main__":
    main()
