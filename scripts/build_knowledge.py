"""Step 1: construct the shared medical knowledge hypergraph."""

import argparse
from collections.abc import Sequence
import asyncio
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_medrag import EEGMedRAG
from eeg_medrag.utils import log_run


def main(argv: Sequence[str] | None = None) -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--working-dir", type=Path, default=PROJECT_ROOT / "exp/workspace"
    )
    command.add_argument(
        "--contexts-root", type=Path, default=PROJECT_ROOT / "contexts"
    )
    command.add_argument("--base_url")
    args = command.parse_args(argv)
    with log_run(args.working_dir.parent / "logs/build_knowledge.log", "build_knowledge") as logger:
        logger.info("Inputs contexts_root=%s working_dir=%s", args.contexts_root, args.working_dir)
        os.environ["OPENAI_BASE_URL"] = (
            args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.apiyi.com/v1"
        )
        os.environ["OPENAI_API_KEY"] = (
            (PROJECT_ROOT / "openai_api_key.txt").read_text(encoding="utf-8").strip()
        )
        rag = EEGMedRAG(args.working_dir)
        documents = rag.pdf_documents(args.contexts_root)
        state = asyncio.run(rag.build_knowledge(documents))
        print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
