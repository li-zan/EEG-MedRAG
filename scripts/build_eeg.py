"""Step 3: encode reference EEG windows; no model API is used."""

import argparse
from collections.abc import Sequence
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eeg_medrag import DatasetCorpus, EEGIndex
from eeg_medrag.storage import Storage
from eeg_medrag.utils import log_run


def main(argv: Sequence[str] | None = None) -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--data-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    command.add_argument(
        "--working-dir", type=Path, default=PROJECT_ROOT / "exp/workspace"
    )
    command.add_argument("--domain", required=True)
    args = command.parse_args(argv)
    corpus = DatasetCorpus.open(args.data_root)
    try:
        if args.domain not in corpus.dataset.domains:
            command.error(
                f"Unknown domain {args.domain!r}. "
                f"Available domains: {', '.join(sorted(corpus.dataset.domains))}"
            )
        with log_run(
            args.working_dir.parent / "logs" / f"build_eeg_{args.domain}.log", "build_eeg",
        ) as logger:
            logger.info(
                "Inputs data_root=%s working_dir=%s domain=%s",
                args.data_root, args.working_dir, args.domain,
            )
            index = EEGIndex(Storage(args.working_dir), corpus)
            index.build(args.domain)
    finally:
        corpus.close()


if __name__ == "__main__":
    main()
