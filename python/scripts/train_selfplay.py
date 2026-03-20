"""Convenience script for launching self-play training from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import main


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training launcher."""

    parser = argparse.ArgumentParser(description="Launch self-play training.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "train.yaml"),
        help="Path to the training config file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config)

    # TODO: Add run naming, seeding, and experiment folder creation.
