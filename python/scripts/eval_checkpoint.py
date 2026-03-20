"""Convenience script for evaluating a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.evaluate import evaluate_policy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("checkpoint", help="Path to a checkpoint file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_policy(args.checkpoint)
    print(result)

    # TODO: Add batch checkpoint evaluation and CSV/JSON output.
