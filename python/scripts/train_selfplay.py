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
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=None,
        help="Optional wall-clock sleep in seconds after each environment step for easier visual debugging.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional PPO checkpoint used to initialize or continue training.",
    )
    parser.add_argument(
        "--train-side",
        choices=["a", "b"],
        default=None,
        help="When using alternating_two_policy mode, choose which side to train first in this run.",
    )
    parser.add_argument(
        "--alternating-cycles",
        type=int,
        default=1,
        help="When using alternating_two_policy mode, run this many automatic A->B cycles.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.config,
        step_sleep_override=args.step_sleep,
        init_checkpoint_override=args.init_checkpoint,
        train_side_override=args.train_side,
        alternating_cycles=args.alternating_cycles,
    )

    # TODO: Add run naming, seeding, and experiment folder creation.
