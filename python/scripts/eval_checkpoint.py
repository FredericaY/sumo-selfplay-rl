"""Convenience script for evaluating a saved checkpoint."""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum steps per episode.")
    parser.add_argument(
        "--opponent-policy",
        choices=["idle", "chase", "flee", "random"],
        default="flee",
        help="Opponent policy used against the evaluated checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Seed for random opponent policy.")
    parser.add_argument("--debug-every", type=int, default=0, help="Print progress every N steps.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_policy(
        args.checkpoint,
        config={
            "host": args.host,
            "port": args.port,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "opponent_policy": args.opponent_policy,
            "seed": args.seed,
            "debug_every": args.debug_every,
        },
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
