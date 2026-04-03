"""Evaluate a saved MLP policy checkpoint in the Unity arena."""

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
    parser = argparse.ArgumentParser(description="Evaluate a saved MLP policy checkpoint in Unity.")
    parser.add_argument("checkpoint_path", help="Path to a saved imitation checkpoint.")
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum episode steps.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to evaluate.")
    parser.add_argument(
        "--opponent-policy",
        choices=["idle", "chase", "flee", "random"],
        default="flee",
        help="Opponent policy used against the loaded MLP.",
    )
    parser.add_argument("--seed", type=int, default=11, help="Seed for random opponent policy.")
    parser.add_argument(
        "--debug-every",
        type=int,
        default=10,
        help="Print a short progress update every N steps during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading checkpoint: {Path(args.checkpoint_path)}")
    result = evaluate_policy(
        args.checkpoint_path,
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
    print("Evaluation finished.")
    print(f"Evaluated checkpoint: {Path(args.checkpoint_path)}")
    print(
        f"Episodes={result['episodes']} win_rate={result['win_rate']:.3f} "
        f"avg_steps={result['avg_episode_steps']:.1f} "
        f"agent_0_wins={result['agent_0_wins']} "
        f"agent_1_wins={result['agent_1_wins']} draws={result['draws']}"
    )

    if result["episode_results"]:
        last = result["episode_results"][-1]
        print(
            f"Last episode: reason={last['terminal_reason']} "
            f"winner={last['winner']} "
            f"return0={last['returns']['agent_0']:.3f} "
            f"return1={last['returns']['agent_1']:.3f}"
        )


if __name__ == "__main__":
    main()
