"""Train a small MLP policy to imitate saved rollout actions."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.imitation_trainer import ImitationTrainConfig, ImitationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an imitation MLP from saved rollout JSON files.")
    parser.add_argument(
        "--rollout-dir",
        default="python/logs/rollouts",
        help="Directory containing saved rollout JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="rollout_*.json",
        help="Glob pattern for selecting rollout files.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--checkpoint-dir",
        default="python/checkpoints/imitation",
        help="Where to save imitation checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    rollout_paths = sorted(rollout_dir.glob(args.pattern))
    if not rollout_paths:
        raise SystemExit(f"No rollout files found in {rollout_dir} matching {args.pattern}")

    trainer = ImitationTrainer(
        ImitationTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
    )
    history = trainer.fit(rollout_paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = Path(args.checkpoint_dir) / f"imitation_mlp_{timestamp}.pt"
    saved_path = trainer.save_checkpoint(
        checkpoint_path,
        metadata={
            "num_rollouts": len(rollout_paths),
            "rollout_files": [str(path) for path in rollout_paths],
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
        },
    )

    print(f"Loaded rollout files: {len(rollout_paths)}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Saved checkpoint to: {saved_path}")


if __name__ == "__main__":
    main()
