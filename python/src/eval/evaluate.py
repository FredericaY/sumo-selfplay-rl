"""Evaluation helpers for self-play experiments."""

from __future__ import annotations

from typing import Any


def evaluate_policy(checkpoint_path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate a checkpoint and return placeholder metrics.

    TODO:
    - Load a real checkpoint
    - Run evaluation episodes
    - Return actual win-rate and reward statistics
    """

    _ = config or {}
    return {
        "checkpoint": checkpoint_path,
        "win_rate": None,
        "notes": "Evaluation not implemented yet.",
    }
