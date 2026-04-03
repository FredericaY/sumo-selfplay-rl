"""Simple self-play manager placeholder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SelfPlayConfig:
    """Configuration for self-play opponent selection."""

    enabled: bool = True
    opponent_sampling: str = "latest"
    opponent_pool_size: int = 8
    opponent_policy: str = "flee"


class SelfPlayManager:
    """Track future self-play responsibilities in one place."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.config = SelfPlayConfig(**config)
        self.opponent_checkpoints: list[str] = []

    def register_checkpoint(self, checkpoint_path: str) -> None:
        """Record a checkpoint for future opponent sampling.

        TODO:
        - Enforce pool size limits
        - Add recency-biased and uniform sampling
        """

        self.opponent_checkpoints.append(checkpoint_path)

    def sample_opponent(self) -> str | None:
        """Return a placeholder opponent reference."""

        if not self.opponent_checkpoints:
            return None
        return self.opponent_checkpoints[-1]

    def summary(self) -> str:
        """Return a short summary string for logs and debugging."""

        return (
            f"enabled={self.config.enabled}, sampling={self.config.opponent_sampling}, "
            f"pool_size={self.config.opponent_pool_size}, "
            f"opponent_policy={self.config.opponent_policy}"
        )
