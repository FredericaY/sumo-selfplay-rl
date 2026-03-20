"""Minimal environment stub for a 1v1 self-play arena.

The real environment can begin as a lightweight Python simulator and later
move closer to Unity parity. For now, this class only defines the intended
responsibilities and a tiny interface for early wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ArenaConfig:
    """Configuration values needed by the first environment prototype."""

    arena_radius: float = 5.0
    max_steps: int = 500
    observation_mode: str = "compact"
    action_mode: str = "hybrid"


class SelfPlayArenaEnv:
    """Stub environment for a symmetric 1v1 sumo-style duel."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.config = ArenaConfig(**config)
        self.current_step = 0

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return the initial observation.

        TODO:
        - Add mirrored spawn states for two agents
        - Return a real observation structure
        """

        self.current_step = 0
        return {"status": "reset_not_implemented"}

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float], bool, dict[str, Any]]:
        """Advance the environment by one step.

        TODO:
        - Validate the action format
        - Apply movement and push logic
        - Compute rewards and termination
        """

        self.current_step += 1
        done = self.current_step >= self.config.max_steps
        observations = {"status": "step_not_implemented", "actions": actions}
        rewards = {"agent_0": 0.0, "agent_1": 0.0}
        info = {"done_reason": "timeout" if done else "running"}
        return observations, rewards, done, info

    def summary(self) -> str:
        """Return a short human-readable description for debugging."""

        return (
            f"name=SelfPlayArena, radius={self.config.arena_radius}, "
            f"max_steps={self.config.max_steps}, action_mode={self.config.action_mode}"
        )
