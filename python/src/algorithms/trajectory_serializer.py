"""Trajectory serialization helpers for rollout data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from algorithms.rollout_collector import RolloutEpisode


class TrajectorySerializer:
    """Save collected rollout episodes in a simple JSON format."""

    def episode_to_dict(self, episode: RolloutEpisode) -> dict[str, Any]:
        """Convert a rollout episode into a JSON-serializable dictionary."""

        return {
            "initial_observations": episode.initial_observations,
            "initial_observation_vectors": episode.initial_observation_vectors,
            "transitions": [self.transition_to_dict(transition) for transition in episode.transitions],
            "returns": episode.total_returns(),
        }

    def transition_to_dict(self, transition) -> dict[str, Any]:
        """Convert one transition into a JSON-serializable dictionary."""

        return {
            "step_idx": transition.step_idx,
            "observations": transition.observations,
            "observation_vectors": transition.observation_vectors,
            "actions": transition.actions,
            "policy_action_vectors": transition.policy_action_vectors,
            "executed_action_vectors": transition.executed_action_vectors,
            "rewards": transition.rewards,
            "done": transition.done,
            "info": transition.info,
        }

    def save_episode(self, episode: RolloutEpisode, path: str | Path) -> Path:
        """Write one episode to disk as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.episode_to_dict(episode), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path
