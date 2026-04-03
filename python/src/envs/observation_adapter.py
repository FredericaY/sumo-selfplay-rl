"""Observation helpers for the Unity self-play arena.

These helpers keep the first training-facing observation interface simple and
explicit. The raw Unity bridge still returns nested dictionaries, while this
module exposes compact vector features that are easier to feed into future
policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AgentObservation = dict[str, Any]
JointObservation = dict[str, AgentObservation]


@dataclass
class ObservationVectorConfig:
    """Feature toggles for vectorizing per-agent observations."""

    include_relative_position: bool = True
    include_relative_velocity: bool = True
    include_push_ready_flag: bool = True


class ObservationAdapter:
    """Convert raw Unity observations into compact vector features."""

    def __init__(self, config: ObservationVectorConfig | None = None) -> None:
        self.config = config or ObservationVectorConfig()

    def vectorize_agent(self, agent_obs: AgentObservation) -> list[float]:
        """Return a flat feature vector for a single agent view."""

        self_pos = agent_obs["selfPosition"]
        self_vel = agent_obs["selfVelocity"]
        opp_pos = agent_obs["opponentPosition"]
        opp_vel = agent_obs["opponentVelocity"]

        vector = [
            float(self_pos["x"]),
            float(self_pos["y"]),
            float(self_vel["x"]),
            float(self_vel["y"]),
            float(opp_pos["x"]),
            float(opp_pos["y"]),
            float(opp_vel["x"]),
            float(opp_vel["y"]),
        ]

        if self.config.include_relative_position:
            vector.extend(
                [
                    float(opp_pos["x"] - self_pos["x"]),
                    float(opp_pos["y"] - self_pos["y"]),
                ]
            )

        if self.config.include_relative_velocity:
            vector.extend(
                [
                    float(opp_vel["x"] - self_vel["x"]),
                    float(opp_vel["y"] - self_vel["y"]),
                ]
            )

        if self.config.include_push_ready_flag:
            vector.append(1.0 if bool(agent_obs["pushReady"]) else 0.0)

        return vector

    def vectorize_joint(self, observations: JointObservation) -> dict[str, list[float]]:
        """Vectorize both agents at once."""

        return {
            "agent_0": self.vectorize_agent(observations["agent_0"]),
            "agent_1": self.vectorize_agent(observations["agent_1"]),
        }
