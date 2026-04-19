"""Observation helpers for the Unity self-play arena.

These helpers keep the first training-facing observation interface simple and
explicit. The raw Unity bridge still returns nested dictionaries, while this
module exposes compact vector features that are easier to feed into future
policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


AgentObservation = dict[str, Any]
JointObservation = dict[str, AgentObservation]
DEFAULT_OBS_DIM = 17


@dataclass
class ObservationVectorConfig:
    """Feature toggles for vectorizing per-agent observations."""

    arena_radius: float = 5.0
    canonicalize_left_right: bool = True
    opponent_position_scale: float = 1.5
    relative_position_scale: float = 2.0
    include_relative_position: bool = True
    include_relative_velocity: bool = True
    include_edge_features: bool = True
    include_push_ready_flag: bool = True


class ObservationAdapter:
    """Convert raw Unity observations into compact vector features."""

    def __init__(self, config: ObservationVectorConfig | None = None) -> None:
        self.config = config or ObservationVectorConfig()

    @property
    def output_dim(self) -> int:
        return DEFAULT_OBS_DIM

    def should_mirror_agent(self, agent_obs: AgentObservation) -> bool:
        """Return whether this agent view should be mirrored into the canonical left-side frame."""

        if not self.config.canonicalize_left_right:
            return False

        self_pos = agent_obs["selfPosition"]
        return float(self_pos["x"]) > 0.0

    def vectorize_agent(self, agent_obs: AgentObservation, mirror_x: bool | None = None) -> list[float]:
        """Return a flat feature vector for a single agent view."""

        self_pos = agent_obs["selfPosition"]
        self_vel = agent_obs["selfVelocity"]
        opp_pos = agent_obs["opponentPosition"]
        opp_vel = agent_obs["opponentVelocity"]

        self_x = float(self_pos["x"])
        self_y = float(self_pos["y"])
        self_vx = float(self_vel["x"])
        self_vy = float(self_vel["y"])
        opp_x = float(opp_pos["x"])
        opp_y = float(opp_pos["y"])
        opp_vx = float(opp_vel["x"])
        opp_vy = float(opp_vel["y"])
        mirror_x = self.should_mirror_agent(agent_obs) if mirror_x is None else bool(mirror_x)

        if mirror_x:
            self_x *= -1.0
            self_vx *= -1.0
            opp_x *= -1.0
            opp_vx *= -1.0

        vector = [
            self_x,
            self_y,
            self_vx,
            self_vy,
            opp_x * self.config.opponent_position_scale,
            opp_y * self.config.opponent_position_scale,
            opp_vx,
            opp_vy,
        ]

        if self.config.include_relative_position:
            rel_x = (opp_x - self_x) * self.config.relative_position_scale
            rel_y = (opp_y - self_y) * self.config.relative_position_scale
            vector.extend(
                [
                    rel_x,
                    rel_y,
                ]
            )

        if self.config.include_relative_velocity:
            vector.extend(
                [
                    opp_vx - self_vx,
                    opp_vy - self_vy,
                ]
            )

        if self.config.include_edge_features:
            self_dist = math.sqrt(self_x * self_x + self_y * self_y)
            opp_dist = math.sqrt(opp_x * opp_x + opp_y * opp_y)
            vector.extend(
                [
                    self_dist,
                    max(0.0, self.config.arena_radius - self_dist),
                    opp_dist,
                    max(0.0, self.config.arena_radius - opp_dist),
                ]
            )

        if self.config.include_push_ready_flag:
            vector.append(1.0 if bool(agent_obs["pushReady"]) else 0.0)

        return vector

    def joint_mirror_flags(self, observations: JointObservation) -> dict[str, bool]:
        """Return per-agent mirror flags for one joint observation."""

        return {
            "agent_0": self.should_mirror_agent(observations["agent_0"]),
            "agent_1": self.should_mirror_agent(observations["agent_1"]),
        }

    def vectorize_joint(self, observations: JointObservation) -> dict[str, list[float]]:
        """Vectorize both agents at once."""

        mirror_flags = self.joint_mirror_flags(observations)
        return {
            "agent_0": self.vectorize_agent(observations["agent_0"], mirror_x=mirror_flags["agent_0"]),
            "agent_1": self.vectorize_agent(observations["agent_1"], mirror_x=mirror_flags["agent_1"]),
        }
