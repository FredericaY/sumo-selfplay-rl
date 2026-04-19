"""Unity-backed environment wrapper for the 1v1 self-play arena.

This module keeps the Python side intentionally small:
- use the existing TCP bridge as the transport
- expose a familiar reset/step/close interface
- return observations, rewards, done, and info in a stable format

The default reward is sparse and mirrors Unity's terminal reward. Optional
Python-side shaping can be enabled to support early RL experiments without
changing the Unity bridge contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .unity_bridge_client import UnityBridgeClient, UnityBridgeConfig


AgentObservation = dict[str, Any]
JointObservation = dict[str, AgentObservation]
JointAction = dict[str, dict[str, Any]]
RewardDict = dict[str, float]
InfoDict = dict[str, Any]


@dataclass
class UnitySelfPlayArenaConfig:
    """Connection and episode settings for the Unity-backed arena."""

    host: str = "127.0.0.1"
    port: int = 5055
    timeout_s: float = 5.0
    max_episode_steps: int = 500
    arena_radius: float = 5.0
    use_shaped_rewards: bool = False
    step_penalty: float = 0.0
    edge_safety_weight: float = 0.0
    outward_pressure_weight: float = 0.0
    terminal_timeout_penalty: float = 0.0
    timeout_center_bias_weight: float = 0.0
    terminal_win_reward: float = 1.0
    terminal_loss_penalty: float = 1.5
    terminal_loss_time_scale: float = 0.5


class UnitySelfPlayArenaEnv:
    """Minimal environment wrapper over the Unity TCP bridge."""

    def __init__(self, config: UnitySelfPlayArenaConfig | None = None) -> None:
        self.config = config or UnitySelfPlayArenaConfig()
        self.client = UnityBridgeClient(
            UnityBridgeConfig(
                host=self.config.host,
                port=self.config.port,
                timeout_s=self.config.timeout_s,
            )
        )
        self.episode_step = 0
        self.last_raw_state: dict[str, Any] | None = None
        self.last_observations: JointObservation | None = None

    def reset(self) -> JointObservation:
        """Reset Unity and return the initial per-agent observations."""

        state = self.client.reset()
        self._ensure_valid_state(state, stage="reset")
        self.episode_step = 0
        self.last_raw_state = state
        observations = self._extract_observations(state)
        self.last_observations = observations
        return observations

    def step(
        self,
        actions: JointAction,
    ) -> tuple[JointObservation, RewardDict, bool, InfoDict]:
        """Apply one action for each agent and return the next transition."""

        self._validate_actions(actions)
        previous_observations = self.last_observations
        state = self.client.step(actions["agent_0"], actions["agent_1"])
        self._ensure_valid_state(state, stage=f"step {self.episode_step}")

        self.episode_step += 1
        self.last_raw_state = state

        observations = self._extract_observations(state)
        base_rewards = self._extract_rewards(state)
        done = bool(state["done"])
        python_timeout = self.episode_step >= self.config.max_episode_steps
        if python_timeout and not done:
            done = True

        terminal_reason = state.get("terminalReason", "running")
        base_rewards = self._apply_terminal_reward_override(
            base_rewards=base_rewards,
            winner=int(state["winner"]),
            done=done,
            unity_done=bool(state["done"]),
            python_timeout=python_timeout,
            terminal_reason=terminal_reason,
            episode_progress=min(1.0, self.episode_step / max(self.config.max_episode_steps, 1)),
        )
        shaped_bonus = self._compute_shaped_bonus(
            previous_observations=previous_observations,
            current_observations=observations,
            unity_done=bool(state["done"]),
            timeout=python_timeout and not bool(state["done"]),
        )
        rewards = {
            agent_id: base_rewards[agent_id] + shaped_bonus[agent_id]
            for agent_id in base_rewards
        }

        info = {
            "winner": int(state["winner"]),
            "status": state["status"],
            "terminal_reason": terminal_reason,
            "episode_step": self.episode_step,
            "timeout": python_timeout and not bool(state["done"]),
            "unity_time_limit": terminal_reason == "time_limit",
            "raw_state": state,
            "base_rewards": base_rewards,
            "shaped_bonus": shaped_bonus,
        }
        self.last_observations = observations
        return observations, rewards, done, info

    def get_state(self) -> JointObservation:
        """Fetch the latest Unity observations without advancing the match."""

        state = self.client.get_state()
        self._ensure_valid_state(state, stage="get_state")
        self.last_raw_state = state
        observations = self._extract_observations(state)
        self.last_observations = observations
        return observations

    def close(self) -> None:
        """Close the underlying bridge client."""

        self.client.close()

    def summary(self) -> str:
        """Return a short human-readable description for debugging."""

        return (
            "name=UnitySelfPlayArenaEnv, "
            f"host={self.config.host}, port={self.config.port}, "
            f"timeout_s={self.config.timeout_s}, "
            f"max_episode_steps={self.config.max_episode_steps}, "
            f"use_shaped_rewards={self.config.use_shaped_rewards}"
        )

    @staticmethod
    def idle_action() -> dict[str, Any]:
        """Return a no-op action compatible with the Unity bridge."""

        return {
            "move": [0.0, 0.0],
            "push": [0.0, 0.0],
            "use_push": False,
        }

    @staticmethod
    def _extract_observations(state: dict[str, Any]) -> JointObservation:
        return {
            "agent_0": state["agent0"],
            "agent_1": state["agent1"],
        }

    @staticmethod
    def _extract_rewards(state: dict[str, Any]) -> RewardDict:
        return {
            "agent_0": float(state["reward0"]),
            "agent_1": float(state["reward1"]),
        }

    def _apply_terminal_reward_override(
        self,
        base_rewards: RewardDict,
        winner: int,
        done: bool,
        unity_done: bool,
        python_timeout: bool,
        terminal_reason: str,
        episode_progress: float,
    ) -> RewardDict:
        if not done:
            return base_rewards

        scaled_loss_penalty = self._scaled_loss_penalty(episode_progress)

        if winner == 0:
            return {
                "agent_0": self.config.terminal_win_reward,
                "agent_1": -scaled_loss_penalty,
            }

        if winner == 1:
            return {
                "agent_0": -scaled_loss_penalty,
                "agent_1": self.config.terminal_win_reward,
            }

        if self.config.timeout_center_bias_weight > 0.0:
            return self._timeout_center_biased_rewards()

        return {
            "agent_0": -self.config.terminal_timeout_penalty,
            "agent_1": -self.config.terminal_timeout_penalty,
        }

    def _compute_shaped_bonus(
        self,
        previous_observations: JointObservation | None,
        current_observations: JointObservation,
        unity_done: bool,
        timeout: bool,
    ) -> RewardDict:
        if not self.config.use_shaped_rewards:
            return {"agent_0": 0.0, "agent_1": 0.0}

        bonus = {
            "agent_0": -self.config.step_penalty,
            "agent_1": -self.config.step_penalty,
        }

        for agent_id, opponent_id in (("agent_0", "agent_1"), ("agent_1", "agent_0")):
            current_agent = current_observations[agent_id]
            current_opponent = current_observations[opponent_id]

            own_edge_ratio = self._distance_to_center(current_agent) / max(self.config.arena_radius, 1e-6)
            opp_edge_ratio = self._distance_to_center(current_opponent) / max(self.config.arena_radius, 1e-6)

            bonus[agent_id] += self.config.outward_pressure_weight * opp_edge_ratio
            bonus[agent_id] -= self.config.edge_safety_weight * own_edge_ratio

        if previous_observations is not None:
            for agent_id, opponent_id in (("agent_0", "agent_1"), ("agent_1", "agent_0")):
                prev_opp = self._distance_to_center(previous_observations[opponent_id])
                curr_opp = self._distance_to_center(current_observations[opponent_id])
                progress = (curr_opp - prev_opp) / max(self.config.arena_radius, 1e-6)
                bonus[agent_id] += 0.5 * self.config.outward_pressure_weight * progress

        if unity_done and timeout:
            pass

        return bonus

    @staticmethod
    def _distance_to_center(agent_obs: AgentObservation) -> float:
        pos = agent_obs["selfPosition"]
        return math.sqrt(pos["x"] * pos["x"] + pos["y"] * pos["y"])

    def _scaled_loss_penalty(self, episode_progress: float) -> float:
        progress = min(1.0, max(0.0, episode_progress))
        return self.config.terminal_loss_penalty * (
            1.0 + self.config.terminal_loss_time_scale * (1.0 - progress)
        )

    def _timeout_center_biased_rewards(self) -> RewardDict:
        if self.last_observations is None:
            return {
                "agent_0": -self.config.terminal_timeout_penalty,
                "agent_1": -self.config.terminal_timeout_penalty,
            }

        d0 = self._distance_to_center(self.last_observations["agent_0"])
        d1 = self._distance_to_center(self.last_observations["agent_1"])
        distance_sum = max(d0 + d1, 1e-6)

        penalty0 = self.config.terminal_timeout_penalty + self.config.timeout_center_bias_weight * (d0 / distance_sum)
        penalty1 = self.config.terminal_timeout_penalty + self.config.timeout_center_bias_weight * (d1 / distance_sum)

        return {
            "agent_0": -penalty0,
            "agent_1": -penalty1,
        }

    @staticmethod
    def _validate_actions(actions: JointAction) -> None:
        missing = [key for key in ("agent_0", "agent_1") if key not in actions]
        if missing:
            raise ValueError(
                "Missing required joint action keys: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _ensure_valid_state(state: dict[str, Any], stage: str) -> None:
        status = str(state.get("status", ""))
        bad_prefixes = ("handler_exception", "missing_", "invalid_", "unknown_")
        if any(status.startswith(prefix) for prefix in bad_prefixes):
            raise RuntimeError(
                f"Unity bridge returned an error during {stage}: {state}"
            )
