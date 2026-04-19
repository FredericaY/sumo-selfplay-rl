"""Vectorized Unity-backed environment wrapper for batched self-play arenas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .unity_bridge_client import UnityBridgeClient, UnityBridgeConfig


AgentObservation = dict[str, Any]
JointObservation = dict[str, AgentObservation]
RewardDict = dict[str, float]
InfoDict = dict[str, Any]


@dataclass
class UnityVecSelfPlayArenaConfig:
    host: str = "127.0.0.1"
    port: int = 5055
    timeout_s: float = 5.0
    num_arenas: int = 4
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


class UnityVecSelfPlayArenaEnv:
    """Batched Unity wrapper for running several self-play arenas at once."""

    def __init__(self, config: UnityVecSelfPlayArenaConfig | None = None) -> None:
        self.config = config or UnityVecSelfPlayArenaConfig()
        self.client = UnityBridgeClient(
            UnityBridgeConfig(
                host=self.config.host,
                port=self.config.port,
                timeout_s=self.config.timeout_s,
            )
        )
        self.last_raw_state: dict[str, Any] | None = None
        self.last_observations: list[JointObservation | None] = [None] * self.config.num_arenas
        self.episode_steps: list[int] = [0] * self.config.num_arenas

    def reset(self, arena_seeds: list[int] | None = None) -> list[JointObservation]:
        state = self.client.reset_batch(arena_seeds=arena_seeds)
        self._ensure_valid_state(state, stage="reset_batch")
        self.last_raw_state = state
        observations = self._extract_observations(state)
        self.last_observations = observations[:]
        self.episode_steps = [0] * len(observations)
        return observations

    def reset_arenas(
        self,
        arena_ids: list[int],
        arena_seeds: list[int] | None = None,
    ) -> list[tuple[int, JointObservation]]:
        state = self.client.reset_arenas(arena_ids, arena_seeds=arena_seeds)
        self._ensure_valid_state(state, stage="reset_arenas")
        self.last_raw_state = state

        observations: list[tuple[int, JointObservation]] = []
        for arena_state in state.get("arenas", []):
            arena_id = int(arena_state["arena_id"])
            joint_observation = {
                "agent_0": arena_state["agent0"],
                "agent_1": arena_state["agent1"],
            }
            self.last_observations[arena_id] = joint_observation
            self.episode_steps[arena_id] = 0
            observations.append((arena_id, joint_observation))
        return observations

    def step(
        self,
        actions: list[dict[str, Any]],
    ) -> tuple[list[JointObservation], list[RewardDict], list[bool], list[InfoDict]]:
        state = self.client.step_batch(actions)
        self._ensure_valid_state(state, stage="step_batch")
        self.last_raw_state = state

        observations: list[JointObservation] = [self._empty_observation() for _ in range(self.config.num_arenas)]
        rewards: list[RewardDict] = [{"agent_0": 0.0, "agent_1": 0.0} for _ in range(self.config.num_arenas)]
        dones: list[bool] = [False] * self.config.num_arenas
        infos: list[InfoDict] = [{"arena_id": idx, "winner": -1, "terminal_reason": "missing"} for idx in range(self.config.num_arenas)]

        for arena_state in state.get("arenas", []):
            arena_id = int(arena_state["arena_id"])
            previous_observation = self.last_observations[arena_id]
            self.episode_steps[arena_id] += 1

            joint_observation = {
                "agent_0": arena_state["agent0"],
                "agent_1": arena_state["agent1"],
            }
            base_rewards = {
                "agent_0": float(arena_state["reward0"]),
                "agent_1": float(arena_state["reward1"]),
            }
            unity_done = bool(arena_state["done"])
            python_timeout = self.episode_steps[arena_id] >= self.config.max_episode_steps
            done = unity_done or python_timeout
            terminal_reason = arena_state.get("terminalReason", "running")

            base_rewards = self._apply_terminal_reward_override(
                base_rewards=base_rewards,
                winner=int(arena_state["winner"]),
                done=done,
                unity_done=unity_done,
                python_timeout=python_timeout,
                terminal_reason=terminal_reason,
                episode_progress=min(1.0, self.episode_steps[arena_id] / max(self.config.max_episode_steps, 1)),
                observations=joint_observation,
            )
            shaped_bonus = self._compute_shaped_bonus(
                previous_observations=previous_observation,
                current_observations=joint_observation,
                unity_done=unity_done,
                timeout=python_timeout and not unity_done,
            )

            reward_dict = {
                "agent_0": base_rewards["agent_0"] + shaped_bonus["agent_0"],
                "agent_1": base_rewards["agent_1"] + shaped_bonus["agent_1"],
            }

            observations[arena_id] = joint_observation
            rewards[arena_id] = reward_dict
            dones[arena_id] = done
            infos[arena_id] = {
                "arena_id": arena_id,
                "winner": int(arena_state["winner"]),
                "terminal_reason": terminal_reason,
                "timeout": python_timeout and not unity_done,
                "unity_time_limit": terminal_reason == "time_limit",
                "episode_step": self.episode_steps[arena_id],
                "raw_state": arena_state,
                "base_rewards": base_rewards,
                "shaped_bonus": shaped_bonus,
            }
            self.last_observations[arena_id] = joint_observation

        return observations, rewards, dones, infos

    def get_state(self) -> list[JointObservation]:
        state = self.client.get_batch_state()
        self._ensure_valid_state(state, stage="get_batch_state")
        self.last_raw_state = state
        observations = self._extract_observations(state)
        self.last_observations = observations[:]
        return observations

    def close(self) -> None:
        self.client.close()

    def summary(self) -> str:
        return (
            "name=UnityVecSelfPlayArenaEnv, "
            f"host={self.config.host}, port={self.config.port}, "
            f"num_arenas={self.config.num_arenas}, "
            f"max_episode_steps={self.config.max_episode_steps}, "
            f"use_shaped_rewards={self.config.use_shaped_rewards}"
        )

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

    def _apply_terminal_reward_override(
        self,
        base_rewards: RewardDict,
        winner: int,
        done: bool,
        unity_done: bool,
        python_timeout: bool,
        terminal_reason: str,
        episode_progress: float,
        observations: JointObservation,
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
            return self._timeout_center_biased_rewards(observations)

        return {
            "agent_0": -self.config.terminal_timeout_penalty,
            "agent_1": -self.config.terminal_timeout_penalty,
        }

    @staticmethod
    def _distance_to_center(agent_obs: AgentObservation) -> float:
        pos = agent_obs["selfPosition"]
        return math.sqrt(pos["x"] * pos["x"] + pos["y"] * pos["y"])

    def _scaled_loss_penalty(self, episode_progress: float) -> float:
        progress = min(1.0, max(0.0, episode_progress))
        return self.config.terminal_loss_penalty * (
            1.0 + self.config.terminal_loss_time_scale * (1.0 - progress)
        )

    def _timeout_center_biased_rewards(self, observations: JointObservation) -> RewardDict:
        d0 = self._distance_to_center(observations["agent_0"])
        d1 = self._distance_to_center(observations["agent_1"])
        distance_sum = max(d0 + d1, 1e-6)

        penalty0 = self.config.terminal_timeout_penalty + self.config.timeout_center_bias_weight * (d0 / distance_sum)
        penalty1 = self.config.terminal_timeout_penalty + self.config.timeout_center_bias_weight * (d1 / distance_sum)

        return {
            "agent_0": -penalty0,
            "agent_1": -penalty1,
        }

    @staticmethod
    def _extract_observations(state: dict[str, Any]) -> list[JointObservation]:
        return [
            {
                "agent_0": arena_state["agent0"],
                "agent_1": arena_state["agent1"],
            }
            for arena_state in state.get("arenas", [])
        ]

    @staticmethod
    def _ensure_valid_state(state: dict[str, Any], stage: str) -> None:
        status = str(state.get("status", ""))
        bad_prefixes = ("handler_exception", "missing_", "invalid_", "unknown_")
        if any(status.startswith(prefix) for prefix in bad_prefixes):
            raise RuntimeError(
                f"Unity batch bridge returned an error during {stage}: {state}"
            )

    @staticmethod
    def _empty_observation() -> JointObservation:
        empty_agent = {
            "selfPosition": {"x": 0.0, "y": 0.0},
            "selfVelocity": {"x": 0.0, "y": 0.0},
            "opponentPosition": {"x": 0.0, "y": 0.0},
            "opponentVelocity": {"x": 0.0, "y": 0.0},
            "pushReady": False,
        }
        return {
            "agent_0": empty_agent,
            "agent_1": empty_agent,
        }
