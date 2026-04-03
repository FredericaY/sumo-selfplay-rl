"""Trajectory collection helpers for Unity self-play rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents import VectorPolicy
from envs import UnitySelfPlayArenaEnv
from envs.action_adapter import ActionAdapter
from envs.observation_adapter import ObservationAdapter


@dataclass
class Transition:
    """Single environment transition for one joint step."""

    step_idx: int
    observations: dict[str, dict[str, Any]]
    observation_vectors: dict[str, list[float]]
    actions: dict[str, dict[str, Any]]
    policy_action_vectors: dict[str, list[float]]
    executed_action_vectors: dict[str, list[float]]
    rewards: dict[str, float]
    done: bool
    info: dict[str, Any]


@dataclass
class RolloutEpisode:
    """Collected data for a full episode."""

    initial_observations: dict[str, dict[str, Any]]
    initial_observation_vectors: dict[str, list[float]]
    transitions: list[Transition] = field(default_factory=list)

    def total_returns(self) -> dict[str, float]:
        """Return undiscounted per-agent episodic returns."""

        returns = {"agent_0": 0.0, "agent_1": 0.0}
        for transition in self.transitions:
            for agent_id in returns:
                returns[agent_id] += float(transition.rewards[agent_id])
        return returns


class RolloutCollector:
    """Collect full episodes from the Unity arena using vector policies."""

    def __init__(
        self,
        env: UnitySelfPlayArenaEnv,
        observation_adapter: ObservationAdapter | None = None,
        action_adapter: ActionAdapter | None = None,
    ) -> None:
        self.env = env
        self.observation_adapter = observation_adapter or ObservationAdapter()
        self.action_adapter = action_adapter or ActionAdapter()

    def collect_episode(
        self,
        policy_agent_0: VectorPolicy,
        policy_agent_1: VectorPolicy,
    ) -> RolloutEpisode:
        """Run one episode and record all transitions."""

        observations = self.env.reset()
        observation_vectors = self.observation_adapter.vectorize_joint(observations)
        episode = RolloutEpisode(
            initial_observations=observations,
            initial_observation_vectors=observation_vectors,
        )

        done = False
        while not done:
            step_idx = self.env.episode_step
            policy_action_vectors = {
                "agent_0": policy_agent_0.act(observation_vectors["agent_0"], step_idx),
                "agent_1": policy_agent_1.act(observation_vectors["agent_1"], step_idx),
            }
            actions = {
                "agent_0": self.action_adapter.action_from_vector(policy_action_vectors["agent_0"]),
                "agent_1": self.action_adapter.action_from_vector(policy_action_vectors["agent_1"]),
            }
            executed_action_vectors = self.action_adapter.vectorize_joint(actions)
            next_observations, rewards, done, info = self.env.step(actions)
            next_observation_vectors = self.observation_adapter.vectorize_joint(next_observations)
            transition = Transition(
                step_idx=step_idx,
                observations=next_observations,
                observation_vectors=next_observation_vectors,
                actions=actions,
                policy_action_vectors=policy_action_vectors,
                executed_action_vectors=executed_action_vectors,
                rewards=rewards,
                done=done,
                info=info,
            )
            episode.transitions.append(transition)
            observations = next_observations
            observation_vectors = next_observation_vectors

        return episode
