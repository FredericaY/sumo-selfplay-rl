"""Evaluation helpers for self-play experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
from typing import Any

from agents import HeuristicVectorPolicy, MLPVectorPolicy, RandomVectorPolicy, VectorPolicy
from agents.imitation_trainer import ImitationTrainConfig, ImitationTrainer
from envs import UnitySelfPlayArenaConfig, UnitySelfPlayArenaEnv
from envs.action_adapter import ActionAdapter
from envs.observation_adapter import DEFAULT_OBS_DIM, ObservationAdapter, ObservationVectorConfig


OBS_DIM = DEFAULT_OBS_DIM


@dataclass
class EvaluationConfig:
    """Configuration for repeated checkpoint evaluation in Unity."""

    checkpoint_path: str
    host: str = "127.0.0.1"
    port: int = 5055
    max_steps: int = 400
    episodes: int = 5
    opponent_policy: str = "flee"
    seed: int = 11
    debug_every: int = 0


def build_policy_from_checkpoint(checkpoint_path: str, obs_dim: int = OBS_DIM) -> MLPVectorPolicy:
    """Load an MLP policy from a saved imitation checkpoint."""

    trainer = ImitationTrainer(ImitationTrainConfig(obs_dim=obs_dim))
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    policy = MLPVectorPolicy(obs_dim=obs_dim)
    policy.load_state_dict(checkpoint["state_dict"])
    return policy


def build_opponent_policy(name: str, seed: int) -> VectorPolicy:
    """Construct a simple baseline opponent policy."""

    if name == "random":
        return RandomVectorPolicy(seed=seed)
    return HeuristicVectorPolicy(mode=name)


def format_agent(agent_obs: dict[str, Any]) -> str:
    """Build a short debug string for one agent observation."""

    pos = agent_obs["selfPosition"]
    vel = agent_obs["selfVelocity"]
    return (
        f"pos=({pos['x']:.2f},{pos['y']:.2f}) "
        f"vel=({vel['x']:.2f},{vel['y']:.2f}) "
        f"push_ready={agent_obs['pushReady']}"
    )


def run_single_evaluation_episode(
    env: UnitySelfPlayArenaEnv,
    policy_agent_0: VectorPolicy,
    policy_agent_1: VectorPolicy,
    debug_every: int = 0,
) -> dict[str, Any]:
    """Run one evaluation episode and return a compact summary."""

    observation_adapter = ObservationAdapter(
        ObservationVectorConfig(arena_radius=env.config.arena_radius)
    )
    action_adapter = ActionAdapter()

    observations = env.reset()
    done = False
    final_info: dict[str, Any] = {
        "winner": -1,
        "terminal_reason": "unknown",
        "base_rewards": {"agent_0": 0.0, "agent_1": 0.0},
        "shaped_bonus": {"agent_0": 0.0, "agent_1": 0.0},
    }

    while not done:
        step_idx = env.episode_step
        mirror_flags = observation_adapter.joint_mirror_flags(observations)
        observation_vectors = observation_adapter.vectorize_joint(observations)
        policy_action_vectors = {
            "agent_0": policy_agent_0.act(observation_vectors["agent_0"], step_idx),
            "agent_1": policy_agent_1.act(observation_vectors["agent_1"], step_idx),
        }
        actions = {
            "agent_0": action_adapter.action_from_vector(
                policy_action_vectors["agent_0"],
                mirror_x=mirror_flags["agent_0"],
            ),
            "agent_1": action_adapter.action_from_vector(
                policy_action_vectors["agent_1"],
                mirror_x=mirror_flags["agent_1"],
            ),
        }
        observations, rewards, done, info = env.step(actions)
        final_info = info

        if debug_every > 0 and (step_idx % debug_every == 0 or done):
            print(
                f"step={step_idx:03d} done={done} winner={info.get('winner', -1)} "
                f"reason={info.get('terminal_reason', 'running')} "
                f"agent0[{format_agent(observations['agent_0'])}] "
                f"agent1[{format_agent(observations['agent_1'])}]"
            )

    returns = {
        "agent_0": final_info["base_rewards"]["agent_0"] + final_info["shaped_bonus"]["agent_0"],
        "agent_1": final_info["base_rewards"]["agent_1"] + final_info["shaped_bonus"]["agent_1"],
    }
    return {
        "winner": int(final_info.get("winner", -1)),
        "terminal_reason": final_info.get("terminal_reason", "unknown"),
        "episode_steps": int(final_info.get("episode_step", env.episode_step)),
        "returns": returns,
    }


def evaluate_policy(checkpoint_path: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one checkpoint over multiple Unity episodes."""

    config = config or {}
    eval_config = EvaluationConfig(checkpoint_path=checkpoint_path, **config)

    policy = build_policy_from_checkpoint(eval_config.checkpoint_path)
    opponent = build_opponent_policy(eval_config.opponent_policy, seed=eval_config.seed)
    env = UnitySelfPlayArenaEnv(
        UnitySelfPlayArenaConfig(
            host=eval_config.host,
            port=eval_config.port,
            timeout_s=5.0,
            max_episode_steps=eval_config.max_steps,
        )
    )

    results: list[dict[str, Any]] = []
    try:
        try:
            for episode_idx in range(eval_config.episodes):
                if eval_config.debug_every > 0:
                    print(f"episode={episode_idx:03d} starting")
                episode_result = run_single_evaluation_episode(
                    env=env,
                    policy_agent_0=policy,
                    policy_agent_1=opponent,
                    debug_every=eval_config.debug_every,
                )
                results.append(episode_result)
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise RuntimeError(
                "Could not connect to the Unity TCP bridge. "
                "Make sure Unity is in Play mode and ArenaTcpBridge is active. "
                f"Original error: {exc}"
            ) from exc
    finally:
        env.close()

    agent_0_wins = sum(1 for result in results if result["winner"] == 0)
    agent_1_wins = sum(1 for result in results if result["winner"] == 1)
    draws = len(results) - agent_0_wins - agent_1_wins
    avg_episode_steps = sum(result["episode_steps"] for result in results) / max(len(results), 1)
    avg_return_agent_0 = sum(result["returns"]["agent_0"] for result in results) / max(len(results), 1)
    avg_return_agent_1 = sum(result["returns"]["agent_1"] for result in results) / max(len(results), 1)
    time_limit_draws = sum(1 for result in results if result["terminal_reason"] == "time_limit")

    return {
        "checkpoint": str(Path(checkpoint_path)),
        "episodes": len(results),
        "opponent_policy": eval_config.opponent_policy,
        "agent_0_wins": agent_0_wins,
        "agent_1_wins": agent_1_wins,
        "draws": draws,
        "time_limit_draws": time_limit_draws,
        "win_rate": agent_0_wins / max(len(results), 1),
        "avg_episode_steps": avg_episode_steps,
        "avg_return_agent_0": avg_return_agent_0,
        "avg_return_agent_1": avg_return_agent_1,
        "episode_results": results,
    }
