"""Minimal PPO trainer for the self-play arena."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from agents import ActorCritic, ActorCriticConfig, HeuristicVectorPolicy, RandomVectorPolicy
from algorithms.ppo_buffer import PPOBatch, PPOBuffer
from envs import (
    UnitySelfPlayArenaConfig,
    UnitySelfPlayArenaEnv,
    UnityVecSelfPlayArenaConfig,
    UnityVecSelfPlayArenaEnv,
)
from envs.action_adapter import ActionAdapter
from envs.observation_adapter import DEFAULT_OBS_DIM, ObservationAdapter, ObservationVectorConfig


@dataclass
class PPOTrainConfig:
    seed: int = 42
    total_updates: int = 10
    steps_per_update: int = 256
    train_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_episode_steps: int = 400
    num_envs: int = 1
    use_shaped_rewards: bool = False
    step_penalty: float = 0.0
    edge_safety_weight: float = 0.0
    outward_pressure_weight: float = 0.0
    terminal_timeout_penalty: float = 0.0
    timeout_center_bias_weight: float = 0.0
    terminal_win_reward: float = 1.0
    terminal_loss_penalty: float = 1.5
    terminal_loss_time_scale: float = 0.5
    canonicalize_left_right: bool = True
    opponent_position_scale: float = 1.5
    relative_position_scale: float = 2.0
    opponent_policy: str = "flee"
    train_mode: str = "single_agent_baseline"
    train_side: str = "a"
    checkpoint_dir: str = "python/checkpoints/ppo"
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 5055
    debug_every: int = 0
    step_sleep_s: float = 0.0
    init_checkpoint: str | None = None
    load_optimizer_state: bool = False
    finish_active_episodes_before_exit: bool = True


class PPOTrainer:
    """Train PPO against a baseline, with shared self-play, or with alternating two-policy self-play."""

    def __init__(self, config: PPOTrainConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for PPOTrainer. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.config = config or PPOTrainConfig()
        valid_modes = {"single_agent_baseline", "shared_selfplay", "alternating_two_policy"}
        if self.config.train_mode not in valid_modes:
            raise ValueError(f"Unsupported train_mode: {self.config.train_mode}")
        if self.config.train_side not in {"a", "b"}:
            raise ValueError(f"Unsupported train_side: {self.config.train_side}")

        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)

        if self.config.num_envs > 1:
            self.env = UnityVecSelfPlayArenaEnv(
                UnityVecSelfPlayArenaConfig(
                    host=self.config.host,
                    port=self.config.port,
                    timeout_s=5.0,
                    num_arenas=self.config.num_envs,
                    max_episode_steps=self.config.max_episode_steps,
                    use_shaped_rewards=self.config.use_shaped_rewards,
                    step_penalty=self.config.step_penalty,
                    edge_safety_weight=self.config.edge_safety_weight,
                    outward_pressure_weight=self.config.outward_pressure_weight,
                    terminal_timeout_penalty=self.config.terminal_timeout_penalty,
                    timeout_center_bias_weight=self.config.timeout_center_bias_weight,
                    terminal_win_reward=self.config.terminal_win_reward,
                    terminal_loss_penalty=self.config.terminal_loss_penalty,
                    terminal_loss_time_scale=self.config.terminal_loss_time_scale,
                )
            )
            arena_radius = self.env.config.arena_radius
        else:
            self.env = UnitySelfPlayArenaEnv(
                UnitySelfPlayArenaConfig(
                    host=self.config.host,
                    port=self.config.port,
                    timeout_s=5.0,
                    max_episode_steps=self.config.max_episode_steps,
                    use_shaped_rewards=self.config.use_shaped_rewards,
                    step_penalty=self.config.step_penalty,
                    edge_safety_weight=self.config.edge_safety_weight,
                    outward_pressure_weight=self.config.outward_pressure_weight,
                    terminal_timeout_penalty=self.config.terminal_timeout_penalty,
                    timeout_center_bias_weight=self.config.timeout_center_bias_weight,
                    terminal_win_reward=self.config.terminal_win_reward,
                    terminal_loss_penalty=self.config.terminal_loss_penalty,
                    terminal_loss_time_scale=self.config.terminal_loss_time_scale,
                )
            )
            arena_radius = self.env.config.arena_radius
        self.observation_adapter = ObservationAdapter(
            ObservationVectorConfig(
                arena_radius=arena_radius,
                canonicalize_left_right=self.config.canonicalize_left_right,
                opponent_position_scale=self.config.opponent_position_scale,
                relative_position_scale=self.config.relative_position_scale,
            )
        )
        self.action_adapter = ActionAdapter()
        self.buffer = PPOBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        self.opponent = self._build_opponent(self.config.opponent_policy)
        self.update_offset = 0

        self.policy = ActorCritic(ActorCriticConfig(obs_dim=DEFAULT_OBS_DIM)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

        self.policy_a = ActorCritic(ActorCriticConfig(obs_dim=DEFAULT_OBS_DIM)).to(self.device)
        self.policy_b = ActorCritic(ActorCriticConfig(obs_dim=DEFAULT_OBS_DIM)).to(self.device)
        self.optimizer_a = torch.optim.Adam(self.policy_a.parameters(), lr=self.config.learning_rate)
        self.optimizer_b = torch.optim.Adam(self.policy_b.parameters(), lr=self.config.learning_rate)

        if self.config.init_checkpoint:
            checkpoint_info = self.load_checkpoint(
                self.config.init_checkpoint,
                load_optimizer_state=self.config.load_optimizer_state,
            )
            metadata = checkpoint_info.get("metadata", {})
            self.update_offset = int(metadata.get("update", -1)) + 1

    def train(self) -> dict[str, Any]:
        if self.config.num_envs > 1:
            return self._train_vectorized()

        return self._train_single()

    def _train_single(self) -> dict[str, Any]:
        history: dict[str, Any] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_agent0_return": [],
            "mean_agent1_return": [],
            "reset_counts": [],
            "reset_events": [],
            "last_checkpoint": None,
        }
        observations = self.env.reset()
        episode_return_agent0 = 0.0
        episode_return_agent1 = 0.0
        completed_returns_agent0: list[float] = []
        completed_returns_agent1: list[float] = []

        try:
            for local_update_idx in range(self.config.total_updates):
                global_update_idx = self.update_offset + local_update_idx
                self.buffer.clear()
                for step_in_update in range(self.config.steps_per_update):
                    step_idx = self.env.episode_step
                    mirror_flags = self.observation_adapter.joint_mirror_flags(observations)
                    observation_vectors = self.observation_adapter.vectorize_joint(observations)

                    if self.config.train_mode == "alternating_two_policy":
                        agent_0_action_vector, agent_0_log_prob, agent_0_value = self._sample_with_policy(
                            self.policy_a,
                            observation_vectors["agent_0"],
                        )
                        agent_1_action_vector, agent_1_log_prob, agent_1_value = self._sample_with_policy(
                            self.policy_b,
                            observation_vectors["agent_1"],
                        )
                    else:
                        agent_0_action_vector, agent_0_log_prob, agent_0_value = self._sample_with_policy(
                            self.policy,
                            observation_vectors["agent_0"],
                        )
                        if self.config.train_mode == "shared_selfplay":
                            agent_1_action_vector, agent_1_log_prob, agent_1_value = self._sample_with_policy(
                                self.policy,
                                observation_vectors["agent_1"],
                            )
                        else:
                            agent_1_action_vector = self.opponent.act(observation_vectors["agent_1"], step_idx)
                            agent_1_log_prob = None
                            agent_1_value = None

                    joint_actions = {
                        "agent_0": self.action_adapter.action_from_vector(
                            agent_0_action_vector,
                            mirror_x=mirror_flags["agent_0"],
                        ),
                        "agent_1": self.action_adapter.action_from_vector(
                            agent_1_action_vector,
                            mirror_x=mirror_flags["agent_1"],
                        ),
                    }

                    next_observations, rewards, done, info = self.env.step(joint_actions)
                    reward_agent0 = float(rewards["agent_0"])
                    reward_agent1 = float(rewards["agent_1"])
                    episode_return_agent0 += reward_agent0
                    episode_return_agent1 += reward_agent1

                    if self.config.train_mode == "alternating_two_policy":
                        if self.config.train_side == "a":
                            self.buffer.add(
                                observation=observation_vectors["agent_0"],
                                action=agent_0_action_vector,
                                reward=reward_agent0,
                                done=done,
                                value=agent_0_value,
                                log_prob=agent_0_log_prob,
                            )
                        else:
                            self.buffer.add(
                                observation=observation_vectors["agent_1"],
                                action=agent_1_action_vector,
                                reward=reward_agent1,
                                done=done,
                                value=agent_1_value,
                                log_prob=agent_1_log_prob,
                            )
                    else:
                        self.buffer.add(
                            observation=observation_vectors["agent_0"],
                            action=agent_0_action_vector,
                            reward=reward_agent0,
                            done=done,
                            value=agent_0_value,
                            log_prob=agent_0_log_prob,
                        )

                        if self.config.train_mode == "shared_selfplay":
                            self.buffer.add(
                                observation=observation_vectors["agent_1"],
                                action=agent_1_action_vector,
                                reward=reward_agent1,
                                done=done,
                                value=agent_1_value,
                                log_prob=agent_1_log_prob,
                            )

                    observations = next_observations
                    if done:
                        completed_returns_agent0.append(episode_return_agent0)
                        completed_returns_agent1.append(episode_return_agent1)
                        episode_return_agent0 = 0.0
                        episode_return_agent1 = 0.0
                        observations = self.env.reset()

                    if self.config.debug_every > 0 and step_in_update % self.config.debug_every == 0:
                        print(
                            f"update={global_update_idx:03d} step={step_in_update:03d} "
                            f"env_step={info.get('episode_step', -1):03d} "
                            f"reward0={reward_agent0:.3f} reward1={reward_agent1:.3f} done={done}"
                        )

                    if self.config.step_sleep_s > 0.0:
                        time.sleep(self.config.step_sleep_s)

                batch = self.buffer.compute_batch(last_value=0.0)
                metrics = self._ppo_update(batch)
                history["policy_loss"].append(metrics["policy_loss"])
                history["value_loss"].append(metrics["value_loss"])
                history["entropy"].append(metrics["entropy"])
                mean_agent0_return = (
                    sum(completed_returns_agent0) / max(len(completed_returns_agent0), 1)
                    if completed_returns_agent0 else 0.0
                )
                mean_agent1_return = (
                    sum(completed_returns_agent1) / max(len(completed_returns_agent1), 1)
                    if completed_returns_agent1 else 0.0
                )
                history["mean_agent0_return"].append(mean_agent0_return)
                history["mean_agent1_return"].append(mean_agent1_return)

                checkpoint_name = (
                    f"twopolicy_update_{global_update_idx:04d}.pt"
                    if self.config.train_mode == "alternating_two_policy"
                    else f"ppo_update_{global_update_idx:04d}.pt"
                )
                checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
                self.save_checkpoint(
                    checkpoint_path,
                    metadata={
                        "update": global_update_idx,
                        "train_mode": self.config.train_mode,
                        "train_side": self.config.train_side,
                        **metrics,
                    },
                )
                history["last_checkpoint"] = str(checkpoint_path)

                print(
                    f"update={global_update_idx:03d} mode={self.config.train_mode} "
                    f"train_side={self.config.train_side if self.config.train_mode == 'alternating_two_policy' else '-'} "
                    f"policy_loss={metrics['policy_loss']:.6f} "
                    f"value_loss={metrics['value_loss']:.6f} entropy={metrics['entropy']:.6f} "
                    f"mean_agent0_return={mean_agent0_return:.3f} "
                    f"mean_agent1_return={mean_agent1_return:.3f}"
                )

            if self.config.finish_active_episodes_before_exit:
                drain_steps = self._drain_active_single_episode(
                    observations=observations,
                    episode_return_agent0=episode_return_agent0,
                    episode_return_agent1=episode_return_agent1,
                    completed_returns_agent0=completed_returns_agent0,
                    completed_returns_agent1=completed_returns_agent1,
                )
                if drain_steps > 0:
                    print(f"Drained active single-arena episode before exit: extra_steps={drain_steps}")

            return history
        finally:
            self.env.close()

    def _train_vectorized(self) -> dict[str, Any]:
        history: dict[str, Any] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_agent0_return": [],
            "mean_agent1_return": [],
            "reset_counts": [],
            "reset_events": [],
            "last_checkpoint": None,
        }

        episode_counts = [0] * self.config.num_envs
        observations = self.env.reset(
            arena_seeds=[self._derive_arena_seed(0, env_idx, episode_counts[env_idx]) for env_idx in range(self.config.num_envs)]
        )
        episode_return_agent0 = [0.0] * self.config.num_envs
        episode_return_agent1 = [0.0] * self.config.num_envs
        completed_returns_agent0: list[float] = []
        completed_returns_agent1: list[float] = []

        try:
            for local_update_idx in range(self.config.total_updates):
                global_update_idx = self.update_offset + local_update_idx
                self.buffer.clear()
                reset_counts_this_update = [0] * self.config.num_envs
                reset_events_this_update: list[list[int]] = []

                for step_in_update in range(self.config.steps_per_update):
                    batched_actions: list[dict[str, Any]] = []

                    for env_idx in range(self.config.num_envs):
                        mirror_flags = self.observation_adapter.joint_mirror_flags(observations[env_idx])
                        observation_vectors = self.observation_adapter.vectorize_joint(observations[env_idx])

                        if self.config.train_mode == "alternating_two_policy":
                            agent_0_action_vector, agent_0_log_prob, agent_0_value = self._sample_with_policy(
                                self.policy_a,
                                observation_vectors["agent_0"],
                            )
                            agent_1_action_vector, agent_1_log_prob, agent_1_value = self._sample_with_policy(
                                self.policy_b,
                                observation_vectors["agent_1"],
                            )
                        else:
                            agent_0_action_vector, agent_0_log_prob, agent_0_value = self._sample_with_policy(
                                self.policy,
                                observation_vectors["agent_0"],
                            )
                            if self.config.train_mode == "shared_selfplay":
                                agent_1_action_vector, agent_1_log_prob, agent_1_value = self._sample_with_policy(
                                    self.policy,
                                    observation_vectors["agent_1"],
                                )
                            else:
                                agent_1_action_vector = self.opponent.act(observation_vectors["agent_1"], step_in_update)
                                agent_1_log_prob = None
                                agent_1_value = None

                        batched_actions.append(
                            {
                                "arena_id": env_idx,
                                "agent0": self.action_adapter.action_from_vector(
                                    agent_0_action_vector,
                                    mirror_x=mirror_flags["agent_0"],
                                ),
                                "agent1": self.action_adapter.action_from_vector(
                                    agent_1_action_vector,
                                    mirror_x=mirror_flags["agent_1"],
                                ),
                            }
                        )

                        batched_actions[-1]["_cached"] = {
                            "obs": observation_vectors,
                            "mirror_flags": mirror_flags,
                            "agent0_action_vector": agent_0_action_vector,
                            "agent0_log_prob": agent_0_log_prob,
                            "agent0_value": agent_0_value,
                            "agent1_action_vector": agent_1_action_vector,
                            "agent1_log_prob": agent_1_log_prob,
                            "agent1_value": agent_1_value,
                        }

                    next_observations, rewards, dones, infos = self.env.step(batched_actions)
                    reset_ids: list[int] = []
                    reset_seeds: list[int] = []

                    for env_idx in range(self.config.num_envs):
                        cached = batched_actions[env_idx]["_cached"]
                        reward_agent0 = float(rewards[env_idx]["agent_0"])
                        reward_agent1 = float(rewards[env_idx]["agent_1"])
                        done = bool(dones[env_idx])
                        info = infos[env_idx]

                        episode_return_agent0[env_idx] += reward_agent0
                        episode_return_agent1[env_idx] += reward_agent1

                        if self.config.train_mode == "alternating_two_policy":
                            if self.config.train_side == "a":
                                self.buffer.add(
                                    observation=cached["obs"]["agent_0"],
                                    action=cached["agent0_action_vector"],
                                    reward=reward_agent0,
                                    done=done,
                                    value=cached["agent0_value"],
                                    log_prob=cached["agent0_log_prob"],
                                )
                            else:
                                self.buffer.add(
                                    observation=cached["obs"]["agent_1"],
                                    action=cached["agent1_action_vector"],
                                    reward=reward_agent1,
                                    done=done,
                                    value=cached["agent1_value"],
                                    log_prob=cached["agent1_log_prob"],
                                )
                        else:
                            self.buffer.add(
                                observation=cached["obs"]["agent_0"],
                                action=cached["agent0_action_vector"],
                                reward=reward_agent0,
                                done=done,
                                value=cached["agent0_value"],
                                log_prob=cached["agent0_log_prob"],
                            )

                            if self.config.train_mode == "shared_selfplay":
                                self.buffer.add(
                                    observation=cached["obs"]["agent_1"],
                                    action=cached["agent1_action_vector"],
                                    reward=reward_agent1,
                                    done=done,
                                    value=cached["agent1_value"],
                                    log_prob=cached["agent1_log_prob"],
                                )

                        if done:
                            completed_returns_agent0.append(episode_return_agent0[env_idx])
                            completed_returns_agent1.append(episode_return_agent1[env_idx])
                            episode_return_agent0[env_idx] = 0.0
                            episode_return_agent1[env_idx] = 0.0
                            episode_counts[env_idx] += 1
                            reset_ids.append(env_idx)
                            reset_seeds.append(self._derive_arena_seed(global_update_idx, env_idx, episode_counts[env_idx]))

                        if self.config.debug_every > 0 and step_in_update % self.config.debug_every == 0:
                            print(
                                f"update={global_update_idx:03d} env={env_idx:02d} step={step_in_update:03d} "
                                f"arena_step={info.get('episode_step', -1):03d} "
                                f"reward0={reward_agent0:.3f} reward1={reward_agent1:.3f} done={done}"
                            )

                    observations = next_observations

                    if reset_ids:
                        reset_events_this_update.append(list(reset_ids))
                        for arena_id in reset_ids:
                            reset_counts_this_update[arena_id] += 1
                        reset_results = self.env.reset_arenas(reset_ids, reset_seeds)
                        for arena_id, arena_observation in reset_results:
                            observations[arena_id] = arena_observation

                    if self.config.step_sleep_s > 0.0:
                        time.sleep(self.config.step_sleep_s)

                batch = self.buffer.compute_batch(last_value=0.0)
                metrics = self._ppo_update(batch)
                history["policy_loss"].append(metrics["policy_loss"])
                history["value_loss"].append(metrics["value_loss"])
                history["entropy"].append(metrics["entropy"])
                mean_agent0_return = (
                    sum(completed_returns_agent0) / max(len(completed_returns_agent0), 1)
                    if completed_returns_agent0 else 0.0
                )
                mean_agent1_return = (
                    sum(completed_returns_agent1) / max(len(completed_returns_agent1), 1)
                    if completed_returns_agent1 else 0.0
                )
                history["mean_agent0_return"].append(mean_agent0_return)
                history["mean_agent1_return"].append(mean_agent1_return)
                history["reset_counts"].append(list(reset_counts_this_update))
                history["reset_events"].append([list(ids) for ids in reset_events_this_update])

                checkpoint_name = (
                    f"twopolicy_update_{global_update_idx:04d}.pt"
                    if self.config.train_mode == "alternating_two_policy"
                    else f"ppo_update_{global_update_idx:04d}.pt"
                )
                checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
                self.save_checkpoint(
                    checkpoint_path,
                    metadata={
                        "update": global_update_idx,
                        "train_mode": self.config.train_mode,
                        "train_side": self.config.train_side,
                        "num_envs": self.config.num_envs,
                        **metrics,
                    },
                )
                history["last_checkpoint"] = str(checkpoint_path)

                print(
                    f"update={global_update_idx:03d} mode={self.config.train_mode} "
                    f"train_side={self.config.train_side if self.config.train_mode == 'alternating_two_policy' else '-'} "
                    f"num_envs={self.config.num_envs} "
                    f"policy_loss={metrics['policy_loss']:.6f} "
                    f"value_loss={metrics['value_loss']:.6f} entropy={metrics['entropy']:.6f} "
                    f"mean_agent0_return={mean_agent0_return:.3f} "
                    f"mean_agent1_return={mean_agent1_return:.3f} "
                    f"reset_counts={reset_counts_this_update} "
                    f"reset_events={reset_events_this_update[:8]}"
                )

            if self.config.finish_active_episodes_before_exit:
                drain_steps, drained_envs = self._drain_active_vectorized_episodes(
                    observations=observations,
                    episode_return_agent0=episode_return_agent0,
                    episode_return_agent1=episode_return_agent1,
                    completed_returns_agent0=completed_returns_agent0,
                    completed_returns_agent1=completed_returns_agent1,
                )
                if drain_steps > 0:
                    print(
                        "Drained active vectorized episodes before exit: "
                        f"extra_steps={drain_steps} finished_envs={drained_envs}"
                    )

            return history
        finally:
            self.env.close()

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.train_mode == "alternating_two_policy":
            payload = {
                "policy_a_state_dict": self.policy_a.state_dict(),
                "policy_b_state_dict": self.policy_b.state_dict(),
                "optimizer_a_state": self.optimizer_a.state_dict(),
                "optimizer_b_state": self.optimizer_b.state_dict(),
                "config": self.config.__dict__,
                "metadata": metadata or {},
            }
        else:
            payload = {
                "state_dict": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "metadata": metadata or {},
            }
        torch.save(payload, output_path)
        return output_path

    def load_checkpoint(self, path: str | Path, load_optimizer_state: bool = False) -> dict[str, Any]:
        checkpoint_path = Path(path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "policy_a_state_dict" in checkpoint and "policy_b_state_dict" in checkpoint:
            self.policy_a.load_state_dict(checkpoint["policy_a_state_dict"])
            self.policy_b.load_state_dict(checkpoint["policy_b_state_dict"])
            if load_optimizer_state:
                if checkpoint.get("optimizer_a_state") is not None:
                    self.optimizer_a.load_state_dict(checkpoint["optimizer_a_state"])
                if checkpoint.get("optimizer_b_state") is not None:
                    self.optimizer_b.load_state_dict(checkpoint["optimizer_b_state"])
        elif "state_dict" in checkpoint:
            self.policy.load_state_dict(checkpoint["state_dict"])
            self.policy_a.load_state_dict(checkpoint["state_dict"])
            self.policy_b.load_state_dict(checkpoint["state_dict"])
            if load_optimizer_state and checkpoint.get("optimizer_state") is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        return checkpoint

    def _sample_with_policy(
        self,
        policy: ActorCritic,
        observation_vector: list[float],
    ) -> tuple[list[float], float, float]:
        obs_tensor = torch.tensor(observation_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob_tensor, _entropy_tensor, value_tensor = policy.sample_action(obs_tensor)
        return (
            action_tensor.squeeze(0).cpu().tolist(),
            float(log_prob_tensor.squeeze(0).cpu()),
            float(value_tensor.squeeze(0).cpu()),
        )

    def _ppo_update(self, batch: PPOBatch) -> dict[str, float]:
        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.log_probs.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)

        if self.config.train_mode == "alternating_two_policy":
            if self.config.train_side == "a":
                target_policy = self.policy_a
                target_optimizer = self.optimizer_a
            else:
                target_policy = self.policy_b
                target_optimizer = self.optimizer_b
        else:
            target_policy = self.policy
            target_optimizer = self.optimizer

        num_samples = observations.shape[0]
        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0

        for _epoch_idx in range(self.config.train_epochs):
            permutation = torch.randperm(num_samples)
            for start in range(0, num_samples, self.config.minibatch_size):
                indices = permutation[start:start + self.config.minibatch_size]
                mini_obs = observations[indices]
                mini_actions = actions[indices]
                mini_old_log_probs = old_log_probs[indices]
                mini_returns = returns[indices]
                mini_advantages = advantages[indices]

                new_log_probs, entropy, values = target_policy.evaluate_actions(mini_obs, mini_actions)
                ratio = torch.exp(new_log_probs - mini_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * mini_advantages, clipped_ratio * mini_advantages).mean()
                value_loss = torch.mean((mini_returns - values) ** 2)
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_bonus
                target_optimizer.zero_grad()
                loss.backward()
                target_optimizer.step()

                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                entropy_value = float(entropy_bonus.detach().cpu())

        return {
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
        }

    def _drain_active_single_episode(
        self,
        observations: dict[str, Any],
        episode_return_agent0: float,
        episode_return_agent1: float,
        completed_returns_agent0: list[float],
        completed_returns_agent1: list[float],
    ) -> int:
        if self.env.last_raw_state is not None and bool(self.env.last_raw_state.get("done", False)):
            return 0
        if self.env.episode_step <= 0:
            return 0

        drain_steps = 0
        max_drain_steps = max(1, self.config.max_episode_steps)
        current_observations = observations
        current_return0 = episode_return_agent0
        current_return1 = episode_return_agent1

        while drain_steps < max_drain_steps:
            step_idx = self.env.episode_step
            mirror_flags = self.observation_adapter.joint_mirror_flags(current_observations)
            observation_vectors = self.observation_adapter.vectorize_joint(current_observations)
            agent_0_action_vector, agent_1_action_vector = self._policy_action_vectors_for_mode(
                observation_vectors=observation_vectors,
                step_idx=step_idx,
            )
            joint_actions = {
                "agent_0": self.action_adapter.action_from_vector(
                    agent_0_action_vector,
                    mirror_x=mirror_flags["agent_0"],
                ),
                "agent_1": self.action_adapter.action_from_vector(
                    agent_1_action_vector,
                    mirror_x=mirror_flags["agent_1"],
                ),
            }
            next_observations, rewards, done, _info = self.env.step(joint_actions)
            current_return0 += float(rewards["agent_0"])
            current_return1 += float(rewards["agent_1"])
            drain_steps += 1
            current_observations = next_observations

            if done:
                completed_returns_agent0.append(current_return0)
                completed_returns_agent1.append(current_return1)
                return drain_steps

        print(
            "Warning: single-arena drain reached safety limit before terminal state. "
            f"limit={max_drain_steps}"
        )
        return drain_steps

    def _drain_active_vectorized_episodes(
        self,
        observations: list[dict[str, Any]],
        episode_return_agent0: list[float],
        episode_return_agent1: list[float],
        completed_returns_agent0: list[float],
        completed_returns_agent1: list[float],
    ) -> tuple[int, list[int]]:
        active = [obs is not None for obs in observations]
        if not any(active):
            return 0, []

        drain_steps = 0
        drained_envs: list[int] = []
        max_drain_steps = max(1, self.config.max_episode_steps * 2)
        current_observations = list(observations)
        current_return0 = list(episode_return_agent0)
        current_return1 = list(episode_return_agent1)

        while any(active) and drain_steps < max_drain_steps:
            batched_actions: list[dict[str, Any]] = []

            for env_idx in range(self.config.num_envs):
                if not active[env_idx]:
                    batched_actions.append(
                        {
                            "arena_id": env_idx,
                            "agent0": self.action_adapter.action_from_vector([0.0, 0.0, 0.0]),
                            "agent1": self.action_adapter.action_from_vector([0.0, 0.0, 0.0]),
                        }
                    )
                    continue

                mirror_flags = self.observation_adapter.joint_mirror_flags(current_observations[env_idx])
                observation_vectors = self.observation_adapter.vectorize_joint(current_observations[env_idx])
                agent_0_action_vector, agent_1_action_vector = self._policy_action_vectors_for_mode(
                    observation_vectors=observation_vectors,
                    step_idx=self.env.episode_steps[env_idx],
                )
                batched_actions.append(
                    {
                        "arena_id": env_idx,
                        "agent0": self.action_adapter.action_from_vector(
                            agent_0_action_vector,
                            mirror_x=mirror_flags["agent_0"],
                        ),
                        "agent1": self.action_adapter.action_from_vector(
                            agent_1_action_vector,
                            mirror_x=mirror_flags["agent_1"],
                        ),
                    }
                )

            next_observations, rewards, dones, _infos = self.env.step(batched_actions)
            drain_steps += 1

            for env_idx in range(self.config.num_envs):
                if not active[env_idx]:
                    continue

                current_return0[env_idx] += float(rewards[env_idx]["agent_0"])
                current_return1[env_idx] += float(rewards[env_idx]["agent_1"])
                current_observations[env_idx] = next_observations[env_idx]

                if bool(dones[env_idx]):
                    completed_returns_agent0.append(current_return0[env_idx])
                    completed_returns_agent1.append(current_return1[env_idx])
                    active[env_idx] = False
                    drained_envs.append(env_idx)

        if any(active):
            print(
                "Warning: vectorized drain reached safety limit before all arenas terminated. "
                f"limit={max_drain_steps} unfinished={[idx for idx, is_active in enumerate(active) if is_active]}"
            )

        return drain_steps, drained_envs

    def _policy_action_vectors_for_mode(
        self,
        observation_vectors: dict[str, list[float]],
        step_idx: int,
    ) -> tuple[list[float], list[float]]:
        if self.config.train_mode == "alternating_two_policy":
            agent_0_action_vector, _agent_0_log_prob, _agent_0_value = self._sample_with_policy(
                self.policy_a,
                observation_vectors["agent_0"],
            )
            agent_1_action_vector, _agent_1_log_prob, _agent_1_value = self._sample_with_policy(
                self.policy_b,
                observation_vectors["agent_1"],
            )
            return agent_0_action_vector, agent_1_action_vector

        agent_0_action_vector, _agent_0_log_prob, _agent_0_value = self._sample_with_policy(
            self.policy,
            observation_vectors["agent_0"],
        )

        if self.config.train_mode == "shared_selfplay":
            agent_1_action_vector, _agent_1_log_prob, _agent_1_value = self._sample_with_policy(
                self.policy,
                observation_vectors["agent_1"],
            )
        else:
            agent_1_action_vector = self.opponent.act(observation_vectors["agent_1"], step_idx)

        return agent_0_action_vector, agent_1_action_vector

    @staticmethod
    def _build_opponent(name: str):
        if name == "random":
            return RandomVectorPolicy(seed=11)
        return HeuristicVectorPolicy(mode=name)

    def _derive_arena_seed(self, update_idx: int, env_idx: int, episode_idx: int) -> int:
        return int(self.config.seed + update_idx * 100_000 + env_idx * 1_000 + episode_idx)

