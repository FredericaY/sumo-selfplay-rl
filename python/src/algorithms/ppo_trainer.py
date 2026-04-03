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
from envs import UnitySelfPlayArenaConfig, UnitySelfPlayArenaEnv
from envs.action_adapter import ActionAdapter
from envs.observation_adapter import ObservationAdapter


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
    opponent_policy: str = "flee"
    train_mode: str = "single_agent_baseline"
    checkpoint_dir: str = "python/checkpoints/ppo"
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 5055
    debug_every: int = 0
    step_sleep_s: float = 0.0


class PPOTrainer:
    """Train PPO either against a fixed baseline or in shared-parameter self-play."""

    def __init__(self, config: PPOTrainConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for PPOTrainer. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.config = config or PPOTrainConfig()
        valid_modes = {"single_agent_baseline", "shared_selfplay"}
        if self.config.train_mode not in valid_modes:
            raise ValueError(f"Unsupported train_mode: {self.config.train_mode}")

        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)

        self.env = UnitySelfPlayArenaEnv(
            UnitySelfPlayArenaConfig(
                host=self.config.host,
                port=self.config.port,
                timeout_s=5.0,
                max_episode_steps=self.config.max_episode_steps,
            )
        )
        self.observation_adapter = ObservationAdapter()
        self.action_adapter = ActionAdapter()
        self.policy = ActorCritic(ActorCriticConfig(obs_dim=13)).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)
        self.buffer = PPOBuffer(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
        self.opponent = self._build_opponent(self.config.opponent_policy)

    def train(self) -> dict[str, list[float]]:
        history = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_agent0_return": [],
            "mean_agent1_return": [],
        }
        observations = self.env.reset()
        episode_return_agent0 = 0.0
        episode_return_agent1 = 0.0
        completed_returns_agent0: list[float] = []
        completed_returns_agent1: list[float] = []

        try:
            for update_idx in range(self.config.total_updates):
                self.buffer.clear()
                for step_in_update in range(self.config.steps_per_update):
                    step_idx = self.env.episode_step
                    observation_vectors = self.observation_adapter.vectorize_joint(observations)

                    agent_0_action_vector, agent_0_log_prob, agent_0_value = self._sample_policy_action(
                        observation_vectors["agent_0"]
                    )

                    if self.config.train_mode == "shared_selfplay":
                        agent_1_action_vector, agent_1_log_prob, agent_1_value = self._sample_policy_action(
                            observation_vectors["agent_1"]
                        )
                    else:
                        agent_1_action_vector = self.opponent.act(observation_vectors["agent_1"], step_idx)
                        agent_1_log_prob = None
                        agent_1_value = None

                    joint_actions = {
                        "agent_0": self.action_adapter.action_from_vector(agent_0_action_vector),
                        "agent_1": self.action_adapter.action_from_vector(agent_1_action_vector),
                    }

                    next_observations, rewards, done, info = self.env.step(joint_actions)
                    reward_agent0 = float(rewards["agent_0"])
                    reward_agent1 = float(rewards["agent_1"])
                    episode_return_agent0 += reward_agent0
                    episode_return_agent1 += reward_agent1

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
                            f"update={update_idx:03d} step={step_in_update:03d} "
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

                checkpoint_path = Path(self.config.checkpoint_dir) / f"ppo_update_{update_idx:04d}.pt"
                self.save_checkpoint(checkpoint_path, metadata={"update": update_idx, **metrics})

                print(
                    f"update={update_idx:03d} mode={self.config.train_mode} "
                    f"policy_loss={metrics['policy_loss']:.6f} "
                    f"value_loss={metrics['value_loss']:.6f} entropy={metrics['entropy']:.6f} "
                    f"mean_agent0_return={mean_agent0_return:.3f} "
                    f"mean_agent1_return={mean_agent1_return:.3f}"
                )

            return history
        finally:
            self.env.close()

    def save_checkpoint(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.policy.state_dict(),
                "config": self.config.__dict__,
                "metadata": metadata or {},
            },
            output_path,
        )
        return output_path

    def _sample_policy_action(self, observation_vector: list[float]) -> tuple[list[float], float, float]:
        obs_tensor = torch.tensor(observation_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, log_prob_tensor, _entropy_tensor, value_tensor = self.policy.sample_action(obs_tensor)
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

                new_log_probs, entropy, values = self.policy.evaluate_actions(mini_obs, mini_actions)
                ratio = torch.exp(new_log_probs - mini_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * mini_advantages, clipped_ratio * mini_advantages).mean()
                value_loss = torch.mean((mini_returns - values) ** 2)
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                policy_loss_value = float(policy_loss.detach().cpu())
                value_loss_value = float(value_loss.detach().cpu())
                entropy_value = float(entropy_bonus.detach().cpu())

        return {
            "policy_loss": policy_loss_value,
            "value_loss": value_loss_value,
            "entropy": entropy_value,
        }

    @staticmethod
    def _build_opponent(name: str):
        if name == "random":
            return RandomVectorPolicy(seed=11)
        return HeuristicVectorPolicy(mode=name)
