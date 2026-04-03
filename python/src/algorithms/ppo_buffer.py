"""PPO rollout buffer for the self-play arena."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


@dataclass
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


@dataclass
class PPOBuffer:
    """Simple on-policy buffer for one-agent PPO training."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    observations: list[list[float]] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)

    def add(
        self,
        observation: list[float],
        action: list[float],
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        self.observations.append(list(observation))
        self.actions.append(list(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.log_probs.clear()

    def compute_batch(self, last_value: float = 0.0) -> PPOBatch:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for PPOBuffer. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        advantages: list[float] = []
        gae = 0.0
        values = self.values + [float(last_value)]
        for idx in reversed(range(len(self.rewards))):
            mask = 0.0 if self.dones[idx] else 1.0
            delta = self.rewards[idx] + self.gamma * values[idx + 1] * mask - values[idx]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.append(gae)
        advantages.reverse()
        returns = [adv + val for adv, val in zip(advantages, self.values)]

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        return PPOBatch(
            observations=torch.tensor(self.observations, dtype=torch.float32),
            actions=torch.tensor(self.actions, dtype=torch.float32),
            log_probs=torch.tensor(self.log_probs, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=advantages_tensor,
            values=torch.tensor(self.values, dtype=torch.float32),
        )
