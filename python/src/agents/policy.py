"""Vector-policy interfaces and simple baselines for the self-play arena."""

from __future__ import annotations

import math
import random
from typing import Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


class VectorPolicy:
    """Minimal policy interface for vector observations and vector actions."""

    def act(self, observation_vector: Sequence[float], step_idx: int) -> list[float]:
        """Map one vector observation to one vector action."""

        raise NotImplementedError


class ConstantVectorPolicy(VectorPolicy):
    """Always returns the same action vector."""

    def __init__(self, action_vector: Sequence[float] | None = None) -> None:
        self.action_vector = list(action_vector or [0.0, 0.0, 0.0])

    def act(self, observation_vector: Sequence[float], step_idx: int) -> list[float]:
        _ = observation_vector, step_idx
        return list(self.action_vector)


class RandomVectorPolicy(VectorPolicy):
    """Produce random movement directions and random push triggers."""

    def __init__(self, push_probability: float = 0.1, seed: int | None = None) -> None:
        self.push_probability = push_probability
        self.rng = random.Random(seed)

    def act(self, observation_vector: Sequence[float], step_idx: int) -> list[float]:
        _ = observation_vector, step_idx
        move_x = self.rng.uniform(-1.0, 1.0)
        move_y = self.rng.uniform(-1.0, 1.0)
        norm = math.sqrt(move_x * move_x + move_y * move_y)
        if norm > 1e-6:
            move_x /= norm
            move_y /= norm

        use_push = 1.0 if self.rng.random() < self.push_probability else 0.0
        return [move_x, move_y, use_push]


class HeuristicVectorPolicy(VectorPolicy):
    """Simple chase/flee/idle baselines using vector observations."""

    def __init__(self, mode: str = "chase") -> None:
        valid_modes = {"idle", "chase", "flee"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported heuristic mode: {mode}")
        self.mode = mode

    def act(self, observation_vector: Sequence[float], step_idx: int) -> list[float]:
        if self.mode == "idle":
            return [0.0, 0.0, 0.0]

        rel_x = float(observation_vector[8]) if len(observation_vector) > 8 else 0.0
        rel_y = float(observation_vector[9]) if len(observation_vector) > 9 else 0.0
        push_ready = float(observation_vector[-1]) > 0.5 if observation_vector else False

        if self.mode == "flee":
            rel_x *= -1.0
            rel_y *= -1.0

        norm = math.sqrt(rel_x * rel_x + rel_y * rel_y)
        if norm > 1e-6:
            rel_x /= norm
            rel_y /= norm
        else:
            rel_x = 0.0
            rel_y = 0.0

        push_period = 12 if self.mode == "chase" else 18
        use_push = 1.0 if push_ready and step_idx % push_period == 0 else 0.0
        return [rel_x, rel_y, use_push]


class MLPPolicyNetwork(nn.Module if nn is not None else object):
    """Small MLP that maps observation vectors to continuous action vectors."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        action_dim: int = 3,
    ) -> None:
        if nn is None:
            raise RuntimeError(
                "PyTorch is required for MLPPolicyNetwork. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            input_dim = int(hidden_size)
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, observation_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(observation_tensor)


class MLPVectorPolicy(VectorPolicy):
    """Torch-backed MLP policy placeholder for future training code."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        action_dim: int = 3,
        seed: int | None = 0,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for MLPVectorPolicy. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        self.network = MLPPolicyNetwork(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            action_dim=action_dim,
        ).to(self.device)
        self.network.eval()

    def act(self, observation_vector: Sequence[float], step_idx: int) -> list[float]:
        _ = step_idx
        observation_tensor = torch.tensor(
            list(observation_vector), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            raw_action = self.network(observation_tensor).squeeze(0)

        move = torch.tanh(raw_action[:2])
        push = torch.sigmoid(raw_action[2:3])
        action = torch.cat([move, push], dim=0)
        return action.detach().cpu().tolist()

    def state_dict(self) -> dict:
        """Expose the underlying torch state dict for future checkpointing."""

        return self.network.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load network weights from a torch-style state dict."""

        self.network.load_state_dict(state_dict)
