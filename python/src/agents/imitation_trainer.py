"""Minimal behavior cloning trainer for the self-play arena MLP policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, random_split
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    DataLoader = None
    random_split = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from agents.imitation_dataset import RolloutImitationDataset
from agents.policy import MLPPolicyNetwork
from envs.observation_adapter import DEFAULT_OBS_DIM


@dataclass
class ImitationTrainConfig:
    """Hyperparameters for the first imitation learning run."""

    obs_dim: int = DEFAULT_OBS_DIM
    action_dim: int = 3
    hidden_sizes: Sequence[int] = (64, 64)
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    train_split: float = 0.9
    seed: int = 7
    device: str = "cpu"


class ImitationTrainer:
    """Train a small MLP to imitate saved rollout actions."""

    def __init__(self, config: ImitationTrainConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for ImitationTrainer. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.config = config or ImitationTrainConfig()
        self.device = torch.device(self.config.device)
        torch.manual_seed(self.config.seed)

        self.network = MLPPolicyNetwork(
            obs_dim=self.config.obs_dim,
            hidden_sizes=self.config.hidden_sizes,
            action_dim=self.config.action_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.move_loss_fn = nn.MSELoss()
        self.push_loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, rollout_paths: Sequence[str | Path]) -> dict[str, list[float]]:
        """Train on one or more rollout files and return loss curves."""

        dataset = RolloutImitationDataset(rollout_paths)
        total_size = len(dataset)
        train_size = max(1, int(total_size * self.config.train_split))
        val_size = max(1, total_size - train_size)
        if train_size + val_size > total_size:
            train_size = total_size - 1
            val_size = 1

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.seed),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}
        for _epoch_idx in range(self.config.epochs):
            train_loss = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader, training=False)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

        return history

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Save a small checkpoint containing weights and config."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "config": self.config.__dict__,
                "metadata": metadata or {},
            },
            output_path,
        )
        return output_path

    def load_checkpoint(self, path: str | Path) -> dict:
        """Load a saved checkpoint and restore network weights."""

        checkpoint = torch.load(Path(path), map_location=self.device)
        self.network.load_state_dict(checkpoint["state_dict"])
        return checkpoint

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        if training:
            self.network.train()
        else:
            self.network.eval()

        total_loss = 0.0
        total_items = 0
        for observations, targets in loader:
            observations = observations.to(self.device)
            targets = targets.to(self.device)
            raw_actions = self.network(observations)

            move_loss = self.move_loss_fn(torch.tanh(raw_actions[:, :2]), targets[:, :2])
            push_loss = self.push_loss_fn(raw_actions[:, 2], targets[:, 2])
            loss = move_loss + push_loss

            if training:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = observations.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size

        return total_loss / max(total_items, 1)
