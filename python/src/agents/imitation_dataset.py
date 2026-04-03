"""Dataset helpers for imitation learning from saved rollout JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    torch = None
    Dataset = object
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


class RolloutImitationDataset(Dataset):
    """Flatten saved rollout transitions into supervised learning samples."""

    def __init__(self, rollout_paths: Iterable[str | Path]) -> None:
        if torch is None:
            raise RuntimeError(
                "PyTorch is required for RolloutImitationDataset. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        self.samples: list[tuple[list[float], list[float]]] = []
        for path in rollout_paths:
            rollout_path = Path(path)
            data = json.loads(rollout_path.read_text(encoding="utf-8"))
            for transition in data.get("transitions", []):
                observation_vectors = transition["observation_vectors"]
                action_vectors = transition.get("executed_action_vectors") or transition.get("action_vectors")
                for agent_id in ("agent_0", "agent_1"):
                    self.samples.append(
                        (
                            list(observation_vectors[agent_id]),
                            list(action_vectors[agent_id]),
                        )
                    )

        if not self.samples:
            raise ValueError("No imitation samples were loaded from the provided rollout files.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        observation_vector, action_vector = self.samples[index]
        return (
            torch.tensor(observation_vector, dtype=torch.float32),
            torch.tensor(action_vector, dtype=torch.float32),
        )
