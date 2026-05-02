"""Actor-critic networks for PPO-style training in the self-play arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Normal
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    Bernoulli = None
    Normal = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

from envs.observation_adapter import DEFAULT_OBS_DIM


@dataclass
class ActorCriticConfig:
    """Network settings for the first PPO policy."""

    obs_dim: int = DEFAULT_OBS_DIM
    hidden_sizes: Sequence[int] = (64, 64)
    init_std: float = 0.4
    use_edge_gate: bool = False
    arena_radius: float = 5.0
    edge_gate_margin: float = 1.25
    edge_gate_min_safety: float = 0.15
    edge_gate_push_penalty: float = 2.0
    edge_gate_hidden_size: int = 16


class MLPBackbone(nn.Module if nn is not None else object):
    """Shared MLP trunk for actor and critic heads."""

    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]) -> None:
        if nn is None:
            raise RuntimeError(
                "PyTorch is required for MLPBackbone. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(hidden_size)))
            layers.append(nn.Tanh())
            input_dim = int(hidden_size)
        self.model = nn.Sequential(*layers)
        self.output_dim = input_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


class ActorCritic(nn.Module if nn is not None else object):
    """Small actor-critic for continuous move + Bernoulli push decisions."""

    def __init__(self, config: ActorCriticConfig | None = None) -> None:
        if nn is None:
            raise RuntimeError(
                "PyTorch is required for ActorCritic. Install dependencies with `pip install -r python/requirements.txt`."
            ) from TORCH_IMPORT_ERROR

        super().__init__()
        self.config = config or ActorCriticConfig()
        self.backbone = MLPBackbone(self.config.obs_dim, self.config.hidden_sizes)
        self.move_mean_head = nn.Linear(self.backbone.output_dim, 2)
        self.push_logit_head = nn.Linear(self.backbone.output_dim, 1)
        self.value_head = nn.Linear(self.backbone.output_dim, 1)
        self.log_std = nn.Parameter(torch.full((2,), float(torch.log(torch.tensor(self.config.init_std)))))
        if self.config.use_edge_gate:
            self.edge_gate_head = nn.Sequential(
                nn.Linear(4, int(self.config.edge_gate_hidden_size)),
                nn.Tanh(),
                nn.Linear(int(self.config.edge_gate_hidden_size), 1),
            )
            self._initialize_edge_gate_head()

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(observations)
        move_mean = torch.tanh(self.move_mean_head(features))
        push_logit = self.push_logit_head(features).squeeze(-1)
        values = self.value_head(features).squeeze(-1)
        if self.config.use_edge_gate:
            move_mean, push_logit = self._apply_edge_gate(observations, move_mean, push_logit)
        return move_mean, push_logit, values

    def _initialize_edge_gate_head(self) -> None:
        final_layer = self.edge_gate_head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, 4.0)

    def _apply_edge_gate(
        self,
        observations: torch.Tensor,
        move_mean: torch.Tensor,
        push_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_features = observations[:, [12, 13, 14, 15]]
        learned_safety = torch.sigmoid(self.edge_gate_head(edge_features)).squeeze(-1)

        self_pos = observations[:, 0:2]
        self_edge_margin = observations[:, 13]
        margin_scale = max(float(self.config.edge_gate_margin), 1e-6)
        boundary_safety = torch.clamp(self_edge_margin / margin_scale, 0.0, 1.0)
        min_safety = float(self.config.edge_gate_min_safety)
        safety_gate = torch.clamp(boundary_safety * learned_safety, min=min_safety, max=1.0)

        radial_dir = self_pos / torch.linalg.norm(self_pos, dim=-1, keepdim=True).clamp_min(1e-6)
        outward_amount = torch.clamp(torch.sum(move_mean * radial_dir, dim=-1, keepdim=True), min=0.0)
        outward_component = outward_amount * radial_dir
        gated_move_mean = move_mean - (1.0 - safety_gate.unsqueeze(-1)) * outward_component

        risk = 1.0 - safety_gate
        gated_push_logit = push_logit - float(self.config.edge_gate_push_penalty) * risk
        return gated_move_mean, gated_push_logit

    def sample_action(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        move_mean, push_logit, values = self.forward(observations)
        move_std = self.log_std.exp().expand_as(move_mean)
        move_dist = Normal(move_mean, move_std)
        push_dist = Bernoulli(logits=push_logit)

        move_action = torch.clamp(move_dist.rsample(), -1.0, 1.0)
        push_action = push_dist.sample().unsqueeze(-1)
        action = torch.cat([move_action, push_action], dim=-1)

        log_prob = move_dist.log_prob(move_action).sum(dim=-1) + push_dist.log_prob(push_action.squeeze(-1))
        entropy = move_dist.entropy().sum(dim=-1) + push_dist.entropy()
        return action, log_prob, entropy, values

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        move_mean, push_logit, values = self.forward(observations)
        move_std = self.log_std.exp().expand_as(move_mean)
        move_dist = Normal(move_mean, move_std)
        push_dist = Bernoulli(logits=push_logit)

        move_actions = actions[:, :2]
        push_actions = actions[:, 2]
        log_prob = move_dist.log_prob(move_actions).sum(dim=-1) + push_dist.log_prob(push_actions)
        entropy = move_dist.entropy().sum(dim=-1) + push_dist.entropy()
        return log_prob, entropy, values
