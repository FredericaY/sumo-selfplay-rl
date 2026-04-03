"""Algorithm scaffolding for training and self-play management."""

from .ppo_buffer import PPOBatch, PPOBuffer
from .ppo_trainer import PPOTrainConfig, PPOTrainer
from .rollout_collector import RolloutCollector, RolloutEpisode, Transition
from .self_play_manager import SelfPlayConfig, SelfPlayManager
from .trajectory_serializer import TrajectorySerializer

__all__ = [
    "PPOBatch",
    "PPOBuffer",
    "PPOTrainConfig",
    "PPOTrainer",
    "RolloutCollector",
    "RolloutEpisode",
    "Transition",
    "SelfPlayConfig",
    "SelfPlayManager",
    "TrajectorySerializer",
]
