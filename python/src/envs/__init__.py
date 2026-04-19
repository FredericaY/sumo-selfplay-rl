"""Environment definitions for DeepRL Self-Play Arena."""

from .self_play_arena_env import UnitySelfPlayArenaConfig, UnitySelfPlayArenaEnv
from .vec_self_play_arena_env import UnityVecSelfPlayArenaConfig, UnityVecSelfPlayArenaEnv

__all__ = [
    "UnitySelfPlayArenaConfig",
    "UnitySelfPlayArenaEnv",
    "UnityVecSelfPlayArenaConfig",
    "UnityVecSelfPlayArenaEnv",
]
