"""Minimal training entry point for DeepRL Self-Play Arena.

This module is intentionally small. It wires together config loading,
environment construction, and a placeholder self-play manager so the
project has a clean starting point without pretending the algorithm is
already implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None

from algorithms.self_play_manager import SelfPlayManager
from envs.self_play_arena_env import SelfPlayArenaEnv


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a Python dictionary."""
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configs. Install dependencies with `pip install -r python/requirements.txt`."
        ) from YAML_IMPORT_ERROR

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(config_path: str | Path = "configs/train.yaml") -> None:
    """Start a placeholder training run.

    TODO:
    - Replace print statements with structured logging
    - Instantiate the real policy and optimizer
    - Add checkpoint saving and evaluation hooks
    """

    config = load_config(config_path)
    env = SelfPlayArenaEnv(config=config.get("env", {}))
    manager = SelfPlayManager(config=config.get("selfplay", {}))

    print(f"Loaded config from: {config_path}")
    print(f"Environment: {env.summary()}")
    print(f"Self-play mode: {manager.summary()}")
    print("Training loop not implemented yet.")


if __name__ == "__main__":
    main()
