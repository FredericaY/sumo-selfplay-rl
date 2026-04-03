"""Minimal training entry point for DeepRL Self-Play Arena."""

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

from algorithms import PPOTrainConfig, PPOTrainer, SelfPlayManager


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a Python dictionary."""

    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configs. Install dependencies with `pip install -r python/requirements.txt`."
        ) from YAML_IMPORT_ERROR

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(config_path: str | Path = "configs/train.yaml", step_sleep_override: float | None = None) -> None:
    """Start the PPO training loop."""

    config = load_config(config_path)
    env_config = config.get("env", {})
    training_config = config.get("training", {})
    selfplay_config = config.get("selfplay", {})
    logging_config = config.get("logging", {})

    manager = SelfPlayManager(config=selfplay_config)
    step_sleep_s = (
        float(step_sleep_override)
        if step_sleep_override is not None
        else float(training_config.get("step_sleep_s", 0.0))
    )
    trainer = PPOTrainer(
        PPOTrainConfig(
            seed=int(config.get("seed", 42)),
            host=env_config.get("host", "127.0.0.1"),
            port=int(env_config.get("port", 5055)),
            max_episode_steps=int(env_config.get("max_episode_steps", 400)),
            total_updates=int(training_config.get("total_updates", 10)),
            steps_per_update=int(training_config.get("steps_per_update", 256)),
            train_epochs=int(training_config.get("train_epochs", 4)),
            minibatch_size=int(training_config.get("minibatch_size", 64)),
            learning_rate=float(training_config.get("learning_rate", 3e-4)),
            clip_ratio=float(training_config.get("clip_ratio", 0.2)),
            value_coef=float(training_config.get("value_coef", 0.5)),
            entropy_coef=float(training_config.get("entropy_coef", 0.01)),
            gamma=float(training_config.get("gamma", 0.99)),
            gae_lambda=float(training_config.get("gae_lambda", 0.95)),
            opponent_policy=str(selfplay_config.get("opponent_policy", "flee")),
            train_mode=str(training_config.get("train_mode", "single_agent_baseline")),
            checkpoint_dir=str(logging_config.get("checkpoint_dir", "python/checkpoints/ppo")),
            debug_every=int(training_config.get("debug_every", 0)),
            step_sleep_s=step_sleep_s,
        )
    )

    print(f"Loaded config from: {config_path}")
    print(f"Self-play mode: {manager.summary()}")
    if step_sleep_s > 0.0:
        print(f"Visual step sleep: {step_sleep_s:.3f}s")
    history = trainer.train()
    print("Training finished.")
    print(f"Final policy loss: {history['policy_loss'][-1]:.6f}")
    print(f"Final value loss: {history['value_loss'][-1]:.6f}")
    print(f"Final mean agent_0 return: {history['mean_agent0_return'][-1]:.3f}")
    print(f"Final mean agent_1 return: {history['mean_agent1_return'][-1]:.3f}")


if __name__ == "__main__":
    main()
