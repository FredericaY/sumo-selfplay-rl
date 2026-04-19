"""Minimal training entry point for DeepRL Self-Play Arena."""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

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


def apply_ema_policy_propagation(
    checkpoint_path: str | Path,
    trained_side: str,
    decay: float,
) -> str:
    """Blend the newly trained policy into the opposite side using EMA and save a new checkpoint."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for EMA checkpoint propagation. Install dependencies with `pip install -r python/requirements.txt`."
        ) from TORCH_IMPORT_ERROR

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "policy_a_state_dict" not in checkpoint or "policy_b_state_dict" not in checkpoint:
        raise ValueError(f"EMA propagation expects a two-policy checkpoint: {checkpoint_path}")

    if trained_side == "a":
        source_key = "policy_a_state_dict"
        target_key = "policy_b_state_dict"
        target_side = "b"
    elif trained_side == "b":
        source_key = "policy_b_state_dict"
        target_key = "policy_a_state_dict"
        target_side = "a"
    else:
        raise ValueError(f"Unsupported trained side for EMA propagation: {trained_side}")

    source_state = checkpoint[source_key]
    target_state = checkpoint[target_key]

    blended_state: dict[str, Any] = {}
    for name, target_tensor in target_state.items():
        source_tensor = source_state[name]
        if torch.is_floating_point(target_tensor):
            blended_state[name] = decay * target_tensor + (1.0 - decay) * source_tensor
        else:
            blended_state[name] = target_tensor.clone()

    checkpoint[target_key] = blended_state
    metadata = dict(checkpoint.get("metadata", {}))
    metadata["ema_propagation"] = {
        "from_side": trained_side,
        "to_side": target_side,
        "decay": decay,
        "source_checkpoint": str(checkpoint_path),
    }
    checkpoint["metadata"] = metadata

    output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_ema_to_{target_side}{checkpoint_path.suffix}")
    torch.save(checkpoint, output_path)
    return str(output_path)


def mutate_two_policy_checkpoint(
    checkpoint_path: str | Path,
    mutate_side: str,
    mutation_std: float,
    mutation_seed: int,
) -> str:
    """Apply a small Gaussian perturbation to one side of a two-policy checkpoint."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for checkpoint mutation. Install dependencies with `pip install -r python/requirements.txt`."
        ) from TORCH_IMPORT_ERROR

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "policy_a_state_dict" not in checkpoint or "policy_b_state_dict" not in checkpoint:
        raise ValueError(f"Mutation expects a two-policy checkpoint: {checkpoint_path}")

    if mutate_side not in {"a", "b"}:
        raise ValueError(f"Unsupported mutate side: {mutate_side}")

    state_key = "policy_a_state_dict" if mutate_side == "a" else "policy_b_state_dict"
    source_state = checkpoint[state_key]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(mutation_seed))

    mutated_state: dict[str, Any] = {}
    for name, tensor in source_state.items():
        if torch.is_floating_point(tensor):
            noise = torch.randn(
                tensor.shape,
                generator=generator,
                dtype=tensor.dtype,
            ) * mutation_std
            mutated_state[name] = tensor.clone() + noise
        else:
            mutated_state[name] = tensor.clone()

    checkpoint[state_key] = mutated_state
    metadata = dict(checkpoint.get("metadata", {}))
    metadata["mutation"] = {
        "side": mutate_side,
        "std": mutation_std,
        "seed": int(mutation_seed),
        "source_checkpoint": str(checkpoint_path),
    }
    checkpoint["metadata"] = metadata

    output_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_mut_{mutate_side}{checkpoint_path.suffix}"
    )
    torch.save(checkpoint, output_path)
    return str(output_path)


def inject_opponent_from_pool(
    checkpoint_path: str | Path,
    train_side: str,
    opponent_checkpoint_path: str | Path,
) -> str:
    """Replace the frozen opponent side in a two-policy checkpoint from a pool sample."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for opponent-pool checkpoint composition. Install dependencies with `pip install -r python/requirements.txt`."
        ) from TORCH_IMPORT_ERROR

    checkpoint_path = Path(checkpoint_path)
    opponent_checkpoint_path = Path(opponent_checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    opponent_checkpoint = torch.load(opponent_checkpoint_path, map_location="cpu")

    if "policy_a_state_dict" not in checkpoint or "policy_b_state_dict" not in checkpoint:
        raise ValueError(f"Opponent-pool composition expects a two-policy checkpoint: {checkpoint_path}")
    if "policy_a_state_dict" not in opponent_checkpoint or "policy_b_state_dict" not in opponent_checkpoint:
        raise ValueError(f"Opponent-pool source must also be a two-policy checkpoint: {opponent_checkpoint_path}")

    if train_side == "a":
        target_key = "policy_b_state_dict"
        opponent_side = "b"
    elif train_side == "b":
        target_key = "policy_a_state_dict"
        opponent_side = "a"
    else:
        raise ValueError(f"Unsupported train side for opponent injection: {train_side}")

    checkpoint[target_key] = {
        name: tensor.clone()
        for name, tensor in opponent_checkpoint[target_key].items()
    }

    metadata = dict(checkpoint.get("metadata", {}))
    metadata["opponent_pool"] = {
        "train_side": train_side,
        "opponent_side": opponent_side,
        "opponent_checkpoint": str(opponent_checkpoint_path),
        "base_checkpoint": str(checkpoint_path),
    }
    checkpoint["metadata"] = metadata

    output_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_pool_opp_{opponent_side}{checkpoint_path.suffix}"
    )
    torch.save(checkpoint, output_path)
    return str(output_path)


def run_once(
    config_path: str | Path,
    step_sleep_override: float | None = None,
    init_checkpoint_override: str | None = None,
    train_side_override: str | None = None,
) -> dict[str, Any]:
    """Run one training segment and return history plus the latest checkpoint path."""

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
    init_checkpoint = init_checkpoint_override or training_config.get("init_checkpoint")
    load_optimizer_state = bool(training_config.get("load_optimizer_state", False))
    train_side = train_side_override or str(training_config.get("train_side", "a"))

    trainer = PPOTrainer(
        PPOTrainConfig(
            seed=int(config.get("seed", 42)),
            host=env_config.get("host", "127.0.0.1"),
            port=int(env_config.get("port", 5055)),
            max_episode_steps=int(env_config.get("max_episode_steps", 400)),
            num_envs=int(env_config.get("num_envs", 1)),
            use_shaped_rewards=bool(env_config.get("use_shaped_rewards", False)),
            step_penalty=float(env_config.get("step_penalty", 0.0)),
            edge_safety_weight=float(env_config.get("edge_safety_weight", 0.0)),
            outward_pressure_weight=float(env_config.get("outward_pressure_weight", 0.0)),
            terminal_timeout_penalty=float(env_config.get("terminal_timeout_penalty", 0.0)),
            timeout_center_bias_weight=float(env_config.get("timeout_center_bias_weight", 0.0)),
            terminal_win_reward=float(env_config.get("terminal_win_reward", 1.0)),
            terminal_loss_penalty=float(env_config.get("terminal_loss_penalty", 1.5)),
            terminal_loss_time_scale=float(env_config.get("terminal_loss_time_scale", 0.5)),
            canonicalize_left_right=bool(env_config.get("canonicalize_left_right", True)),
            opponent_position_scale=float(env_config.get("opponent_position_scale", 1.5)),
            relative_position_scale=float(env_config.get("relative_position_scale", 2.0)),
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
            train_side=train_side,
            checkpoint_dir=str(logging_config.get("checkpoint_dir", "python/checkpoints/ppo")),
            debug_every=int(training_config.get("debug_every", 0)),
            step_sleep_s=step_sleep_s,
            init_checkpoint=str(init_checkpoint) if init_checkpoint else None,
            load_optimizer_state=load_optimizer_state,
            finish_active_episodes_before_exit=bool(
                training_config.get("finish_active_episodes_before_exit", True)
            ),
        )
    )

    print(f"Loaded config from: {config_path}")
    print(f"Self-play mode: {manager.summary()}")
    print(f"Train mode: {trainer.config.train_mode}")
    if trainer.config.train_mode == "alternating_two_policy":
        print(f"Train side: {trainer.config.train_side}")
    if init_checkpoint:
        print(f"Initialized from checkpoint: {init_checkpoint}")
        print(f"Load optimizer state: {load_optimizer_state}")
    if step_sleep_s > 0.0:
        print(f"Visual step sleep: {step_sleep_s:.3f}s")

    history = trainer.train()
    print("Training finished.")
    print(f"Final policy loss: {history['policy_loss'][-1]:.6f}")
    print(f"Final value loss: {history['value_loss'][-1]:.6f}")
    print(f"Final mean agent_0 return: {history['mean_agent0_return'][-1]:.3f}")
    print(f"Final mean agent_1 return: {history['mean_agent1_return'][-1]:.3f}")
    if history.get("last_checkpoint"):
        print(f"Last checkpoint: {history['last_checkpoint']}")
    return history


def main(
    config_path: str | Path = "configs/train.yaml",
    step_sleep_override: float | None = None,
    init_checkpoint_override: str | None = None,
    train_side_override: str | None = None,
    alternating_cycles: int = 1,
) -> None:
    """Start the PPO training loop, optionally chaining A->B cycles automatically."""

    config = load_config(config_path)
    training_config = config.get("training", {})
    selfplay_config = config.get("selfplay", {})
    ema_propagation_enabled = bool(training_config.get("ema_propagation_enabled", False))
    ema_propagation_decay = float(training_config.get("ema_propagation_decay", 0.8))
    mutation_enabled = bool(training_config.get("mutation_enabled", False))
    mutation_every_segments = int(training_config.get("mutation_every_segments", 0))
    mutation_std = float(training_config.get("mutation_std", 0.01))
    opponent_sampling = str(selfplay_config.get("opponent_sampling", "latest"))
    opponent_pool_size = int(selfplay_config.get("opponent_pool_size", 0))
    rng = random.Random(int(config.get("seed", 42)))

    if alternating_cycles <= 1:
        run_once(
            config_path,
            step_sleep_override=step_sleep_override,
            init_checkpoint_override=init_checkpoint_override,
            train_side_override=train_side_override,
        )
        return

    current_checkpoint = init_checkpoint_override
    current_side = train_side_override or "a"
    opponent_pool: deque[str] = deque(maxlen=max(opponent_pool_size, 1))
    segment_counter = 0

    for cycle_idx in range(alternating_cycles):
        for side in (current_side, "b" if current_side == "a" else "a"):
            prepared_checkpoint = current_checkpoint

            if prepared_checkpoint and opponent_sampling == "pool" and len(opponent_pool) > 0:
                sampled_opponent_checkpoint = rng.choice(list(opponent_pool))
                prepared_checkpoint = inject_opponent_from_pool(
                    prepared_checkpoint,
                    train_side=side,
                    opponent_checkpoint_path=sampled_opponent_checkpoint,
                )
                print(
                    f"Injected opponent from pool before side={side}: "
                    f"sample={sampled_opponent_checkpoint} prepared={prepared_checkpoint}"
                )

            if (
                prepared_checkpoint
                and mutation_enabled
                and mutation_every_segments > 0
                and segment_counter > 0
                and segment_counter % mutation_every_segments == 0
            ):
                mutation_seed = int(config.get("seed", 42)) + segment_counter * 1009 + (0 if side == "a" else 1)
                prepared_checkpoint = mutate_two_policy_checkpoint(
                    prepared_checkpoint,
                    mutate_side=side,
                    mutation_std=mutation_std,
                    mutation_seed=mutation_seed,
                )
                print(
                    f"Applied mutation before side={side}: std={mutation_std:.4f} "
                    f"seed={mutation_seed} prepared={prepared_checkpoint}"
                )

            print(f"=== Alternating cycle {cycle_idx + 1}/{alternating_cycles} side={side} ===")
            history = run_once(
                config_path,
                step_sleep_override=step_sleep_override,
                init_checkpoint_override=prepared_checkpoint,
                train_side_override=side,
            )
            current_checkpoint = history.get("last_checkpoint")
            if current_checkpoint:
                opponent_pool.append(current_checkpoint)
            if ema_propagation_enabled and current_checkpoint:
                current_checkpoint = apply_ema_policy_propagation(
                    current_checkpoint,
                    trained_side=side,
                    decay=ema_propagation_decay,
                )
                print(
                    f"Applied EMA propagation after side={side}: "
                    f"decay={ema_propagation_decay:.3f} checkpoint={current_checkpoint}"
                )
                opponent_pool.append(current_checkpoint)
            segment_counter += 1
        current_side = "a"


if __name__ == "__main__":
    main()
