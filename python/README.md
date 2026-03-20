# Python Training Workspace

This folder contains the training, evaluation, and experiment-facing code for DeepRL Self-Play Arena.

## Design Goals

- Keep the first training scaffold minimal
- Make it easy to extend toward PPO and self-play
- Separate environment logic, policy logic, evaluation, and experiment scripts

## Suggested Workflow

1. Finalize the first environment API in `src/envs/`
2. Add a simple policy baseline in `src/agents/`
3. Implement self-play checkpoint bookkeeping in `src/algorithms/`
4. Use `scripts/` for experiment entry points

## Current Status

- The code is intentionally skeletal
- Most modules contain placeholder classes or functions
- TODO markers indicate the next implementation steps

## TODO

- Choose a configuration library strategy or keep YAML parsing lightweight
- Add PPO training dependencies only when the training loop is ready
- Add tests once the environment interface stabilizes
