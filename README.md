# DeepRL Self-Play Arena

DeepRL Self-Play Arena is a solo course project for building a small 1v1 deep reinforcement learning environment with a Unity-based final demo.

The game is a toy sumo-style duel in a circular arena. Two agents try to push each other out using a minimal action set:

- Continuous movement in 2D
- A higher-impact push/dash action with cooldown

The repo intentionally keeps the gameplay simple so the main project focus stays on:

- self-play training
- reward design
- policy evaluation
- experiment iteration
- Unity integration for the final presentation

## Project Status

The project has passed the initial integration milestone.

Implemented and verified:

- Unity minimal playable prototype in `unity/SelfPlaySumoArena`
- TCP bridge between Python and Unity
- Python environment wrapper with `reset / step / close`
- Batched 4-arena Unity training support
- Observation and action vector adapters
- Rollout collection and JSON trajectory saving
- Heuristic, random, and MLP vector policies
- Imitation learning training and checkpoint evaluation
- Multi-episode evaluation utilities
- PPO baseline training loop
- Alternating two-policy PPO training
- Human-vs-policy realtime demo mode

This means the current codebase already supports:

- collecting trajectories from Unity
- training a small neural policy in Python
- evaluating checkpoints back inside the Unity environment
- running early PPO/self-play experiments

## Why Self-Play

Self-play is a natural fit for a symmetric 1v1 arena:

- it removes the need for a strong hand-authored opponent
- it lets the policy improve against increasingly strong versions of itself
- it exposes stability issues that are central to multi-agent RL

The current plan is to start from a shared-parameter self-play baseline and then extend toward stronger experiment variants if time allows.

## Why Unity

Unity is used for the environment presentation and final demo side:

- fast iteration on arena feel and collisions
- easy visual debugging of policy behavior
- straightforward path from RL environment to playable demo

Python is used for training so experimentation stays lightweight and easy to extend.

## Current Training Stack

The current training-side workflow is:

1. Unity runs the arena and match logic
2. Python connects through the TCP bridge
3. `UnitySelfPlayArenaEnv` exposes a training-friendly interface
4. Policies consume observation vectors and output action vectors
5. Rollouts can be collected, saved, trained on, and evaluated

Current observation vector size:

- `17`

Current action vector size:

- `3`
- `move_x`
- `move_y`
- `use_push`

## Quick Start

### 1. Create the Python environment

```powershell
conda create -n rl_sumo python=3.10 -y
conda activate rl_sumo
python -m pip install --upgrade pip
pip install -r python/requirements.txt
```

### 2. Open Unity

Open the project in:

- `unity/SelfPlaySumoArena`

For all Python-driven workflows, make sure the arena scene has:

- `MatchController`
- `ArenaTcpBridge`
- `Application.runInBackground = true` through the bridge

### 3. Optional bridge smoke test

```powershell
python python/scripts/test_unity_bridge.py --max-steps 100 --agent1-policy flee --sleep 0.05
```

## Main Workflow 1: PPO / Self-Play Training

This is the primary training path used in the project.

### Unity setup for training

Use the training scene / arena setup and make sure:

- `MatchController -> Use Manual Physics Simulation = true`
- `MatchController -> Auto Simulate Bridge Steps = false`
- `MatchController -> Step Duration = 0.05`
- `MatchController -> Episode Duration = 400`
- `ArenaTcpBridge` is enabled
- Unity is in `Play` mode before launching Python

If you are using the 4-arena batch setup, also make sure:

- `ArenaBatchManager` is assigned on `TcpBridge`
- each arena instance has its own `ArenaMatchController`

### Start training

Run the default config:

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml
```

Run alternating A/B training for multiple cycles:

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml --alternating-cycles 2
```

Continue from an existing PPO checkpoint:

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml --init-checkpoint python/checkpoints/ppo/twopolicy_update_0092.pt --alternating-cycles 2
```

Slow visible stepping for debugging:

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml --step-sleep 0.05
```

### Training config

The main config file is:

- `python/configs/train.yaml`

The most frequently adjusted fields are:

- `env.num_envs`
- `env.max_episode_steps`
- `env.edge_safety_weight`
- `env.outward_pressure_weight`
- `env.terminal_timeout_penalty`
- `env.terminal_loss_penalty`
- `env.terminal_loss_time_scale`
- `env.opponent_position_scale`
- `env.relative_position_scale`
- `training.total_updates`
- `training.steps_per_update`
- `training.learning_rate`
- `training.entropy_coef`
- `training.finish_active_episodes_before_exit`
- `training.ema_propagation_enabled`
- `training.ema_propagation_decay`

## Main Workflow 2: Human vs Policy

This is the main demo / presentation path: you control `agent_0` in Unity and Python drives `agent_1` from a checkpoint.

### Unity setup for human-vs-policy

For the human-controlled arena, set it up like this:

#### `Agent0`

- keep `AgentMotor2D`
- keep `HumanInputAgent2D`

#### `Agent1`

- keep `AgentMotor2D`
- keep `RealtimeExternalAgentController2D`
- do **not** use `HumanInputAgent2D` on `Agent1`

#### `MatchController`

- assign `Realtime Agent1 Controller`
- `Use Manual Physics Simulation = false`
- `Auto Simulate Bridge Steps = false`

Human-vs-policy is a realtime mode. It should not use the step-driven training setting with manual physics simulation enabled.

### Play against the latest policy

Start Unity in `Play` mode, then run:

```powershell
python python/scripts/play_human_vs_policy.py python/checkpoints/ppo/<checkpoint>.pt --policy-side b --reset-on-start --auto-reset
```

If you want to slow policy updates slightly for easier viewing:

```powershell
python python/scripts/play_human_vs_policy.py python/checkpoints/ppo/<checkpoint>.pt --policy-side b --reset-on-start --auto-reset --sleep 0.03
```

Use `--policy-side a` or `--policy-side b` when loading a two-policy PPO checkpoint.

### Find the newest checkpoint in PowerShell

If you want the newest PPO checkpoint quickly:

```powershell
Get-ChildItem python/checkpoints/ppo/*.pt | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

Then plug the newest path into `play_human_vs_policy.py`.

## Other Utilities

### Collect rollouts

```powershell
python python/scripts/collect_rollout.py --agent0-policy chase --agent1-policy flee --max-steps 400 --save
```

### Train imitation policy

```powershell
python python/scripts/train_imitation.py --epochs 20 --batch-size 64 --lr 1e-3
```

### Evaluate a checkpoint

```powershell
python python/scripts/eval_checkpoint.py python/checkpoints/imitation/<checkpoint>.pt --episodes 10 --opponent-policy flee
```

## Repository Structure

```text
sumo-selfplay-rl/
|-- README.md
|-- docs/
|   |-- DeepRL_final_proposal.pdf
|   `-- design/
|-- unity/
|   |-- README.md
|   |-- project_structure.md
|   |-- scripts_plan.md
|   `-- SelfPlaySumoArena/
|-- python/
|   |-- README.md
|   |-- configs/
|   |-- scripts/
|   |-- src/
|   |-- checkpoints/
|   `-- logs/
|-- experiments/
|-- models/
`-- tools/
```

## Important Scripts

Training and rollout utilities:

- `python/scripts/test_unity_bridge.py`
- `python/scripts/collect_rollout.py`
- `python/scripts/train_imitation.py`
- `python/scripts/eval_mlp_policy.py`
- `python/scripts/eval_checkpoint.py`
- `python/scripts/train_selfplay.py`

Key Python modules:

- `python/src/envs/self_play_arena_env.py`
- `python/src/envs/observation_adapter.py`
- `python/src/envs/action_adapter.py`
- `python/src/agents/policy.py`
- `python/src/agents/actor_critic.py`
- `python/src/algorithms/rollout_collector.py`
- `python/src/algorithms/ppo_trainer.py`
- `python/src/eval/evaluate.py`

## Current Limitations

- The PPO implementation is still an early experiment framework rather than a polished benchmark suite
- Self-play currently focuses on alternating two-policy training and does not yet include a full opponent pool / league setup
- Reward shaping and reward attribution are still under active iteration
- Unity presentation polish is still behind the core training/debugging functionality

## Next Milestones

- stabilize alternating two-policy PPO training
- improve reward design and symmetry handling
- compare trained checkpoints against stronger baselines
- add checkpoint pool / opponent sampling variants if time allows
- finalize a stronger policy for the Unity demo flow

## Author

Julie Yang  
Deep reinforcement learning course project
