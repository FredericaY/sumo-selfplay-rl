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
- Observation and action vector adapters
- Rollout collection and JSON trajectory saving
- Heuristic, random, and MLP vector policies
- Imitation learning training and checkpoint evaluation
- Multi-episode evaluation utilities
- PPO baseline training loop
- Shared-parameter self-play PPO skeleton

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

- `13`

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

Make sure the arena scene is configured with:

- `MatchController`
- `ArenaTcpBridge`
- `Application.runInBackground = true` through the bridge

### 3. Run a bridge smoke test

```powershell
python python/scripts/test_unity_bridge.py --max-steps 100 --agent1-policy flee --sleep 0.05
```

### 4. Collect rollouts

```powershell
python python/scripts/collect_rollout.py --agent0-policy chase --agent1-policy flee --max-steps 400 --save
```

### 5. Train imitation policy

```powershell
python python/scripts/train_imitation.py --epochs 20 --batch-size 64 --lr 1e-3
```

### 6. Evaluate a checkpoint

```powershell
python python/scripts/eval_checkpoint.py python/checkpoints/imitation/<checkpoint>.pt --episodes 10 --opponent-policy flee
```

### 7. Run PPO / self-play training

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml
```

To slow down visible stepping for debugging:

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml --step-sleep 0.05
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

- The PPO implementation is an early baseline, not a full experiment framework yet
- Self-play currently uses a shared-parameter setup before any opponent-pool logic
- Reward shaping and experiment tracking are still minimal
- Unity presentation polish is still ahead of the current training milestone

## Next Milestones

- stabilize shared-parameter PPO training
- compare trained policies against multiple baselines
- add checkpoint pool / opponent sampling variants
- refine reward design and experiment logging
- export a stronger final policy back into the Unity demo flow

## Author

Julie Yang  
Deep reinforcement learning course project
