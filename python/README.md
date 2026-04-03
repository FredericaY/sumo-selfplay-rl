# Python Training Workspace

This folder contains the current Python-side training and evaluation pipeline for DeepRL Self-Play Arena.

## What Works Now

The Python workspace is no longer just a skeleton. The following pieces are implemented and already exercised against the Unity environment:

- TCP bridge client for Unity
- environment wrapper with `reset / step / close`
- observation and action vector adapters
- rollout collection and trajectory serialization
- heuristic, random, MLP, and actor-critic policies
- imitation learning training
- checkpoint evaluation over multiple episodes
- PPO baseline training loop
- shared-parameter self-play PPO skeleton

## Main Workflow

### Smoke test Unity connection

```powershell
python python/scripts/test_unity_bridge.py --max-steps 100 --agent1-policy flee --sleep 0.05
```

### Collect demonstration rollouts

```powershell
python python/scripts/collect_rollout.py --agent0-policy chase --agent1-policy flee --max-steps 400 --save
```

### Train imitation policy

```powershell
python python/scripts/train_imitation.py --epochs 20 --batch-size 64 --lr 1e-3
```

### Evaluate an imitation checkpoint

```powershell
python python/scripts/eval_checkpoint.py python/checkpoints/imitation/<checkpoint>.pt --episodes 10 --opponent-policy flee
```

### Run PPO / shared self-play training

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml
```

### Slow down visible stepping for debugging

```powershell
python python/scripts/train_selfplay.py --config python/configs/train.yaml --step-sleep 0.05
```

## Key Directories

- `configs/`: YAML configs
- `scripts/`: runnable entry points
- `src/envs/`: Unity environment wrappers and adapters
- `src/agents/`: policy and model definitions
- `src/algorithms/`: rollout, PPO, and training utilities
- `src/eval/`: evaluation helpers
- `checkpoints/`: saved imitation and PPO checkpoints
- `logs/`: rollout logs and serialized trajectories

## Key Files

- `src/envs/self_play_arena_env.py`
- `src/envs/unity_bridge_client.py`
- `src/envs/observation_adapter.py`
- `src/envs/action_adapter.py`
- `src/agents/policy.py`
- `src/agents/actor_critic.py`
- `src/agents/imitation_trainer.py`
- `src/algorithms/rollout_collector.py`
- `src/algorithms/ppo_trainer.py`
- `src/eval/evaluate.py`

## Notes

- The current observation vector size is `13`
- The current action vector size is `3`
- PPO is still an early baseline and should be treated as an experiment scaffold, not a final trainer
- Shared-parameter self-play is the current main training direction
