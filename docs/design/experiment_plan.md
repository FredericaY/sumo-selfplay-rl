# Experiment Plan

## Goal

Build a small sequence of experiments that is realistic for a solo course project and supports a clean final report.

## Main Questions

- Can a simple policy learn useful duel behavior in this environment?
- How stable is standard self-play in this setting?
- Does an opponent pool improve robustness or stability?

## Experiment Roadmap

### Exp001: Baseline Random or Scripted Opponent

- Purpose: verify the environment and reward loop
- Output: sanity-check training curves and first qualitative behavior

### Exp002: PPO Self-Play

- Purpose: establish the main baseline
- Output: win-rate snapshots against historical checkpoints

### Exp003: Reward Shaping

- Purpose: test whether limited shaping improves sample efficiency
- Output: compare learning speed and final behavior quality

### Exp004: Opponent Pool

- Purpose: reduce overfitting to the latest opponent
- Output: compare stability and generalization against prior versions

## Metrics

- Episode reward
- Win rate
- Average episode length
- Evaluation vs historical checkpoints
- Qualitative behavior notes from recorded rollouts

## Deliverables

- One clean baseline run
- One self-play result
- One opponent-pool comparison
- One Unity demo scene or match playback for presentation

## Practical Constraints

- Keep the number of major experiment branches small
- Favor interpretable comparisons over large hyperparameter sweeps
- Save enough logs and checkpoints to support final plots

## TODO

- Define evaluation schedule frequency
- Define checkpoint naming convention
- Add a lightweight experiment result template in each `experiments/expXXX_*` folder
