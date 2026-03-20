# DeepRL Self-Play Arena

DeepRL Self-Play Arena is a solo deep reinforcement learning course project focused on building a small 1v1 self-play environment and presenting the final result in Unity.

The core scenario is a toy sumo-style duel. Two agents compete in a circular arena and try to push each other out using only a small action set:

- Move in a chosen direction with short cooldown
- Push or dash in a chosen direction with longer cooldown

This project intentionally keeps the environment simple so the main challenge becomes competitive learning, self-play stability, reward design, and opponent diversity.

## Why Self-Play

Self-play is a natural fit for a symmetric 1v1 game:

- It removes the need to hand-design a strong scripted opponent first
- It allows the agent to continually face stronger policies over time
- It makes it easier to study instability, cycling behavior, and opponent-pool ideas

The longer-term research direction for this repo is to compare baseline latest-policy self-play against simple opponent-pool variants.

## Why Unity

Unity is used for the final demo and gameplay-facing side of the project:

- Fast iteration on arena layout, collisions, feel, and presentation
- Clear visualization for the final course presentation
- Easy path from training environment ideas to a playable prototype

The training code is kept on the Python side so experiments remain easy to run, log, compare, and extend.

## Planned Milestones

1. Finalize environment rules, observations, and reward design
2. Build a minimal playable Unity prototype for the arena loop
3. Implement a simple Python training scaffold and baseline policy loop
4. Run first baseline experiments with random or scripted opponents
5. Add PPO self-play and checkpoint evaluation
6. Add opponent-pool experiments and compare stability
7. Export a trained policy and connect the final demo flow in Unity

## Repository Structure

```text
sumo-selfplay-rl/
├─ README.md
├─ .gitignore
├─ docs/
│  ├─ DeepRL_final_proposal.pdf
│  ├─ design/
│  │  ├─ game_design.md
│  │  ├─ env_spec.md
│  │  ├─ reward_design.md
│  │  └─ experiment_plan.md
│  ├─ meetings/
│  ├─ paper_notes/
│  └─ proposal/
├─ unity/
│  ├─ README.md
│  ├─ project_structure.md
│  ├─ scripts_plan.md
│  └─ SelfPlaySumoArena/
├─ python/
│  ├─ README.md
│  ├─ requirements.txt
│  ├─ pyproject.toml
│  ├─ configs/
│  │  └─ train.yaml
│  ├─ src/
│  │  ├─ agents/
│  │  ├─ algorithms/
│  │  ├─ envs/
│  │  ├─ eval/
│  │  └─ train.py
│  ├─ scripts/
│  │  ├─ train_selfplay.py
│  │  └─ eval_checkpoint.py
│  ├─ checkpoints/
│  ├─ logs/
│  └─ videos/
├─ experiments/
├─ models/
└─ tools/
```

## Current Status

- A Unity project already exists in `unity/SelfPlaySumoArena`
- The repository has now been initialized with documentation and Python training skeletons
- Most files contain TODO markers instead of full implementations on purpose

## Next Steps

- Use the proposal PDF in `docs/` to refine the environment specification
- Decide whether the Python training loop will first use a toy local simulator or connect directly to Unity
- Implement the first playable duel prototype in Unity
- Add the first baseline experiment under `experiments/`

## Author

Julie Yang  
Deep reinforcement learning course project
