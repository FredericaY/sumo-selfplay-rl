# Sumo Self-Play RL (Unity ML-Agents)

A Unity-based multi-agent reinforcement learning project that studies training stability in competitive self-play using opponent pool mechanisms.

---

## 🧠 Overview

This project investigates how to stabilize training in competitive multi-agent reinforcement learning (MARL) environments.

In standard self-play, agents continuously train against their latest versions, which introduces a highly non-stationary opponent distribution. This often leads to:

- Policy oscillation  
- Training instability  
- Overfitting to recent opponents  

To address this, we propose an **opponent pool mechanism**, where agents train against a diverse set of previously saved policies instead of only the latest one.

---

## 🎮 Environment

We design a minimal 1v1 physics-based "sumo-style" environment in Unity:

- Two agents compete in a bounded arena  
- Objective: push the opponent out of the arena  

### Actions

Each agent has two core actions:

- **Move**: short-range movement (low cooldown)  
- **Push**: long-range movement with stronger force (high cooldown)  

This setup creates a simple yet expressive competitive environment that supports strategic behavior such as positioning, timing, and aggression.

---

## 🤖 Method

We use **PPO (Proximal Policy Optimization)** with self-play in Unity ML-Agents.

### Baseline

- Standard self-play  
- Agent always plays against the latest policy  

### Our Approach: Opponent Pool

We maintain a pool of historical policies:

- Periodically save checkpoints  
- Sample opponents from the pool during training  

We explore multiple sampling strategies:

- Latest-only (baseline)
- Uniform sampling
- Recency-biased sampling

---

## 📊 Evaluation

We evaluate performance using:

- Win rate against historical checkpoints  
- Training stability (reward curves)  
- Generalization under environment variations  

We also analyze how opponent diversity affects convergence behavior.

---

## 🧪 Project Structure

```
sumo-selfplay-rl/
├── UnityEnv/          # Unity project (ML-Agents environment)
├── training/          # Training scripts and configs
├── models/            # Saved checkpoints
├── results/           # Logs, plots, evaluation results
└── README.md
```

---

## 🚀 Setup

### Requirements

- Unity (2022 LTS recommended)
- ML-Agents Toolkit
- Python 3.10+
- PyTorch

### Training

```bash
mlagents-learn config/ppo.yaml --run-id=sumo_selfplay
```

---

## 🎥 Demo (Coming Soon)

- Self-play training progression  
- Agent behaviors and strategies  
- Comparison between baseline and opponent pool  

---

## 📚 References

- AlphaGo Zero (Silver et al., 2017)  
- OpenAI Five (OpenAI, 2019)  
- Emergent Complexity via Multi-Agent Competition (Bansal et al., 2018)  

---

## 👤 Author

**Julie Yang**

This project is developed independently as part of a Deep Reinforcement Learning course project, with the goal of building a research-oriented and portfolio-ready system.
