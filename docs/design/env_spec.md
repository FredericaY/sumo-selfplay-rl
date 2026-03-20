# Environment Specification

## Purpose

This document defines the training-facing environment API and the gameplay rules that should stay consistent between Python experiments and the Unity demo.

## Scenario

- Environment name: `SelfPlayArena`
- Mode: two-player symmetric duel
- Objective: push the opponent out of the arena

## Agents

- Number of agents: 2
- Roles: symmetric
- Spawn: mirrored starting positions near the arena center

## Observation Sketch

Each agent should observe compact information that is easy to reproduce in both Python and Unity.

### Candidate Observation Features

- Self position relative to arena center
- Self velocity
- Self facing direction
- Opponent position relative to self
- Opponent velocity
- Opponent facing direction
- Distance to arena edge
- Own push cooldown state
- Opponent push availability estimate if exposed by design

## Action Space Sketch

Keep the action space small and practical for PPO.

### Option A: Hybrid

- Continuous move direction vector
- Continuous push direction vector
- Binary trigger for push

### Option B: Discrete Branches

- Discrete move direction bucket
- Discrete push direction bucket
- Binary push usage

Initial implementation should choose the simpler option for debugging and training stability.

## Episode Termination

- Opponent ring-out
- Self ring-out
- Time limit reached
- Optional invalid physics state reset

## Rewards

Primary signal:

- Win: positive terminal reward
- Loss: negative terminal reward

Possible shaping:

- Small reward for pushing opponent outward
- Small penalty for drifting toward the edge
- Time penalty to discourage stalling

Detailed shaping choices belong in `reward_design.md`.

## Reset Rules

- Reset both agents to mirrored start states
- Reset cooldown timers
- Reset environment timers
- Optional randomization of spawn orientation and minor spawn jitter

## Logging Needs

- Episode length
- Winner
- Ring-out side or termination reason
- Cooldown usage counts
- Optional contact or push statistics

## Compatibility Notes

- The Python environment can begin as a lightweight prototype
- Unity does not need to be fully coupled on day one
- Rule parity should be improved incrementally as the project matures

## TODO

- Choose the first observation vector definition
- Lock the first action space design
- Define exact arena radius and timestep assumptions
- Add a short section comparing Python simulator vs Unity-driven training
