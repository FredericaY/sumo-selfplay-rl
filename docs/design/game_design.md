# Game Design

## Goal

Define a simple and readable 1v1 competitive arena game that is small enough for a solo course project but still rich enough for self-play experiments.

## Core Loop

- Two agents spawn inside a bounded arena
- Each agent tries to control space and force the opponent outward
- A round ends when one agent is pushed out of bounds or another terminal condition is reached
- Multiple rounds can be chained into matches for evaluation

## Intended Feel

- Easy to understand visually in a presentation
- Fast rounds with clear outcomes
- Enough positional depth that timing and orientation matter

## Agent Actions

### Action 1: Move

- Input: direction
- Usage: frequent
- Purpose: repositioning, spacing, edge recovery, baiting

### Action 2: Push or Dash

- Input: direction
- Usage: limited by longer cooldown
- Purpose: burst displacement, ring-out pressure, punish windows

## Arena Rules

- Arena shape: circular by default
- Arena size: moderate, small enough that interactions happen quickly
- Ring-out: leaving the playable boundary causes loss
- Symmetry: both agents use the same rules and action space

## Minimum Viable Prototype

- One arena
- One round win condition
- Simple physics interactions
- Visible cooldown feedback for debugging

## Design Constraints

- Keep mechanics minimal
- Avoid adding health bars, inventories, or many action types early
- Prefer a stable prototype before adding complexity

## Open Questions

- Should move and push share the same direction parameter format?
- Should push apply self-lockout, recoil, or vulnerability?
- Should there be a time limit with a fallback winner rule?

## TODO

- Finalize the exact action parameterization
- Decide the first round length limit
- Define whether collisions are fully physics-driven or partly scripted
- Align the Unity prototype rules with the Python environment spec
