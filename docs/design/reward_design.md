# Reward Design

## Goal

Use the simplest reward design that still teaches competitive behavior without making training unnecessarily fragile.

## Recommended Starting Point

Start with sparse terminal rewards and add shaping only if the baseline fails to learn useful behavior.

### Baseline Rewards

- Win: `+1.0`
- Loss: `-1.0`
- Draw or timeout: `0.0` or small symmetric penalty

## Candidate Shaping Terms

These should be introduced one at a time.

### Outward Pressure

- Reward the agent for moving the opponent closer to the arena boundary
- Risk: can reward meaningless contact farming

### Edge Safety

- Penalize the agent for being too close to the edge
- Risk: can make policies overly passive

### Time Penalty

- Small per-step penalty to encourage engagement
- Risk: can create reckless behavior if too large

### Push Efficiency

- Small bonus when push meaningfully changes opponent position
- Risk: can overfit to a shaped metric instead of winning

## Reward Design Principles

- Prefer sparse rewards first
- Add shaping only to solve a specific learning problem
- Keep shaping interpretable
- Track whether shaping changes actual win rate or only training curves

## Experimental Plan

1. Run sparse terminal reward baseline
2. If no progress, add small time penalty
3. If still weak, test one positional shaping term
4. Compare final win rates against earlier checkpoints

## Failure Modes To Watch

- Agents learn to spin or jitter without engagement
- Agents farm shaping rewards without winning
- Agents become too conservative near the center
- Training becomes unstable after reward additions

## TODO

- Define timeout handling reward
- Decide the first shaping term to test, if any
- Add concrete coefficient candidates after first baseline results
