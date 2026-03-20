# Unity Scripts Plan

## Objective

Build the smallest possible playable duel loop first, then add training-facing hooks only after the game rules feel correct.

## First Wave

### `ArenaBounds`

- Detect whether an agent leaves the valid arena area
- Notify the match controller of ring-out events

### `AgentMotor`

- Apply move input
- Apply push or dash impulse
- Handle cooldown timers

### `AgentState`

- Store status needed by gameplay and debugging
- Track whether the agent can push

### `MatchController`

- Spawn or reset both agents
- Start and end rounds
- Decide winner on ring-out or timeout

## Second Wave

### `PlayerInputAdapter`

- Let a human test the prototype locally

### `SimpleBotController`

- Provide a weak scripted opponent for sanity checks

### `HUDController`

- Show timer, cooldowns, and winner text for demos

## Third Wave

### `TrainingAgentAdapter`

- Translate policy actions into gameplay commands
- Expose observations if Unity-side training is used

### `ReplayOrEvaluationHooks`

- Support loading checkpoints or replaying evaluation matches

## TODO

- Turn this plan into concrete C# scripts after the arena prototype scene exists
- Decide whether `AgentMotor` owns cooldown logic or delegates it to `AgentState`
- Add a short testing checklist for each script once implementation begins
