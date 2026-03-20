# Unity Project Structure Plan

This file describes the intended organization inside `unity/SelfPlaySumoArena`.

## Recommended Layout

```text
SelfPlaySumoArena/
├─ Assets/
│  ├─ Art/
│  ├─ Audio/
│  ├─ Materials/
│  ├─ Prefabs/
│  ├─ Scenes/
│  │  ├─ Bootstrap.unity
│  │  ├─ Arena_Prototype.unity
│  │  └─ Demo.unity
│  ├─ Scripts/
│  │  ├─ Core/
│  │  ├─ Agents/
│  │  ├─ Gameplay/
│  │  ├─ Arena/
│  │  ├─ TrainingBridge/
│  │  ├─ UI/
│  │  └─ Utils/
│  ├─ ScriptableObjects/
│  ├─ Settings/
│  └─ Gizmos/
├─ Packages/
└─ ProjectSettings/
```

## Folder Responsibilities

### `Assets/Scenes`

- `Bootstrap`: startup scene for loading and global managers
- `Arena_Prototype`: main development scene for testing duel mechanics
- `Demo`: polished scene for the final presentation

### `Assets/Scripts/Core`

- Shared bootstrap code
- Game state flow
- Global configuration references

### `Assets/Scripts/Agents`

- Agent controller
- Action buffering
- Cooldown logic
- Observation collection if Unity is used for training

### `Assets/Scripts/Gameplay`

- Match flow
- Win and loss handling
- Round reset logic

### `Assets/Scripts/Arena`

- Arena bounds
- Ring-out detection
- Spawn points

### `Assets/Scripts/TrainingBridge`

- ML-Agents integration or custom bridge code
- Policy loading hooks
- Debug interfaces for evaluation

## Practical Guidance

- Keep the first scene minimal
- Do not build a large architecture before the duel loop works
- Prefer one clean prototype scene over many partially finished scenes

## TODO

- Compare this plan against the existing Unity project contents
- Create the target folders inside `Assets/` when Unity-side implementation starts
- Decide whether ML-Agents will be used directly in Unity or only as a reference
