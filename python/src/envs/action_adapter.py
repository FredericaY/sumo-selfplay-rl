"""Action helpers for the Unity self-play arena.

The environment still consumes dictionary actions for readability, but this
module defines a compact vector form so future policies can work with fixed
size tensors more easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ActionVectorConfig:
    """Feature toggles for vectorizing and restoring actions."""

    include_push_direction: bool = False
    push_threshold: float = 0.5


class ActionAdapter:
    """Convert between dict actions and compact vector actions."""

    def __init__(self, config: ActionVectorConfig | None = None) -> None:
        self.config = config or ActionVectorConfig()

    def vectorize_action(self, action: dict[str, Any], mirror_x: bool = False) -> list[float]:
        """Convert a Unity bridge action dict into a flat vector."""

        move = action.get("move", [0.0, 0.0])
        push = action.get("push", [0.0, 0.0])
        use_push = 1.0 if bool(action.get("use_push", False)) else 0.0
        move_x = float(move[0])
        move_y = float(move[1])
        push_x = float(push[0])
        push_y = float(push[1])

        if mirror_x:
            move_x *= -1.0
            push_x *= -1.0

        vector = [move_x, move_y, use_push]
        if self.config.include_push_direction:
            vector.extend([push_x, push_y])
        return vector

    def vectorize_joint(self, actions: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
        """Vectorize both agent actions at once."""

        return {
            "agent_0": self.vectorize_action(actions["agent_0"]),
            "agent_1": self.vectorize_action(actions["agent_1"]),
        }

    def action_from_vector(self, vector: list[float], mirror_x: bool = False) -> dict[str, Any]:
        """Convert a compact vector back into a Unity bridge action dict."""

        move_x = float(vector[0]) if len(vector) > 0 else 0.0
        move_y = float(vector[1]) if len(vector) > 1 else 0.0
        use_push = float(vector[2]) >= self.config.push_threshold if len(vector) > 2 else False

        if self.config.include_push_direction and len(vector) >= 5:
            push = [float(vector[3]), float(vector[4])]
        else:
            # Current Unity semantics ignore push direction and use velocity direction,
            # but keeping the field present avoids changing the bridge contract.
            push = [move_x, move_y]

        if mirror_x:
            move_x *= -1.0
            push[0] *= -1.0

        return {
            "move": [move_x, move_y],
            "push": push,
            "use_push": use_push,
        }
