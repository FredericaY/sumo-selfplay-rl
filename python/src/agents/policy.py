"""Policy interface placeholders.

The first version does not implement a real neural network. This module
simply defines the shape of a policy object so the rest of the codebase can
be wired cleanly.
"""

from __future__ import annotations

from typing import Any


class Policy:
    """Minimal policy interface used by training and evaluation code."""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Produce an action from an observation.

        TODO:
        - Replace this placeholder with a torch-based policy
        - Support batched inference if needed
        """

        return {
            "move_direction": [0.0, 0.0],
            "push_direction": [0.0, 0.0],
            "use_push": False,
        }
