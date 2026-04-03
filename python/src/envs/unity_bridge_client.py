"""Small TCP client for talking to the Unity arena bridge.

The bridge protocol is intentionally simple:
- one JSON request per connection
- one JSON response per connection

This keeps the first Unity/Python integration robust and easy to debug.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any


def _normalize_vector(vector: list[float] | tuple[float, float]) -> list[float]:
    """Clamp vector-like inputs into a JSON-friendly 2D list."""

    if len(vector) < 2:
        return [0.0, 0.0]

    return [float(vector[0]), float(vector[1])]


@dataclass
class UnityBridgeConfig:
    """Connection settings for the local Unity TCP bridge."""

    host: str = "127.0.0.1"
    port: int = 5055
    timeout_s: float = 5.0


class UnityBridgeClient:
    """Minimal request/response TCP client for the Unity arena bridge."""

    def __init__(self, config: UnityBridgeConfig | None = None) -> None:
        self.config = config or UnityBridgeConfig()

    def connect(self) -> None:
        """Retained for compatibility with earlier caller code."""

        return

    def close(self) -> None:
        """Retained for compatibility with earlier caller code."""

        return

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON request over a short-lived TCP connection."""

        message = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"

        with socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.timeout_s,
        ) as sock:
            sock.settimeout(self.config.timeout_s)
            sock.sendall(message)
            response_line = self._recv_line(sock)

        return json.loads(response_line.decode("utf-8"))

    def reset(self) -> dict[str, Any]:
        """Reset the Unity match and return the initial state."""

        return self.request({"command": "reset"})

    def get_state(self) -> dict[str, Any]:
        """Fetch the current Unity state without applying an action."""

        return self.request({"command": "get_state"})

    def step(self, agent0: dict[str, Any], agent1: dict[str, Any]) -> dict[str, Any]:
        """Apply one action for each agent and return the next state."""

        return self.request(
            {
                "command": "step",
                "agent0": self._format_action(agent0),
                "agent1": self._format_action(agent1),
            }
        )

    @staticmethod
    def _recv_line(sock: socket.socket) -> bytes:
        """Read until a newline-delimited response is available."""

        buffer = b""
        while b"\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                raise RuntimeError("Unity bridge closed the connection unexpectedly.")
            buffer += chunk

        line, _rest = buffer.split(b"\n", 1)
        return line.rstrip(b"\r")

    @staticmethod
    def _format_action(action: dict[str, Any]) -> dict[str, Any]:
        """Convert a loose Python action dict into the Unity bridge format."""

        return {
            "move": _normalize_vector(action.get("move", [0.0, 0.0])),
            "push": _normalize_vector(action.get("push", [0.0, 0.0])),
            "use_push": bool(action.get("use_push", False)),
        }
