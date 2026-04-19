"""Persistent TCP client for talking to the Unity arena bridge."""

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
    """Persistent request/response TCP client for the Unity arena bridge."""

    def __init__(self, config: UnityBridgeConfig | None = None) -> None:
        self.config = config or UnityBridgeConfig()
        self._socket: socket.socket | None = None
        self._recv_buffer = b""

    def connect(self) -> None:
        """Open the persistent TCP connection if needed."""

        if self._socket is not None:
            return

        sock = socket.create_connection(
            (self.config.host, self.config.port),
            timeout=self.config.timeout_s,
        )
        sock.settimeout(self.config.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = sock
        self._recv_buffer = b""

    def close(self) -> None:
        """Close the persistent connection and clear local buffers."""

        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
        self._recv_buffer = b""

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one JSON request over the persistent TCP connection."""

        message = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"

        for attempt_idx in range(2):
            try:
                self.connect()
                assert self._socket is not None
                self._socket.sendall(message)
                response_line = self._recv_line()
                return json.loads(response_line.decode("utf-8"))
            except (OSError, RuntimeError, json.JSONDecodeError):
                self.close()
                if attempt_idx == 1:
                    raise

        raise RuntimeError("Unity bridge request failed after reconnect attempts.")

    def reset(self) -> dict[str, Any]:
        return self.request({"command": "reset"})

    def get_state(self) -> dict[str, Any]:
        return self.request({"command": "get_state"})

    def step(self, agent0: dict[str, Any], agent1: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            {
                "command": "step",
                "agent0": self._format_action(agent0),
                "agent1": self._format_action(agent1),
            }
        )

    def reset_batch(self, arena_seeds: list[int] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"command": "reset_batch"}
        if arena_seeds is not None:
            payload["arena_seeds"] = [int(seed) for seed in arena_seeds]
        return self.request(payload)

    def reset_arenas(self, arena_ids: list[int], arena_seeds: list[int] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "command": "reset_arenas",
            "arena_ids": [int(arena_id) for arena_id in arena_ids],
        }
        if arena_seeds is not None:
            payload["arena_seeds"] = [int(seed) for seed in arena_seeds]
        return self.request(payload)

    def get_batch_state(self) -> dict[str, Any]:
        return self.request({"command": "get_batch_state"})

    def step_batch(self, arenas: list[dict[str, Any]]) -> dict[str, Any]:
        return self.request(
            {
                "command": "step_batch",
                "arenas": [
                    {
                        "arena_id": int(arena["arena_id"]),
                        "agent0": self._format_action(arena["agent0"]),
                        "agent1": self._format_action(arena["agent1"]),
                    }
                    for arena in arenas
                ],
            }
        )

    def set_agent1_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return self.request(
            {
                "command": "set_agent1_action",
                "agent1": self._format_action(action),
            }
        )

    def _recv_line(self) -> bytes:
        assert self._socket is not None

        while b"\n" not in self._recv_buffer:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RuntimeError("Unity bridge closed the persistent connection unexpectedly.")
            self._recv_buffer += chunk

        line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
        return line.rstrip(b"\r")

    @staticmethod
    def _format_action(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "move": _normalize_vector(action.get("move", [0.0, 0.0])),
            "push": _normalize_vector(action.get("push", [0.0, 0.0])),
            "use_push": bool(action.get("use_push", False)),
        }
