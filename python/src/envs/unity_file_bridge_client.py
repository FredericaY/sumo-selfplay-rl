"""File-based bridge client for Unity.

This is a pragmatic fallback transport for early integration work. It keeps
the same request/response payload shape as the TCP bridge while replacing the
transport layer with JSON files on disk.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _normalize_vector(vector: list[float] | tuple[float, float]) -> list[float]:
    """Clamp vector-like inputs into a JSON-friendly 2D list."""

    if len(vector) < 2:
        return [0.0, 0.0]

    return [float(vector[0]), float(vector[1])]


@dataclass
class UnityFileBridgeConfig:
    """Filesystem paths and wait policy for file-based bridge I/O."""

    bridge_dir: Path
    timeout_s: float = 5.0
    poll_interval_s: float = 0.02

    def request_path(self, request_id: str) -> Path:
        return self.bridge_dir / f"request_{request_id}.json"

    def response_path(self, request_id: str) -> Path:
        return self.bridge_dir / f"response_{request_id}.json"


class UnityFileBridgeClient:
    """Very small file-based request/response client for Unity."""

    def __init__(self, config: UnityFileBridgeConfig) -> None:
        self.config = config
        self.config.bridge_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        return

    def close(self) -> None:
        return

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = payload.get("request_id") or uuid.uuid4().hex
        full_payload = dict(payload)
        full_payload["request_id"] = request_id

        request_path = self.config.request_path(request_id)
        response_path = self.config.response_path(request_id)

        if request_path.exists():
            request_path.unlink()
        if response_path.exists():
            response_path.unlink()

        temp_request_path = request_path.with_suffix(".json.tmp")
        temp_request_path.write_text(
            json.dumps(full_payload, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_request_path, request_path)

        deadline = time.time() + self.config.timeout_s
        last_response_text = None
        while time.time() < deadline:
            if response_path.exists():
                last_response_text = response_path.read_text(encoding="utf-8")
                try:
                    response = json.loads(last_response_text)
                except json.JSONDecodeError:
                    time.sleep(self.config.poll_interval_s)
                    continue

                if response.get("request_id") == request_id:
                    return response

            time.sleep(self.config.poll_interval_s)

        raise TimeoutError(
            "Timed out waiting for Unity file bridge response. "
            f"bridge_dir={self.config.bridge_dir}, expected_request_id={request_id}, "
            f"response_path={response_path}, last_response_text={last_response_text!r}"
        )

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

    @staticmethod
    def _format_action(action: dict[str, Any]) -> dict[str, Any]:
        return {
            "move": _normalize_vector(action.get("move", [0.0, 0.0])),
            "push": _normalize_vector(action.get("push", [0.0, 0.0])),
            "use_push": bool(action.get("use_push", False)),
        }
