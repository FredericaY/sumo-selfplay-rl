"""Smoke test for the Unity TCP bridge.

This script does not train anything. It verifies that:
- Python can connect to Unity
- reset/step/get_state all work
- both agents can be controlled externally
- terminal conditions and winner reporting are wired correctly
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs.unity_bridge_client import UnityBridgeClient, UnityBridgeConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the Unity bridge smoke test."""

    parser = argparse.ArgumentParser(description="Smoke test the Unity arena TCP bridge.")
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum number of control steps to run before stopping.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional wall-clock sleep between step requests.",
    )
    parser.add_argument(
        "--agent1-policy",
        choices=["idle", "chase", "flee"],
        default="flee",
        help="Behavior policy used for agent1 during the smoke test.",
    )
    parser.add_argument(
        "--stop-on-done",
        action="store_true",
        help="Stop immediately after the first terminal state.",
    )
    parser.add_argument(
        "--auto-reset",
        action="store_true",
        help="Reset the match automatically after terminal states.",
    )
    return parser.parse_args()


def normalize(vector: tuple[float, float]) -> list[float]:
    """Normalize a 2D vector into a unit-length list."""

    x, y = vector
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        return [0.0, 0.0]
    return [x / norm, y / norm]


def direction_to_opponent(agent_obs: dict[str, Any]) -> list[float]:
    """Compute a normalized direction from self to opponent."""

    self_pos = agent_obs["selfPosition"]
    opp_pos = agent_obs["opponentPosition"]
    return normalize((opp_pos["x"] - self_pos["x"], opp_pos["y"] - self_pos["y"]))


def negate(vector: list[float]) -> list[float]:
    """Return the opposite direction of a 2D vector."""

    return [-vector[0], -vector[1]]


def chase_policy(agent_obs: dict[str, Any], step_idx: int) -> dict[str, Any]:
    """Move toward the opponent and push periodically when ready."""

    direction = direction_to_opponent(agent_obs)
    push_ready = bool(agent_obs["pushReady"])

    return {
        "move": direction,
        "push": direction,
        "use_push": push_ready and step_idx % 12 == 0,
    }


def flee_policy(agent_obs: dict[str, Any], step_idx: int) -> dict[str, Any]:
    """Move away from the opponent and only push occasionally."""

    away = negate(direction_to_opponent(agent_obs))
    push_ready = bool(agent_obs["pushReady"])

    return {
        "move": away,
        "push": away,
        "use_push": push_ready and step_idx % 18 == 0,
    }


def idle_policy() -> dict[str, Any]:
    """Return a no-op action."""

    return {
        "move": [0.0, 0.0],
        "push": [0.0, 0.0],
        "use_push": False,
    }


def select_agent1_action(agent_obs: dict[str, Any], step_idx: int, policy_name: str) -> dict[str, Any]:
    """Dispatch the requested smoke-test policy for agent1."""

    if policy_name == "idle":
        return idle_policy()
    if policy_name == "chase":
        return chase_policy(agent_obs, step_idx)
    return flee_policy(agent_obs, step_idx)


def format_agent_status(agent_obs: dict[str, Any]) -> str:
    """Build a short human-readable summary for console logging."""

    pos = agent_obs["selfPosition"]
    vel = agent_obs["selfVelocity"]
    return (
        f"pos=({pos['x']:.2f},{pos['y']:.2f}) "
        f"vel=({vel['x']:.2f},{vel['y']:.2f}) "
        f"push_ready={agent_obs['pushReady']}"
    )


def ensure_ok_response(state: dict[str, Any], stage: str) -> None:
    """Fail fast with the full Unity response when the bridge returns an error."""

    status = state.get("status", "")
    bad_prefixes = ("handler_exception", "missing_", "invalid_", "unknown_")
    if any(status.startswith(prefix) for prefix in bad_prefixes):
        raise SystemExit(
            f"Unity bridge returned an error during {stage}:\n"
            f"{json.dumps(state, indent=2, ensure_ascii=False)}"
        )


def main() -> None:
    """Run the Unity bridge smoke test."""

    args = parse_args()
    client = UnityBridgeClient(
        UnityBridgeConfig(host=args.host, port=args.port, timeout_s=5.0)
    )

    try:
        try:
            client.connect()
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise SystemExit(
                "Could not connect to the Unity TCP bridge. "
                "Make sure Unity is in Play mode, `ArenaTcpBridge` is attached, "
                "and the port matches the script arguments. "
                f"Original error: {exc}"
            ) from exc

        state = client.reset()
        ensure_ok_response(state, "reset")
        print(f"Connected. Reset status: {state['status']}")

        terminal_count = 0
        for step_idx in range(args.max_steps):
            agent0_action = chase_policy(state["agent0"], step_idx)
            agent1_action = select_agent1_action(state["agent1"], step_idx, args.agent1_policy)
            state = client.step(agent0_action, agent1_action)
            ensure_ok_response(state, f"step {step_idx}")

            print(
                f"step={step_idx:03d} "
                f"done={state['done']} winner={state['winner']} "
                f"agent0[{format_agent_status(state['agent0'])}] "
                f"agent1[{format_agent_status(state['agent1'])}]"
            )

            if state["done"]:
                terminal_count += 1
                print(
                    f"Terminal state reached at step {step_idx}. "
                    f"winner={state['winner']} total_terminals={terminal_count}"
                )

                if args.stop_on_done:
                    break

                if args.auto_reset:
                    state = client.reset()
                    ensure_ok_response(state, "post-match reset")
                    print(f"Reset status: {state['status']}")
                else:
                    break

            if args.sleep > 0:
                time.sleep(args.sleep)

        final_state = client.get_state()
        ensure_ok_response(final_state, "get_state")
        print(f"Final state status: {final_state['status']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
