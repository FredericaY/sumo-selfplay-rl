"""Smoke test for the Unity file bridge."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs.unity_file_bridge_client import UnityFileBridgeClient, UnityFileBridgeConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the Unity file bridge smoke test."""

    parser = argparse.ArgumentParser(description="Smoke test the Unity arena file bridge.")
    parser.add_argument(
        "--bridge-dir",
        default=str(ROOT.parent / "unity" / "bridge_io"),
        help="Shared directory used by Unity and Python for request/response files.",
    )
    parser.add_argument("--steps", type=int, default=120, help="Number of control steps to run.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional wall-clock sleep between step requests.",
    )
    return parser.parse_args()


def normalize(vector: tuple[float, float]) -> list[float]:
    """Normalize a 2D vector into a unit-length list."""

    x, y = vector
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        return [0.0, 0.0]
    return [x / norm, y / norm]


def chase_policy(agent_obs: dict[str, Any], step_idx: int) -> dict[str, Any]:
    """Very small policy: move toward the opponent and push periodically."""

    self_pos = agent_obs["selfPosition"]
    opp_pos = agent_obs["opponentPosition"]
    direction = normalize((opp_pos["x"] - self_pos["x"], opp_pos["y"] - self_pos["y"]))
    push_ready = bool(agent_obs["pushReady"])

    return {
        "move": direction,
        "push": direction,
        "use_push": push_ready and step_idx % 12 == 0,
    }


def idle_policy() -> dict[str, Any]:
    """Return a no-op action for the second agent during the first smoke test."""

    return {
        "move": [0.0, 0.0],
        "push": [0.0, 0.0],
        "use_push": False,
    }


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
            f"Unity file bridge returned an error during {stage}:\n"
            f"{json.dumps(state, indent=2, ensure_ascii=False)}"
        )


def main() -> None:
    """Run the Unity file bridge smoke test."""

    args = parse_args()
    client = UnityFileBridgeClient(
        UnityFileBridgeConfig(bridge_dir=Path(args.bridge_dir), timeout_s=5.0)
    )

    state = client.reset()
    ensure_ok_response(state, "reset")
    print(f"Connected through files. Reset status: {state['status']}")

    for step_idx in range(args.steps):
        agent0_action = chase_policy(state["agent0"], step_idx)
        agent1_action = idle_policy()
        state = client.step(agent0_action, agent1_action)
        ensure_ok_response(state, f"step {step_idx}")

        print(
            f"step={step_idx:03d} "
            f"done={state['done']} winner={state['winner']} "
            f"agent0[{format_agent_status(state['agent0'])}] "
            f"agent1[{format_agent_status(state['agent1'])}]"
        )

        if state["done"]:
            print("Match ended. Sending reset.")
            state = client.reset()
            ensure_ok_response(state, "post-match reset")
            print(f"Reset status: {state['status']}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    final_state = client.get_state()
    ensure_ok_response(final_state, "get_state")
    print(f"Final state status: {final_state['status']}")


if __name__ == "__main__":
    main()
