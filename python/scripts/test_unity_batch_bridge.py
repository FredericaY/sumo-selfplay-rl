"""Smoke test for the 4-arena Unity batch bridge."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import UnityVecSelfPlayArenaConfig, UnityVecSelfPlayArenaEnv


def chase_action(obs: dict[str, object]) -> dict[str, object]:
    self_pos = obs["selfPosition"]
    opp_pos = obs["opponentPosition"]
    dx = float(opp_pos["x"]) - float(self_pos["x"])
    dy = float(opp_pos["y"]) - float(self_pos["y"])
    mag = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    move = [dx / mag, dy / mag]
    return {
        "move": move,
        "push": move,
        "use_push": bool(obs.get("pushReady", False)),
    }


def idle_action() -> dict[str, object]:
    return {"move": [0.0, 0.0], "push": [0.0, 0.0], "use_push": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the Unity batch bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = UnityVecSelfPlayArenaEnv(
        UnityVecSelfPlayArenaConfig(
            host=args.host,
            port=args.port,
            num_arenas=4,
        )
    )
    try:
        observations = env.reset()
        print(f"Connected. reset_batch returned {len(observations)} arenas.")

        for step_idx in range(args.steps):
            actions = []
            for arena_id, arena_obs in enumerate(observations):
                actions.append(
                    {
                        "arena_id": arena_id,
                        "agent0": chase_action(arena_obs["agent_0"]),
                        "agent1": idle_action(),
                    }
                )

            observations, rewards, dones, infos = env.step(actions)
            summary = ", ".join(
                [
                    (
                        f"arena={info['arena_id']} "
                        f"done={done} winner={info['winner']} "
                        f"agent0=({arena_obs['agent_0']['selfPosition']['x']:.2f},"
                        f"{arena_obs['agent_0']['selfPosition']['y']:.2f})"
                    )
                    for arena_obs, done, info in zip(observations, dones, infos)
                ]
            )
            print(f"step={step_idx:03d} {summary}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
