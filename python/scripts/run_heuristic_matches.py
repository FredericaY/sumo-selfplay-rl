"""Run repeated heuristic-vs-heuristic matches against the Unity arena."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import socket
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from envs import UnitySelfPlayArenaConfig, UnitySelfPlayArenaEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated heuristic matches in the Unity self-play arena."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum number of steps allowed per episode.",
    )
    parser.add_argument(
        "--agent0-policy",
        choices=["idle", "chase", "flee"],
        default="chase",
        help="Heuristic policy for agent 0.",
    )
    parser.add_argument(
        "--agent1-policy",
        choices=["idle", "chase", "flee"],
        default="flee",
        help="Heuristic policy for agent 1.",
    )
    parser.add_argument(
        "--use-shaped-rewards",
        action="store_true",
        help="Enable lightweight Python-side reward shaping.",
    )
    parser.add_argument(
        "--step-penalty",
        type=float,
        default=0.0,
        help="Small per-step penalty applied symmetrically when shaping is enabled.",
    )
    parser.add_argument(
        "--edge-safety-weight",
        type=float,
        default=0.0,
        help="Penalty weight for drifting toward the arena edge.",
    )
    parser.add_argument(
        "--outward-pressure-weight",
        type=float,
        default=0.0,
        help="Reward weight for pushing the opponent outward.",
    )
    parser.add_argument(
        "--terminal-timeout-penalty",
        type=float,
        default=0.0,
        help="Extra symmetric penalty applied when an episode times out.",
    )
    return parser.parse_args()


def normalize(vector: tuple[float, float]) -> list[float]:
    x, y = vector
    norm = math.sqrt(x * x + y * y)
    if norm < 1e-6:
        return [0.0, 0.0]
    return [x / norm, y / norm]


def direction_to_opponent(agent_obs: dict[str, Any]) -> list[float]:
    self_pos = agent_obs["selfPosition"]
    opp_pos = agent_obs["opponentPosition"]
    return normalize((opp_pos["x"] - self_pos["x"], opp_pos["y"] - self_pos["y"]))


def negate(vector: list[float]) -> list[float]:
    return [-vector[0], -vector[1]]


def idle_policy(_agent_obs: dict[str, Any], _step_idx: int) -> dict[str, Any]:
    return UnitySelfPlayArenaEnv.idle_action()


def chase_policy(agent_obs: dict[str, Any], step_idx: int) -> dict[str, Any]:
    direction = direction_to_opponent(agent_obs)
    push_ready = bool(agent_obs["pushReady"])
    return {
        "move": direction,
        "push": direction,
        "use_push": push_ready and step_idx % 12 == 0,
    }


def flee_policy(agent_obs: dict[str, Any], step_idx: int) -> dict[str, Any]:
    away = negate(direction_to_opponent(agent_obs))
    push_ready = bool(agent_obs["pushReady"])
    return {
        "move": away,
        "push": away,
        "use_push": push_ready and step_idx % 18 == 0,
    }


def select_policy(name: str):
    if name == "idle":
        return idle_policy
    if name == "chase":
        return chase_policy
    return flee_policy


def winner_label_from_info(info: dict[str, Any]) -> str:
    winner = int(info["winner"])
    if winner == 0:
        return "agent_0"
    if winner == 1:
        return "agent_1"

    terminal_reason = info.get("terminal_reason", "unknown")
    if terminal_reason == "time_limit":
        return "time_limit_draw"
    if terminal_reason == "double_ring_out":
        return "double_ring_out_draw"
    return "draw"


def main() -> None:
    args = parse_args()
    env = UnitySelfPlayArenaEnv(
        UnitySelfPlayArenaConfig(
            host=args.host,
            port=args.port,
            timeout_s=5.0,
            max_episode_steps=args.max_steps,
            use_shaped_rewards=args.use_shaped_rewards,
            step_penalty=args.step_penalty,
            edge_safety_weight=args.edge_safety_weight,
            outward_pressure_weight=args.outward_pressure_weight,
            terminal_timeout_penalty=args.terminal_timeout_penalty,
        )
    )

    policy0 = select_policy(args.agent0_policy)
    policy1 = select_policy(args.agent1_policy)
    win_counts = {"agent_0": 0, "agent_1": 0, "time_limit_draw": 0, "double_ring_out_draw": 0, "draw": 0}

    try:
        try:
            observations = env.reset()
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise SystemExit(
                "Could not connect to the Unity TCP bridge. "
                "Make sure Unity is in Play mode and ArenaTcpBridge is active. "
                f"Original error: {exc}"
            ) from exc

        print(env.summary())
        print(
            f"Running {args.episodes} episodes: "
            f"agent_0={args.agent0_policy} vs agent_1={args.agent1_policy}"
        )

        for episode_idx in range(args.episodes):
            if episode_idx > 0:
                observations = env.reset()

            done = False
            info: dict[str, Any] = {"winner": -1, "timeout": False, "terminal_reason": "running"}
            rewards = {"agent_0": 0.0, "agent_1": 0.0}
            returns = {"agent_0": 0.0, "agent_1": 0.0}
            shaping_totals = {"agent_0": 0.0, "agent_1": 0.0}

            while not done:
                step_idx = env.episode_step
                actions = {
                    "agent_0": policy0(observations["agent_0"], step_idx),
                    "agent_1": policy1(observations["agent_1"], step_idx),
                }
                observations, rewards, done, info = env.step(actions)
                for agent_id in returns:
                    returns[agent_id] += rewards[agent_id]
                    shaping_totals[agent_id] += info["shaped_bonus"][agent_id]

            outcome_label = winner_label_from_info(info)
            win_counts[outcome_label] = win_counts.get(outcome_label, 0) + 1

            print(
                f"episode={episode_idx:03d} "
                f"steps={info['episode_step']:03d} "
                f"outcome={outcome_label} "
                f"terminal_reason={info['terminal_reason']} "
                f"python_timeout={info['timeout']} "
                f"return0={returns['agent_0']:.3f} "
                f"return1={returns['agent_1']:.3f} "
                f"shape0={shaping_totals['agent_0']:.3f} "
                f"shape1={shaping_totals['agent_1']:.3f}"
            )

        print("Summary:")
        print(
            f"agent_0_wins={win_counts['agent_0']} "
            f"agent_1_wins={win_counts['agent_1']} "
            f"time_limit_draws={win_counts['time_limit_draw']} "
            f"double_ring_out_draws={win_counts['double_ring_out_draw']} "
            f"other_draws={win_counts['draw']}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
