"""Collect one rollout and optionally save it to disk."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import HeuristicVectorPolicy, MLPVectorPolicy, RandomVectorPolicy
from algorithms.rollout_collector import RolloutCollector
from algorithms.trajectory_serializer import TrajectorySerializer
from envs import UnitySelfPlayArenaConfig, UnitySelfPlayArenaEnv
from envs.action_adapter import ActionAdapter
from envs.observation_adapter import DEFAULT_OBS_DIM, ObservationAdapter, ObservationVectorConfig


OBS_DIM = DEFAULT_OBS_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one rollout episode from the Unity self-play arena."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument("--max-steps", type=int, default=400, help="Maximum episode steps.")
    parser.add_argument(
        "--agent0-policy",
        choices=["idle", "chase", "flee", "random", "mlp"],
        default="chase",
        help="Vector policy for agent 0.",
    )
    parser.add_argument(
        "--agent1-policy",
        choices=["idle", "chase", "flee", "random", "mlp"],
        default="flee",
        help="Vector policy for agent 1.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print a compact JSON summary of the collected rollout.",
    )
    parser.add_argument(
        "--save-dir",
        default="python/logs/rollouts",
        help="Directory for saving collected rollout JSON files.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist the collected rollout to disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for random or MLP policy initialization.",
    )
    return parser.parse_args()


def build_policy(name: str, seed: int):
    if name == "random":
        return RandomVectorPolicy(seed=seed)
    if name == "mlp":
        return MLPVectorPolicy(obs_dim=OBS_DIM, seed=seed)
    return HeuristicVectorPolicy(mode=name)


def main() -> None:
    args = parse_args()
    env = UnitySelfPlayArenaEnv(
        UnitySelfPlayArenaConfig(
            host=args.host,
            port=args.port,
            timeout_s=5.0,
            max_episode_steps=args.max_steps,
        )
    )
    observation_adapter = ObservationAdapter(
        ObservationVectorConfig(arena_radius=env.config.arena_radius)
    )
    collector = RolloutCollector(
        env=env,
        observation_adapter=observation_adapter,
        action_adapter=ActionAdapter(),
    )
    serializer = TrajectorySerializer()
    policy0 = build_policy(args.agent0_policy, seed=args.seed)
    policy1 = build_policy(args.agent1_policy, seed=args.seed + 1)

    try:
        try:
            episode = collector.collect_episode(policy0, policy1)
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise SystemExit(
                "Could not connect to the Unity TCP bridge. "
                "Make sure Unity is in Play mode and ArenaTcpBridge is active. "
                f"Original error: {exc}"
            ) from exc

        last_transition = episode.transitions[-1]
        returns = episode.total_returns()
        print(f"Collected transitions: {len(episode.transitions)}")
        print(
            f"Terminal reason: {last_transition.info.get('terminal_reason', 'unknown')} "
            f"winner={last_transition.info.get('winner', -1)}"
        )
        print(
            f"Returns: agent_0={returns['agent_0']:.3f} "
            f"agent_1={returns['agent_1']:.3f}"
        )
        print(
            f"Vector dims: obs={len(episode.initial_observation_vectors['agent_0'])} "
            f"policy_action={len(episode.transitions[0].policy_action_vectors['agent_0'])} "
            f"executed_action={len(episode.transitions[0].executed_action_vectors['agent_0'])}"
        )

        if args.save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(args.save_dir) / f"rollout_{timestamp}.json"
            saved_path = serializer.save_episode(episode, output_path)
            print(f"Saved rollout to: {saved_path}")

        if args.dump_json:
            payload = {
                "num_transitions": len(episode.transitions),
                "initial_observation_vectors": episode.initial_observation_vectors,
                "initial_policy_action_vector_example": episode.transitions[0].policy_action_vectors["agent_0"],
                "initial_executed_action_vector_example": episode.transitions[0].executed_action_vectors["agent_0"],
                "final_info": last_transition.info,
                "returns": returns,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
