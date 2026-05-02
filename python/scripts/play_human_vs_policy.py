"""Play human-vs-policy by controlling agent_1 from Python while agent_0 is human-controlled in Unity."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required for play_human_vs_policy.py. Activate the rl_sumo environment first."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents import ActorCritic, ActorCriticConfig, MLPVectorPolicy
from envs.observation_adapter import DEFAULT_OBS_DIM, ObservationAdapter, ObservationVectorConfig
from envs.action_adapter import ActionAdapter
from envs.unity_bridge_client import UnityBridgeClient, UnityBridgeConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run human-vs-policy mode with Unity controlling agent_0 by keyboard.")
    parser.add_argument("checkpoint", help="Path to a .pt checkpoint file.")
    parser.add_argument("--host", default="127.0.0.1", help="Unity bridge host.")
    parser.add_argument("--port", type=int, default=5055, help="Unity bridge port.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Wall-clock sleep between policy updates.")
    parser.add_argument("--policy-side", choices=["a", "b"], default="b", help="Which PPO sub-policy to use when loading a two-policy checkpoint.")
    parser.add_argument("--reset-on-start", action="store_true", help="Reset the match once before starting realtime control.")
    parser.add_argument("--auto-reset", action="store_true", help="Automatically reset after terminal states.")
    parser.add_argument("--debug-every", type=int, default=30, help="Print a short debug line every N updates.")
    return parser.parse_args()


class PPOVectorPolicy:
    def __init__(self, checkpoint_path: Path, policy_side: str = "b", device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.policy_side = policy_side
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        self.network = ActorCritic(
            ActorCriticConfig(
                obs_dim=DEFAULT_OBS_DIM,
                use_edge_gate=bool(config.get("use_edge_gate", False)),
                arena_radius=float(config.get("arena_radius", 5.0)),
                edge_gate_margin=float(config.get("edge_gate_margin", 1.25)),
                edge_gate_min_safety=float(config.get("edge_gate_min_safety", 0.15)),
                edge_gate_push_penalty=float(config.get("edge_gate_push_penalty", 2.0)),
                edge_gate_hidden_size=int(config.get("edge_gate_hidden_size", 16)),
            )
        ).to(self.device)

        if "policy_a_state_dict" in checkpoint and "policy_b_state_dict" in checkpoint:
            state_dict = checkpoint["policy_a_state_dict"] if policy_side == "a" else checkpoint["policy_b_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

        self.network.load_state_dict(state_dict, strict=False)
        self.network.eval()

    def act(self, observation_vector: list[float]) -> list[float]:
        observation_tensor = torch.tensor(observation_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor, _log_prob, _entropy, _value = self.network.sample_action(observation_tensor)
        return action_tensor.squeeze(0).detach().cpu().tolist()


def load_policy(checkpoint_path: Path, policy_side: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "policy_a_state_dict" in checkpoint or "policy_b_state_dict" in checkpoint:
        return PPOVectorPolicy(checkpoint_path, policy_side=policy_side)

    if "state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        hidden_sizes = tuple(config.get("hidden_sizes", (64, 64)))
        action_dim = int(config.get("action_dim", 3))
        policy = MLPVectorPolicy(obs_dim=DEFAULT_OBS_DIM, hidden_sizes=hidden_sizes, action_dim=action_dim)
        policy.load_state_dict(checkpoint["state_dict"])
        return policy

    raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")


def format_obs(agent_obs: dict) -> str:
    pos = agent_obs["selfPosition"]
    vel = agent_obs["selfVelocity"]
    return f"pos=({pos['x']:.2f},{pos['y']:.2f}) vel=({vel['x']:.2f},{vel['y']:.2f}) push_ready={agent_obs['pushReady']}"


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    policy = load_policy(checkpoint_path, policy_side=args.policy_side)
    client = UnityBridgeClient(UnityBridgeConfig(host=args.host, port=args.port, timeout_s=5.0))
    observation_adapter = ObservationAdapter(ObservationVectorConfig(arena_radius=5.0))
    action_adapter = ActionAdapter()

    try:
        try:
            client.connect()
            if args.reset_on_start:
                client.reset()

            loop_idx = 0
            while True:
                state = client.get_state()
                if bool(state.get("done", False)):
                    if args.auto_reset:
                        print(
                            f"terminal: winner={state.get('winner', -1)} reason={state.get('terminalReason', 'unknown')} -> reset"
                        )
                        client.reset()
                        time.sleep(args.sleep)
                        continue

                    print(
                        f"terminal: winner={state.get('winner', -1)} reason={state.get('terminalReason', 'unknown')}"
                    )
                    time.sleep(args.sleep)
                    continue

                mirror_x = observation_adapter.should_mirror_agent(state["agent1"])
                observation_vector = observation_adapter.vectorize_agent(state["agent1"], mirror_x=mirror_x)
                action_vector = policy.act(observation_vector)
                action = action_adapter.action_from_vector(action_vector, mirror_x=mirror_x)
                response = client.set_agent1_action(action)

                if args.debug_every > 0 and loop_idx % args.debug_every == 0:
                    print(
                        f"loop={loop_idx:05d} status={response.get('status', 'unknown')} "
                        f"agent0[{format_obs(response['agent0'])}] agent1[{format_obs(response['agent1'])}]"
                    )

                loop_idx += 1
                time.sleep(args.sleep)
        except (ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise SystemExit(
                "Could not connect to the Unity TCP bridge. "
                "Make sure Unity is in Play mode and ArenaTcpBridge is active. "
                f"Original error: {exc}"
            ) from exc
    finally:
        client.close()


if __name__ == "__main__":
    main()
