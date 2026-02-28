
# eval.py
# =======
# Evaluation utilities for AR-QL on Gymnasium MuJoCo tasks.

import argparse

import numpy as np
import torch
import gymnasium as gym

from utils import flatten_obs, get_device, set_seed
from AR_QL import ARQLAgent, ARQLConfig


@torch.no_grad()
def eval_policy(env: gym.Env, agent: ARQLAgent, episodes: int, mass_param: float, friction_param: float) -> float:
    total = 0.0
    env = env.unwrapped
    env.model.body_mass = env.model.body_mass * mass_param # mass paramter 변화
    env.model.geom_friction = env.model.geom_friction * friction_param # Friction parameter 변화화
    for _ in range(episodes):
        obs, _info = env.reset()
        done = False
        ep = 0.0
        while not done:
            obs_np = flatten_obs(obs)
            _aP, _aA, a_tilde = agent.act_components(obs_np)
            obs, r, terminated, truncated, _info = env.step(a_tilde)
            done = bool(terminated or truncated)
            ep += float(r)
        total += ep
    return total / float(episodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--ckpt", type=str, default="arql_mujoco_checkpoint.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    env = gym.make(args.env, render_mode="human" if args.render else None)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    act_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    payload = torch.load(args.ckpt, map_location=device)
    cfg_dict = payload["cfg"]
    cfg = ARQLConfig(**cfg_dict)

    agent = ARQLAgent(obs_dim, act_dim, act_low, act_high, cfg, device=device)
    agent.q.load_state_dict(payload["q"])
    agent.pi.load_state_dict(payload["pi"])
    agent.q_targ.load_state_dict(payload["q"])
    agent.pi_targ.load_state_dict(payload["pi"])

    avg = eval_policy(env, agent, episodes=args.episodes, mass_param=args.mass_param, friction_param=args.friction_param)
    print(f"Average return over {args.episodes} episodes: {avg:.2f}")
    env.close()


if __name__ == "__main__":
    main()
