
# train.py
# ========
# Train AR-QL on a Gymnasium MuJoCo task.

import argparse

import numpy as np
import torch
import gymnasium as gym

from utils import set_seed, get_device, flatten_obs, action_bounds_from_env
from AR_QL import ARQLConfig, ARQLAgent, ReplayBuffer
from eval import eval_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4",
                        help="MuJoCo task id (gymnasium). Examples: Hopper-v4, Walker2d-v4, HalfCheetah-v4, Ant-v4.")
    parser.add_argument("--total_steps", type=int, default=1_000_000)

    # Algorithm parameters
    parser.add_argument("--alpha", type=float, default=0.1)         # perturbation coefficient α
    parser.add_argument("--gamma", type=float, default=0.99)        # discount γ
    parser.add_argument("--tau", type=float, default=0.005)         # target update τ

    # Optimizers
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--actor_lr", type=float, default=3e-4)

    # Protagonist sampler
    parser.add_argument("--sampler", type=str, choices=["rs", "ld"], default="rs")
    parser.add_argument("--L", type=float, default=0.7)             # RS: u ~ U(L,1)
    parser.add_argument("--rs_n_est", type=int, default=256)        # RS: Qmin/Qmax estimation samples
    parser.add_argument("--rs_max_trials", type=int, default=1024)  # RS: acceptance trials
    parser.add_argument("--epsilon", type=float, default=0.03)      # LD step size ε
    parser.add_argument("--ld_steps", type=int, default=30)         # LD steps

    # Replay / updates
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup_steps", type=int, default=10_000)
    parser.add_argument("--N", type=int, default=1, help="Algorithm 4 lines 12-17: critic updates per env step")

    # Exploration (Algorithm 4 line 9)
    parser.add_argument("--exploration_std", type=float, default=0.1)

    # Eval
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=5)

    # Network size
    parser.add_argument("--hidden", type=int, default=256)

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default="arql_mujoco_checkpoint.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # ------------------------------------------------------------
    # Algorithm 4 (AR-QL) — line-by-line mapping (as code comments)
    # ------------------------------------------------------------

    # Algorithm 4, line 1: Initialize: Qθ, πϕ, α, γ, Replay buffer R
    env = gym.make(args.env)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_low, act_high = action_bounds_from_env(env)

    cfg = ARQLConfig(
        alpha=args.alpha,             # α
        gamma=args.gamma,             # γ
        tau=args.tau,                 # τ
        critic_lr=args.critic_lr,     # learning rate for Qθ
        actor_lr=args.actor_lr,       # learning rate for πϕ
        sampler=args.sampler,         # Algorithm 2 or 3 for protagonist sampling
        L=args.L,                     # RS hyperparameter
        rs_n_est=args.rs_n_est,       # RS estimation samples
        rs_max_trials=args.rs_max_trials,
        epsilon=args.epsilon,         # LD step size
        ld_steps=args.ld_steps,       # LD iterations
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        N=args.N,                     # number of critic updates per env step
        exploration_std=args.exploration_std,
        hidden=args.hidden,
    )

    agent = ARQLAgent(obs_dim, act_dim, act_low, act_high, cfg, device=device)
    rb = ReplayBuffer(obs_dim, act_dim, capacity=cfg.replay_size, device=device)

    # Algorithm 4, line 2: Initialize target networks: Q̄θ ← Qθ, πϕ− ← πϕ
    # (Done in ARQLAgent.__init__ via state_dict copy.)

    # Algorithm 4, line 3: for episode in 0...M do
    obs, _info = env.reset(seed=args.seed)  # Algorithm 4, line 4: receive initial state s0
    obs = flatten_obs(obs)

    best_eval = -1e18

    for t_global in range(1, args.total_steps + 1):
        aP_t, aA_t, a_tilde = agent.act_components(obs)

        # Algorithm 4, line 10: Execute ã_t, observe r_t, s_{t+1}
        obs2, r, terminated, truncated, _info = env.step(a_tilde)
        done = bool(terminated or truncated)
        obs2 = flatten_obs(obs2)

        # Algorithm 4, line 11: Store (s_t, aP_t, aA_t, r_t, s_{t+1}) in replay buffer R
        rb.add(obs, aP_t, aA_t, float(r), obs2, done)

        obs = obs2

        if done:
            # Algorithm 4, line 22: end for (episode time-step loop) — reset next episode
            obs, _info = env.reset()
            obs = flatten_obs(obs)

        # Algorithm 4, lines 12-21: Parameter updates (after sufficient warmup data)
        if rb.size >= cfg.warmup_steps:
            info = agent.update(rb)

        # Periodic evaluation
        if (t_global % args.eval_every == 0) and (rb.size >= cfg.warmup_steps):
            eval_env = gym.make(args.env)
            avg = eval_policy(eval_env, agent, episodes=args.eval_episodes)
            eval_env.close()

            print(f"[eval @ step {t_global}] avg_return={avg:.2f}")

            if avg > best_eval:
                best_eval = avg
                torch.save(
                    {
                        "cfg": cfg.to_dict(),
                        "q": agent.q.state_dict(),
                        "pi": agent.pi.state_dict(),
                    },
                    args.save,
                )
                print(f"  saved best checkpoint to {args.save}")
    torch.save(
        {
            "cfg": cfg.to_dict(),
            "q": agent.q.state_dict(),
            "pi": agent.pi.state_dict(),
        },
        args.save,
    )
    print(f"Training finished. Saved checkpoint to {args.save}")
    env.close()


if __name__ == "__main__":
    main()
