
# AR_QL.py
# ========
# Core implementation of AR-QL (Action Robust Q-Learning) for Gymnasium + MuJoCo.

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_tensor, soft_update_


# ============================================================
# Replay Buffer: stores (s, aP, aA, r, s', done)
# ============================================================

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ap = np.zeros((capacity, action_dim), dtype=np.float32)
        self.aa = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, s, aP, aA, r, s2, done):
        i = self.ptr
        self.s[i] = s
        self.ap[i] = aP
        self.aa[i] = aA
        self.r[i] = r
        self.s2[i] = s2
        self.d[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            to_tensor(self.s[idx], self.device),
            to_tensor(self.ap[idx], self.device),
            to_tensor(self.aa[idx], self.device),
            to_tensor(self.r[idx], self.device),
            to_tensor(self.s2[idx], self.device),
            to_tensor(self.d[idx], self.device),
        )


# ============================================================
# Networks: Critic Qθ(s,a) and Antagonist πφ(s)
# ============================================================

class CriticQ(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class ActorDet(nn.Module):
    """
    Deterministic antagonist πφ(s) with tanh + scaling to action bounds.
    """
    def __init__(self, state_dim: int, action_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden: int = 256):
        super().__init__()
        # store bounds as buffers (moved with .to(device))
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a01 = (self.net(s) + 1.0) * 0.5  # in [0,1]
        return self.act_low + a01 * (self.act_high - self.act_low)


# ============================================================
# Protagonist sampling: Algorithm 2 (RS) or Algorithm 3 (LD)
# ============================================================

@torch.no_grad()
def estimate_qmin_qmax(
    q: CriticQ,
    s: torch.Tensor,
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    n_est: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate Qmin/Qmax for each state in the batch.
    For MuJoCo multi-D actions, we approximate via uniform random sampling.
    Returns qmin,qmax with shape [B,1].
    """
    B = s.shape[0]
    A = act_low.numel()

    a = act_low + torch.rand((B, n_est, A), device=s.device) * (act_high - act_low)  # [B,n_est,A]
    s_rep = s.unsqueeze(1).repeat(1, n_est, 1)                                        # [B,n_est,S]
    qv = q(s_rep.reshape(B * n_est, -1), a.reshape(B * n_est, A)).view(B, n_est)      # [B,n_est]

    qmin = qv.min(dim=1, keepdim=True).values
    qmax = qv.max(dim=1, keepdim=True).values
    return qmin, qmax


@torch.no_grad()
def sample_action_rs(
    q: CriticQ,
    s: torch.Tensor,
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    L: float,
    n_est: int,
    max_trials: int,
) -> torch.Tensor:
    """
    Algorithm 2 (Rejection Sampling):
      sample a ~ U(amin,amax), u ~ U(L,1)
      accept if u <= (Q(s,a)-Qmin)/(Qmax-Qmin)
    """
    if not (0.0 < L < 1.0):
        raise ValueError(f"L must be in (0,1), got {L}")

    B = s.shape[0]
    A = act_low.numel()

    qmin, qmax = estimate_qmin_qmax(q, s, act_low, act_high, n_est=n_est)
    denom = (qmax - qmin).clamp_min(1e-6)

    accepted = torch.zeros((B, 1), dtype=torch.bool, device=s.device)
    a_out = torch.zeros((B, A), dtype=torch.float32, device=s.device)

    for _ in range(max_trials):
        a = act_low + torch.rand((B, A), device=s.device) * (act_high - act_low)
        u = torch.rand((B, 1), device=s.device) * (1.0 - L) + L  # U(L,1)
        ratio = (q(s, a) - qmin) / denom
        ok = u <= ratio
        newly = (~accepted) & ok
        if newly.any():
            a_out[newly.squeeze(1)] = a[newly.squeeze(1)]
            accepted = accepted | newly
        if accepted.all():
            return a_out

    # Fallback: pick the best among n_est uniform samples (greedy)
    a = act_low + torch.rand((B, n_est, A), device=s.device) * (act_high - act_low)
    s_rep = s.unsqueeze(1).repeat(1, n_est, 1)
    qv = q(s_rep.reshape(B * n_est, -1), a.reshape(B * n_est, A)).view(B, n_est)
    idx = qv.argmax(dim=1)
    return a[torch.arange(B, device=s.device), idx, :]


def sample_action_ld(
    q: CriticQ,
    s: torch.Tensor,
    act_low: torch.Tensor,
    act_high: torch.Tensor,
    epsilon: float,
    n_steps: int,
) -> torch.Tensor:
    """
    Algorithm 3 (Langevin Dynamics):
      a_{n+1} = a_n + ε ∇_a Q(s,a)|_{a_n} + sqrt(2ε) z

    NOTE: No decorator used; gradients are required.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    B = s.shape[0]
    A = act_low.numel()
    a = act_low + torch.rand((B, A), device=s.device) * (act_high - act_low)

    for _ in range(n_steps):
        a.requires_grad_(True)
        qsum = q(s, a).sum()
        grad = torch.autograd.grad(qsum, a, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            z = torch.randn_like(a)
            a = a + float(epsilon) * grad + math.sqrt(2.0 * float(epsilon)) * z
            a = torch.max(torch.min(a, act_high), act_low)  # clip to bounds
        a = a.detach()

    return a


# ============================================================
# Hyperparameters container (no decorators)
# ============================================================

class ARQLConfig:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        tau: float,
        critic_lr: float,
        actor_lr: float,
        sampler: str,
        L: float,
        rs_n_est: int,
        rs_max_trials: int,
        epsilon: float,
        ld_steps: int,
        replay_size: int,
        batch_size: int,
        warmup_steps: int,
        N: int,
        exploration_std: float,
        hidden: int = 256,
    ):
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.critic_lr = float(critic_lr)
        self.actor_lr = float(actor_lr)

        self.sampler = str(sampler)
        self.L = float(L)
        self.rs_n_est = int(rs_n_est)
        self.rs_max_trials = int(rs_max_trials)

        self.epsilon = float(epsilon)
        self.ld_steps = int(ld_steps)

        self.replay_size = int(replay_size)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.N = int(N)

        self.exploration_std = float(exploration_std)
        self.hidden = int(hidden)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ============================================================
# AR-QL Agent
# ============================================================

class ARQLAgent:
    def __init__(self, state_dim: int, action_dim: int, act_low: np.ndarray, act_high: np.ndarray,
                 cfg: ARQLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.act_low_np = act_low.astype(np.float32)
        self.act_high_np = act_high.astype(np.float32)
        self.act_low = torch.as_tensor(self.act_low_np, dtype=torch.float32, device=device)
        self.act_high = torch.as_tensor(self.act_high_np, dtype=torch.float32, device=device)

        self.q = CriticQ(state_dim, action_dim, hidden=cfg.hidden).to(device)        # Qθ
        self.q_targ = CriticQ(state_dim, action_dim, hidden=cfg.hidden).to(device)   # Q̄θ
        self.q_targ.load_state_dict(self.q.state_dict())

        self.pi = ActorDet(state_dim, action_dim, self.act_low_np, self.act_high_np, hidden=cfg.hidden).to(device)      # πφ
        self.pi_targ = ActorDet(state_dim, action_dim, self.act_low_np, self.act_high_np, hidden=cfg.hidden).to(device) # πφ−
        self.pi_targ.load_state_dict(self.pi.state_dict())

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.actor_lr)

    def sample_aP(self, s: torch.Tensor, use_target_q: bool) -> torch.Tensor:
        qnet = self.q_targ if use_target_q else self.q
        if self.cfg.sampler.lower() == "rs":
            return sample_action_rs(
                qnet, s, self.act_low, self.act_high,
                L=self.cfg.L, n_est=self.cfg.rs_n_est, max_trials=self.cfg.rs_max_trials
            )
        if self.cfg.sampler.lower() == "ld":
            return sample_action_ld(
                qnet, s, self.act_low, self.act_high,
                epsilon=self.cfg.epsilon, n_steps=self.cfg.ld_steps
            )
        raise ValueError(f"Unknown sampler: {self.cfg.sampler}")

    @torch.no_grad()
    def act_components(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (aP_t, aA_t, ã_t) for environment stepping.
        """
        s = to_tensor(obs.reshape(1, -1), self.device)

        # Algorithm 4, line 6: Sample aP_t from Qθ(·, s_t) using Algorithm 2 or 3
        aP = self.sample_aP(s, use_target_q=False)

        # Algorithm 4, line 7: aA_t = πφ(s_t)
        aA = self.pi(s)

        # Algorithm 4, line 8: a_t = (1-α)aP_t + αaA_t
        a = (1.0 - self.cfg.alpha) * aP + self.cfg.alpha * aA

        # Algorithm 4, line 9: ã_t = a_t + exploration noise
        if self.cfg.exploration_std > 0.0:
            a_tilde = a + torch.randn_like(a) * float(self.cfg.exploration_std)
        else:
            a_tilde = a

        a_tilde = torch.max(torch.min(a_tilde, self.act_high), self.act_low)

        return (
            aP.cpu().numpy().reshape(-1),
            aA.cpu().numpy().reshape(-1),
            a_tilde.cpu().numpy().reshape(-1),
        )

    def update(self, rb: ReplayBuffer) -> dict:
        """
        Implements Algorithm 4 lines 12-21 (updates after collecting experience).
        This function assumes the replay buffer already contains valid transitions.
        """
        cfg = self.cfg
        info = {}

        # Algorithm 4, line 12: for i in 0...N do
        for _i in range(cfg.N):

            # Algorithm 4, line 13: Sample batch from R
            s, aP, aA, r, s2, d = rb.sample(cfg.batch_size)

            # Algorithm 4, line 14: Sample aP from Q̄θ(·, s') using Algorithm 2 or 3
            aP2 = self.sample_aP(s2, use_target_q=True).detach()

            # Algorithm 4, line 15: aA = πφ−(s')
            aA2 = self.pi_targ(s2).detach()

            # (helper for line 16) build next mixed action
            a_mix2 = (1.0 - cfg.alpha) * aP2 + cfg.alpha * aA2

            with torch.no_grad():
                # (helper for line 16) TD target y
                y = r + cfg.gamma * (1.0 - d) * self.q_targ(s2, a_mix2)

            # Algorithm 4, line 16: Update critic (MSE) using mixed current action from replay
            a_mix = (1.0 - cfg.alpha) * aP + cfg.alpha * aA
            qv = self.q(s, a_mix)
            q_loss = F.mse_loss(qv, y)

            self.q_opt.zero_grad(set_to_none=True)
            q_loss.backward()
            self.q_opt.step()

            info["q_loss"] = float(q_loss.item())
            info["q_mean"] = float(qv.mean().item())

        # Algorithm 4, line 17: end for

        # Algorithm 4, line 18: Sample batch from R
        s, aP, _aA_buf, _r, _s2, _d = rb.sample(cfg.batch_size)

        # Algorithm 4, line 19: Update adversary (minimize Qθ(s, (1-α)aP + α πφ(s)))
        aA_new = self.pi(s)
        a_mix_adv = (1.0 - cfg.alpha) * aP + cfg.alpha * aA_new
        adv_loss = self.q(s, a_mix_adv).mean()

        self.pi_opt.zero_grad(set_to_none=True)
        adv_loss.backward()
        self.pi_opt.step()
        info["adv_loss"] = float(adv_loss.item())

        # Algorithm 4, line 20: Update critic (one extra TD update after adversary update)
        s, aP, aA, r, s2, d = rb.sample(cfg.batch_size)
        aP2 = self.sample_aP(s2, use_target_q=True).detach()
        aA2 = self.pi_targ(s2).detach()
        a_mix2 = (1.0 - cfg.alpha) * aP2 + cfg.alpha * aA2
        with torch.no_grad():
            y = r + cfg.gamma * (1.0 - d) * self.q_targ(s2, a_mix2)
        a_mix = (1.0 - cfg.alpha) * aP + cfg.alpha * aA
        qv2 = self.q(s, a_mix)
        q_loss2 = F.mse_loss(qv2, y)

        self.q_opt.zero_grad(set_to_none=True)
        q_loss2.backward()
        self.q_opt.step()

        info["q_loss_post_adv"] = float(q_loss2.item())

        # Algorithm 4, line 21: Update target networks (soft update)
        soft_update_(self.pi_targ, self.pi, cfg.tau)   # φ− ← τ φ + (1-τ) φ−
        soft_update_(self.q_targ, self.q, cfg.tau)     # θ̄ ← τ θ + (1-τ) θ̄

        return info
