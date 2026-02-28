
# utils.py
# ========
# Shared utilities for AR-QL (Gymnasium + MuJoCo)

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str.lower() == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def flatten_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def soft_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)
            
def action_bounds_from_env(env) -> Tuple[np.ndarray, np.ndarray]:
    # MuJoCo Gymnasium tasks use Box action space
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    return low, high