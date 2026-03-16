# load_trl_env_from_yaml.py
import os
import yaml
import numpy as np
import torch

from tensordict import TensorDict
from typing import Any, Dict, Optional

# from main.env import BatchedDiffDES
# from main.env_views import RLViewDiffDES

# ---------------- Utils ----------------
# ---- helpers ----
def _as_np(x):
    return np.asarray(x, dtype=np.float32)

def _broadcast_to_len(x, L):
    x = _as_np(x).reshape(-1)
    if x.size == 1:
        return np.full((L,), x.item(), dtype=np.float32)
    if x.size != L:
        raise ValueError(f"Expected length-{L} or scalar, got {x.shape}")
    return x

