# load_trl_env_from_yaml.py
import os
import yaml
import numpy as np
import torch

from tensordict import TensorDict
from typing import Any, Dict, Optional

# 你的环境类（按你给的代码路径导入）
# from main.env import BatchedDiffDES
# from main.env_views import RLViewDiffDES
from RL.env.rl_env import RLViewDiffDES  # 若文件同目录

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

# ---------------- Loader ----------------
# ---------------- Loader ----------------
def load_rl_env(env_config, policy_config, project_root, seed, batch) -> RLViewDiffDES:
    env_name = env_config["name"]
    policy_name = policy_config["name"]

    if "env_type" in env_config:
        env_type = env_config["env_type"]
    else:
        env_type = env_name

    # 从 env 中读取
    if env_config["network"] is None:
        network_path = os.path.join(
            project_root, "configs", "env_data", env_type, f"{env_type}_network.npy"
        )
        env_config["network"] = np.load(network_path)
    env_config["network"] = torch.tensor(env_config["network"]).float()

    if env_config["mu"] is None:
        mu_path = os.path.join(
            project_root, "configs", "env_data", env_type, f"{env_type}_mu.npy"
        )
        env_config["mu"] = np.load(mu_path)
    env_config["mu"] = torch.tensor(env_config["mu"]).float()

    orig_s, orig_q = env_config["network"].size()
    network = env_config["network"].repeat_interleave(1, dim=0)
    mu = env_config["mu"].repeat_interleave(1, dim=0)

    lam_type = env_config["lam_type"]
    lam_params = env_config["lam_params"]
    h = torch.tensor(env_config["h"]).float()

    queue_event_options = env_config["queue_event_options"]
    if queue_event_options is not None:
        if queue_event_options == "custom":
            queue_event_options_path = os.path.join(
                project_root, "configs", "env_data", env_type, f"{env_type}_delta.npy"
            )
            queue_event_options = torch.tensor(np.load(queue_event_options_path))
        else:
            queue_event_options = torch.tensor(queue_event_options)

    if lam_params["val"] is None:
        lam_r_path = os.path.join(
            project_root, "configs", "env_data", env_type, f"{env_type}_lam.npy"
        )
        lam_r = np.load(lam_r_path)
    else:
        lam_r = lam_params["val"]

    def lam(t):
        if lam_type == "constant":
            lam = lam_r
        elif lam_type == "step":
            is_surge = 1 * (t.data.cpu().numpy() <= lam_params["t_step"])
            lam = is_surge * np.array(lam_params["val1"]) + (1 - is_surge) * np.array(
                lam_params["val2"]
            )
        else:
            return "Nonvalid arrival rate"
        return lam

    # 从 policy 中读取
    device = policy_config["env"]["device"]
    temp = policy_config["env"]["temperature"]
    time_f = policy_config["env"]["env_temp"]

    # ---- 抽样器（回到 torch 张量） ----
    def draw_service(env, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        rate = torch.ones(B, orig_q, device=env.device)
        return torch.distributions.Exponential(rate=rate).sample()

    def draw_inter_arrivals(env, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_now = float(t[0, 0].item()) if t.numel() > 0 else 0.0
            lam_vec = np.array(lam(t_now), dtype=np.float32)
            lam_vec = np.maximum(lam_vec, 1e-8)
        lam_t = torch.as_tensor(lam_vec, device=env.device).view(1, orig_q).expand(t.size(0), orig_q)
        return torch.distributions.Exponential(rate=lam_t).sample()

    # ---- 构造环境 ----
    env = RLViewDiffDES(
        network=network,
        mu=mu,
        h=h,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        temp=temp,
        device=device,
        seed=seed,
        default_B=batch,
        queue_event_options=queue_event_options,
        time_f=time_f,
    ).to(device)

    return env
