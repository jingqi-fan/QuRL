import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

from RL.env.rl_env import TRLContinuousEnv
from main.env import BatchedDiffDES

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def load_trl_env(env_config: dict,
                 temp: float,
                 batch: int,
                 seed: int,
                 device: str = "cpu",
                 reward_scale: float = 1.0,
                 time_f: bool = False):
    """
    加载纯连续动作的 TorchRL 环境（不依赖 gym），返回 TRLContinuousEnv 实例。
    - env_config 需要包含至少: name, (可选)env_type, h, lam_type, lam_params, network/mu 路径或数组
      例如：
        env_config = {
            "name": "myenv",
            "env_type": "myenv",               # 可缺省，默认同 name
            "network": None,                   # 或 numpy.ndarray
            "mu": None,                        # 或 numpy.ndarray
            "h": [1.0, 1.0, ...],              # 长度 = Q
            "lam_type": "constant"|"step",
            "lam_params": {"val": None 或 数组[Q], "t_step": 100, "val1": [...], "val2": [...]}
        }
    - temp: ST-argmin 温度
    - batch: 批次大小（reset 时会用）
    - seed: 随机种子
    - device: "cuda" 或 "cpu"
    - reward_scale: 奖励缩放
    - time_f: True 时 observation = cat(queues, time)
    """
    name = env_config["name"]
    env_type = env_config.get("env_type", name)

    # ---- 读取 network / mu ----
    net = env_config.get("network", None)
    if net is None:
        network_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_network.npy")
        net = np.load(network_path)
    network = torch.as_tensor(net, dtype=torch.float32)

    mu_arr = env_config.get("mu", None)
    if mu_arr is None:
        mu_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_mu.npy")
        mu_arr = np.load(mu_path)
    mu = torch.as_tensor(mu_arr, dtype=torch.float32)

    S, Q = network.shape
    h = torch.as_tensor(env_config["h"], dtype=torch.float32)
    assert h.numel() == Q, f"h length must be {Q}, got {h.numel()}"

    # ---- λ 设置 ----
    lam_type = env_config["lam_type"]
    lam_params = env_config["lam_params"]
    if lam_params.get("val", None) is None:
        lam_r_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_lam.npy")
        lam_r = np.load(lam_r_path)  # 期望形状 [Q] 或标量
    else:
        lam_r = lam_params["val"]

    lam_r = np.array(lam_r, dtype=np.float32).reshape(-1)  # -> [Q] 或 [1]
    if lam_r.size not in (1, Q):
        raise ValueError(f"lam_params['val'] shape must be scalar or [Q={Q}]")

    def lam_numpy(t_scalar: float) -> np.ndarray:
        """返回当前时刻的到达率向量 (np.float32, shape [Q])"""
        if lam_type == "constant":
            return lam_r if lam_r.size == Q else np.full((Q,), lam_r.item(), dtype=np.float32)
        elif lam_type == "step":
            t_step = lam_params["t_step"]
            val1 = np.array(lam_params["val1"], dtype=np.float32).reshape(-1)
            val2 = np.array(lam_params["val2"], dtype=np.float32).reshape(-1)
            if val1.size not in (1, Q) or val2.size not in (1, Q):
                raise ValueError("val1/val2 must be scalar or length-Q")
            rate1 = val1 if val1.size == Q else np.full((Q,), val1.item(), dtype=np.float32)
            rate2 = val2 if val2.size == Q else np.full((Q,), val2.item(), dtype=np.float32)
            return rate1 if t_scalar <= t_step else rate2
        else:
            raise ValueError("lam_type must be 'constant' or 'step'")

    # ---- 采样器：在 env.device 上返回张量 ----
    def draw_service(env: BatchedDiffDES, t: torch.Tensor) -> torch.Tensor:
        # 指数(1.0) 服务时间，[B,Q]
        B = t.size(0)
        rate = torch.ones(B, Q, device=env.device)
        return torch.distributions.Exponential(rate=rate).sample()

    def draw_inter_arrivals(env: BatchedDiffDES, t: torch.Tensor) -> torch.Tensor:
        # 指数(λ) 到达间隔，[B,Q]；每个 batch 共享同一时刻的 λ 配置
        # t: [B,1]，取第一条的时间作为“当前物理时间”近似
        with torch.no_grad():
            t_now = float(t[0, 0].item()) if t.numel() > 0 else 0.0
            lam_vec = lam_numpy(t_now)  # np [Q]
        # 安全下限避免 0
        lam_vec = np.maximum(lam_vec, 1e-8).astype(np.float32)
        lam = torch.as_tensor(lam_vec, device=env.device).view(1, Q).expand(t.size(0), Q)  # [B,Q]
        return torch.distributions.Exponential(rate=lam).sample()

    # ---- 构造 TorchRL 环境（纯连续动作包装）----
    env = TRLContinuousEnv(
        network=network,
        mu=mu,
        h=h,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        max_jobs=64,            # 或从 env_config 读取
        temp=temp,
        device=device,
        seed=seed,
        verbose=False,
        reward_scale=reward_scale,
        time_f=time_f,
    ).to(device)

    # ---- 按教程风格设置 batch（非 batch-locked，可动态切换）----
    if batch is not None and batch > 0:
        _ = env.reset(env.gen_params(batch_size=[batch]))
    else:
        _ = env.reset()

    return env
