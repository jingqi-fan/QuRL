# rl_env.py  — TorchRL-only wrapper that inherits DiffDiscreteEventSystemTorch

import os
import numpy as np
import torch
from typing import Optional, Dict, Any

# 路径注入（和你原来一致）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
import sys
sys.path.append(project_root)

from main.env import DiffDiscreteEventSystemTorch


class RLQueueEnv(DiffDiscreteEventSystemTorch):
    """Pure TorchRL env that *inherits* DiffDiscreteEventSystemTorch.
    提供 from_config 工厂方法，把 yaml/dict 配置转成 GPU 上可用的 env。
    """

    def __init__(
        self,
        *,
        network: torch.Tensor,            # [s, q]
        mu: torch.Tensor,                 # [s, q]
        h: torch.Tensor,                  # [q]
        draw_service,                     # (env, time) -> [1, q] torch
        draw_inter_arrivals,              # (env, time) -> [1, q] torch
        queue_event_options: Optional[torch.Tensor] = None,  # [2q, q]
        temp: float = 1.0,
        device: str | torch.device = "cuda",
        seed: int = 3003,
    ):
        super().__init__(
            network=network,
            mu=mu,
            h=h,
            draw_service=draw_service,
            draw_inter_arrivals=draw_inter_arrivals,
            queue_event_options=queue_event_options,
            temp=temp,
            device=device,
            seed=seed,
        )

    # -------- convenience: build from config dict --------
    @classmethod
    def from_config(
        cls,
        env_config: Dict[str, Any],
        *,
        temp: float = 1.0,
        seed: int = 3003,
        device: str | torch.device = "cuda",
    ) -> "RLQueueEnv":
        """把你原来的 env_config（yaml -> dict）转换成 TorchRL 环境。
        - 全部 tensor 放到 device（默认 cuda）
        - 采样函数 (service / inter-arrivals) 用 torch.distributions 实现
        """
        dev = torch.device(device)

        name = env_config.get("name")
        env_type = env_config.get("env_type", name)

        # ---- load network ----
        if env_config.get("network") is None:
            network_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_network.npy")
            env_config["network"] = np.load(network_path)
        network = torch.as_tensor(env_config["network"], dtype=torch.float32, device=dev)

        # ---- load mu ----
        if env_config.get("mu") is None:
            mu_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_mu.npy")
            env_config["mu"] = np.load(mu_path)
        mu = torch.as_tensor(env_config["mu"], dtype=torch.float32, device=dev)

        # ---- h (queue cost weights) ----
        h = torch.as_tensor(env_config["h"], dtype=torch.float32, device=dev)

        s, q = network.shape

        # ---- queue_event_options ----
        queue_event_options = env_config.get("queue_event_options")
        if queue_event_options is not None:
            if queue_event_options == "custom":
                queue_event_options_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_delta.npy")
                queue_event_options = torch.as_tensor(np.load(queue_event_options_path), dtype=torch.float32, device=dev)
            else:
                queue_event_options = torch.as_tensor(queue_event_options, dtype=torch.float32, device=dev)

        # ---- arrival rate (lambda) config ----
        lam_type = env_config.get("lam_type", "constant")
        lam_params = env_config.get("lam_params", {"val": None})

        if lam_params.get("val") is None:
            lam_r_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_lam.npy")
            lam_r_np = np.load(lam_r_path)
        else:
            lam_r_np = np.asarray(lam_params["val"])
        lam_r = torch.as_tensor(lam_r_np, dtype=torch.float32, device=dev)  # [q]

        # torch 版 λ(t)
        def lam_rate_at(t: torch.Tensor) -> torch.Tensor:
            # t: [1,1]
            if lam_type == "constant":
                return lam_r
            elif lam_type == "step":
                val1 = torch.as_tensor(lam_params["val1"], dtype=torch.float32, device=dev)
                val2 = torch.as_tensor(lam_params["val2"], dtype=torch.float32, device=dev)
                is_surge = (t.squeeze() <= float(lam_params["t_step"]))
                return val1 if is_surge else val2
            else:
                raise ValueError(f"Unsupported lam_type: {lam_type}")

        # ---- torch-native sampling functions ----
        def draw_inter_arrivals(env: DiffDiscreteEventSystemTorch, time: torch.Tensor) -> torch.Tensor:
            rate = lam_rate_at(time).unsqueeze(0)  # [1, q]
            return torch.distributions.Exponential(rate=rate).sample()

        def draw_service(env: DiffDiscreteEventSystemTorch, time: torch.Tensor) -> torch.Tensor:
            rate = torch.ones(1, q, device=dev)
            return torch.distributions.Exponential(rate=rate).sample()

        # ---- build env (on device) ----
        return cls(
            network=network,
            mu=mu,
            h=h,
            draw_service=draw_service,
            draw_inter_arrivals=draw_inter_arrivals,
            queue_event_options=queue_event_options,
            temp=temp,
            device=dev,
            seed=seed,
        )

def load_rl_p_env(
    env_config: Dict[str, Any],
    temp: float,
    batch: int,           # 兼容旧签名：此参数此处不再使用
    seed: int,
    policy_name: str,     # 兼容旧签名：此参数此处不再使用
    device: str | torch.device,
):
    """
    兼容旧接口的“薄封装”：
    - 忽略 old-SB3 流程里用到的 batch / policy_name（TorchRL 里建议用 collector/ParallelEnv 管理并行）
    - 直接调用纯 TorchRL 的构造器，返回一个 GPU 就绪的环境实例
    """
    dev = torch.device(device)
    env = RLQueueEnv.from_config(env_config, temp=temp, seed=seed, device=dev)
    return env

# -----------------------------
# Minimal example
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 这里用一个最小配置；实际使用时从你的 yaml 读进来
    env_config = {
        "name": "demo",
        "network": np.ones((3, 4), dtype=np.float32),
        "mu": (np.random.rand(3, 4).astype(np.float32) + 0.5),
        "h": np.ones(4, dtype=np.float32),
        "lam_type": "constant",
        "lam_params": {"val": [1.0, 1.0, 1.0, 1.0]},
        "queue_event_options": None,
    }

    env = RLQueueEnv.from_config(env_config, device=device, seed=3003, temp=1.0)

    td = env.reset()
    # print("reset:", td.get("queues"), td.get("time"))

    from tensordict import TensorDict
    for t in range(5):
        action = env.action_spec.rand()  # TorchRL 原生动作采样
        td = env.step(TensorDict({"action": action}, batch_size=[]))
        # print(
        #     f"t={t+1:02d} | time={float(td.get('time').item()):.4f} | "
        #     f"reward={float(td.get('reward').item()):.4f} | queues={td.get('queues').tolist()}"
        # )
