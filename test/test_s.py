import argparse
import os
import random
import yaml
import torch
from typing import Dict, Any

from env_s_test import BatchedDiffDES


# ---------- 从 YAML 读取配置 ----------
def load_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ---------- 根据 cfg 构造采样函数 ----------
def make_samplers(cfg, Q, device):
    lam_type = cfg.get("lam_type", "constant")
    lam_params = cfg.get("lam_params", {})
    if lam_type != "constant" or "val" not in lam_params:
        raise ValueError("当前示例仅支持 lam_type=constant 且提供 lam_params.val")

    lams = torch.tensor(lam_params["val"], dtype=torch.float32, device=device)  # [Q]
    if lams.numel() != Q:
        raise ValueError(f"lam_params.val 长度应为 Q={Q}，但得到 {lams.numel()}")

    # inter-arrival：各队列独立 Exp(lam_q)
    def draw_inter_arrivals(env, t: torch.Tensor) -> torch.Tensor:
        # t: [B,1] -> 返回 [B,Q]
        B = t.shape[0]
        U = torch.rand(B, Q, device=env.device).clamp_min(1e-8)
        return -torch.log(U) / lams  # Exp(lam)

    # service requirement：这里给一个简单的 Exp(1.0)，当然你也可以换成别的分布
    def draw_service(env, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        U = torch.rand(B, Q, device=env.device).clamp_min(1e-8)
        return -torch.log(U)  # Exp(1.0)

    def draw_due_date(env, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        Q = env.Q
        rate = 1.0 / 2.0  # mean = 10 → λ = 0.1
        return torch.distributions.Exponential(rate).sample((B, Q)).to(env.device)

    return draw_service, draw_inter_arrivals, draw_due_date

# ---------- 主测试逻辑 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=str, default="n_model_mm_5.yaml",
                        help="同目录下的 YAML 文件名")
    parser.add_argument("--steps", type=int, default=None,
                        help="仿真步数；默认用 YAML 中的 train_T")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    yaml_path = args.yaml
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"找不到 YAML 文件：{yaml_path}")

    # 读配置
    cfg = load_config(yaml_path)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 解析配置 -> 张量
    network = torch.tensor(cfg["network"], dtype=torch.float32)
    mu = torch.tensor(cfg["mu"], dtype=torch.float32)
    h = torch.tensor(cfg["h"], dtype=torch.float32)
    init_queues = torch.tensor(cfg.get("init_queues", [0]*len(cfg["h"])), dtype=torch.float32).unsqueeze(0)  # [1,Q]
    queue_event_options = cfg.get("queue_event_options", None)
    queue_event_options2 = cfg.get("queue_event_options2", None)
    train_T = int(cfg.get("train_T", 5))
    steps = args.steps or train_T

    # 维度检查
    S, Q = network.shape

    # 构造采样器
    draw_service, draw_inter_arrivals, draw_due_date = make_samplers(cfg, Q, device)

    # 实例化环境
    env = BatchedDiffDES(
        network=network,
        mu=mu,
        h=h,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        draw_due_date=draw_due_date,
        max_jobs=10,
        temp=1.0,
        device=device,
        seed=args.seed,
        default_B=1,
        queue_event_options=None if queue_event_options is None else torch.tensor(queue_event_options, dtype=torch.float32),
        queue_event_options2=None if queue_event_options2 is None else torch.tensor(queue_event_options2, dtype=torch.float32),
        reentrant=0,
        verbose=False,
    )

    B_train = 1
    td = env.reset(env.gen_params(batch_size=[B_train]))
    B = 1
    for t in range(1, steps + 1):
        # 随机动作：[B,S,Q]；环境内部会按 server 维对 Q 做归一化
        # action = torch.rand(B, S, Q, device=device)
        action = torch.randint(0, 2, (B, S, Q), device=device)

        # 组装输入 TensorDict
        step_in = td.clone() if hasattr(td, "clone") else td
        step_in["action"] = action

        # 调用环境
        out = env.step(step_in)
        td = out

if __name__ == "__main__":
    main()
