# ppo_min_torchrl06.py
import torch
import torch.nn as nn

from torch.distributions import Categorical

from torchrl.envs import GymEnv
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE

# 1) 环境：离散 action，连续 state（CartPole-v1）
env = GymEnv("CartPole-v1")          # obs: shape [4], action: Discrete(2)
eval_env = GymEnv("CartPole-v1")     # 简单评估用（可选）

obs_dim = env.observation_spec.shape[-1]
n_actions = env.action_spec.space.n

device = torch.device("cpu")

# 2) 策略与价值网络（尽量小）
actor_net = MLP(
    in_features=obs_dim,
    out_features=n_actions,
    num_cells=[64, 64],
    activation_class=nn.Tanh,
)
actor_td = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])
actor = ProbabilisticActor(
    module=actor_td,
    in_keys=["logits"],
    dist_class=Categorical,          # 离散分布
    return_log_prob=True,            # 训练 PPO 需要 log_prob
    default_interaction_type=None,   # 训练时用 sample
)
critic = ValueOperator(
    module=MLP(in_features=obs_dim, out_features=1, num_cells=[64, 64], activation_class=nn.Tanh),
    in_keys=["observation"],
)

actor.to(device)
critic.to(device)

# 3) 收集器（同步采样）
collector = SyncDataCollector(
    env,
    policy=actor,
    frames_per_batch=16,  # 每个 batch 的采样步数（小点儿跑得快）
    total_frames=16 * 2, # 总步数（演示足够）
    device=device,
)

# 4) 优化器与 loss/优势估计
optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
gae = GAE(value_network=critic, gamma=0.99, lmbda=0.95)
ppo_loss = ClipPPOLoss(actor=actor, critic=critic, clip_epsilon=0.2, entropy_coef=0.01)

# 5) 训练循环（极简：每个 batch 只做少量 epoch，不做复杂的分片/打乱）
epochs_per_batch = 2

def evaluate(env, policy, episodes=2):
    import numpy as np
    with torch.no_grad():
        rews = []
        for _ in range(episodes):
            td = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                td = policy(td)                     # 生成 "action"
                td = env.step(td)                   # 与环境交互
                ep_r += float(td["reward"])
                done = bool(td.get("done", td.get("terminated", False)))
                td = td.get("next")                 # 下一个时刻
            rews.append(ep_r)
    return np.mean(rews)

print("Start training…")
for batch_idx, td in enumerate(collector):
    # td 形状约为 [T, B] 或 [frames_per_batch]（取决于环境/collector），统一展开成一维
    # 先做 GAE（会在 td 中写入 "advantage" / "value_target"）
    td = td.to(device)
    gae(td)

    # 简单扁平化（时间和批次合并），极简写法
    flat_td = td.reshape(-1)

    for _ in range(epochs_per_batch):
        optim.zero_grad()
        loss_vals = ppo_loss(flat_td)
        loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
        loss.backward()
        optim.step()

    if (batch_idx + 1) % 5 == 0:
        avg_ret = evaluate(eval_env, actor, episodes=5)
        print(f"[Batch {batch_idx+1}] loss={loss.item():.3f}  eval_return={avg_ret:.1f}")

print("Done.")
