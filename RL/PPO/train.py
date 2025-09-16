# train_torchrl.py  — TorchRL PPO (GPU) trainer for your env

import os, sys, time, json, yaml, math
import numpy as np
import torch
from torch import nn
from tensordict import TensorDict
from torchrl.envs import ParallelEnv
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import (
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import AddStateIndependentNormalScale

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torch.optim import Adam

# 项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RL_root = os.path.join(project_root, 'RL')
sys.path.extend([project_root, RL_root])

# 你刚做好的纯 TorchRL 环境加载器（兼容旧签名）
from RL.utils.rl_env import load_rl_p_env  # 返回 RLQueueEnv 实例（TorchRL EnvBase）

# -----------------------------
# Utils
# -----------------------------

def cosine_lr(initial_lr, min_lr=1e-5, progress=1.0, warmup=0.03):
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress > (1 - warmup):
        warm = (1 - progress) / warmup
        return min_lr + (initial_lr - min_lr) * warm
    adj = (progress - (1 - warmup)) / (1 - warmup)
    cos_decay = 0.5 * (1 + math.cos(math.pi * adj))
    return initial_lr * ((1 - min_lr / initial_lr) * cos_decay + min_lr / initial_lr)

def concat_obs(td, time_in_obs: bool):
    # 你的 env obs 是 Composite: {"queues":[q], "time":[1]}
    q = td.get("queues")
    if time_in_obs:
        t = td.get("time")
        return torch.cat([q, t], dim=-1)
    return q

# -----------------------------
# Model builders
# -----------------------------

def build_actor_value(env, pi_arch, vf_arch, time_f: bool, device):
    """构建 Actor/Value 网络（MLP），Actor 用 TanhNormal + AddStateIndependentNormalScale。
       最后由 ProbabilisticActor 自动 clamp 到 action_spec 边界。"""
    obs_dim = env.observation_spec["queues"].shape[-1] + (1 if time_f else 0)
    act_shape = env.action_spec.shape   # [s, q]
    act_dim = int(torch.prod(torch.tensor(act_shape)).item())

    # feature extractor
    def mlp(sizes):
        layers = []
        for i in range(len(sizes)-1):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.Tanh()]
        return nn.Sequential(*layers[:-1])  # 去掉最后一个 Tanh
    actor_net = mlp([obs_dim] + pi_arch + [act_dim])   # 输出动作 mean（未tanh）
    value_net = mlp([obs_dim] + vf_arch + [1])

    # wrap into TensorDictModules
    def obs_to_feat(td):
        x = concat_obs(td, time_f)
        return {"obs_vec": x}
    obs_module = TensorDictModule(lambda td: td.update(obs_to_feat(td)), in_keys=[], out_keys=["obs_vec"])

    actor_module = TensorDictModule(
        module=nn.Sequential(
            nn.LayerNorm(obs_dim),  # 稍微稳定
            nn.Identity(),          # obs_vec 在 td 中
        ),
        in_keys=["obs_vec"],
        out_keys=["obs_vec_norm"],
    )

    # 平滑点：把 obs_vec_norm 直接送入 actor_net/value_net
    class Head(nn.Module):
        def __init__(self, net): super().__init__(); self.net=net
        def forward(self, x): return self.net(x)

    actor_mean = TensorDictModule(
        module=Head(actor_net),
        in_keys=["obs_vec_norm"],
        out_keys=["loc"],  # mean
    )
    # state-independent log_std 参数
    actor_scale = AddStateIndependentNormalScale(
        in_keys=["loc"], out_keys=["scale"], init_scale=0.5, min_scale=1e-4
    )
    # TanhNormal -> 映射到 (-1,1)，ProbActor 会根据 action_spec 再缩放到 [low, high]
    actor = ProbabilisticActor(
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        return_log_prob=True,
        default_interaction_type=None,  # "random" during collect, "mean" during eval 可切换
    )

    # Value
    value_module = TensorDictModule(
        module=Head(value_net),
        in_keys=["obs_vec_norm"],
        out_keys=["state_value"],
    )
    value = ValueOperator(module=value_module, in_keys=["obs_vec_norm"])

    # 串起来：obs_module -> actor_module -> actor_mean -> actor_scale -> actor
    actor_tdseq = TensorDictSequential(obs_module, actor_module, actor_mean, actor_scale, actor).to(device)
    value_tdseq = TensorDictSequential(obs_module, actor_module, value).to(device)

    return actor_tdseq, value_tdseq

# -----------------------------
# Trainer
# -----------------------------

def main():
    # 命令行参数
    cfg_name = sys.argv[1]   # policy_configs/*.yaml
    env_cfg_name = sys.argv[2]  # configs/env/<name>.yaml

    # 读取配置
    config_path = os.path.join(RL_root, 'policy_configs', cfg_name if cfg_name.endswith('.yaml') else f"{cfg_name}.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    env_cfg_path = os.path.join(project_root, 'configs', 'env', f'{env_cfg_name}.yaml')
    with open(env_cfg_path, 'r', encoding='utf-8') as f:
        env_config = yaml.safe_load(f)

    # device & seeds
    device = torch.device(config["env"]["device"])
    train_seed = int(config["env"]["train_seed"])
    test_seed = int(config["env"]["test_seed"])
    time_f = bool(config["env"]["time_f"])
    env_temp = float(config["env"]["env_temp"])
    actors = int(config["training"]["actors"])

    # PPO / GAE 超参
    episode_steps = int(config["training"]["episode_steps"])
    num_epochs = int(config["training"]["num_epochs"])
    ppo_epochs = int(config["training"]["ppo_epochs"])
    gamma = float(config["training"]["gamma"])
    gae_lambda = float(config["training"]["gae_lambda"])
    clip_range = 0.2
    ent_coef = float(config["training"]["ent_coef"])
    vf_coef = float(config["training"]["vf_coef"])
    target_kl = config["training"]["target_kl"]  # 可能是 None

    # 学习率
    lr_policy = float(config["training"]["lr_policy"])
    lr_value = float(config["training"]["lr_value"])
    min_lr_policy = float(config["training"]["min_lr_policy"])
    min_lr_value = float(config["training"]["min_lr_value"])

    # 网络规模
    scale = int(config["model"]["scale"])
    # 根据 env 的 s, q 设置网络宽度（读取一次 env 来拿 shape）
    env_probe = load_rl_p_env(env_config, temp=env_temp, batch=1, seed=train_seed, policy_name="torchrl", device=device)
    q = env_probe.observation_spec["queues"].shape[-1]
    s, qq = env_probe.action_spec.shape
    assert qq == q, "action_spec last dim != queues dim"
    gm = int(math.sqrt(s * q))
    pi_arch = [scale * q, scale * gm, scale * s]
    vf_arch = [scale * q, scale * gm, scale * s]
    del env_probe  # 释放探针环境

    # 环境构造器
    def make_env(i):
        # 不使用 DummyVecEnv，直接返回 TorchRL Env
        return load_rl_p_env(env_config, temp=env_temp, batch=1, seed=train_seed + i, policy_name="torchrl", device=device)

    # 并行环境
    env = ParallelEnv(actors, lambda: make_env(0))  # torchrl 会多次调用 factory；seed 我们在 collector 中设置
    env.set_seed(train_seed)

    # 构建 Actor/Value
    actor, value = build_actor_value(env, pi_arch, vf_arch, time_f=time_f, device=device)

    # GAE & PPO Loss
    advantage = GAE(gamma=gamma, lmbda=gae_lambda, value_network=value, average_gae=True)
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic=value,
        clip_epsilon=clip_range,
        entropy_coef=ent_coef,
        critic_coef=vf_coef,
        normalize_advantage=True,
    ).to(device)

    # 优化器
    optim_policy = Adam(actor.parameters(), lr=lr_policy)
    optim_value = Adam(value.parameters(), lr=lr_value)

    # 采样器（每轮采样 episode_steps * actors 条数据）
    collector = SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=episode_steps * actors,
        total_frames=num_epochs * episode_steps * actors,
        device=device,
        storing_device=device,
        reset_at_each_iter=False,
        split_trajs=True,  # 更利于 GAE
    )

    # Replay-like buffer（on device）
    storage = LazyTensorStorage(episode_steps * actors, device=device)
    rb = TensorDictReplayBuffer(storage=storage)

    global_frames = 0
    for epoch, tensordict_data in enumerate(collector):
        # tensordict_data: ["next","observation", "action", "reward", "done", "terminated", "truncated", "log_prob"]
        frames = tensordict_data.numel()
        global_frames += frames

        # 计算 advantage / returns
        with torch.no_grad():
            tensordict_data = advantage(tensordict_data)
        rb.extend(tensordict_data)

        # PPO 多 epoch
        for ppo_iter in range(ppo_epochs):
            # 这里做简单整包训练（也可切成 minibatch）
            batch = rb.sample(len(rb))
            # 更新 loss 的 KL 参考
            loss_module.update_sampled_log_prob(batch)

            # policy 更新
            optim_policy.zero_grad()
            loss_pi = loss_module.actor_loss(batch)
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optim_policy.step()

            # value 更新
            optim_value.zero_grad()
            loss_v = loss_module.critic_loss(batch)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), max_norm=1.0)
            optim_value.step()

            # 可选：early stop based on target_kl
            if target_kl is not None:
                with torch.no_grad():
                    approx_kl = loss_module.approx_kl(batch)
                if torch.mean(approx_kl).item() > 1.5 * float(target_kl):
                    print(f"[epoch {epoch}] early stop PPO iters: KL={approx_kl.mean().item():.4f}")
                    break

        # 清空 buffer
        rb.empty()

        # 余弦学习率
        progress = 1.0 - (epoch + 1) / float(num_epochs)
        for pg in optim_policy.param_groups:
            pg["lr"] = cosine_lr(lr_policy, min_lr=min_lr_policy, progress=progress)
        for pg in optim_value.param_groups:
            pg["lr"] = cosine_lr(lr_value, min_lr=min_lr_value, progress=progress)

        # 监控
        ep_reward = tensordict_data.get(("next", "reward")).mean().item()
        ep_len = frames / actors
        print(f"[{epoch+1:04d}/{num_epochs}] frames={global_frames} | ep_len≈{ep_len:.0f} | "
              f"rew={ep_reward:.4f} | Lpi={loss_pi.item():.4f} | Lv={loss_v.item():.4f} | "
              f"lr_pi={optim_policy.param_groups[0]['lr']:.2e} lr_v={optim_value.param_groups[0]['lr']:.2e}")

        if epoch + 1 >= num_epochs:
            break

    # 可选：保存模型参数
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict(), "value": value.state_dict()}, f"checkpoints/ppo_final.pt")
    print("Training finished.")

if __name__ == "__main__":
    main()
