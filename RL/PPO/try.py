# QGymGPU/RL/train.py
import os
import sys
import time

import numpy as np
import yaml
import torch
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE
# ===== 简化版：不做异常兜底，直接从 spec 读取 =====
from torchrl.data.tensor_specs import DiscreteTensorSpec
from RL.env.rl_env import RLViewDiffDES
from RL.policies.WC_policy import WC_Policy
from RL.policies.vanilla_policy import Vanilla_Policy
from RL.utils.count_time import count_time
from torch.distributions import Normal, Categorical


def load_rl_env(seed, batch):
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
        temp=env_temp,
        device=device,
        seed=seed,
        default_B=batch,
        queue_event_options=queue_event_options,
        time_f=time_f,
    ).to(device)

    return env


def train_ppo():
    ct = count_time(time.time())

    # ======== 可根据你的环境改的常量（键名） ========
    OBS_KEY = "obs"  # 你的 env 若输出 "obs"，这里改成 "obs"
    ACTION_KEY = "action"
    REWARD_KEY = "reward"
    DONE_KEY = "done"

    # === 加载环境 ===
    env = load_rl_env(policy_config['env']["train_seed"], policy_config['training']["train_batch"])

    obs_spec = env.observation_spec
    act_spec = env.action_spec
    # 观测维度：优先用单一张量的 shape，否则当作 CompositeSpec 用 OBS_KEY 取子规格
    if hasattr(obs_spec, "shape") and obs_spec.shape is not None and len(obs_spec.shape) > 0:
        obs_dim = int(obs_spec.shape[-1])
    else:
        obs_dim = int(obs_spec[OBS_KEY].shape[-1])

    # 动作：离散用 n，连续用最后一维
    is_discrete = isinstance(act_spec, DiscreteTensorSpec)
    action_dim = int(act_spec.n) if is_discrete else int(act_spec.shape[-1])
    # 采样/更新规模
    frames_per_batch = 4  # 每次收集的步数
    minibatch_size = 4  # PPO 小批
    ppo_epochs = 1  # 每次更新轮数
    total_frames = 20 # 总步数

    # PPO/GAE
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.0
    critic_coef = 0.5
    lr = 3e-4
    hidden_sizes = [16, 16]


    # # Actor: 对离散/连续分别输出 logits 或 (loc, scale)
    # if is_discrete:
    #     # 离散：输出 logits
    #     actor_net = MLP(
    #         in_features=obs_dim, out_features=action_dim,
    #         depth=len(hidden_sizes), num_cells=hidden_sizes, activation_class=nn.Tanh
    #     )
    #     actor_td = TensorDictModule(
    #         actor_net, in_keys=[OBS_KEY], out_keys=["logits"]
    #     )
    #     actor = ProbabilisticActor(
    #         module=actor_td,
    #         in_keys=["logits"],
    #         spec=env.action_spec,
    #         distribution_class=Categorical,
    #         return_log_prob=True,
    #         default_interaction_mode="random",  # "random" 训练时采样
    #     ).to(device)
    # else:
    # 连续：输出 (loc, scale)
    # out = 2 * action_dim (前 action_dim 为均值，后 action_dim 经过 softplus 变 scale)
    actor_net = MLP(
        in_features=obs_dim, out_features=2 * action_dim,
        depth=len(hidden_sizes), num_cells=hidden_sizes, activation_class=nn.Tanh
    )

    class _SplitLocScale(nn.Module):
        def __init__(self, action_dim):
            super().__init__()
            self.action_dim = action_dim
            self.softplus = nn.Softplus()

        def forward(self, td):
            x = td.get(OBS_KEY)
            out = actor_net(x)
            loc, raw_scale = out[..., :self.action_dim], out[..., self.action_dim:]
            scale = self.softplus(raw_scale) + 1e-5
            td.set("loc", loc)
            td.set("scale", scale)
            return td

    actor_td = TensorDictModule(
        _SplitLocScale(action_dim), in_keys=[OBS_KEY], out_keys=["loc", "scale"]
    )
    actor = ProbabilisticActor(
        module=actor_td,
        in_keys=["loc", "scale"],
        spec=env.action_spec if hasattr(env, "action_spec") else None,
        distribution_class=Normal,
        distribution_kwargs={"validate_args": False},
        return_log_prob=True,
        default_interaction_mode="random",
    ).to(device)

    # Critic: 值函数
    critic_net = MLP(
        in_features=obs_dim, out_features=1,
        depth=len(hidden_sizes), num_cells=hidden_sizes, activation_class=nn.Tanh
    )
    critic = ValueOperator(
        module=critic_net,
        in_keys=[OBS_KEY],
    ).to(device)

    # ======== 数据收集器 ========
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # ======== GAE + PPO 损失 ========
    advantage = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=critic,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        normalize_advantage=True,  # 简洁稳妥
    )

    optim_actor = optim.Adam(actor.parameters(), lr=lr)
    optim_critic = optim.Adam(critic.parameters(), lr=lr)

    # ======== 训练循环（极简） ========
    log_interval = 10
    frame_count = 0
    iter_idx = 0

    for tensordict_data in collector:
        iter_idx += 1
        frame_count += tensordict_data.numel()

        # 计算 advantage / returns
        with torch.no_grad():
            advantage(tensordict_data)

        # 做个简单的随机打乱，切成 minibatch
        td = tensordict_data.shuffle()

        for _ in range(ppo_epochs):
            for i in range(0, td.numel(), minibatch_size):
                sub = td[i: i + minibatch_size]

                loss_vals = loss_module(sub)

                # 先清零
                optim_actor.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

                # 总损失 = policy + critic - entropy（已经在 ClipPPOLoss 内组合）
                loss_vals["loss_objective"].backward()
                # 简单裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)

                optim_actor.step()
                optim_critic.step()

        # 简单日志
        if iter_idx % log_interval == 0:
            # 近似打印回报（平均 reward）
            ep_reward = float(td.get(REWARD_KEY).mean().cpu())
            approx_kl = float(loss_vals.get("approx_kl", torch.tensor(0.)).mean().cpu())
            ent = float(loss_vals.get("entropy", torch.tensor(0.)).mean().cpu())
            vf = float(loss_vals.get("loss_critic", torch.tensor(0.)).mean().cpu())
            print(
                f"[iter {iter_idx:04d}] frames={frame_count} "
                f"reward~{ep_reward:.2f} kl~{approx_kl:.4f} "
                f"ent~{ent:.3f} vloss~{vf:.3f}"
            )

    print("Training done.")



if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 输出为 QGymGPU 项目目录
    policy_file_name = sys.argv[1]
    env_file_name = sys.argv[2]
    print(f'Policy file: {policy_file_name}, Env file: {env_file_name}')
    if not policy_file_name.endswith('.yaml'):
        policy_file_name += '.yaml'
    if not env_file_name.endswith('.yaml'):
        env_file_name += '.yaml'
    policy_file_path = os.path.join(project_root, "RL", 'policy_configs', policy_file_name)
    env_file_path = os.path.join(project_root, 'configs', 'env', env_file_name)
    with open(policy_file_path, 'r') as f:
        policy_config = yaml.safe_load(f)
    with open(env_file_path, 'r', encoding='UTF-8') as f:
        env_config = yaml.safe_load(f)

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

    # env hyperparameters
    device = policy_config['env']['device']
    print(f'device: {device}')

    env_temp = policy_config['env']['env_temp']
    randomize = policy_config['env']['randomize']
    time_f = policy_config['env']['time_f']

    # training hyperparameters
    actors = policy_config['training']['actors']
    normalize_advantage = policy_config['training']['normalize_advantage']
    normalize_value = policy_config['training']['normalize_value']
    normalize_reward = policy_config['training']['normalize_reward']
    rescale_v = policy_config['training']['rescale_v']
    truncation = policy_config['training']['truncation']
    num_epochs = policy_config['training']['num_epochs']
    amp_value = policy_config['training']['amp_value']
    var_scaler = policy_config['training']['var_scaler']
    per_iter_normal_obs = policy_config['training']['per_iter_normal_obs']
    per_iter_normal_value = policy_config['training']['per_iter_normal_value']

    # learning rates:
    lr = policy_config['training']['lr']
    lr_policy = policy_config['training']['lr_policy']
    lr_value = policy_config['training']['lr_value']
    min_lr_policy =policy_config['training']['min_lr_policy']
    min_lr_value = policy_config['training']['min_lr_value']

    episode_steps = policy_config['training']['episode_steps']
    gae_lambda = policy_config['training']['gae_lambda']
    gamma = policy_config['training']['gamma']
    target_kl = policy_config['training']['target_kl']
    vf_coef = policy_config['training']['vf_coef']
    ppo_batch_size = policy_config['training']['batch_size']
    ppo_epochs = policy_config['training']['ppo_epochs']
    clip_range_vf = policy_config['training']['clip_range_vf']
    ent_coef = policy_config['training']['ent_coef']
    bc = policy_config['training']['behavior_cloning']

    # model hyperparameters:
    scale = policy_config['model']['scale']
    # policy hyperparameters
    test_policy = policy_config['policy']['test_policy']
    # total steps
    total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps
    test_T = env_config['test_T']



    train_ppo()

