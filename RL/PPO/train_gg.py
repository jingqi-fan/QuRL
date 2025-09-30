# QGymGPU/RL/train.py
import json
import os
import sys
import time

import numpy as np
import yaml
import torch

from RL.PPO.trainer_pathwise import PathwiseTrainerTorchRL, PathwiseArgs
from RL.PPO.trainer_wc import PPOTrainerTorchRL, PPOArgs
from RL.PPO.trainer_vanilla import PPOTrainerTorchRL_Vanilla
from RL.env.rl_env import RLViewDiffDES
from RL.utils.count_time import count_time


def load_rl_env(seed, batch):
    # ------- 小工具 ------- #
    def _expand_param_to_1Q(param, Q, device):
        """
        把标量 / list / 1D tensor 统一成形状 [1, Q]、在 device 上的 float32 张量，便于 broadcast。
        """
        if isinstance(param, (list, tuple)):
            t = torch.tensor(param, dtype=torch.float32, device=device)
        elif isinstance(param, torch.Tensor):
            t = param.to(device=device, dtype=torch.float32)
        else:
            t = torch.tensor([param], dtype=torch.float32, device=device).expand(Q)
        if t.ndim == 0:
            t = t.expand(Q)
        return t.view(1, Q)  # [1,Q]

    def _truncnorm_pos(mean_1Q, std_1Q, shape_BQ, device, max_retries=3):
        """
        截断正态到 >0：先采样 Normal(mean, std)，若 <=0 则重采几次，最后 clamp 到 epsilon。
        mean_1Q, std_1Q: [1,Q]
        返回 shape_BQ: [B,Q]
        """
        B, Q = shape_BQ
        # 逐队列不同 mean/std 的采样：标准正态 * std + mean
        x = torch.randn((B, Q), device=device) * std_1Q + mean_1Q
        for _ in range(max_retries):
            mask = x <= 0
            if not mask.any():
                break
            # 只在负的位置重采
            rs = torch.randn(mask.sum().item(), device=device)
            # 需要匹配对应列的 std/mean
            # 将 rs reshape 为 [k,1]，按列 broadcast 到正确的 std/mean 上
            # 简化：再整块重采（便于实现；对统计影响极小）
            x = torch.where(mask, (torch.randn((B, Q), device=device) * std_1Q + mean_1Q), x)
        return x.clamp_min(1e-6).float()

    # ------- G/G/N 到达 & 服务 ------- #
    def draw_inter_arrivals(env, time: torch.Tensor) -> torch.Tensor:
        """
        G/G/N - 到达间隔采样（truncnorm）
        从 env_config['arrival_dist'] 读取:
          type: "truncnorm"
          mean: 标量 / 长度Q 的 list/array
          std : 标量 / 长度Q 的 list/array
        返回 [B,Q] 正数，到达间隔（多久后下一次到达）。
        """
        device = env.device
        B, Q = time.shape[0], env.Q

        spec = env_config.get('arrival_dist', None)
        if spec is None:
            raise ValueError("env_config['arrival_dist'] 未配置。")

        dist_type = str(spec.get('type', 'truncnorm')).lower()
        if dist_type != 'truncnorm':
            raise ValueError(f"当前示例仅实现 truncnorm，到达 dist_type={dist_type}")

        mean_1Q = _expand_param_to_1Q(spec.get('mean', 1.0), Q, device)
        std_1Q = _expand_param_to_1Q(spec.get('std', 0.5), Q, device)
        return _truncnorm_pos(mean_1Q, std_1Q, (B, Q), device)

    def draw_service(env, time: torch.Tensor) -> torch.Tensor:
        """
        G/G/N - 服务“工作量”采样（truncnorm）
        从 env_config['service_dist'] 读取:
          type: "truncnorm"
          mean: 标量 / 长度Q 的 list/array
          std : 标量 / 长度Q 的 list/array
        返回 [B,Q] 正数，“工作量/服务需求”W（不含 mu）。
        注意：Env 内部会用 job_rates（含 mu 和名额）来消费工作量：
              完成时间 = W / job_rates
        """
        device = env.device
        B, Q = time.shape[0], env.Q

        spec = env_config.get('service_dist', None)
        if spec is None:
            raise ValueError("env_config['service_dist'] 未配置。")

        dist_type = str(spec.get('type', 'truncnorm')).lower()
        if dist_type != 'truncnorm':
            raise ValueError(f"当前示例仅实现 truncnorm，服务 dist_type={dist_type}")

        mean_1Q = _expand_param_to_1Q(spec.get('mean', 1.0), Q, device)
        std_1Q = _expand_param_to_1Q(spec.get('std', 0.5), Q, device)
        return _truncnorm_pos(mean_1Q, std_1Q, (B, Q), device)

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

    act_spec = env.action_spec
    obs_spec = env.observation_spec
    obs_dim = obs_spec["obs"].shape[-1]

    return env, act_spec, obs_dim


def train_ppo():

    train_env, train_act_spec, train_obs_dim = load_rl_env(train_seed, train_batch)
    eval_env, eval_act_spec, eval_obs_dim = load_rl_env(test_seed, test_batch)

    # 2) 组装 Trainer 的参数（对齐你原来 SB3 的超参语义）
    ppo_args = PPOArgs(
        device=device,
        obs_dim=int(train_obs_dim),
        S=int(orig_s),
        Q=int(orig_q),
        hidden=int(scale * int(np.sqrt(orig_q * orig_s))),  # 与原来 scale * sqrt(L*J) 对齐
        # rollout
        episode_steps=int(episode_steps),
        train_batch=int(train_batch),
        test_batch=int(test_batch),
        # PPO
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_eps=0.2,  # 原来 SB3 里固定 0.2
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        max_grad_norm=1.0,  # 与原来一致
        ppo_epochs=int(ppo_epochs),
        minibatch_size=int(ppo_batch_size),  # 原 config['training']['batch_size']
        target_kl=(None if target_kl in [None, "None"] else float(target_kl)),
        # LR（分别对 policy / value）
        lr_policy=float(lr_policy),
        lr_value=float(lr_value),
        min_lr_policy=float(min_lr_policy),
        min_lr_value=float(min_lr_value),
        warmup=0.03,  # 与之前实现相同的 warmup 比例
        # 训练轮次
        total_epochs=int(num_epochs),
        # 其他
        normalize_advantage=bool(normalize_advantage),
        rescale_value=bool(rescale_v),
        behavior_cloning=bool(bc),
        bc_samples=1000,  # 与原 parallel_eval.BCD 一致
        bc_lr=3e-4,  # 与原 BC 优化器一致
        # 评估
        eval_every=1,  # 每个 epoch 评估一次（原来每个 episode_steps 调一次）
        eval_T=int(test_T),
        randomize=randomize,
        time_f=time_f
    )

    # 2) 组装 Trainer 的参数（对齐你原来 SB3 的超参语义）
    pathwise_args = PathwiseArgs(
        device=device,
        obs_dim=int(train_obs_dim),
        S=int(orig_s),
        Q=int(orig_q),
        hidden=int(scale * int(np.sqrt(orig_q * orig_s))),  # 与原来 scale * sqrt(L*J) 对齐
        # rollout
        episode_steps=int(episode_steps),
        train_batch=int(train_batch),
        test_batch=int(test_batch),
        # PPO
        gamma=float(gamma),
        max_grad_norm=1.0,  # 与原来一致
        # LR（分别对 policy / value）
        lr_policy=float(lr_policy),
        lr_value=float(lr_value),
        min_lr_policy=float(min_lr_policy),
        min_lr_value=float(min_lr_value),
        warmup=0.03,  # 与之前实现相同的 warmup 比例
        # 训练轮次
        total_epochs=int(num_epochs),
        rescale_value=bool(rescale_v),
        behavior_cloning=bool(bc),
        bc_samples=1000,  # 与原 parallel_eval.BCD 一致
        bc_lr=3e-4,  # 与原 BC 优化器一致
        # 评估
        eval_every=1,  # 每个 epoch 评估一次（原来每个 episode_steps 调一次）
        eval_T=int(test_T),
        randomize=randomize,
        tau=env_temp,
        cost_is_negative_reward=False
    )
    ct = count_time(time.time())
    if policy_file_name == 'WC.yaml' or policy_file_name == 'WC':
        # 运行 WC 的
        trainer = PPOTrainerTorchRL(
            train_env=train_env,
            eval_env=eval_env,
            args=ppo_args,
            network_mask=network if network.dim() == 2 else network[0],  # [S,Q] or按需处理
            ct=ct
        )
    # elif policy_file_name == 'small.yaml':
    #     # 运行 WC 的
    #     trainer = PPOTrainerTorchRL(
    #         train_env=train_env,
    #         eval_env=eval_env,
    #         args=ppo_args,
    #         network_mask=network if network.dim() == 2 else network[0],  # [S,Q] or按需处理
    #         ct=ct
    #     )
    elif policy_file_name == 'pathwise.yaml' or policy_file_name == 'pathwise':
        # # 运行 pathwise 的
        trainer = PathwiseTrainerTorchRL(
            train_env=train_env,
            eval_env=eval_env,
            args=pathwise_args,
            network_mask=network if network.dim() == 2 else network[0],  # [S,Q] or按需处理
            ct=ct
        )
    else:
        # # 运行 vanilla 和 vanilla bc 的
        trainer = PPOTrainerTorchRL_Vanilla(
            train_env=train_env,
            eval_env=eval_env,
            args=ppo_args,
            # network_mask=network if network.dim() == 2 else network[0],  # [S,Q] or按需处理
            ct=ct
        )


    # 是否进行行为克隆预训练：由 config 控制（与原流程一致）
    if bc:
        trainer.pre_train() # 可选：如果 policy_config['training']['behavior_cloning'] 为 True
    trainer.learn()


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
    with open(policy_file_path, 'r', encoding='UTF-8') as f:
        policy_config = yaml.safe_load(f)
    with open(env_file_path, 'r', encoding='UTF-8') as f:
        env_config = yaml.safe_load(f)

    env_name = env_config["name"]
    policy_name = policy_config["name"]
    model_name = policy_config['model']["policy_name"]

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

    # if lam_params["val"] is None:
    #     lam_r_path = os.path.join(
    #         project_root, "configs", "env_data", env_type, f"{env_type}_lam.npy"
    #     )
    #     lam_r = np.load(lam_r_path)
    # else:
    #     lam_r = lam_params["val"]
    #
    #
    # def lam(t):
    #     if lam_type == "constant":
    #         lam = lam_r
    #     elif lam_type == "step":
    #         is_surge = 1 * (t.data.cpu().numpy() <= lam_params["t_step"])
    #         lam = is_surge * np.array(lam_params["val1"]) + (1 - is_surge) * np.array(
    #             lam_params["val2"]
    #         )
    #     else:
    #         return "Nonvalid arrival rate"
    #     return lam

    # env hyperparameters
    device = policy_config['env']['device']
    print(f'device: {device}')

    env_temp = policy_config['env']['env_temp']
    randomize = policy_config['env']['randomize']
    time_f = policy_config['env']['time_f']

    # training hyperparameters
    # actors = policy_config['training']['actors']
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
    train_seed = policy_config['env']['train_seed']
    test_seed = policy_config['env']['test_seed']
    train_batch = policy_config['training']['train_batch']
    test_batch = policy_config['training']['test_batch']

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
    # total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps
    test_T = env_config['test_T']

    # ===== 新增：输出重定向到 results/rl/ =====
    timestamp = time.strftime("%m%d_%H%M")
    results_dir = os.path.join(project_root, "results", "rl")
    os.makedirs(results_dir, exist_ok=True)

    log_file = os.path.join(results_dir, f"{timestamp}_{policy_file_name}_{env_file_name}.log")
    sys.stdout = open(log_file, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout  # 错误也写入同一个文件

    print(f"[INFO] Logging to {log_file}")

    train_ppo()

