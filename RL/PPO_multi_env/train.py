# QGymGPU/RL/train.py
import json
import os
import sys
import time
from pprint import pprint

import numpy as np
import yaml
import torch

from RL.PPO_multi_env.trainer_pathwise import PathwiseTrainerTorchRL, PathwiseArgs
# from RL.PPO_multi_env.trainer_wc import PPOTrainerTorchRL, PPOArgs
from RL.PPO_multi_env.trainer_vanilla import PPOTrainerTorchRL_Vanilla
from RL.env.rl_env import RLViewDiffDES
from RL.utils.count_time import count_time

from RL.PPO_multi_env.trainer_wc2 import PPOTrainerTorchRL, PPOArgs

def load_rl_env(seed, batch):
    # ---- 抽样器（回到 torch 张量） ----
    # def draw_service(env, t: torch.Tensor) -> torch.Tensor:
    #     B = t.shape[0]
    #     rate = torch.ones(B, orig_q, device=env.device)
    #     return torch.distributions.Exponential(rate=rate).sample()

    def draw_service(env, t: torch.Tensor) -> torch.Tensor:
        B = t.shape[0]
        rate = torch.full((B, orig_q), 1.0, device=env.device)  # rate = 0.5 → 更大数值
        return torch.distributions.Exponential(rate=rate).sample()

    def draw_inter_arrivals(env, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            t_now = float(t[0, 0].item()) if t.numel() > 0 else 0.0
            lam_vec = np.array(lam(t_now), dtype=np.float32)
            lam_vec = np.maximum(lam_vec, 1e-8)
        lam_t = torch.as_tensor(lam_vec, device=env.device).view(1, orig_q).expand(t.size(0), orig_q)
        return torch.distributions.Exponential(rate=lam_t).sample()

    # ---- 构造环境 ----
    # env = RLViewDiffDES(
    #     network=network,
    #     mu=mu,
    #     h=h,
    #     draw_service=draw_service,
    #     draw_inter_arrivals=draw_inter_arrivals,
    #     temp=env_temp,
    #     device=device,
    #     seed=seed,
    #     default_B=batch,
    #     queue_event_options=queue_event_options,
    #     time_f=time_f,
    # ).to(device)
    env = (RLViewDiffDES(
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
        queue_event_options2=queue_event_options2,
        time_f=time_f,
        reentrant=reentrant
    ))
    # env.reset()
    # print(td)
    # env =env.to("cuda:0")

    act_spec = env.action_spec
    obs_spec = env.observation_spec
    obs_dim = obs_spec["obs"].shape[-1]

    return env, act_spec, obs_dim


def train_ppo():
    # ---- multi-env: 外部创建 N 个 env，每个 env 内 batch=1（不多进程） ----
    train_envs = []
    for i in range(int(train_batch)):
        env_i, train_act_spec, train_obs_dim = load_rl_env(train_seed + i, batch=1)
        train_envs.append(env_i)

    eval_envs = []
    for i in range(int(test_batch)):
        env_i, eval_act_spec, eval_obs_dim = load_rl_env(test_seed + i, batch=1)
        eval_envs.append(env_i)

    # 后面 args 不用改：train_batch/test_batch 仍表示“并行条数”
    ppo_args = PPOArgs(
        device=device,
        obs_dim=int(train_obs_dim),
        S=int(orig_s),
        Q=int(orig_q),
        hidden=int(scale * int(np.sqrt(orig_q * orig_s))),
        episode_steps=int(episode_steps),
        train_batch=int(train_batch),   # 现在表示 env 数量
        test_batch=int(test_batch),     # 现在表示 env 数量
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
        clip_eps=0.2,
        ent_coef=float(ent_coef),
        vf_coef=float(vf_coef),
        max_grad_norm=1.0,
        ppo_epochs=int(ppo_epochs),
        minibatch_size=int(ppo_batch_size),
        target_kl=(None if target_kl in [None, "None"] else float(target_kl)),
        lr_policy=float(lr_policy),
        lr_value=float(lr_value),
        min_lr_policy=float(min_lr_policy),
        min_lr_value=float(min_lr_value),
        warmup=0.03,
        total_epochs=int(num_epochs),
        normalize_advantage=bool(normalize_advantage),
        rescale_value=bool(rescale_v),
        behavior_cloning=bool(bc),
        bc_samples=1000,
        bc_lr=3e-4,
        eval_every=1,
        eval_T=int(test_T),
        randomize=randomize,
        time_f=time_f
    )

    ct = count_time(time.time())

    # vanilla trainer：把 env list 传进去
    trainer = PPOTrainerTorchRL_Vanilla(
        train_env=train_envs,
        eval_env=eval_envs,
        args=ppo_args,
        ct=ct
    )

    if bc:
        trainer.pre_train()
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

    queue_event_options2 = env_config["queue_event_options2"]
    if queue_event_options2 is not None:
        if queue_event_options2 == "custom":
            queue_event_options_path = os.path.join(
                project_root, "configs", "env_data", env_type, f"{env_type}_delta2.npy"
            )
            queue_event_options2 = torch.tensor(np.load(queue_event_options_path))
        else:
            queue_event_options2 = torch.tensor(queue_event_options2)


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
    reentrant = env_config.get('reentrant', 0)

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
    # lr = policy_config['training']['lr']
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

    log_file = os.path.join(results_dir, f"{timestamp}_{policy_file_name}_{env_file_name}_lrpolicy_{lr_policy}.log")
    sys.stdout = open(log_file, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout  # 错误也写入同一个文件

    print(f"[INFO] Logging to {log_file}")


    print("\n========== Policy Config ==========")
    pprint(policy_config)
    print("===================================")


    train_ppo()

