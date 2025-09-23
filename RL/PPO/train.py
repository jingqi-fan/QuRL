# QGymGPU/RL/train.py
import os
import sys
import time

import numpy as np
import yaml
import torch
import torch.nn as nn
from torchrl.collectors import SyncDataCollector
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE

from RL.env.rl_env import RLViewDiffDES
from RL.utils.count_time import count_time


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

    # === 加载环境 ===
    train_env = load_rl_env(policy_config["train_seed"], policy_config["train_batch"])
    eval_env = load_rl_env(policy_config["test_seed"], policy_config["test_batch"])





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
    env_file_path = os.path.join(project_root, 'configs', 'env', policy_file_name)
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

