import sys
from policies.cmu import *
from utils.switchplot import *


import yaml
import argparse
from utils.switchplot import *
from main.trainer_multi_env import Trainer

import torch
import torch.optim as optim

import json

import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', type=str)
parser.add_argument('-m', type=str)
parser.add_argument('-experiment_name', type=str)




args = parser.parse_args()

with open(f'configs/env/{args.e}', 'r', encoding='utf-8') as f:
    env_config = yaml.safe_load(f)

with open(f'configs/model/{args.m}', 'r', encoding='utf-8') as f:
    model_config = yaml.safe_load(f)


experiment_name = args.experiment_name


name = env_config['name']
if env_config['network'] is None:
    if env_config['lam_type'] == 'hyper':
        env_config['network'] = np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_network.npy')
    else:
        env_config['network'] = np.load(f'configs/env_data/{name}/{name}_network.npy')
env_config['network'] = torch.tensor(env_config['network']).float()


if env_config['mu'] is None:
    if env_config['lam_type'] == 'hyper':
        env_config['mu'] = np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_mu.npy')
    else:
        env_config['mu'] = np.load(f'configs/env_data/{name}/{name}_mu.npy')
env_config['mu'] = torch.tensor(env_config['mu']).float()

orig_s, orig_q = env_config['network'].size()

policy = MaxWeightCMuPolicy()

env_config['network'] = env_config['network'].repeat_interleave(1, dim = 0)
env_config['mu'] = env_config['mu'].repeat_interleave(1, dim = 0)
if 'server_pool_size' in env_config.keys():
    env_config['server_pool_size'] = torch.tensor(env_config['server_pool_size']).to(model_config['env']['device'])
else:
    env_config['server_pool_size'] = torch.ones(orig_s).to(model_config['env']['device'])



# lam_r 仍按你现在的方式读取，得到 numpy 或 list 都可
lam_type   = env_config['lam_type']
lam_params = env_config['lam_params']
# 把基础到达率转成 torch [Q]，后面广播到 [B,Q]
if lam_params['val'] is None:
    if lam_type == 'hyper':
        lam_r = np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_lam.npy')
    else:
        lam_r = np.load(f'configs/env_data/{name}/{name}_lam.npy')
else:
    lam_r = lam_params['val']
lam_r_base = torch.as_tensor(lam_r, dtype=torch.float32)


def lam_torch(env, t: torch.Tensor) -> torch.Tensor:
    """
    t: [B,1], 返回 [B,Q] 的到达率（>0），在 env.device 上
    """
    device = env.device
    B = t.shape[0]
    Q = env.Q
    lam_r = lam_r_base.to(device).view(1, Q).expand(B, Q)

    if lam_type == 'constant':
        lam = lam_r
    elif lam_type == 'step':
        is_surge = (t.to(device) <= lam_params['t_step']).to(torch.float32)  # [B,1]
        val1 = torch.as_tensor(lam_params['val1'], dtype=torch.float32, device=device).view(1, Q)
        val2 = torch.as_tensor(lam_params['val2'], dtype=torch.float32, device=device).view(1, Q)
        lam = is_surge * val1 + (1.0 - is_surge) * val2                      # [B,Q]
    elif lam_type == 'hyper':
        scale = float(lam_params['scale'])
        # 每个 batch 单元随机切换一种缩放
        coins = torch.bernoulli(0.5 * torch.ones(B, 1, device=device))       # [B,1] in {0,1}
        lam_hi = lam_r / (1.0 + scale)
        lam_lo = lam_r / (1.0 - scale)
        lam = coins * lam_hi + (1.0 - coins) * lam_lo                         # [B,Q]
    else:
        raise ValueError(f"Invalid lam_type: {lam_type}")

    return lam.clamp_min(1e-6)

    

if env_config['queue_event_options'] == 'custom':
    if env_config['lam_type'] == 'hyper':
        env_config['queue_event_options'] = torch.tensor(np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_delta.npy'))
    else:
        env_config['queue_event_options'] = torch.tensor(np.load(f'configs/env_data/{name}/{name}_delta.npy'))
if type(env_config['queue_event_options']) == list:
    env_config['queue_event_options'] = torch.tensor(env_config['queue_event_options']).float()

if env_config['queue_event_options2'] == 'custom':
    if env_config['lam_type'] == 'hyper':
        env_config['queue_event_options2'] = torch.tensor(np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_delta2.npy'))
    else:
        env_config['queue_event_options2'] = torch.tensor(np.load(f'configs/env_data/{name}/{name}_delta2.npy'))
if type(env_config['queue_event_options2']) == list:
    env_config['queue_event_options2'] = torch.tensor(env_config['queue_event_options2']).float()


def draw_service(env, time: torch.Tensor) -> torch.Tensor:
    """
    返回 [B,Q] 的正值张量（float32，env.device）
    若 env_config['service_type']=='hyper'，用两种均值的指数混合；否则默认 Exp(mean=1)
    这个生成的是工作量，在env中除以（action乘mu服务速率）
    """
    device = env.device
    B, Q = time.shape[0], env.Q
    U = torch.rand((B, Q), device=device).clamp_min(1e-6)

    if env_config.get('service_type') == 'hyper':
        scale = 0.8  # 你原来的常量
        coins = torch.bernoulli(0.5 * torch.ones(B, Q, device=device))
        mean_a, mean_b = 1.0 + scale, 1.0 - scale
        a = -torch.log(U) * mean_a
        b = -torch.log(U) * mean_b
        return (coins * a + (1 - coins) * b).to(torch.float32)

    return (-torch.log(U)).to(torch.float32)  # Exp(mean=1)


def draw_inter_arrivals(env, time: torch.Tensor) -> torch.Tensor:
    """
    返回 [B,Q] 的下一次到达间隔(多久之后下一次到达发生)：Exp(rate=λ) => Exp(1)/λ
    """
    device = env.device
    B, Q = time.shape[0], env.Q
    U = torch.rand((B, Q), device=device).clamp_min(1e-6)
    exp1 = -torch.log(U)                        # Exp(rate=1)
    lam  = lam_torch(env, time)                 # [B,Q]
    return (exp1 / lam).to(torch.float32)



optimizer = None
trainer = Trainer(model_config, env_config, policy, optimizer, experiment_name = experiment_name, draw_service = draw_service, draw_inter_arrivals = draw_inter_arrivals)


trainer.test_epoch(0)

with open(f'{trainer.loss_dir}/loss.json', 'w') as f:
    json.dump(trainer.test_loss, f)





