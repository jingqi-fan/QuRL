import sys

sys.path.append('/user/hc3295/queue-learning')

from policies.fluid import *
from utils.switchplot import *

import yaml
import argparse
from utils.switchplot import *
from main.trainer import Trainer

import torch
import torch.optim as optim

import json

import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-e', type=str)
parser.add_argument('-m', type=str)
parser.add_argument('-experiment_name', type=str)

import torch.nn.functional as F

args = parser.parse_args()

with open(f'configs/env/{args.e}', 'r') as f:
    env_config = yaml.safe_load(f)

with open(f'configs/model/{args.m}', 'r') as f:
    model_config = yaml.safe_load(f)

experiment_name = args.experiment_name

name = env_config['name']
if env_config['network'] is None:
    if env_config['lam_type'] == 'hyper':
        env_config['network'] = np.load(
            f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_network.npy')
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

if env_config['queue_event_options'] == 'custom':
    if env_config['lam_type'] == 'hyper':
        env_config['queue_event_options'] = torch.tensor(
            np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_delta.npy'))
    else:
        env_config['queue_event_options'] = torch.tensor(np.load(f'configs/env_data/{name}/{name}_delta.npy'))
default_event_mat = torch.cat((F.one_hot(torch.arange(0, orig_q)), -F.one_hot(torch.arange(0, orig_q)))).float().to(
    model_config['env']['device'])
if env_config['queue_event_options'] is None:
    env_config['queue_event_options'] = default_event_mat
if type(env_config['queue_event_options']) == list:
    env_config['queue_event_options'] = torch.tensor(env_config['queue_event_options']).float()
policy = FluidPolicy(queue_event_options=env_config['queue_event_options'])

env_config['network'] = env_config['network'].repeat_interleave(1, dim=0)
env_config['mu'] = env_config['mu'].repeat_interleave(1, dim=0)
if 'server_pool_size' in env_config.keys():
    env_config['server_pool_size'] = torch.tensor(env_config['server_pool_size']).to(model_config['env']['device'])
else:
    env_config['server_pool_size'] = torch.ones(orig_s).to(model_config['env']['device'])

if env_config['queue_event_options2'] == 'custom':
    if env_config['lam_type'] == 'hyper':
        env_config['queue_event_options2'] = torch.tensor(np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_delta2.npy'))
    else:
        env_config['queue_event_options2'] = torch.tensor(np.load(f'configs/env_data/{name}/{name}_delta2.npy'))
if type(env_config['queue_event_options2']) == list:
    env_config['queue_event_options2'] = torch.tensor(env_config['queue_event_options2']).float()

if env_config['queue_event_options'] == 'custom':
    if env_config['lam_type'] == 'hyper':
        env_config['queue_event_options'] = torch.tensor(
            np.load(f'configs/env_data/{env_config["env_type"]}/{env_config["env_type"]}_delta.npy'))
    else:
        env_config['queue_event_options'] = torch.tensor(np.load(f'configs/env_data/{name}/{name}_delta.npy'))
if type(env_config['queue_event_options']) == list:
    env_config['queue_event_options'] = torch.tensor(env_config['queue_event_options']).float()


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
    std_1Q  = _expand_param_to_1Q(spec.get('std',  0.5), Q, device)
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
    std_1Q  = _expand_param_to_1Q(spec.get('std',  0.5), Q, device)
    return _truncnorm_pos(mean_1Q, std_1Q, (B, Q), device)


optimizer = None
trainer = Trainer(model_config, env_config, policy, optimizer, experiment_name=experiment_name,
                  draw_service=draw_service, draw_inter_arrivals=draw_inter_arrivals)

trainer.test_epoch(0)

with open(f'{trainer.loss_dir}/loss.json', 'w') as f:
    json.dump(trainer.test_loss, f)





