# train_torchrl.py

import sys
import os
import time
import json
import yaml
import numpy as np
import torch
from torch import nn

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RL_root = os.path.join(project_root, 'RL')
sys.path.append(project_root)
sys.path.append(RL_root)

from RL.utils.rl_env import load_rl_p_env
from RL.policies.WC_policy import WC_Policy
from RL.policies.vanilla_policy import Vanilla_Policy
from RL.PPO.trainer import TorchRLPPOTrainer
from RL.utils.eval import ParallelEvalTorchRL
from RL.utils.count_time import count_time

# 关键：TorchRL 0.6 的 TensorDictModule 在 tensordict.nn
from tensordict.nn import TensorDictModule


def main():
    ct = count_time(time.time())

    config_file_name = sys.argv[1]
    env_config_name = sys.argv[2]
    if not config_file_name.endswith(".yaml"):
        config_file_name += ".yaml"

    # === 读取配置 ===
    config_file_path = os.path.join(RL_root, "policy_configs", config_file_name)
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"env_config: {env_config_name}")
    env_config_path = os.path.join(project_root, "configs", "env", f"{env_config_name}.yaml")
    with open(env_config_path, "r", encoding="UTF-8") as f:
        env_config = yaml.safe_load(f)

    name = env_config["name"]
    env_type = env_config.get("env_type", name)
    print(f"name: {name}, env_type: {env_type}")

    # === 环境数据 ===
    if env_config["network"] is None:
        network_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_network.npy")
        network = np.load(network_path)
    else:
        network = env_config["network"]

    if env_config["mu"] is None:
        mu_path = os.path.join(project_root, "configs", "env_data", env_type, f"{env_type}_mu.npy")
        mu = np.load(mu_path)
    else:
        mu = env_config["mu"]

    network = torch.tensor(network, dtype=torch.float32)
    mu = torch.tensor(mu, dtype=torch.float32)
    orig_s, orig_q = network.size()

    # server pools
    num_pool = env_config["num_pool"]
    network = network.repeat_interleave(num_pool, dim=0)
    mu = mu.repeat_interleave(num_pool, dim=0)

    init_test_queues = torch.tensor([env_config["init_queues"]], dtype=torch.float32)

    # === 超参 ===
    device = torch.device(config["env"]["device"])
    test_seed = config["env"]["test_seed"]
    train_seed = config["env"]["train_seed"]
    env_temp = config["env"]["env_temp"]
    randomize = config["env"]["randomize"]
    time_f = config["env"]["time_f"]
    policy_name = config["model"]["policy_name"]

    print(f"device={device}, test_seed={test_seed}, train_seed={train_seed}, policy_name={policy_name}")

    actors = config["training"]["actors"]
    num_epochs = config["training"]["num_epochs"]
    episode_steps = config["training"]["episode_steps"]
    gae_lambda = config["training"]["gae_lambda"]
    gamma = config["training"]["gamma"]
    target_kl = config["training"]["target_kl"]
    vf_coef = config["training"]["vf_coef"]
    ppo_epochs = config["training"]["ppo_epochs"]
    ent_coef = config["training"]["ent_coef"]
    bc = config["training"]["behavior_cloning"]

    lr_policy = config["training"]["lr_policy"]
    lr_value = config["training"]["lr_value"]
    min_lr_policy = config["training"]["min_lr_policy"]
    min_lr_value = config["training"]["min_lr_value"]

    scale = config["model"]["scale"]
    test_policy = config["policy"]["test_policy"]
    test_batch = config["training"]["test_batch"]
    test_T = env_config["test_T"]

    total_steps = num_epochs * episode_steps * actors
    eval_freq = episode_steps

    print(f"total_steps={total_steps}, test_T={test_T}, eval_freq={eval_freq}")

    # === Env builders ===
    def make_env(seed):
        return load_rl_p_env(
            env_config=env_config,
            temp=env_temp,
            batch=1,
            seed=seed,
            policy_name=policy_name,  # 这个参数在你的 loader 里现在无所谓了
            device=device,
        )

    dq_raw = make_env(train_seed)

    # === 策略网络结构 ===
    L = orig_q
    J = orig_s
    gmLJ = int(np.sqrt(L * J))
    pi_arch = [scale * L, scale * gmLJ, scale * J]
    vi_arch = [scale * L, scale * gmLJ, scale * J]
    print(f"pi_arch={pi_arch}, vi_arch={vi_arch}")

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        network=network,            # (s,q)
        time_f=time_f,
        randomize=randomize,
        scale=scale,
        rescale_v=config["training"]["rescale_v"],
        alpha=0,                    # 你的 WC_Policy 内部会扩展成 (q,)
        D=dq_raw.queue_event_options,  # (2q,q)
        mu=mu,                      # (s,q)
        net_arch=dict(pi=pi_arch, vf=vi_arch),
    )

    # === 基类策略（保持你的实现）===
    if policy_name == "WC":
        BasePolicyClass = WC_Policy
    elif policy_name == "vanilla":
        BasePolicyClass = Vanilla_Policy
    else:
        raise ValueError(f"Unknown policy {policy_name}")

    base_policy = BasePolicyClass(**policy_kwargs).to(device)

    # === 用 TensorDictModule 适配成 TorchRL 可用的 actor / value ===
    # 注意：Env 的 observation keys 是 {"queues","time"}，我们只用 queues
    class PolicyTDAdapter(nn.Module):
        def __init__(self, pol):
            super().__init__()
            self.pol = pol
        def forward(self, queues: torch.Tensor):
            # 返回 action(one-hot, B,s,q) & sample_log_prob(B,)
            action, _, log_prob = self.pol.forward(queues, deterministic=False)
            return action, log_prob

    actor_td = TensorDictModule(
        module=PolicyTDAdapter(base_policy),
        in_keys=["queues"],                      # 只喂 queues
        out_keys=["action", "sample_log_prob"],  # ClipPPOLoss 期望的 key
    ).to(device)

    class ValueHead(nn.Module):
        def __init__(self, pol):
            super().__init__()
            self.pol = pol
        def forward(self, queues: torch.Tensor):
            v = self.pol.predict_values(queues)
            return v

    value_td = TensorDictModule(
        module=ValueHead(base_policy),
        in_keys=["queues"],
        out_keys=["state_value"],
    ).to(device)

    # === Trainer（传入 TensorDict 版本的 actor/value）===
    trainer = TorchRLPPOTrainer(
        actor=actor_td,
        value=value_td,
        env=make_env(train_seed),
        lr_policy=lr_policy,
        lr_value=lr_value,
        min_lr_policy=min_lr_policy,
        min_lr_value=min_lr_value,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ppo_epochs=ppo_epochs,
        num_epochs=num_epochs,
        episode_steps=episode_steps,
        actors=actors,
        device=device,
        target_kl=target_kl,
        ct=ct,
    )

    # === Eval（这里仍然用你原来的基类策略，避免改 ParallelEvalTorchRL）===
    evaler = ParallelEvalTorchRL(
        actor=base_policy,  # 注意这里传基类策略（它有 predict/get_prob_act）
        make_env_fn=lambda seed: make_env(seed),
        eval_freq=eval_freq,
        eval_t=test_T,
        test_seed=test_seed,
        test_batch=test_batch,
        device=device,
        bc=bc,
        per_iter_normal_obs=config["training"]["per_iter_normal_obs"],
        verbose=1,
    )

    evaler.eval()  # pre-train eval

    # === Train ===
    _, _ = trainer.learn()

    # === Final Eval ===
    q_mean, q_std, avg_reward, overall_ql_mean, overall_ql_se = evaler.eval()

    results = {
        "q_mean": float(q_mean.item()),
        "q_std": float(q_std.item()),
        "avg_reward": avg_reward,
        "overall_ql_mean": overall_ql_mean,
        "overall_ql_se": overall_ql_se,
    }
    with open("test_cost_list.json", "w") as f:
        json.dump(results, f)

    print("Training finished. Results saved.")


if __name__ == "__main__":
    main()
