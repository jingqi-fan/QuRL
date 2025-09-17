import os, sys, yaml, time
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv, Compose, StepCounter, RewardScaling
from torchrl.modules import ProbabilisticActor, TanhNormal, MLP
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torch.optim import Adam

# ==== 你自己的 env loader ====
from RL.utils.load_rl_env import load_trl_env   # <- 你要把上次写的 load_trl_env 放到这里

def train_ppo(config_file, env_config_file):
    # === 加载参数 ===
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    with open(env_config_file, "r") as f:
        env_config = yaml.safe_load(f)

    device = torch.device(config["env"]["device"])
    seed = config["env"]["train_seed"]
    torch.manual_seed(seed)

    # === 构造环境 ===
    env = load_trl_env(
        env_config,
        temp=config["env"]["env_temp"],
        batch=1,
        seed=seed,
        device=device,
        reward_scale=config["env"]["reward_scale"],
        time_f=config["env"]["time_f"],
    )
    # PPO 常见的 env transform：reward scaling、step 限制
    env = TransformedEnv(env, Compose(
        RewardScaling(loc=0.0, scale=1.0),
        StepCounter(max_steps=200),  # 你可以改成 episode_steps
    ))

    # === 构造 policy & value 网络 ===
    # obs_dim = env.observation_spec.shape[0]

    obs_spec = env.observation_spec
    if hasattr(obs_spec, "keys"):
        obs_dim = obs_spec["queues"].shape[-1]
    else:
        obs_dim = obs_spec.shape[-1]

    act_dim = env.action_spec.shape[-1]

    policy_net = MLP(in_features=obs_dim, out_features=2 * act_dim, num_cells=[64, 64])
    value_net = MLP(in_features=obs_dim, out_features=1, num_cells=[64, 64])

    # 包装成 TorchRL 模块
    policy_module = TensorDictModule(
        policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        distribution_class=TanhNormal,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        return_log_prob=True,
    ).to(device)

    value = TensorDictModule(
        value_net, in_keys=["observation"], out_keys=["state_value"]
    ).to(device)

    # # === PPO loss ===
    # advantage_module = GAE(
    #     gamma=config["training"].get("gamma", 0.99),
    #     lmbda=config["training"].get("gae_lambda", 0.95),
    #     value_network=value,
    # )
    # loss_module = ClipPPOLoss(
    #     actor=policy,
    #     critic=value,
    #     advantage_key="advantage",
    #     value_target_key="value_target",
    #     clip_epsilon=0.2,
    # )

    # GAE 构造不变
    advantage_module = GAE(
        gamma=float(config["training"].get("gamma", 0.99)),
        lmbda=float(config["training"].get("gae_lambda", 0.95)),
        value_network=value,  # 你的 value TDModule，out_keys 里要包含 "state_value"
    )

    # 只有当你的键名与默认不一致时才需要 set_keys
    # 默认：advantage="advantage", value_target="value_target", value="state_value",
    #      reward="reward", done="done", terminated="terminated"
    advantage_module.set_keys(
        advantage="advantage",
        value_target="value_target",
        value="state_value",  # 必须与 value 网络的 out_keys 匹配
        # reward="reward",
        # done="done",
        # terminated="terminated",
    )

    # --- PPO Loss: 先实例化，再 set_keys ---
    loss_module = ClipPPOLoss(
        actor=policy,  # 你的 ProbabilisticActor（return_log_prob=True）
        critic=value,  # 你的 value TDModule
        clip_epsilon=float(config["training"].get("clip_epsilon", 0.2)),
        entropy_bonus=True,
        entropy_coef=float(config["training"].get("ent_coef", 0.01)),
        critic_coef=float(config["training"].get("vf_coef", 0.5)),
        normalize_advantage=bool(config["training"].get("normalize_advantage", True)),
        # value 裁剪：False / True / 浮点
        clip_value=(float(config["training"]["clip_range_vf"])
                    if "clip_range_vf" in config["training"] and config["training"]["clip_range_vf"]
                    else False),
    )

    # 让 PPO loss 知道各字段的键名
    loss_module.set_keys(
        action="action",
        sample_log_prob="sample_log_prob",
        value="state_value",
        value_target="value_target",
        advantage="advantage",
        # 如 env 的键名不同，也可以一起对齐：
        # reward="reward",
        # done="done",
        # terminated="terminated",
    )

    # === Replay buffer ===
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=10000, device=device)
    )

    # === Collector ===
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=200,
        total_frames=2000,
        device=device,
    )

    optim = Adam(loss_module.parameters(), lr=3e-4)

    # === 训练循环 ===
    for i, data in enumerate(collector):
        # data: [batch, time, ...]
        data = data.to(device)

        # 计算 advantage
        advantage_module(data)

        # 存入 buffer
        rb.extend(data.reshape(-1))

        # 采样 batch 训练
        for _ in range(4):  # ppo_epochs
            subdata = rb.sample(64)
            loss_vals = loss_module(subdata)
            loss = loss_vals["loss_objective"] + 0.5 * loss_vals["loss_critic"] - 0.01 * loss_vals["loss_entropy"]

            optim.zero_grad()
            loss.backward()
            optim.step()

        if i % 10 == 0:
            print(f"Iter {i}, loss: {loss.item():.4f}")

    print("训练完成 ✅")


if __name__ == "__main__":
    # 调用方式：
    # python train_ppo.py PPO.yaml myenv.yaml
    if len(sys.argv) < 3:
        print("用法: python RL/PPO/train.py <policy_config.yaml> <env_name>")
        sys.exit(1)

        # 命令行参数
    policy_config_name = sys.argv[1]
    env_config_name = sys.argv[2]

    # 获取项目根目录
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 拼接 policy config 路径
    if not policy_config_name.endswith(".yaml"):
        policy_config_name += ".yaml"
    print(f'project_root: {project_root}')
    policy_config_path = os.path.join(project_root, "RL", "policy_configs", policy_config_name)

    # 拼接 env config 路径
    env_config_path = os.path.join(project_root, "configs", "env", f"{env_config_name}.yaml")

    train_ppo(policy_config_path, env_config_path)
