import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.utils import step_mdp

from main.env import BatchedDiffDES


# 假设你已有 BatchedDiffDES（上一版可批处理的 TorchRL 环境）
# from your_module import BatchedDiffDES

class TRLContinuousEnv(BatchedDiffDES):
    """
    纯连续动作的 TorchRL 包装环境（不依赖 gym）。
    - 继承 BatchedDiffDES
    - 观测里新增 "observation": queues 或 cat(queues, time)
    - 奖励缩放 reward_scale
    """

    def __init__(
        self,
        network: torch.Tensor,          # [S,Q]
        mu: torch.Tensor,               # [S,Q]
        h: torch.Tensor,                # [Q]
        draw_service,                   # fn(env, t:[B,1]) -> [B,Q]
        draw_inter_arrivals,            # fn(env, t:[B,1]) -> [B,Q]
        *,
        max_jobs: int = 64,
        temp: float = 1.0,
        device: str = "cpu",
        seed: int | None = None,
        verbose: bool = False,
        # wrapper 选项：
        reward_scale: float = 1.0,
        time_f: bool = False,           # True: observation=cat(queues, time)
    ):
        super().__init__(
            network=network,
            mu=mu,
            h=h,
            draw_service=draw_service,
            draw_inter_arrivals=draw_inter_arrivals,
            max_jobs=max_jobs,
            temp=temp,
            device=device,
            seed=seed,
            verbose=verbose,
        )
        self.reward_scale = float(reward_scale)
        self.time_f = bool(time_f)

        # —— 扩展 observation_spec，新增一个 "observation" 键 —— #
        obs_dim = self.Q + (1 if self.time_f else 0)
        self.observation_spec = Composite(
            **self.observation_spec._specs,            # 继承父类已有的 queues/time/params
            observation=Unbounded(shape=(obs_dim,), dtype=torch.float32),
            shape=(),
        )
        # action_spec / reward_spec / done_spec 直接沿用父类

    # --- 构造 observation 张量 ---
    def _build_observation(self, queues: torch.Tensor, time_now: torch.Tensor) -> torch.Tensor:
        # queues: [B,Q], time_now: [B,1]
        if not self.time_f:
            return queues
        return torch.cat([queues, time_now], dim=-1)  # [B,Q+1]

    # --- 覆盖 _reset：把 observation 填进去 ---
    def _reset(self, tensordict: TensorDictBase | None) -> TensorDictBase:
        td = super()._reset(tensordict)
        td.set("observation", self._build_observation(td["queues"], td["time"]))
        return td

    # --- 覆盖 _step：奖励缩放 + observation ---
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = super()._step(tensordict)  # 读 "action":[B,S,Q]，返回 queues/time/reward/done/params
        if self.reward_scale != 1.0:
            out.set("reward", out.get("reward") * self.reward_scale)
        out.set("observation", self._build_observation(out["queues"], out["time"]))
        return out


# ----------------- 用法示例 -----------------
if __name__ == "__main__":
    S, Q, J = 3, 2, 16
    device = "cuda"

    network = torch.tensor([[1, 1],
                            [1, 0],
                            [0, 1]], dtype=torch.float32)
    mu = torch.ones(S, Q)
    h = torch.tensor([1.0, 1.0])

    # 采样器：指数分布（确保在 env.device 上）
    def draw_service(env, t):
        B = t.size(0)
        return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()

    def draw_inter_arrivals(env, t):
        B = t.size(0)
        return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()

    env = TRLContinuousEnv(
        network, mu, h,
        draw_service, draw_inter_arrivals,
        max_jobs=J, temp=1.0, device=device, seed=0,
        reward_scale=0.1,
        time_f=True,        # observation = cat(queues, time)
    ).to(device)

    # 按教程风格动态 batch
    BATCH = 3
    td0 = env.reset(env.gen_params(batch_size=[BATCH]))
    print("reset:", td0.batch_size, td0["observation"].shape, td0["observation"].device)

    # 连续动作： [B,S,Q]
    action = torch.rand(BATCH, S, Q, device=env.device)
    td1 = env.step(TensorDict({"action": action}, batch_size=[BATCH]))
    # print("step reward:", td1.get("reward").shape, td1.get("reward").device)

    # 手写 rollout（batch 自适应）
    def simple_rollout(env: EnvBase, steps=10):
        _data = env.reset()                           # 确保在 env.to(device) 之后调用
        B = _data.batch_size
        traj = TensorDict({}, [*B, steps], device=env.device)
        for t in range(steps):
            _data["action"] = env.action_spec.rand(B).to(env.device)  # 连续动作采样
            _data = env.step(_data)
            traj[..., t] = _data
            _data = step_mdp(_data, keep_other=True)
        return traj

    ro = simple_rollout(env, steps=5)
    print("rollout obs:", ro.get(("next", "observation")).shape)
