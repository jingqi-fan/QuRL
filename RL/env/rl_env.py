# env_views.py
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Bounded, Unbounded

from main.env import BatchedDiffDES  # 确保导入路径指向你刚才发的那个 env 定义


class RLViewDiffDES(BatchedDiffDES):
    """
    继承 BatchedDiffDES，加入 time 可见性开关：
      - time_f=True: 观测包含 'time'
      - time_f=False: 观测不包含 'time'（仅 'queues' 与 'params'）
    其他行为与父类一致。
    """

    def __init__(
        self,
        *args,
        time_f: bool = True,
        **kwargs,
    ):
        self.time_f = time_f
        super().__init__(*args, **kwargs)

    # 重新构造 spec，使 observation_spec 与可见字段一致
    def _make_spec(self):
        # 先调用父类，拿到完整 spec
        super()._make_spec()

        # 根据开关裁剪 observation_spec
        if not self.time_f:
            # 仅保留 queues 与 params
            self.observation_spec = Composite(
                queues=Bounded(low=0.0, high=float("inf"), shape=(self.Q,), dtype=torch.float32),
                # params=Composite(
                #     max_jobs=Bounded(low=1, high=self.J, shape=(), dtype=torch.int64),
                #     shape=(),
                # ),
                shape=(),
            )
            # state_spec 可以保持与 observation_spec 一致（也可保留父类，不影响内部状态）
            self.state_spec = self.observation_spec.clone()

    # 工具：按需裁剪观测（去掉 time）
    def _filter_obs(self, td: TensorDictBase) -> TensorDictBase:
        if self.time_f:
            return td
        # 仅选择 queues / params；reward/done 不在 obs 里，这里不动
        if "queues" in td.keys() or "params" in td.keys():
            return td.select("queues", "params")
        # 如果是 step 的整体输出，需要只裁剪“观测相关”的键
        # 这里假设父类 reset/step 返回的是顶层 obs（而非 next 包装）
        keys = []
        if "queues" in td.keys(): keys.append("queues")
        if "params" in td.keys(): keys.append("params")
        # 其他键原样带出
        out = TensorDict({}, batch_size=td.batch_size)
        for k, v in td.items():
            if k in ("queues", "time", "params"):
                # 观测相关：按开关处理
                if k == "time":
                    continue
                out.set(k, v)
            else:
                # 非观测（如 reward / done 等）保留
                out.set(k, v)
        return out

    # 覆盖 reset：裁剪 time
    def _reset(self, tensordict: TensorDictBase | None) -> TensorDictBase:
        td = super()._reset(tensordict)
        return self._filter_obs(td)

    # 覆盖 step：裁剪 time
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        out = super()._step(tensordict)
        return self._filter_obs(out)


# 假设 RLViewDiffDES 已经在同目录导入
# from your_env_file import RLViewDiffDES

def dummy_draw_service(env, t):
    """返回正态分布的 service times"""
    B = t.shape[0]
    return torch.ones(B, env.Q, device=env.device)

def dummy_draw_inter_arrivals(env, t):
    """返回常数 inter-arrivals"""
    B = t.shape[0]
    return torch.ones(B, env.Q, device=env.device) * 2.0

if __name__ == "__main__":
    # 配置环境
    S, Q = 2, 3  # 2 个服务器，3 个队列
    network = torch.ones(S, Q)  # fully connected
    mu = torch.ones(S, Q)       # unit service rate
    h = torch.arange(1, Q+1).float()  # [1,2,3] as weights

    env = RLViewDiffDES(
        network=network,
        mu=mu,
        h=h,
        draw_service=dummy_draw_service,
        draw_inter_arrivals=dummy_draw_inter_arrivals,
        max_jobs=4,
        temp=1.0,
        device="cuda",
        default_B=2,   # batch = 2
        time_f=False,   # 👈 开关控制是否在 obs 里包含 time
    )

    # reset
    td = env.reset()
    print("=== Reset obs ===")
    print(td)

    # 随机动作 step
    for i in range(5):
        B = td.batch_size[0]
        action = torch.rand(B, S, Q)
        # td = env.step(TensorDict({"action": action}, batch_size=[B]))
        # print(f"\n=== Step {i} ===")
        # print("queues:", td["next", "queues"])
        # print("time:", td["next", "time"])
        # print("reward:", td["next", "reward"])
        # print("done:", td["next", "done"])
