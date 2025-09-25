# env_views.py
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Bounded, Unbounded

from main.env import BatchedDiffDES  # 确保导入路径指向你刚才发的那个 env 定义


class RLViewDiffDES(BatchedDiffDES):
    def __init__(self, *args, time_f: bool = True, **kwargs):
        self.time_f = time_f
        super().__init__(*args, **kwargs)

    def _make_spec(self):
        # 先让父类生成 action/reward/done 等 spec
        super()._make_spec()

        # ——统一观测为 obs——
        if self.time_f:
            # obs = [queues, time]
            self.observation_spec = Composite(
                obs=Unbounded(shape=(self.Q + 1,), dtype=torch.float32),
                shape=(),
            )
        else:
            # obs = queues
            self.observation_spec = Composite(
                obs=Unbounded(shape=(self.Q,), dtype=torch.float32),
                shape=(),
            )
        # 与 observation_spec 对齐
        self.state_spec = self.observation_spec.clone()
        # action_spec / reward_spec / done_spec 沿用父类已生成的

    def _filter_obs(self, td: TensorDictBase) -> TensorDictBase:
        out = TensorDict({}, batch_size=td.batch_size)

        # 拷贝非观测字段（reward / done / event_time 等）
        for k, v in td.items():
            if k not in ("queues", "time"):
                out.set(k, v)

        q = td.get("queues", None)  # [B,Q]
        if q is None:
            return td

        if self.time_f and "time" in td.keys():
            t = td.get("time")  # [B,1]
            obs = torch.cat([q, t], dim=-1)  # [B, Q+1]
        else:
            obs = q  # [B, Q]

        out.set("obs", obs)
        return out

    def _reset(self, tensordict: TensorDictBase | None) -> TensorDictBase:
        td = super()._reset(tensordict)
        return self._filter_obs(td)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        td = super()._step(tensordict)
        return self._filter_obs(td)



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
