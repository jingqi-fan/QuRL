# env_views.py
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Composite, Bounded, Unbounded

from main.env import BatchedDiffDES


class RLViewDiffDES(BatchedDiffDES):
    def __init__(self, *args, time_f: bool = True, **kwargs):
        self.time_f = time_f
        super().__init__(*args, **kwargs)

    def _make_spec(self):
        super()._make_spec()

        if self.time_f:
            self.observation_spec = Composite(
                obs=Unbounded(shape=(self.Q + 1,), dtype=torch.float32),
                shape=(),
            )
        else:
            self.observation_spec = Composite(
                obs=Unbounded(shape=(self.Q,), dtype=torch.float32),
                shape=(),
            )

        self.state_spec = self.observation_spec.clone()

    def _filter_obs(self, td: TensorDictBase) -> TensorDictBase:
        out = TensorDict({}, batch_size=td.batch_size)

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


def dummy_draw_service(env, t):
    B = t.shape[0]
    return torch.ones(B, env.Q, device=env.device)


def dummy_draw_inter_arrivals(env, t):
    B = t.shape[0]
    return torch.ones(B, env.Q, device=env.device) * 2.0


if __name__ == "__main__":
    # Configure environment
    S, Q = 2, 3  # 2 servers, 3 queues
    network = torch.ones(S, Q)  # fully connected
    mu = torch.ones(S, Q)  # unit service rate
    h = torch.arange(1, Q + 1).float()  # [1,2,3] as weights

    env = RLViewDiffDES(
        network=network,
        mu=mu,
        h=h,
        draw_service=dummy_draw_service,
        draw_inter_arrivals=dummy_draw_inter_arrivals,
        max_jobs=4,
        temp=1.0,
        device="cuda",
        default_B=5,  # batch = 2
        time_f=False,  # Switch to control whether time is included in obs
    )

    # reset
    td = env.reset()
    print("=== Reset obs ===")
    print(td)

    # Random action step
    for i in range(5):
        B = td.batch_size[0]
        action = torch.rand(B, S, Q)
        print('8')
        # td = env.step(TensorDict({"action": action}, batch_size=[B]))
        # print(f"\n=== Step {i} ===")
        # print("queues:", td["next", "queues"])
        # print("time:", td["next", "time"])
        # print("reward:", td["next", "reward"])
        # print("done:", td["next", "done"])