from __future__ import annotations
from typing import List, Dict, Callable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    Categorical
)
from torchrl.envs.common import EnvBase


# -----------------------------
# Straight-through argmin (kept from your code)
# -----------------------------
class STargmin(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, K] (we keep batch_size==1 below; extend to batched if needed)
        return (
            F.one_hot(torch.argmin(x), num_classes=x.size(1))
            - self.softmax(-x / self.temp).detach()
            + self.softmax(-x / self.temp)
        )


# -----------------------------
# Torch (GPU-friendly) allocator (replaces numpy usage)
# -----------------------------
@torch.no_grad()
def allocator(action: torch.Tensor, mu: torch.Tensor, queue_service_times: List[List[torch.Tensor]]):
    """
    Args:
        action: [1, s, q], non-negative floats; will be multiplied by network before calling here
        mu:     [1, s, q]
        queue_service_times: python list of length q; each entry is a list of per-job tensors
                             where each per-job tensor has shape [1, 1, q]
    Returns:
        allocated_a: list length q; each element is a list of length m (jobs allocated), entries are tensors on same device
        queue_nonzero_inds: dict q -> list[[s_idx, q_idx], ...]
        num_allocated: list length q; number of jobs allocated
    """
    device = action.device

    adj_const = action.clone()
    adj_const[adj_const < 1] = 1

    mu_with_grad = mu * action / adj_const
    a = mu * action  # [1, s, q]
    num_q = a.size(-1)

    # non-zero locations at batch 0
    nonzero_inds = (a[0] > 0).nonzero(as_tuple=False).tolist()  # [[s, q], ...]

    queue_nonzero_inds: Dict[int, List[List[int]]] = {i: [] for i in range(num_q)}
    for s_idx, q_idx in nonzero_inds:
        k = int(torch.round(action[0, s_idx, q_idx]).item())
        for _ in range(k):
            queue_nonzero_inds[q_idx].append([s_idx, q_idx])

    allocated_a: List[List[torch.Tensor]] = []
    num_allocated: List[int] = []
    for q in range(num_q):
        idxs = queue_nonzero_inds[q]
        if idxs:
            scores = torch.tensor(
                [mu_with_grad[0, i[0], i[1]].item() for i in idxs], device=device
            )
            order = torch.argsort(scores, descending=True).tolist()
            idxs = [idxs[i] for i in order]

        m = min(len(queue_service_times[q]), len(idxs))
        num_allocated.append(m)

        q_alloc: List[torch.Tensor] = []
        for j in range(m):
            s_idx, q_idx = idxs[j]
            q_alloc.append(mu_with_grad[0, s_idx, q_idx])
        allocated_a.append(q_alloc)

    return allocated_a, queue_nonzero_inds, num_allocated


# -----------------------------
# TorchRL-only Env
# -----------------------------
class DiffDiscreteEventSystemTorch(EnvBase):
    """
    TorchRL-native environment converted from your Gym version.

    Notes
    -----
    * Implemented for batch_size == 1 for clarity (TorchRL supports batched envs; extend if needed).
    * Internals (service_times lists) are kept as Python lists, computations are torch on `device`.
    * No Stable-Baselines nor Gym dependency.
    * Observation is Composite: {"queues": [q], "time": [1]}.
    * Action spec is bounded continuous [s, q] with lower bound 0.
    """

    def __init__(
        self,
        network: torch.Tensor,    # [s, q]
        mu: torch.Tensor,         # [s, q]
        h: torch.Tensor,          # [q]
        draw_service: Callable[["DiffDiscreteEventSystemTorch", torch.Tensor], torch.Tensor],
        draw_inter_arrivals: Callable[["DiffDiscreteEventSystemTorch", torch.Tensor], torch.Tensor],
        *,
        queue_event_options: Optional[torch.Tensor] = None,  # [2q, q] with +I and -I by default
        temp: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int = 3003,
    ):
        self.device = torch.device(device)
        super().__init__(device=device, batch_size=torch.Size([]))

        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # Make tensors with batch dim = 1 inside (for compatibility with your code)
        self.network = network.to(self.device).unsqueeze(0)  # [1, s, q]
        self.mu = mu.to(self.device).unsqueeze(0)            # [1, s, q]
        self.q = self.network.size(-1)
        self.s = self.network.size(-2)
        self.h = h.to(self.device)                           # [q]
        self.temp = temp
        self.st_argmin = STargmin(temp=temp)

        self.eps = 1e-8
        self.inv_eps = 1.0 / self.eps

        if queue_event_options is None:
            eye = torch.eye(self.q, device=self.device)
            self.queue_event_options = torch.cat([eye, -eye], dim=0).float()  # [2q, q]
        else:
            self.queue_event_options = queue_event_options.to(self.device).float()

        # user-provided sampling callables
        self.draw_service_core = draw_service
        self.draw_inter_arrivals_core = draw_inter_arrivals

        # internal state holders (batch_size=1)
        self._queues = torch.zeros(1, self.q, device=self.device)
        self._time = torch.zeros(1, 1, device=self.device)
        self._service_times: List[List[torch.Tensor]] = [[] for _ in range(self.q)]
        self._arrival_times = torch.zeros(1, self.q, device=self.device)
        self._time_weight_queue_len = torch.zeros(self.q, device=self.device)

        # ---- Specs ----
        self.observation_spec = CompositeSpec(
            queues=UnboundedContinuousTensorSpec(shape=torch.Size([self.q]), device=self.device),
            time=UnboundedContinuousTensorSpec(shape=torch.Size([1]), device=self.device),
        ).to(self.device)

        # actions >= 0 (set a large upper bound to satisfy spec finiteness)
        self.action_spec = BoundedTensorSpec(
            low=torch.zeros(self.s, self.q, device=self.device),
            high=torch.full((self.s, self.q), 1e6, device=self.device),
            dtype=torch.float32,
            device=self.device,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([1]), device=self.device)
        self.done_spec = Categorical(2, shape=torch.Size([1]), dtype=torch.bool, device=self.device)

    # ------------------ helpers ------------------
    def draw_service(self, time: torch.Tensor) -> torch.Tensor:
        return self.draw_service_core(self, time)

    def draw_inter_arrivals(self, time: torch.Tensor) -> torch.Tensor:
        return self.draw_inter_arrivals_core(self, time)

    # ------------------ EnvBase API ------------------
    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        self._time.zero_()
        self._queues.zero_()
        self._time_weight_queue_len.zero_()
        self._service_times = [[] for _ in range(self.q)]
        self._arrival_times = self.draw_inter_arrivals(self._time)  # [1, q]

        obs = TensorDict(
            {
                "queues": self._queues.clone().squeeze(0),  # [q]
                "time": self._time.clone().squeeze(0),      # [1]
            },
            batch_size=torch.Size([]),
            device=self.device,
        )
        return obs

    def _step(self, tensordict: TensorDict) -> TensorDict:
        # action: [s, q] or [1, s, q]
        action = tensordict.get("action")
        if action.dim() == 2:
            action = action.unsqueeze(0)
        action = action.to(self.device)

        queues = self._queues  # [1, q]
        time = self._time      # [1, 1]

        # respect network connectivity and queue availability
        action = action * self.network
        action = torch.minimum(action, queues.unsqueeze(1).expand(-1, self.s, -1))

        allocated_work, queue_nonzero_inds, num_allocated = allocator(action, self.mu, self._service_times)

        # effective service times per queue
        eff_service_times = [torch.tensor([self.inv_eps], device=self.device)] * self.q
        for q in range(self.q):
            if num_allocated[q] > 0:
                st = torch.stack(self._service_times[q][:num_allocated[q]])[:, 0, q]  # [m]
                mw = torch.stack(allocated_work[q])                                   # [m]
                eff_service_times[q] = st / torch.clamp(mw, min=self.eps)

        min_eff_service_times = torch.stack([torch.min(eff) for eff in eff_service_times]).unsqueeze(0)  # [1, q]

        # event pool: arrivals vs services
        event_times = torch.cat([self._arrival_times, min_eff_service_times], dim=1).float()  # [1, 2q]

        # straight-through argmin outcome in one-hot over 2q
        outcome = self.st_argmin(event_times)  # [2q]

        # translate outcome into delta_q in R^q (arrive +1, service -1)
        delta_q = outcome @ self.queue_event_options  # [q]

        # min event time
        event_time = event_times.min()

        # updates
        self._time_weight_queue_len = self._time_weight_queue_len + event_time * queues.squeeze(0)
        time.add_(event_time)
        cost = (event_time * queues) @ self.h  # [1, 1] * [q] via broadcasting -> [1, 1]

        queues = F.relu(queues + delta_q)
        self._queues.copy_(queues)

        # update service times for jobs with positive work
        for q in range(self.q):
            if num_allocated[q] > 0:
                updated = (
                    torch.stack(self._service_times[q][:num_allocated[q]])
                    - event_time * torch.stack(allocated_work[q]).unsqueeze(1).unsqueeze(-1).expand(-1, 1, self.q)
                )
                self._service_times[q][:num_allocated[q]] = list(torch.unbind(updated))

        # arrival clock ticks down
        self._arrival_times = self._arrival_times - event_time

        # bookkeeping: decide which job/queue completed, manage arrivals/departures
        delta = delta_q.to(torch.int)
        delta_arrived = (delta == 1).to(torch.int)
        delta_left = (delta == -1).to(torch.int)

        if (delta != 0).sum() == 0:
            # avoid deadlock: bump the smallest arrival far away
            idx = torch.argmax(outcome)
            self._arrival_times[0, idx] = self._arrival_times[0, idx] + torch.tensor(1e8, device=self.device)

        if delta_arrived.sum() > 0:
            new_arrival_times = self.draw_inter_arrivals(time)  # [1, q]
            new_service_time = self.draw_service(time)          # [1, q]

            if delta_arrived.sum() == 1:
                self._arrival_times = self._arrival_times + torch.nan_to_num(
                    new_arrival_times.reshape(1, self.q) * delta_arrived.view(1, self.q),
                    nan=self.inv_eps,
                )

            which_arrival = torch.argmax(delta_arrived).item()
            self._service_times[which_arrival].append(new_service_time)

        if delta_left.sum() > 0:
            # select which queue and which job among allocated ones finished
            which_job = [0] * self.q
            for q in range(self.q):
                if num_allocated[q] > 0:
                    ratios = (
                        torch.stack(self._service_times[q][:num_allocated[q]])[:, 0, q]
                        / torch.stack(allocated_work[q])
                    )
                    which_job[q] = int(torch.argmin(ratios).item())
            which_queue = int(torch.argmin(min_eff_service_times).item())
            # remove the served job
            _ = self._service_times[which_queue].pop(which_job[which_queue])

        # build next tensordict
        next_td = TensorDict(
            {
                "queues": self._queues.squeeze(0).clone(),   # [q]
                "time": self._time.squeeze(0).clone(),       # [1]
                "reward": (-cost).reshape(1).clone(),        # [1]
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=torch.Size([]),
            device=self.device,
        )
        return next_td

    def _set_seed(self, seed: Optional[int]) -> int:
        if seed is None:
            return int(self.gen.initial_seed())
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        return seed


# -----------------------------
# Example usage (pseudo)
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s, q = 3, 4
    network = torch.ones(s, q)  # fully connected
    mu = torch.rand(s, q) + 0.5
    h = torch.ones(q)

    # Simple torch-native sampling fns
    def draw_service(env: DiffDiscreteEventSystemTorch, time: torch.Tensor) -> torch.Tensor:
        return torch.distributions.Exponential(rate=torch.ones(1, env.q, device=env.device)).sample()

    def draw_inter_arrivals(env: DiffDiscreteEventSystemTorch, time: torch.Tensor) -> torch.Tensor:
        # constant rate 1.0 (customize as needed)
        exps = torch.distributions.Exponential(rate=torch.ones(1, env.q, device=env.device)).sample()
        return exps

    env = DiffDiscreteEventSystemTorch(
        network=network,
        mu=mu,
        h=h,
        draw_service=draw_service,
        draw_inter_arrivals=draw_inter_arrivals,
        device=device,
    )

    td = env.reset()
    print("reset:", td.get("queues"), td.get("time"))  # 这里只看观测

    for t in range(5):
        action = env.network.squeeze(0).clone()
        td = env.step(TensorDict({"action": action}, batch_size=[]))

        # 取 next 子字典里的值
        queues = td["next", "queues"]
        now_t = td["next", "time"]
        reward = td["next", "reward"]
        done = td["next", "done"]

        print(f"t={t + 1:02d} | time={float(now_t.item()):.4f} | "
              f"reward={float(reward.item()):.4f} | queues={queues.tolist()}")

        # 如果你后面要把“下一状态”作为输入继续迭代，可以把 td 递进为 next：
        td = td.get("next")
