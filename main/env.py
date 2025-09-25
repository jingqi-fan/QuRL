import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, Unbounded
from typing import Optional
from torchrl.envs.utils import step_mdp

class STargmin(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hard = F.one_hot(torch.argmin(x, dim=-1), num_classes=x.size(-1)).to(x.dtype)
        soft = self.softmax(-x / self.temp)
        return hard - soft.detach() + soft


def masked_min(x: torch.Tensor, mask: torch.Tensor, large: float = 1e9):
    x_masked = torch.where(mask, x, torch.full_like(x, large))
    vals, idx = x_masked.min(dim=-1)
    return vals, idx


class BatchedDiffDES(EnvBase):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    batch_locked = False

    def __init__(
        self,
        network: torch.Tensor,     # [S,Q] 0/1
        mu: torch.Tensor,          # [S,Q]
        h: torch.Tensor,           # [Q]
        draw_service,              # fn(env, t:[B,1]) -> [B,Q]
        draw_inter_arrivals,       # fn(env, t:[B,1]) -> [B,Q]
        max_jobs: int = 64,
        temp: float = 1.0,
        device: str = "cuda",
        seed: Optional[int] = None,
        default_B: int = 1,
        queue_event_options=None,
        verbose: bool = False,
    ):
        super().__init__(batch_size=[])
        self.to(device)

        self.network = network.to(self.device).float()  # [S,Q]
        self.mu = mu.to(self.device).float()            # [S,Q]
        self.h = h.to(self.device).float()              # [Q]
        self.S, self.Q = self.network.shape
        self.J = int(max_jobs)
        self.default_B = default_B

        self.draw_service_core = draw_service
        self.draw_inter_arrivals_core = draw_inter_arrivals

        self.temp = temp
        self.st_argmin = STargmin(temp)
        self.verbose = verbose
        self.eps = 1e-8
        self.big = 1e12

        if queue_event_options is None:
            eye = torch.eye(self.Q, device=self.device)
            self.queue_event_options = torch.cat([eye, -eye], dim=0)  # [2Q,Q]
        else:
            self.queue_event_options = queue_event_options.to(self.device).float()

        self._queues = None
        self._time = None
        self._arrival_times = None
        self._service_times = None
        self._job_counts = None

        self._make_spec()

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # ---------- specs ----------
    def _make_spec(self):
        self.observation_spec = Composite(
            queues=Bounded(low=0.0, high=float("inf"), shape=(self.Q,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.float32),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = Bounded(low=0.0, high=1e6, shape=(self.S, self.Q), dtype=torch.float32)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _set_seed(self, seed: Optional[int]):
        torch.manual_seed(int(seed) if seed is not None else 0)

    def gen_params(self, batch_size=None) -> TensorDictBase:
        if batch_size is None or batch_size == []:
            batch_size = torch.Size([self.default_B])
        return TensorDict({}, batch_size)

    def draw_service(self, t: torch.Tensor) -> torch.Tensor:
        return self.draw_service_core(self, t)

    def draw_inter_arrivals(self, t: torch.Tensor) -> torch.Tensor:
        return self.draw_inter_arrivals_core(self, t)

    # ---------- reset ----------
    def _reset(self, tensordict: Optional[TensorDictBase]) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, [])
        if "params" in tensordict.keys():
            B = tensordict["params"].batch_size
        elif tensordict.batch_size != torch.Size([]):
            B = tensordict.batch_size
        else:
            B = torch.Size([self.default_B])
        queues = tensordict.get("queues", torch.zeros(B + (self.Q,), device=self.device)).float()
        time_now = tensordict.get("time", torch.zeros(B + (1,), device=self.device)).float()

        arrival_times = self.draw_inter_arrivals(time_now)                     # [B,Q]
        service_times = torch.zeros(B + (self.Q, self.J), device=self.device)  # [B,Q,J]
        job_counts = torch.clamp(torch.round(queues).long(), min=0)
        job_counts = torch.minimum(job_counts, torch.full_like(job_counts, self.J))

        if job_counts.max() > 0:
            for j in range(self.J):
                mask_j = (job_counts > j)  # [B,Q]
                if mask_j.any():
                    samp = self.draw_service(time_now)  # [B,Q]
                    service_times[..., j] = torch.where(mask_j, samp, service_times[..., j])

        self._queues = queues
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts

        out = TensorDict(
            {
                "queues": queues.clone(),
                "time": time_now.clone(),
                # "params": TensorDict({"max_jobs": torch.full(B, self.J, dtype=torch.int64, device=self.device)}, B),
            },
            batch_size=B,
        )
        return out

    # ---------- 离散名额分配（代码2语义的张量化实现） ----------
    def _discrete_job_rates_from_action(self, action: torch.Tensor, job_counts: torch.Tensor) -> torch.Tensor:
        """
        输入:
          action: [B,S,Q]  (连续)
          job_counts: [B,Q]
        过程:
          1) slots = round(action) ∈ {0,1,2,...}，每个 (s,q) 的离散名额，cap 到 J
          2) mu_with_grad = mu*action/adj_const，其中 adj_const=1{action<1}?1:action
          3) 为每个 (s,q) 复制 'slots' 个槽位，得到 [B,Q,S*J] 的槽位速率矩阵
          4) 每个 (b,q) 取 top-J 槽位作为前 J 个作业的速率，得到 [B,Q,J]
        """
        B = action.size(0)
        mu = self.mu.view(1, self.S, self.Q)  # [1,S,Q] -> broadcast to [B,S,Q]

        # step-2: mu_with_grad（与代码2一致）
        adj_const = action.clone()
        adj_const = torch.where(adj_const < 1.0, torch.ones_like(adj_const), adj_const)
        mu_with_grad = mu * action / adj_const  # [B,S,Q]

        # step-1: slots（名额），每 server-queue 不超过 J
        slots = torch.round(action).to(torch.int64).clamp(min=0, max=self.J)  # [B,S,Q]

        # 构造每个 server-queue 的 J 个“虚拟槽位”：k < slots ? mu_with_grad : 0
        k = torch.arange(self.J, device=action.device).view(1, 1, 1, self.J)  # [1,1,1,J]
        mask_slots = (k < slots.unsqueeze(-1))                                # [B,S,Q,J]
        rates_per_slot = mu_with_grad.unsqueeze(-1) * mask_slots.float()      # [B,S,Q,J]

        # 把所有 server 的槽位摊平到每个 queue 的一维： [B,Q,S*J]
        rates_flat = rates_per_slot.permute(0, 2, 1, 3).contiguous().view(B, self.Q, self.S * self.J)

        # 对每个 (b,q) 取 top-J，缺的自然补 0
        topk_vals, _ = torch.topk(rates_flat, k=self.J, dim=-1, largest=True)  # [B,Q,J]

        # 只给实际存在的作业（前 job_counts 个），多余位置最终会被 mask 掉
        return topk_vals  # [B,Q,J]

    # ---------- step ----------
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        queues = self._queues             # [B,Q]
        time_now = self._time             # [B,1]
        arrival_times = self._arrival_times  # [B,Q]
        service_times = self._service_times  # [B,Q,J]


        service_times_calc = service_times


        job_counts = self._job_counts        # [B,Q]
        B = queues.shape[0]

        # 动作 [B,S,Q]，与连通性相乘并裁剪
        action = tensordict["action"].to(self.device).float()  # [B,S,Q]
        net = self.network.view(1, self.S, self.Q)
        action = torch.clamp(action * net, min=0.0)

        # === 关键改动：使用“离散名额分配”的 job_rates（代码2语义） ===
        job_rates = self._discrete_job_rates_from_action(action, job_counts)  # [B,Q,J]

        # 有效作业 mask：pos<job_counts
        pos = torch.arange(self.J, device=self.device).view(1, 1, self.J)
        valid_job_mask = pos < job_counts.unsqueeze(-1)  # [B,Q,J]

        # 有效服务速率，避免除零
        job_rates = torch.where(valid_job_mask, job_rates, torch.zeros_like(job_rates))

        # 有效完成时间：service_time / rate（未分配或不存在的作业置为 big）
        eff_times = torch.where(
            valid_job_mask,
            torch.where(job_rates > 0, service_times_calc / (job_rates + self.eps), torch.full_like(service_times_calc, self.big)),
            torch.full_like(service_times_calc, self.big),
        )  # [B,Q,J]

        # 队列最早完成时间
        q_done_time, which_job = masked_min(eff_times, valid_job_mask, large=self.big)  # [B,Q], [B,Q]

        # 事件组合：到达 vs 完成
        event_times = torch.cat([arrival_times, q_done_time], dim=-1)  # [B,2Q]
        outcome = self.st_argmin(event_times)                           # [B,2Q] onehot

        # 队列增量 Δq
        delta_q = outcome @ self.queue_event_options  # [B,Q]

        # 事件时间
        event_time = torch.min(event_times, dim=-1, keepdim=True).values  # [B,1]

        # 成本与奖励
        cost = (event_time * queues) @ self.h  # [B,1]
        reward = -cost

        # 推进时间与队列
        time_now = time_now + event_time
        queues = F.relu(queues + delta_q)

        # 更新到达计时器
        arrival_times = arrival_times - event_time

        # 到达/离开 mask
        arrived_mask = outcome[..., : self.Q] > 0.5   # [B,Q]
        left_mask = outcome[..., self.Q :] > 0.5      # [B,Q]

        # 到达：重采该队列 inter-arrival & 新作业 service time（未满才写入）
        if arrived_mask.any():
            new_inter = self.draw_inter_arrivals(time_now)  # [B,Q]
            arrival_times = torch.where(arrived_mask, arrival_times + new_inter, arrival_times)

            new_service = self.draw_service(time_now)       # [B,Q]
            write_pos = job_counts.clamp(max=self.J - 1)    # [B,Q]
            can_write = arrived_mask & (job_counts < self.J)

            # 在 write_pos 写入新作业时间
            # service_times.scatter_(2, write_pos.unsqueeze(-1), new_service.unsqueeze(-1))

            service_times = service_times.scatter(2, write_pos.unsqueeze(-1), new_service.unsqueeze(-1))

            # 计数 +1（未满）
            job_counts = torch.where(can_write, job_counts + 1, job_counts)

        # 离开：移除 which_job（与尾元素交换并清零）
        if left_mask.any():
            has_job = job_counts > 0
            effective_left = left_mask & has_job
            if effective_left.any():
                idx = which_job.clamp(min=0, max=self.J - 1)
                last_pos = (job_counts - 1).clamp(min=0)  # [B,Q]
                gather_last = service_times.gather(2, last_pos.unsqueeze(-1))  # [B,Q,1]
                # service_times.scatter_(2, idx.unsqueeze(-1), gather_last)

                service_times = service_times.scatter(2, idx.unsqueeze(-1), gather_last)


                zero_src = torch.zeros_like(gather_last)
                # service_times.scatter_(2, last_pos.unsqueeze(-1), zero_src)

                service_times = service_times.scatter(2, last_pos.unsqueeze(-1), zero_src)

                job_counts = torch.where(effective_left, (job_counts - 1).clamp(min=0), job_counts)

        # 写回内部状态
        self._queues = queues
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts

        done = torch.zeros_like(reward, dtype=torch.bool)
        B = queues.shape[0]
        bs = torch.Size([B])
        out = TensorDict(
            {
                "queues": queues,
                "time": time_now,
                "reward": reward,
                "event_time": event_time,
                "done": done,
            },
            batch_size=bs,
        )
        return out
        # return step_mdp(out, keep_other=True)
