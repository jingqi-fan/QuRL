import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, Unbounded
from typing import Optional

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
        queue_event_options2=None,
        reentrant = 0,
        verbose: bool = False,
        # draw_due_date=None
    ):
        super().__init__(batch_size=[])
        self.to(device)
        random.seed(seed)

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

        # Generate event_map_full: contains arrival events + J completion events
        # queue_event_options shape [2Q, Q], first Q rows +1, last Q rows -1
        arrive_map = self.queue_event_options[: self.Q]  # [Q, Q]
        leave_map = self.queue_event_options[self.Q:]  # [Q, Q]
        # Each queue has J departure events
        leave_map_expanded = leave_map.repeat_interleave(self.J, dim=0)  # [Q*J, Q]
        self.event_map_full = torch.cat([arrive_map, leave_map_expanded], dim=0).to(self.device)

        if reentrant == 1:
            print(f'in env, 1')
            self.queue_event_options2 = torch.as_tensor(
                queue_event_options2, dtype=torch.float32, device=self.device
            )
            # Generate event_map_full: contains arrival events + J completion events
            # queue_event_options2 shape [2Q, Q], first Q rows +1, last Q rows -1
            arrive_map2 = self.queue_event_options2[: self.Q]  # [Q, Q]
            leave_map2 = self.queue_event_options2[self.Q:]  # [Q, Q]
            # Each queue has J departure events
            leave_map_expanded2 = leave_map2.repeat_interleave(self.J, dim=0)  # [Q*J, Q]
            self.event_map_full2 = torch.cat([arrive_map2, leave_map_expanded2], dim=0).to(self.device)
        else:
            self.event_map_full2 = self.event_map_full.to(self.device)

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
        self.action_spec = Bounded(low=0.0, high=1.0, shape=(self.S, self.Q), dtype=torch.float32)

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

        B, Q, J = queues.size(0), self.Q, self.J
        new_service = self.draw_service(time_now).unsqueeze(-1).expand(B, Q, J)  # [B,Q,J]
        # Only positions where j < job_counts are written
        pos = torch.arange(J, device=self.device).view(1, 1, J)
        mask = pos < job_counts.unsqueeze(-1)  # [B,Q,J]
        service_times = torch.where(mask, new_service, service_times)

        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts

        out = TensorDict(
            {
                "queues": queues.clone(),
                "time": time_now.clone(),
            },
            batch_size=B,
        )
        return out

    # # ---------- Allocation and Rates (Tensorized implementation of "B semantics") ----------
    def _alloc_job_rates_and_counts(self,
                                    action: torch.Tensor,  # [B,S,Q]
                                    mu: torch.Tensor,  # [1,S,Q] or [B,S,Q]
                                    job_counts: torch.Tensor):  # [B,Q] (int)
        """
        Vectorized Capacity sharing allocation:
          - mu_eff = mu * action  -> [B,S,Q]
          - For each (b,q), select the top k rates (descending) in the server dimension,
            where k = min(job_counts[b,q], number of available servers, J)
          - Generate job_rates[b,q,j]: j=0..k-1 are the corresponding rates, the rest are 0
        Returns:
          job_rates: [B,Q,J]
          num_alloc: [B,Q] (long)
        """
        assert action.dim() == 3, "action must be [B,S,Q]"
        B, S, Q = action.shape
        J = self.J
        device = action.device
        dtype = action.dtype

        # 1) Effective rates (<=0 are directly considered unavailable)
        mu_eff = (mu * action).clamp_min(0)  # [B,S,Q]

        # 2) Number of available servers for each (b,q)
        avail_cnt = (mu_eff > 0).sum(dim=1)  # [B,Q]

        # 3) First take top-K in the server dimension (K = min(J,S)), obtaining [B,Q,K]
        K = min(J, S)
        # Swap dimensions to [B,Q,S] to facilitate topk on the last dimension
        mu_bqs = mu_eff.permute(0, 2, 1).contiguous()  # [B,Q,S]
        topk_vals, _ = torch.topk(mu_bqs, k=K, dim=-1, largest=True)

        # 4) If J>S, zero-padding to J positions is needed at the tail
        if J > K:
            pad = torch.zeros(B, Q, J - K, device=device, dtype=dtype)
            topj_vals = torch.cat([topk_vals, pad], dim=-1)  # [B,Q,J]
        else:
            topj_vals = topk_vals  # [B,Q,J]

        # 5) num_alloc = min(job_counts, avail_cnt, J)
        num_alloc = torch.minimum(job_counts, avail_cnt)
        num_alloc = torch.minimum(num_alloc, torch.full_like(num_alloc, J))

        # 6) Keep only the first num_alloc positions, clear the rest to zero
        pos = torch.arange(J, device=device).view(1, 1, J)  # [1,1,J]
        keep = pos < num_alloc.unsqueeze(-1)  # [B,Q,J]
        job_rates = torch.where(keep, topj_vals, torch.zeros_like(topj_vals))

        return job_rates, num_alloc.long()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # === Read state (merged semantics: job_counts is queues) ===
        time_now = self._time  # [B,1]
        arrival_times = self._arrival_times  # [B,Q]
        service_times = self._service_times  # [B,Q,J]
        job_counts = self._job_counts  # [B,Q]  (≡ queues)
        B, Q, J = job_counts.shape[0], self.Q, self.J
        S = self.S
        dev = self.device
        eps, BIG = self.eps, self.big

        # === Action clipping and normalization (rate limiting only for "number of people in service") ===
        action = tensordict["action"].to(dev).float()  # [B,S,Q]
        net = self.network.view(1, S, Q)
        action = torch.clamp(action * net, min=0.0)
        action = torch.minimum(action, job_counts.unsqueeze(1).expand(-1, S, -1))
        per_server_sum = action.sum(dim=2, keepdim=True).clamp_min(1e-8)
        action = action / per_server_sum

        # === Allocate service rates (only for the first job_counts slots) ===
        job_rates, num_alloc = self._alloc_job_rates_and_counts(
            action, self.mu.view(1, S, Q), job_counts
        )  # [B,Q,J], [B,Q]

        pos = torch.arange(J, device=dev).view(1, 1, J)  # [1,1,J]
        valid_job_mask = (pos < job_counts.unsqueeze(-1))  # [B,Q,J]
        has_rate = (job_rates > 0)

        # === Candidate completion times (not actually deducted) ===
        eff_times = torch.where(
            valid_job_mask,
            torch.where(has_rate, service_times / (job_rates + eps), torch.full_like(service_times, BIG)),
            torch.full_like(service_times, BIG),
        )  # [B,Q,J]

        # === ST-argmin select event & dt ===
        cand_complete = eff_times.view(B, -1)  # [B,Q*J]
        event_times = torch.cat([arrival_times, cand_complete], dim=-1)  # [B,Q + QJ]
        outcome = self.st_argmin(event_times)  # [B,Q + QJ] (Hard one-hot forward)
        dt = (outcome * event_times).sum(dim=-1, keepdim=True)  # [B,1]

        # Event index (tensorized parsing; no python if)
        idx = outcome.argmax(dim=-1)  # [B]
        is_arrival = (idx < Q)  # [B] bool

        # === Pre-event snapshot (used to locate write/pop) ===
        job_counts_prev = job_counts.clone()  # [B,Q]
        allocated_mask_prev = ((pos < num_alloc.unsqueeze(-1)) & valid_job_mask)  # [B,Q,J]

        # === Update time and arrival timer (structural changes moved to the end) ===
        time_now = time_now + dt
        arrival_times = arrival_times - dt

        # === Use event_map_full to calculate Δq and update job_count (merged semantics) ===
        # delta_q = outcome @ self.event_map_full  # [B,Q] (Arrival +1 / Completion -1 / Routing according to matrix)

        if random.random() <= 0.9:
            delta_q = outcome @ self.event_map_full
        else:
            delta_q = outcome @ self.event_map_full2

        # delta_q = outcome @ self.event_map_full

        job_counts = job_counts + delta_q

        job_counts = torch.clamp(job_counts, min=0, max=J)

        # === External arrival: register new service_time (write position before event; use mask tensorization, no if) ===
        arrived_mask_q = outcome[..., :Q] > 0.5  # [B,Q] Arrived queue one-hot
        write_pos_arr = job_counts_prev  # [B,Q]
        can_write_arr = arrived_mask_q & (write_pos_arr < J)  # [B,Q]
        # Sample new service time (sample for all queues, then select using mask)
        new_service_all = self.draw_service(time_now)  # [B,Q]
        # Add arrival timer resampling only to positions where arrival occurred
        new_inter_all = self.draw_inter_arrivals(time_now)  # [B,Q]
        arrival_times = torch.where(
            arrived_mask_q, arrival_times + new_inter_all, arrival_times
        )
        # Write service_times[b,q,write_pos_arr] = new_service_all[b,q] (only at can_write_arr)
        target_eq_wp = (pos == write_pos_arr.unsqueeze(-1))  # [B,Q,J]
        write_mask = can_write_arr.unsqueeze(-1) & target_eq_wp  # [B,Q,J]
        service_times = torch.where(
            write_mask, new_service_all.unsqueeze(-1), service_times
        )

        # === Completion event: pop (q_src,j_src) (swap to "pre-event" tail slot and clear) ===
        dep_mask_b = (~is_arrival).unsqueeze(-1).unsqueeze(-1)  # [B,1,1] bool
        # "Pre-event" last_pos
        last_pos_prev = (job_counts_prev.clamp_min(1) - 1)  # [B,Q]
        last_sel_mask = (pos == last_pos_prev.unsqueeze(-1))  # [B,Q,J] bool
        # Extract the "pre-event tail slot value" for each (b,q)
        last_vals = torch.sum(service_times * last_sel_mask, dim=-1, keepdim=True)  # [B,Q,1]
        # Completion slot one-hot (from the completion part of outcome)
        left_tail = (outcome[..., Q:].view(B, Q, J) > 0.5)  # [B,Q,J] bool

        # Queue mask corresponding to the completion event: [B, Q, 1]
        dep_q_mask = left_tail.any(dim=-1, keepdim=True)  # True for the queue where completion occurred, False otherwise
        # 1) Write the remaining service time of the "pre-event tail job" to the completion slot (only for this queue)
        service_times = torch.where(dep_mask_b & left_tail,
                                    last_vals.expand_as(service_times),
                                    service_times)
        # 2) Clear the "pre-event tail slot" to zero (only for this queue)
        service_times = torch.where(dep_mask_b & dep_q_mask & last_sel_mask,
                                    torch.zeros_like(service_times),
                                    service_times)

        # === Deduct dt of this step: deduct dt*rate for slots "allocated rate before event", excluding slots completed in this step ===
        dec = dt.view(B, 1, 1) * job_rates  # [B,Q,J]
        # Exclude slots completed in this step
        allocated_mask_prev = allocated_mask_prev & (~left_tail)
        service_times = torch.where(
            allocated_mask_prev, torch.clamp(service_times - dec, min=0.0), service_times
        )

        # === Write back (merged semantics: queues = job_counts) ===
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts
        self._queues = job_counts

        # Cost (calculate holding cost using pre-event headcount; can be changed to post-event if needed)
        cost = (dt * job_counts_prev) @ self.h
        reward = -cost
        done = torch.zeros_like(reward, dtype=torch.bool)

        out = TensorDict(
            {
                "queues": job_counts,  # Merged queue length = number of people in service
                "time": time_now,
                "reward": reward,
                "event_time": dt,
                "done": done,
            },
            batch_size=torch.Size([B]),
        )
        return out