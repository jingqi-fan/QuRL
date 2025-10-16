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

        # 生成 event_map_full：包含到达事件 + J 个完成事件
        # queue_event_options 形状 [2Q, Q]，前 Q 行 +1，后 Q 行 -1
        arrive_map = self.queue_event_options[: self.Q]  # [Q, Q]
        leave_map = self.queue_event_options[self.Q:]  # [Q, Q]
        # 每个队列有 J 个离开事件
        leave_map_expanded = leave_map.repeat_interleave(self.J, dim=0)  # [Q*J, Q]
        self.event_map_full = torch.cat([arrive_map, leave_map_expanded], dim=0).to(self.device)

        if reentrant == 1:
            print(f'1')
            # self.queue_event_options2 = queue_event_options2.to(self.device).float()
            self.queue_event_options2 = torch.as_tensor(
                queue_event_options2, dtype=torch.float32, device=self.device
            )
            # 生成 event_map_full：包含到达事件 + J 个完成事件
            # queue_event_options2 形状 [2Q, Q]，前 Q 行 +1，后 Q 行 -1
            arrive_map2 = self.queue_event_options2[: self.Q]  # [Q, Q]
            leave_map2 = self.queue_event_options2[self.Q:]  # [Q, Q]
            # 每个队列有 J 个离开事件
            leave_map_expanded2 = leave_map2.repeat_interleave(self.J, dim=0)  # [Q*J, Q]
            self.event_map_full2 = torch.cat([arrive_map2, leave_map_expanded2], dim=0).to(self.device)
        else:
            self.event_map_full2 = self.event_map_full.to(self.device)

            # self._queues = None
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
        # self.action_spec = Bounded(low=0.0, high=1, shape=(self.S, self.Q), dtype=torch.float32)

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

        # if job_counts.max() > 0:
        #     for j in range(self.J):
        #         mask_j = (job_counts > j)  # [B,Q]
        #         if mask_j.any():
        #             samp = self.draw_service(time_now)  # [B,Q]
        #             service_times[..., j] = torch.where(mask_j, samp, service_times[..., j])

        # 现在的做法: 先 if job_counts.max()>0: 再 for j: if mask_j.any(): ...
        # 改为一次性采样 J 份，然后用掩码写入：
        B, Q, J = queues.size(0), self.Q, self.J
        new_service = self.draw_service(time_now).unsqueeze(-1).expand(B, Q, J)  # [B,Q,J]
        # 只有 j<job_counts 的位置被写入
        pos = torch.arange(J, device=self.device).view(1, 1, J)
        mask = pos < job_counts.unsqueeze(-1)  # [B,Q,J]
        service_times = torch.where(mask, new_service, service_times)

        # self._queues = queues
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

    # # ---------- 分配与速率（“B 语义”的张量化实现） ----------
    def _alloc_job_rates_and_counts(self,
                                    action: torch.Tensor,  # [B,S,Q]
                                    mu: torch.Tensor,  # [1,S,Q] 或 [B,S,Q]
                                    job_counts: torch.Tensor):  # [B,Q] (int)
        """
        向量化的 Capacity sharing 分配：
          - mu_eff = mu * action  -> [B,S,Q]
          - 对每个 (b,q) 在 server 维度选出前 k 个速率 (降序)，
            其中 k = min(job_counts[b,q], 可用server数, J)
          - 生成 job_rates[b,q,j]：j=0..k-1 为对应速率，其余为 0
        返回:
          job_rates: [B,Q,J]
          num_alloc: [B,Q] (long)
        """
        assert action.dim() == 3, "action must be [B,S,Q]"
        B, S, Q = action.shape
        J = self.J
        device = action.device
        dtype = action.dtype

        # 1) 有效速率（<=0 的直接视为不可用）
        mu_eff = (mu * action).clamp_min(0)  # [B,S,Q]

        # 2) 每个 (b,q) 的可用 server 数
        avail_cnt = (mu_eff > 0).sum(dim=1)  # [B,Q]

        # 3) 先在 server 维度取 top-K（K = min(J,S)），得到 [B,Q,K]
        K = min(J, S)
        # 把维度换到 [B,Q,S]，便于在最后一维 topk
        mu_bqs = mu_eff.permute(0, 2, 1).contiguous()  # [B,Q,S]
        topk_vals, _ = torch.topk(mu_bqs, k=K, dim=-1, largest=True)

        # 4) 若 J>S，需要在尾部补零到 J 位
        if J > K:
            pad = torch.zeros(B, Q, J - K, device=device, dtype=dtype)
            topj_vals = torch.cat([topk_vals, pad], dim=-1)  # [B,Q,J]
        else:
            topj_vals = topk_vals  # [B,Q,J]

        # 5) num_alloc = min(job_counts, avail_cnt, J)
        num_alloc = torch.minimum(job_counts, avail_cnt)
        num_alloc = torch.minimum(num_alloc, torch.full_like(num_alloc, J))

        # 6) 只保留前 num_alloc 个位置，其余清零
        pos = torch.arange(J, device=device).view(1, 1, J)  # [1,1,J]
        keep = pos < num_alloc.unsqueeze(-1)  # [B,Q,J]
        job_rates = torch.where(keep, topj_vals, torch.zeros_like(topj_vals))

        return job_rates, num_alloc.long()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # === 读状态（合并语义：job_counts 就是 queues） ===
        time_now = self._time  # [B,1]
        arrival_times = self._arrival_times  # [B,Q]
        service_times = self._service_times  # [B,Q,J]
        job_counts = self._job_counts  # [B,Q]  (≡ queues)
        B, Q, J = job_counts.shape[0], self.Q, self.J
        S = self.S
        dev = self.device
        eps, BIG = self.eps, self.big

        # === 动作裁剪与归一（只对“在服人数”限流） ===
        action = tensordict["action"].to(dev).float()  # [B,S,Q]
        net = self.network.view(1, S, Q)
        action = torch.clamp(action * net, min=0.0)
        action = torch.minimum(action, job_counts.unsqueeze(1).expand(-1, S, -1))
        per_server_sum = action.sum(dim=2, keepdim=True).clamp_min(1e-8)
        action = action / per_server_sum

        # === 分配服务率（仅对前 job_counts 个槽位） ===
        job_rates, num_alloc = self._alloc_job_rates_and_counts(
            action, self.mu.view(1, S, Q), job_counts
        )  # [B,Q,J], [B,Q]

        pos = torch.arange(J, device=dev).view(1, 1, J)  # [1,1,J]
        valid_job_mask = (pos < job_counts.unsqueeze(-1))  # [B,Q,J]
        has_rate = (job_rates > 0)

        # === 候选完成时间（不真正扣减） ===
        eff_times = torch.where(
            valid_job_mask,
            torch.where(has_rate, service_times / (job_rates + eps), torch.full_like(service_times, BIG)),
            torch.full_like(service_times, BIG),
        )  # [B,Q,J]

        # === ST-argmin 选事件 & dt ===
        cand_complete = eff_times.view(B, -1)  # [B,Q*J]
        event_times = torch.cat([arrival_times, cand_complete], dim=-1)  # [B,Q + QJ]
        outcome = self.st_argmin(event_times)  # [B,Q + QJ] (硬 one-hot 前向)
        dt = (outcome * event_times).sum(dim=-1, keepdim=True)  # [B,1]

        # 事件索引（张量化解析；无 python if）
        idx = outcome.argmax(dim=-1)  # [B]
        is_arrival = (idx < Q)  # [B] bool
        flat = (idx - Q).clamp_min(0)
        q_src = (flat // J)  # [B] 完成源队列
        j_src = (flat % J)  # [B] 完成槽位

        # === 事件前快照（用于定位写入/弹出） ===
        job_counts_prev = job_counts.clone()  # [B,Q]
        allocated_mask_prev = ((pos < num_alloc.unsqueeze(-1)) & valid_job_mask)  # [B,Q,J]

        # === 更新时间与到达计时器（结构性变动放到后面） ===
        time_now = time_now + dt
        arrival_times = arrival_times - dt

        # === 用 event_map_full 计算 Δq 并更新 job_count（合并语义） ===
        # delta_q = outcome @ self.event_map_full  # [B,Q]（到达+1 / 完成-1 / 路由按矩阵）

        if random.random() < 0.9:
            delta_q = outcome @ self.event_map_full
        else:
            delta_q = outcome @ self.event_map_full2

        # delta_q = outcome @ self.event_map_full

        job_counts = job_counts + delta_q

        # 容量警告与夹取
        # over_J = (job_counts > J)
        # if over_J.any():
        #     num_over = over_J.sum().item()
            # print(f"[WARN] {num_over} job_count entries exceeded J={J}; clamping to J (overflow dropped).")

        job_counts = torch.clamp(job_counts, min=0, max=J)

        # === 外到达：登记新 service_time（事件前的写入位置；用掩码张量化，无 if） ===
        arrived_mask_q = outcome[..., :Q] > 0.5  # [B,Q]  到达的队列 one-hot
        write_pos_arr = job_counts_prev  # [B,Q]
        can_write_arr = arrived_mask_q & (write_pos_arr < J)  # [B,Q]
        # 抽样新服务时间（对所有队列采样，再用掩码挑选）
        new_service_all = self.draw_service(time_now)  # [B,Q]
        # 把到达计时器重采样只加到发生到达的位置
        new_inter_all = self.draw_inter_arrivals(time_now)  # [B,Q]
        arrival_times = torch.where(
            arrived_mask_q, arrival_times + new_inter_all, arrival_times
        )
        # 写入 service_times[b,q,write_pos_arr] = new_service_all[b,q]（只在 can_write_arr 处）
        target_eq_wp = (pos == write_pos_arr.unsqueeze(-1))  # [B,Q,J]
        write_mask = can_write_arr.unsqueeze(-1) & target_eq_wp  # [B,Q,J]
        service_times = torch.where(
            write_mask, new_service_all.unsqueeze(-1), service_times
        )

        # === 完成事件：弹出 (q_src,j_src)（交换到“事件前”的尾槽位并清零；人数已由 Δq 改过） ===
        # === 完成事件：弹出 (q_src,j_src)（交换到“事件前”的尾槽位并清零；人数已由 Δq 改过） ===
        dep_mask_b = (~is_arrival).unsqueeze(-1).unsqueeze(-1)  # [B,1,1] bool
        # “事件前”的 last_pos
        last_pos_prev = (job_counts_prev.clamp_min(1) - 1)  # [B,Q]
        last_sel_mask = (pos == last_pos_prev.unsqueeze(-1))  # [B,Q,J] bool
        # 取出每个 (b,q) 的“事件前尾槽位值”
        last_vals = torch.sum(service_times * last_sel_mask, dim=-1, keepdim=True)  # [B,Q,1]
        # 完成槽位 one-hot（来自 outcome 的完成部分）
        left_tail = (outcome[..., Q:].view(B, Q, J) > 0.5)  # [B,Q,J] bool
        # 交换 + 清零
        service_times = torch.where(dep_mask_b & left_tail, last_vals.expand_as(service_times), service_times)
        service_times = torch.where(dep_mask_b & last_sel_mask, torch.zeros_like(service_times), service_times)

        # === 扣减本步 dt：对“事件前被分配速率”的槽位扣 dt*rate，排除本步完成的槽位 ===
        dec = dt.view(B, 1, 1) * job_rates  # [B,Q,J]
        # 排除本步完成槽位
        allocated_mask_prev = allocated_mask_prev & (~left_tail)
        service_times = torch.where(
            allocated_mask_prev, torch.clamp(service_times - dec, min=0.0), service_times
        )

        # === 写回（合并语义：queues = job_counts） ===
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts
        self._queues = job_counts

        # 成本（用事件前人数算 holding cost；如需事件后可自行改）
        cost = (dt * job_counts_prev) @ self.h
        reward = -cost
        done = torch.zeros_like(reward, dtype=torch.bool)

        out = TensorDict(
            {
                "queues": job_counts,  # 合并后的队长 = 在服人数
                "time": time_now,
                "reward": reward,
                "event_time": dt,
                "done": done,
            },
            batch_size=torch.Size([B]),
        )
        return out

    # # ---------- step ----------
    # def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
    #     # queues = self._queues                  # [B,Q]
    #     time_now = self._time                  # [B,1]
    #     arrival_times = self._arrival_times    # [B,Q]
    #     service_times = self._service_times    # [B,Q,J]
    #     job_counts = self._job_counts          # [B,Q]
    #     B = job_counts.shape[0]
    #
    #     # 动作 [B,S,Q]：连通性 & 按 B 的语义限制到队列可服务作业数
    #     action = tensordict["action"].to(self.device).float()  # [B,S,Q]
    #     net = self.network.view(1, self.S, self.Q)
    #     action = torch.clamp(action * net, min=0.0)
    #     action = torch.minimum(action, job_counts.unsqueeze(1).expand(-1, self.S, -1))
    #
    #
    #     # # ---- 加上容量守恒归一化 ----
    #     per_server_sum = action.sum(dim=2, keepdim=True).clamp_min(1e-8)
    #     action = action / torch.maximum(per_server_sum, torch.ones_like(per_server_sum))
    #
    #     # # 名额分配得每个作业的分配速率（只给前 num_alloc 个）
    #     # job_rates, num_alloc = self._alloc_job_rates_and_counts(action, job_counts)  # [B,Q,J], [B,Q]
    #     # --- 按 Capacity Sharing 分配比例速率 ---
    #     job_rates, num_alloc = self._alloc_job_rates_and_counts(
    #         action, self.mu.view(1, self.S, self.Q), job_counts
    #     )
    #
    #     # 有效作业 mask
    #     pos = torch.arange(self.J, device=self.device).view(1, 1, self.J)
    #     valid_job_mask = pos < job_counts.unsqueeze(-1)  # [B,Q,J]
    #
    #     # 只在已存在作业且被分配速率>0 的位置定义有效完成时间，否则置大
    #     eps = self.eps
    #     has_rate = job_rates > 0
    #     eff_times = torch.where(
    #         valid_job_mask,
    #         torch.where(has_rate, service_times / (job_rates + eps), torch.full_like(service_times, self.big)),
    #         torch.full_like(service_times, self.big),
    #     )  # [B,Q,J]
    #
    #     # 队列最早完成时间
    #     q_done_time, which_job = masked_min(eff_times, valid_job_mask, large=self.big)  # [B,Q], [B,Q]
    #
    #     # 事件：到达 vs 完成
    #     # event_times = torch.cat([arrival_times, q_done_time], dim=-1)  # [B,2Q]
    #     # outcome = self.st_argmin(event_times)                           # [B,2Q] onehot
    #
    #     # 2) 拼成 [B, Q + Q*J] 的候选集合（全局 m+mJ）
    #     cand_complete = eff_times.view(B, -1)  # [B, Q*J]
    #     event_times = torch.cat([arrival_times,  # [B, Q]
    #                              cand_complete], dim=-1)  # [B, Q + Q*J]
    #
    #     # 3) ST-argmin：forward=hard one-hot, backward=softmin 梯度
    #     outcome = self.st_argmin(event_times)  # [B, Q + Q*J]
    #     # 4) 事件时间（建议用 dot，而不是再取 min，确保与 ST 对齐）
    #     event_time = (outcome * event_times).sum(dim=-1, keepdim=True)  # [B,1]
    #
    #     # 5) Δq：用一个 (Q+QJ)×Q 的映射矩阵把 one-hot 变成每队列 +1/-1
    #     #    可在 __init__ 里预生成 self.event_map_full，形状 [(Q+Q*J), Q]
    #     delta_q = outcome @ self.event_map_full  # [B, Q]
    #
    #     # 6) 到达/离开掩码（hard）
    #     arrived_mask = outcome[..., :self.Q] > 0.5  # [B, Q]
    #     left_tail = outcome[..., self.Q:].view(B, self.Q, self.J)  # [B, Q, J]
    #     left_mask = left_tail.any(dim=-1)  # [B, Q]
    #
    #
    #
    #     # 队列增量 Δq
    #     # delta_q = outcome @ self.queue_event_options  # [B,Q]
    #
    #     # 事件时间
    #     # event_time = torch.min(event_times, dim=-1, keepdim=True).values  # [B,1]
    #
    #     # 成本与奖励（与 A/B 一致）
    #     cost = (event_time * job_counts) @ self.h  # [B,1]
    #     reward = -cost
    #
    #     # ====== 关键：对所有“已分配但未完成”的作业扣减剩余工时（B 的语义）======
    #     # 已分配位置：pos < num_alloc
    #     allocated_mask = (pos < num_alloc.unsqueeze(-1)) & valid_job_mask  # [B,Q,J]
    #     # 本步扣减量：event_time * job_rates
    #     dec = (event_time.view(B, 1, 1)) * job_rates                        # [B,Q,J]
    #     service_times = torch.where(allocated_mask, torch.clamp(service_times - dec, min=0.0), service_times)
    #
    #     # 推进时间与队列
    #     time_now = time_now + event_time
    #     job_counts = F.relu(job_counts + delta_q)
    #
    #     # 更新到达计时器
    #     arrival_times = arrival_times - event_time
    #
    #     # 到达/离开 mask（硬事件）
    #     # arrived_mask = outcome[..., : self.Q] > 0.5   # [B,Q]
    #     # left_mask = outcome[..., self.Q :] > 0.5      # [B,Q]
    #
    #     # 到达：重采 inter-arrival & 新作业 service time（有容量才写入）
    #     if arrived_mask.any():
    #         new_inter = self.draw_inter_arrivals(time_now)  # [B,Q]
    #         arrival_times = torch.where(arrived_mask, arrival_times + new_inter, arrival_times)
    #
    #         new_service = self.draw_service(time_now)       # [B,Q]
    #         write_pos = job_counts.clamp(max=self.J - 1)    # [B,Q]
    #         can_write = arrived_mask & (job_counts < self.J)
    #
    #         service_times = service_times.scatter(2, write_pos.unsqueeze(-1), new_service.unsqueeze(-1))
    #         job_counts = torch.where(can_write, job_counts + 1, job_counts)
    #
    #     # 8) 离开：直接用 left_tail 定位 (q, j)
    #     if left_mask.any():
    #         # 只有一个 (q,j) 会是 1；hard forward 下没歧义
    #         which_job = left_tail.float().argmax(dim=-1)  # [B, Q]  (在 left_mask 为 True 的那条队列有效)
    #         idx = which_job.clamp(min=0, max=self.J - 1)  # [B, Q]
    #         last_pos = (job_counts - 1).clamp(min=0)  # [B, Q]
    #
    #         # 与尾交换 + 清零（和你现有代码一致）
    #         gather_last = service_times.gather(2, last_pos.unsqueeze(-1))  # [B,Q,1]
    #         service_times = service_times.scatter(2, idx.unsqueeze(-1), gather_last)
    #         zero_src = torch.zeros_like(gather_last)
    #         service_times = service_times.scatter(2, last_pos.unsqueeze(-1), zero_src)
    #         job_counts = torch.where(left_mask & (job_counts > 0), (job_counts - 1).clamp(min=0), job_counts)
    #
    #     # 写回内部状态
    #     # self._queues = job_counts
    #
    #     # ---- 警告：检测是否有队列触顶 J ----
    #     over_cap = (self._job_counts >= self.J)
    #     if over_cap.any():
    #         # max_over = (self._queues - self._job_counts).max().item()
    #         print(f"[WARN] Some queues exceeded max_jobs={self.J}, "
    #               f"extra jobs = {self._job_counts-self.J}. These jobs are effectively dropped.")
    #
    #
    #     self._time = time_now
    #     self._arrival_times = arrival_times
    #     self._service_times = service_times
    #     self._job_counts = job_counts
    #
    #     done = torch.zeros_like(reward, dtype=torch.bool)
    #     bs = torch.Size([job_counts.shape[0]])
    #     out = TensorDict(
    #         {
    #             "queues": job_counts,
    #             "time": time_now,
    #             "reward": reward,
    #             "event_time": event_time,
    #             "done": done,
    #         },
    #         batch_size=bs,
    #     )
    #     return out
