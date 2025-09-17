from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs.utils import check_env_specs, step_mdp


# ---------- utils ----------
class STargmin(nn.Module):
    def __init__(self, temp: float = 1.0):
        super().__init__()
        self.temp = temp
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., K)
        hard = F.one_hot(torch.argmin(x, dim=-1), num_classes=x.size(-1)).to(x.dtype)
        soft = self.softmax(-x / self.temp)
        return hard - soft.detach() + soft


def masked_min(x: torch.Tensor, mask: torch.Tensor, large: float = 1e9):
    """按 mask 在最后一维做最小值（False 的置大，从而不被选择）。"""
    x_masked = torch.where(mask, x, torch.full_like(x, large))
    vals, idx = x_masked.min(dim=-1)
    return vals, idx


# ---------- env ----------
class BatchedDiffDES(EnvBase):
    """
    可批处理（非 batch-locked）的可微分队列离散事件环境（TorchRL EnvBase）。
    - B: batch size（运行时由 reset 时传入的 params 决定）
    - S: servers, Q: queues, J: max_jobs(每队列的容量上界)
    观测: {"queues":[B,Q], "time":[B,1], "params": <Composite>}
    动作: [B,S,Q] 连续非负（将与 network 掩码相乘）
    奖励: - (Δt * queues)·h  —— 与你原始定义一致
    终止: 永不终止（恒 False）
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}
    batch_locked = False

    def __init__(
        self,
        network: torch.Tensor,          # [S, Q] 0/1连通
        mu: torch.Tensor,               # [S, Q] 服务速率
        h: torch.Tensor,                # [Q] 代价权重
        draw_service,                   # fn(env, t:[B,1]) -> [B,Q]  新作业的服务时间样本
        draw_inter_arrivals,            # fn(env, t:[B,1]) -> [B,Q]  每队列下一次到达间隔
        max_jobs: int = 64,             # 每队列最多作业数
        temp: float = 1.0,
        device: str = "cuda",
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        super().__init__(batch_size=[])  # 关键：不在 __init__ 固定 device/batch
        self.to(device)

        # 静态结构（不含 batch）
        self.network = network.to(self.device).float()  # [S,Q]
        self.mu = mu.to(self.device).float()            # [S,Q]
        self.h = h.to(self.device).float()              # [Q]
        self.S, self.Q = self.network.shape
        self.J = int(max_jobs)

        # 随机采样器
        self.draw_service_core = draw_service
        self.draw_inter_arrivals_core = draw_inter_arrivals

        # ST-argmin / 其他
        self.temp = temp
        self.st_argmin = STargmin(temp)
        self.verbose = verbose
        self.eps = 1e-8
        self.big = 1e12

        # (+e_i, -e_i) in R^Q  —— 事件的队列增量
        eye = torch.eye(self.Q, device=self.device)
        self.queue_event_options = torch.cat([eye, -eye], dim=0)  # [2Q,Q]

        # 内部状态（含 batch 维），在 reset 时创建：
        self._queues = None          # [B,Q]
        self._time = None            # [B,1]
        self._arrival_times = None   # [B,Q]
        self._service_times = None   # [B,Q,J]
        self._job_counts = None      # [B,Q]

        # specs
        self._make_spec()

        # 随机种子
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # ----- specs -----
    def _make_spec(self):
        self.observation_spec = Composite(
            queues=Bounded(low=0.0, high=float("inf"), shape=(self.Q,), dtype=torch.float32),
            time=Unbounded(shape=(1,), dtype=torch.float32),
            params=Composite(  # 方便与教程风格一致：把运行参数放到obs里传递
                max_jobs=Bounded(low=1, high=self.J, shape=(), dtype=torch.int64),
                shape=(),
            ),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()

        self.action_spec = Bounded(
            low=0.0, high=1e6, shape=(self.S, self.Q), dtype=torch.float32
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    # ----- seeding -----
    def _set_seed(self, seed: Optional[int]):
        torch.manual_seed(int(seed) if seed is not None else 0)

    # ----- params helper（与教程一致）-----
    def gen_params(self, batch_size=None) -> TensorDictBase:
        if batch_size is None:
            batch_size = []
        td = TensorDict(
            {
                "params": TensorDict(
                    {
                        "max_jobs": torch.tensor(self.J, dtype=torch.int64, device=self.device),
                    },
                    [],
                )
            },
            [],
        )
        if batch_size:
            td = td.expand(batch_size).contiguous()
        return td

    # ----- draw helpers -----
    def draw_service(self, t: torch.Tensor) -> torch.Tensor:
        # 期望返回 [B,Q] 的样本
        return self.draw_service_core(self, t)

    def draw_inter_arrivals(self, t: torch.Tensor) -> torch.Tensor:
        # 返回 [B,Q]
        return self.draw_inter_arrivals_core(self, t)

    # ----- reset -----
    def _reset(self, tensordict: Optional[TensorDictBase]) -> TensorDictBase:
        if tensordict is None:
            tensordict = TensorDict({}, [])

        # 获取 batch 大小：优先 params，其次 queues/time
        if "params" in tensordict.keys():
            B = tensordict["params"].batch_size
        else:
            B = tensordict.batch_size
        if len(B) == 0:
            B = torch.Size([1])

        # 初始 queues/time
        queues = tensordict.get("queues", torch.zeros(B + (self.Q,), device=self.device)).float()
        time_now = tensordict.get("time", torch.zeros(B + (1,), device=self.device)).float()

        # 初始化状态张量
        arrival_times = self.draw_inter_arrivals(time_now)  # [B,Q]
        service_times = torch.zeros(B + (self.Q, self.J), device=self.device)  # padding 0
        job_counts = torch.clamp(torch.round(queues).long(), min=0)  # [B,Q]
        job_counts = torch.minimum(job_counts, torch.full_like(job_counts, self.J))

        # 为每个队列填充前 count 个位置的 service_times
        if job_counts.max() > 0:
            # 批量抽样每队列的初始作业服务时间（逐个位置填充）
            # 这里简单：对每个 “位置 j” 批量抽一次，然后只在 j < count 的地方写入
            for j in range(self.J):
                mask_j = (job_counts > j)  # [B,Q]
                if mask_j.any():
                    samp = self.draw_service(time_now)  # [B,Q]
                    service_times[..., j] = torch.where(mask_j, samp, service_times[..., j])

        # 写回内部状态
        self._queues = queues
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts

        # 输出 obs（带 params，和教程风格一致）
        out = TensorDict(
            {
                "queues": queues.clone(),
                "time": time_now.clone(),
                "params": TensorDict({"max_jobs": torch.full(B, self.J, dtype=torch.int64, device=self.device)}, B),
            },
            batch_size=B,
        )
        return out

    # ----- step -----
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        queues = self._queues             # [B,Q]
        time_now = self._time             # [B,1]
        arrival_times = self._arrival_times  # [B,Q]
        service_times = self._service_times  # [B,Q,J]
        job_counts = self._job_counts        # [B,Q]

        B = queues.shape[0]

        # 动作 [B,S,Q]，与连通性相乘并裁剪不超过队列长度
        action = tensordict["action"].to(self.device).float()  # [B,S,Q]
        net = self.network.view(1, self.S, self.Q)             # [1,S,Q]
        action = action * net
        # 不超过队列长度（每个队列可“激活”的服务通道数上界）
        # 这里允许连续值，不做取整；只是保证非负与有限
        action = torch.clamp(action, min=0.0)
        # 也可以按需上界：每个队列最多“总动作”不超过 job_counts（软约束可不加）
        max_per_q = job_counts.unsqueeze(1).clamp(min=1).float()  # [B,1,Q]
        action = torch.minimum(action, max_per_q * 1e6)  # 让动作不至于极端爆炸

        # 服务器对每个队列的“有效速率”：mu * action
        mu = self.mu.view(1, self.S, self.Q)  # [1,S,Q] -> broadcast
        server_rate = mu * action             # [B,S,Q]

        # —— 为每个队列的前 k 个作业分配“最优的 k 个服务器速率”（Top-k）——
        # k_q = min(job_counts[b,q], 有速率的服务器数)
        # 计算每队列的 topJ 服务器速率（降序），再取前 job_counts 个作为作业速率
        topk_vals, _ = torch.topk(server_rate, k=min(self.S, self.J), dim=1, largest=True)  # [B, min(S,J), Q]
        # pad 到 J（作业维）
        if topk_vals.size(1) < self.J:
            pad = self.J - topk_vals.size(1)
            topk_vals = torch.cat([topk_vals, torch.zeros(B, pad, self.Q, device=self.device)], dim=1)  # [B,J,Q]
        else:
            topk_vals = topk_vals[:, : self.J, :]  # [B,J,Q]
        # 变换形状与 service_times 对齐： [B,Q,J]
        job_rates = topk_vals.permute(0, 2, 1).contiguous()  # [B,Q,J]

        # 有效作业 mask：pos<job_counts
        pos = torch.arange(self.J, device=self.device).view(1, 1, self.J)  # [1,1,J]
        valid_job_mask = pos < job_counts.unsqueeze(-1)                    # [B,Q,J]

        # 有效服务速率，避免除零
        job_rates = torch.where(valid_job_mask, job_rates, torch.zeros_like(job_rates))
        # 有效服务时间
        eff_times = torch.where(valid_job_mask,
                                torch.where(job_rates > 0, service_times / (job_rates + self.eps), torch.full_like(service_times, self.big)),
                                torch.full_like(service_times, self.big))  # [B,Q,J]

        # 每个队列最早完成时间（在 J 上取 min）
        q_done_time, which_job = masked_min(eff_times, valid_job_mask, large=self.big)  # [B,Q], [B,Q]

        # 事件组合：到达 vs 服务完成
        event_times = torch.cat([arrival_times, q_done_time], dim=-1)  # [B, 2Q]

        # ST-argmin -> [B,2Q] onehot
        outcome = self.st_argmin(event_times)

        # 队列增量 Δq
        delta_q = outcome @ self.queue_event_options  # [B,Q]

        # 事件时间（标量/每个 batch 单元一个）：min 或 straight-through
        event_time = torch.min(event_times, dim=-1, keepdim=True).values  # [B,1]

        # 成本与奖励
        cost = (event_time * queues) @ self.h  # [B,1]
        reward = -cost  # [B,1]

        # 推进时间与队列
        time_now = time_now + event_time
        queues = F.relu(queues + delta_q)

        # —— 更新 service_times / arrival_times / job_counts —— #
        # 1) 所有到达时间递减
        arrival_times = arrival_times - event_time

        # 2) 根据 outcome 判定到达或离开（逐队列 one-hot）
        arrived_mask = outcome[..., : self.Q] > 0.5   # [B,Q]
        left_mask = outcome[..., self.Q :] > 0.5      # [B,Q]

        # 到达：重抽该队列 inter-arrival，并在该队列的下一个空位写入一个新作业的 service_time
        if arrived_mask.any():
            new_inter = self.draw_inter_arrivals(time_now)  # [B,Q]
            arrival_times = torch.where(arrived_mask, arrival_times + new_inter, arrival_times)

            new_service = self.draw_service(time_now)       # [B,Q]
            # 写入位置 = job_counts（旧）
            write_pos = job_counts.clamp(max=self.J - 1)    # [B,Q]
            # 只在未满 & 确实有到达的位置写
            can_write = arrived_mask & (job_counts < self.J)
            service_times.scatter_(
                dim=2,
                index=write_pos.unsqueeze(-1),
                src=new_service.unsqueeze(-1),
            )
            # 计数 +1（未满处）
            job_counts = torch.where(can_write, job_counts + 1, job_counts)

        # 离开：移除 which_job[b,q] 位置的作业（把其 service_times 清零，并把尾部前移一位）
        if left_mask.any():
            # 只有在该队列有作业时才有效
            has_job = job_counts > 0
            effective_left = left_mask & has_job
            if effective_left.any():
                # 读到该队列的 which_job 索引
                # which_job: [B,Q] ∈ [0,J)
                idx = which_job.clamp(min=0, max=self.J - 1)

                # 把 “最后一个有效作业”(pos = job_counts-1) 移到 idx 位置，再把最后一个置零；计数减一
                last_pos = (job_counts - 1).clamp(min=0)  # [B,Q]
                gather_last = service_times.gather(  # 取最后一个
                    2, last_pos.unsqueeze(-1)
                )  # [B,Q,1]

                # 写回到 idx
                service_times.scatter_(2, idx.unsqueeze(-1), gather_last)

                # 把 last_pos 位置清零
                zero_src = torch.zeros_like(gather_last)
                service_times.scatter_(2, last_pos.unsqueeze(-1), zero_src)

                # 计数 -1
                job_counts = torch.where(effective_left, (job_counts - 1).clamp(min=0), job_counts)

        # 写回内部状态
        self._queues = queues
        self._time = time_now
        self._arrival_times = arrival_times
        self._service_times = service_times
        self._job_counts = job_counts

        done = torch.zeros_like(reward, dtype=torch.bool)

        # out = TensorDict(
        #     {
        #         "queues": queues,
        #         "time": time_now,
        #         "params": TensorDict({"max_jobs": torch.full((queues.size(0),), self.J, dtype=torch.int64, device=self.device)}, queues.batch_size),
        #         "reward": reward,
        #         "done": done,
        #     },
        #     batch_size=queues.batch_size,
        # )

        # 取 batch 维（这里是 B）
        B = queues.shape[0]  # int
        bs = torch.Size([B])  # TensorDict 需要 torch.Size
        out = TensorDict(
            {
                "queues": queues,  # [B, Q]
                "time": time_now,  # [B, 1]
                "params": TensorDict(
                    {"max_jobs": torch.full((B,), self.J, dtype=torch.int64, device=self.device)},
                    batch_size = bs,
                ),
              "reward": reward,  # [B, 1]
               "done": done,  # [B, 1]
            },
            batch_size = bs,
        )

        return out


# # ----------- 在cpu上测试 -----------
# if __name__ == "__main__":
#     S, Q, J = 3, 2, 16
#     device = "cpu"
#
#     network = torch.tensor([[1, 1],
#                             [1, 0],
#                             [0, 1]], dtype=torch.float32)
#     mu = torch.full((S, Q), 1.0)
#     h = torch.tensor([1.0, 1.0])
#
#     # 采样器：指数分布
#     def draw_service(env, t):
#         # t: [B,1] -> return [B,Q]
#         B = t.size(0)
#         return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()
#
#     def draw_inter_arrivals(env, t):
#         B = t.size(0)
#         return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()
#
#     env = BatchedDiffDES(network, mu, h, draw_service, draw_inter_arrivals, max_jobs=J, device=device, temp=1.0)
#     env.to(device)
#
#     # ✅ 按教程风格动态设 batch
#     BATCH = 3
#     td = env.reset(env.gen_params(batch_size=[BATCH]))
#     print("reset(batch=10):", td.shape, td["queues"].shape)
#
#     # 随机动作
#     action = torch.rand(BATCH, S, Q)
#     td_step = env.step(TensorDict({"action": action}, batch_size=[BATCH]))
#     print("step:", td_step.shape)
#
#
#     def simple_rollout(env, steps=100):
#         # 先 reset，拿到当前批次大小 B（非 batch-locked 可随时变）
#         _data = env.reset()  # 或者传 env.reset(env.gen_params(batch_size=[B]))
#         B = _data.batch_size  # e.g. torch.Size([1]) 或 torch.Size([B])
#         data = TensorDict({}, [*B, steps])  # 预分配 [B, T] 轨迹容器
#
#         for t in range(steps):
#             # 关键：根据当前批次采样动作 -> 形状 [B, S, Q]
#             _data["action"] = env.action_spec.rand(B).to(env.device)
#
#             _data = env.step(_data)  # 根下产生 "next/*"
#             data[..., t] = _data  # 写入第 t 步
#             _data = step_mdp(_data, keep_other=True)  # 提升 next，为下一步做输入
#
#         return data
#
#
#     rollout_data = simple_rollout(env, steps=100)
#     print(rollout_data)

# 在gpu上测试
if __name__ == "__main__":
    S, Q, J = 3, 2, 16
    device = "cuda"

    network = torch.tensor([[1, 1],
                            [1, 0],
                            [0, 1]], dtype=torch.float32)
    mu = torch.full((S, Q), 1.0)
    h = torch.tensor([1.0, 1.0])

    # 采样器：指数分布
    def draw_service(env, t):
        B = t.size(0)
        return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()

    def draw_inter_arrivals(env, t):
        B = t.size(0)
        return torch.distributions.Exponential(rate=torch.ones(B, Q, device=env.device)).sample()

    env = BatchedDiffDES(network, mu, h, draw_service, draw_inter_arrivals, max_jobs=J, device=device, temp=1.0)
    env.to(device)

    # ✅ 按教程风格动态设 batch
    BATCH = 3
    td = env.reset(env.gen_params(batch_size=[BATCH]))
    print("reset(batch=3):", td.shape, td["queues"].shape, td["queues"].device)

    # ✅ 随机动作放到 GPU
    action = torch.rand(BATCH, S, Q, device=env.device)
    td_step = env.step(TensorDict({"action": action}, batch_size=[BATCH]))
    print("step:", td_step.shape, td_step.get(("next", "queues")).device)


    def simple_rollout(env, steps=100):
        _data = env.reset()
        B = _data.batch_size
        data = TensorDict({}, [*B, steps], device=env.device)

        for t in range(steps):
            # ✅ 正确：先采样，再搬到 env.device
            _data["action"] = env.action_spec.rand(B).to(env.device)
            _data = env.step(_data)
            data[..., t] = _data
            _data = step_mdp(_data, keep_other=True)

        return data


    rollout_data = simple_rollout(env, steps=100)
    print(rollout_data)
