# pathwise_trainer_torchrl.py — TorchRL(0.6) 单环境-多batch，Pathwise 风格
import math, time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs import EnvBase


# ==========================
# Utilities (SB3-style LR 进度)
# ==========================
def cosine_with_warmup_sb3_style(initial_lr: float, min_lr: float, progress_remaining: float, warmup: float) -> float:
    pr = max(0.0, min(1.0, progress_remaining))
    if pr > (1.0 - warmup):
        warmup_progress = (1.0 - pr) / max(1e-12, warmup)
        return min_lr + (initial_lr - min_lr) * warmup_progress
    adj = (pr - (1.0 - warmup)) / max(1e-12, (1.0 - warmup))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * adj))
    return min_lr + (initial_lr - min_lr) * cos_decay


# ==========================
# Networks
# ==========================
class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out),
        )
    def forward(self, x): return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, S: int, Q: int, hidden: int):
        super().__init__()
        self.S, self.Q = S, Q
        self.pi = MLP(obs_dim, hidden, S * Q)
        self.v  = MLP(obs_dim, hidden, 1)
    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        logits = self.pi(obs).view(B, self.S, self.Q)  # [B,S,Q]
        value  = self.v(obs).squeeze(-1)               # [B]
        return logits, value


# ==========================
# Config
# ==========================
@dataclass
class PathwiseArgs:
    device: str
    obs_dim: int
    S: int
    Q: int
    hidden: int

    # rollout
    episode_steps: int
    train_batch: int
    test_batch: int

    # discounts
    gamma: float

    # opt
    max_grad_norm: float
    lr_policy: float
    lr_value: float
    min_lr_policy: float
    min_lr_value: float
    warmup: float            # 0~1
    total_epochs: int

    # extras
    behavior_cloning: bool   # 可选：预训练
    bc_samples: int
    bc_lr: float

    # eval
    eval_every: int
    eval_T: int
    randomize: bool          # 评估是否采样（否则贪心）

    # pathwise
    tau: float               # 温度
    rescale_value: bool      # 是否把 value 反标尺到 returns 量级
    cost_is_negative_reward: bool  # 环境 reward 是否表示 -cost（常见做法：cost = -reward）


# ==========================
# Trainer (Pathwise)
# ==========================
class PathwiseTrainerTorchRL:
    """
    - 训练：Pathwise/可微，传连续动作(容量分配)给环境；policy_loss = 折扣代价均值
    - 值函数：与策略图断开，用返回的 detach 版做 MSE
    - 概率：WC-Softmax（温度 τ，1{x>0} 与 network 掩码；不与 obs 取 min）
    - 环境：TorchRL 单环境，但内部 batch=B（env.step 接收 [B,S,Q] 动作；推荐环境保持可微）
    """
    def __init__(self,
                 train_env: EnvBase,
                 eval_env: EnvBase,
                 args: PathwiseArgs,
                 network_mask: torch.Tensor,   # [S,Q] in {0,1}
                 print_fn: Callable[[str], None] = print,
                 ct=None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.args = args
        self.net_mask = network_mask.float()
        self.print = print_fn
        self.ct = ct

        device = torch.device(args.device)
        self.policy = ActorCritic(args.obs_dim, args.S, args.Q, args.hidden).to(device)

        # 两个优化器（策略/价值分离）
        self.opt_pi = torch.optim.Adam(self.policy.pi.parameters(), lr=args.lr_policy)
        self.opt_v  = torch.optim.Adam(self.policy.v.parameters(),  lr=args.lr_value)

        # SB3-style 进度：1→0
        self.total_updates = self.args.total_epochs
        self.update_idx = 0

        # value 反标尺（与 pathwise 代码的 returns_mean/std 用法对齐）
        self.returns_mean = torch.tensor(0.0, device=device)
        self.returns_std  = torch.tensor(1.0, device=device)

        # 观测标准化（队列均值/方差）
        self.mean_queue = torch.tensor(0.0, device=device)
        self.std_queue  = torch.tensor(1.0, device=device)

        # ---- 预计算 discounts 缓存（改进点 4）----
        self._discounts = None

    # ====== 队列标准化（与 pathwise/vanilla 一致的接口） ======
    def update_queue_stats(self, mean_queue_length: float, std_queue_length: float):
        self.mean_queue = torch.tensor(float(mean_queue_length), device=self.mean_queue.device)
        self.std_queue  = torch.tensor(max(1e-8, float(std_queue_length)), device=self.std_queue.device)

    def _standardize(self, obs: torch.Tensor) -> torch.Tensor:
        return ((obs - self.mean_queue) / self.std_queue).float()

    # ---- 折扣缓存获取（改进点 4）----
    def _get_discounts(self, T: int, device: torch.device) -> torch.Tensor:
        if (self._discounts is None) or (self._discounts.numel() != T) or (self._discounts.device != device):
            self._discounts = torch.pow(torch.tensor(self.args.gamma, device=device),
                                        torch.arange(T, device=device))
        return self._discounts

    # ====== WC-Softmax（更稳的 masked logsumexp 写法；改进点 2） ======
    def _wc_softmax(self, logits: torch.Tensor, obs: torch.Tensor, tau: float, eps: float = 1e-8) -> torch.Tensor:
        """
        numerator/denominator 通过对不可行位置置 -inf，再 log_softmax 得到；
        若某行全不可行，回退为网络掩码上的均匀分布（避免 NaN）。
        """
        B, S, Q = logits.shape
        pos_mask = (obs[:, :Q] > 0).unsqueeze(1)                    # [B,1,Q] (bool)
        net_mask = (self.net_mask > 0).to(logits.device).unsqueeze(0)  # [1,S,Q] (bool)
        feas = pos_mask & net_mask                                   # [B,S,Q] (bool)

        scaled = logits / max(tau, 1e-6)
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(feas, scaled, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
        logp = F.log_softmax(masked, dim=-1)                         # [B,S,Q]
        p = logp.exp()                                                # [B,S,Q]

        # 行全不可行时（和为0），回退到网络掩码均匀分布
        row_sum = p.sum(dim=-1, keepdim=True)                        # [B,S,1]
        fallback = net_mask.float() / net_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)  # [1,S,1]→[1,S,1]
        fallback = fallback.expand_as(p)
        return torch.where(row_sum > 0, p, fallback)

    # ====== Policy/Value 前向（含可选 value 反标尺） ======
    def _pi_v(self, obs: torch.Tensor):
        logits, v = self.policy(obs)
        if self.args.rescale_value:
            v = v * self.returns_std + self.returns_mean
        return logits, v

    # ====== 单个 epoch 的 pathwise 训练 ======
    def _train_one_epoch_pathwise(self):
        device = torch.device(self.args.device)
        B = self.args.train_batch
        T = self.args.episode_steps
        self.policy.train()

        # reset & 初始 obs
        td = self.train_env.reset()
        obs = td["obs"]                 # 已在 GPU，无需 .to(device)

        # 轨迹缓存（用于 value 训练；与策略图断开）
        obs_traj = []
        cost_traj = []

        # ===== rollout（连续动作，可微）=====
        for t in range(T):
            obs_traj.append(obs.detach())          # 断图，留给 value 分支

            std_obs = self._standardize(obs)       # 与 pathwise/vanilla 一致的标准化
            logits, _ = self._pi_v(std_obs)        # logits:[B,S,Q]
            probs = self._wc_softmax(logits, obs, tau=self.args.tau)  # [B,S,Q]
            action = probs                          # 训练时直接把概率当“容量分配”传环境（连续）

            # 环境一步（保持同一 GPU；改进点 6：无多余 .to(...)）
            out = self.train_env.step(TensorDict({"action": action}, batch_size=[B]))
            nxt = out["next"]
            obs_next = nxt["obs"]                  # [B, obs_dim]
            rew = nxt["reward"].reshape(B)

            # 代价
            step_cost = -rew if self.args.cost_is_negative_reward else rew
            cost_traj.append(step_cost)            # [B]

            obs = obs_next

        # ===== 策略损失：折扣代价均值（直接从 cost 反传）=====
        ep_cost = torch.stack(cost_traj, dim=1)    # [B,T]
        discounts = self._get_discounts(T, device) # 改进点 4
        discounted_cost = (ep_cost * discounts.unsqueeze(0)).sum(dim=1)   # [B]
        policy_loss = discounted_cost.mean()       # 最小化 cost → 最大化 reward

        # ---- policy step（分头裁剪；改进点 6）----
        self.opt_pi.zero_grad(set_to_none=True)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.pi.parameters(), self.args.max_grad_norm)
        self.opt_pi.step()

        # ===== 值函数损失：用“与策略图断开”的目标 =====
        with torch.no_grad():
            # 从 ep_cost(detached) 计算 returns（逐时刻 G_t）
            G = []
            running = torch.zeros(B, device=device)
            ep_cost_detach = ep_cost.detach()
            for t in reversed(range(T)):
                running = ep_cost_detach[:, t] + self.args.gamma * running
                G.append(running)
            G = torch.stack(list(reversed(G)), dim=1)  # [B,T]

            # 用本轮 returns 的统计更新反标尺参数（若启用）
            self.returns_mean = G.mean()
            self.returns_std  = G.std().clamp_min(1e-6)

        # ---- value step：一次性前向（改进点 5）----
        self.opt_v.zero_grad(set_to_none=True)

        obs_tb = torch.stack(obs_traj, dim=0)                 # [T,B,obs_dim]
        std_obs_tb = self._standardize(obs_tb)                # [T,B,obs_dim]
        # 仅取 value 分支做一次性前向
        _, v_tb = self.policy(std_obs_tb.view(-1, self.args.obs_dim))  # v_tb: [T*B]
        v_tb = v_tb.view(obs_tb.shape[0], obs_tb.shape[1])    # [T,B]

        if self.args.rescale_value:
            v_tb = v_tb * self.returns_std + self.returns_mean

        value_loss = F.mse_loss(v_tb, G.transpose(0, 1))      # [T,B] vs [T,B]
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.v.parameters(), self.args.max_grad_norm)
        self.opt_v.step()

        return policy_loss.item(), value_loss.item()

    # ====== 学习主循环 ======
    def learn(self):
        self.print("Start training (Pathwise, single-env, batched)")
        for epoch in range(self.args.total_epochs):
            t0 = time.time()

            # LR 调度（SB3 风格 progress_remaining）
            self._lr_step()

            # 改进点 3：移除 set_detect_anomaly(True)（该开关会极大拖慢训练）

            pl, vl = self._train_one_epoch_pathwise()

            self.ct.get_end_time(time.time())
            print(f'get total time {self.ct.get_total_time():.2f}s')
            self.print(f"[Epoch {epoch+1}/{self.args.total_epochs}] "
                       f"policy_loss={pl:.5f}, value_loss={vl:.5f}")

            if (epoch + 1) % self.args.eval_every == 0:
                self.evaluate()

    # ====== 评估（离散动作：采样/贪心，向量化；改进点 1） ======
    @torch.no_grad()
    def evaluate(self):
        device = torch.device(self.args.device)
        td = self.eval_env.reset()
        B, Q = self.args.test_batch, self.args.Q

        total_r = torch.zeros(B, device=device)

        # —— 时间积分累计（右端点）
        time_weight_queue_len = torch.zeros(B, Q, device=device)  # ∑(next_queues * dt)
        time_now = torch.zeros(B, device=device)  # ∑ dt

        for _ in range(self.args.eval_T):
            # 标准化仅用于策略前向；统计一律用“右端点 next”
            obs = td["obs"]                                # 已在 GPU
            std_obs = self._standardize(obs)

            # 策略动作
            logits, _ = self._pi_v(std_obs)
            probs = self._wc_softmax(logits, obs, tau=self.args.tau)  # [B,S,Q]

            if self.args.randomize:
                B_, S_, Q_ = probs.shape
                idx = torch.multinomial(probs.view(-1, Q_), 1).view(B_, S_)   # [B,S]
                a = F.one_hot(idx, num_classes=Q_).float()                    # [B,S,Q]
            else:
                idx = probs.argmax(dim=-1)                                    # [B,S]
                a = F.one_hot(idx, num_classes=self.args.Q).float()

            # 环境前进一步（右端点统计）；无多余 .to(...)（改进点 1/6）
            out = self.eval_env.step(TensorDict({"action": a}, batch_size=[B]))
            nxt = out["next"]

            # 奖励累计（保持原有返回）
            r_t = nxt["reward"].reshape(B)
            total_r += r_t

            # 步长 Δt：event_time > time 差分 > 1
            dt = nxt["event_time"].reshape(B)
            queues_next = nxt["obs"][:, :Q]     # [B,Q]
            # print(f'queue next {queues_next}, shap {queues_next.shape}')
            # if "event_time" in nxt.keys():
            #     dt = nxt["event_time"].reshape(B)
            # elif "time" in nxt.keys() and "time" in td.keys():
            #     dt = (nxt["time"].reshape(B) - td["time"].reshape(B)).clamp_min(0)
            # else:
            #     dt = torch.ones(B, device=device)
            #
            # # 右端点队列：优先 next["queues"]，否则从 next["obs"] 取前 Q 维
            # if "queues" in nxt.keys():
            #     queues_next = nxt["queues"][:, :Q]  # [B,Q]
            # else:
            #     queues_next = nxt["obs"][:, :Q]     # [B,Q]

            # —— 时间积分累计
            time_weight_queue_len += queues_next * dt.view(B, 1)  # [B,Q]
            time_now += dt  # [B]

            td = nxt

        # 每并行环境的“每队列时间平均长度”：[B,Q]
        qlen_per_env = time_weight_queue_len / time_now.view(B, 1).clamp_min(1e-12)
        qlen_overall_per_env = qlen_per_env.mean(dim=1)  # [B]

        # 跨 B 的统计
        q_mean = qlen_overall_per_env.mean().item()
        q_se = (qlen_overall_per_env.std(unbiased=True) / math.sqrt(B)).item()

        # 保持原有的 return 统计/返回
        ret_mean = total_r.mean().item()
        ret_std = total_r.std(unbiased=True).item()

        self.print(f"Eval (B={B}): queue length mean (overall): {q_mean:.4f}")
        self.print(f"se (overall): {q_se:.4f}")
        # return ret_mean, ret_std

    # ====== （可选）BC：teacher = WC-Softmax(softmax(logits/tau), 仅网络&正队列掩码) ======
    def pre_train(self):
        if not self.args.behavior_cloning:
            return
        self.print("[BC] start (pathwise teacher)")
        device = torch.device(self.args.device)
        opt = torch.optim.Adam(self.policy.pi.parameters(), lr=self.args.bc_lr)
        self.policy.train()

        # 简单随机合成 batch（你也可以替换成自己的数据）
        for _ in range(max(1, self.args.bc_samples // self.args.test_batch)):
            # 随机 obs（Q 或 Q+额外特征），这里假设 obs_dim==Q 或包含其它特征时，队列在前 Q 维
            obs = torch.randint(0, 101, (self.args.test_batch, self.args.obs_dim), device=device).float()
            std_obs = self._standardize(obs)
            logits, _ = self._pi_v(std_obs)
            with torch.no_grad():
                target = self._wc_softmax(logits, obs, tau=self.args.tau)  # [B,S,Q]

            pred_logits, _ = self._pi_v(std_obs)
            pred = self._wc_softmax(pred_logits, obs, tau=self.args.tau)
            loss = F.mse_loss(pred, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            opt.step()
        self.print("[BC] done")

    # ====== LR 调度 ======
    def _lr_step(self):
        progress = self.update_idx / max(1, self.total_updates - 1)
        pr = 1.0 - progress
        for pg in self.opt_pi.param_groups:
            pg["lr"] = cosine_with_warmup_sb3_style(self.args.lr_policy, self.args.min_lr_policy, pr, self.args.warmup)
        for pg in self.opt_v.param_groups:
            pg["lr"] = cosine_with_warmup_sb3_style(self.args.lr_value, self.args.min_lr_value, pr, self.args.warmup)
        self.update_idx += 1
