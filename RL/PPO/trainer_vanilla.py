# vanilla_trainer_torchrl.py — TorchRL(0.6) 单环境-多batch，纯 softmax 策略
import math, time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Categorical
from torchrl.envs import EnvBase


# ==========================
# Utilities (SB3-style LR进度)
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
        logits = self.pi(obs).view(B, self.S, self.Q)
        value  = self.v(obs).squeeze(-1)
        return logits, value


# ==========================
# GAE
# ==========================
@torch.no_grad()
def compute_gae(reward, done, value, next_value, gamma, lam):
    T, B = reward.shape
    adv = torch.zeros(T, B, device=reward.device)
    last = torch.zeros(B, device=reward.device)
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = reward[t] + gamma * next_value[t] * mask - value[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    ret = adv + value
    return adv, ret


# ==========================
# Behavior Cloning dataset（vanilla：纯 softmax）
# ==========================
class BCDVanilla(torch.utils.data.Dataset):
    """
    随机生成 obs（队列长度），teacher 概率为 softmax(base)，按 S 行复制。
    """
    def __init__(self, num_samples: int, S: int, Q: int, time_f: bool = False):
        self.num_samples = num_samples
        self.S, self.Q = S, Q
        self.time_f = time_f
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        queues = torch.randint(0, 101, (self.Q,), dtype=torch.float32)
        if self.time_f:
            obs = torch.cat([queues, torch.randint(0, 101, (1,), dtype=torch.float32)], dim=0)
        else:
            obs = queues
        base = F.softmax(queues, dim=-1)            # [Q]
        p = base.unsqueeze(0).repeat(self.S, 1)     # [S,Q]
        return obs, p


# ==========================
# Trainer (Single-Env, batched inside)
# ==========================
@dataclass
class PPOArgs:
    device: str
    obs_dim: int
    S: int
    Q: int
    hidden: int

    # rollout
    episode_steps: int
    train_batch: int
    test_batch: int

    # PPO
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    ppo_epochs: int
    minibatch_size: int
    target_kl: Optional[float]

    # LR (separate)
    lr_policy: float
    lr_value: float
    min_lr_policy: float
    min_lr_value: float
    warmup: float  # 0~1

    # training
    total_epochs: int

    # extras
    normalize_advantage: bool
    rescale_value: bool      # == vanilla_policy.rescale_v
    behavior_cloning: bool
    randomize: bool          # == vanilla_policy.randomize
    time_f: bool             # == vanilla_policy.time_f

    # eval
    eval_every: int
    eval_T: int
    bc_samples: int = 1000
    bc_lr: float = 3e-4


class PPOTrainerTorchRL_Vanilla:
    """
    计算逻辑模仿 vanilla_policy.py（纯 softmax）：
      - probs = softmax(logits)
      - 随机/贪心由 randomize 控制
      - log_prob: 选中概率取 log 并在 S 维求和
      - 可选 value 反标尺
      - 保持 TorchRL 单环境 + 内部多 batch rollout
    """
    def __init__(self,
                 train_env: EnvBase,
                 eval_env: EnvBase,
                 args: PPOArgs,
                 print_fn: Callable[[str], None] = print,
                 ct=None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.args = args
        self.print = print_fn
        self.ct = ct

        device = torch.device(args.device)
        self.policy = ActorCritic(args.obs_dim, args.S, args.Q, args.hidden).to(device)

        # separate optimizers
        self.opt_pi = torch.optim.Adam(self.policy.pi.parameters(), lr=args.lr_policy)
        self.opt_v  = torch.optim.Adam(self.policy.v.parameters(),  lr=args.lr_value)

        # SB3-style progress_remaining: 1.0 → 0.0
        self.total_updates = self._estimate_total_updates()
        self.update_idx = 0

        # rollout returns stats for value rescale
        self.returns_mean = torch.tensor(0.0, device=device)
        self.returns_std  = torch.tensor(1.0, device=device)

        # queue standardization (to mimic vanilla_policy)
        self.mean_queue = torch.tensor(0.0, device=device)
        self.std_queue  = torch.tensor(1.0, device=device)

    # ==== vanilla：队列标准化 ====
    def update_queue_stats(self, mean_queue_length: float, std_queue_length: float):
        self.mean_queue = torch.tensor(float(mean_queue_length), device=self.mean_queue.device)
        self.std_queue  = torch.tensor(max(1e-8, float(std_queue_length)), device=self.std_queue.device)

    def _standardize_queues(self, obs: torch.Tensor) -> torch.Tensor:
        # 若 obs 包含 time_f 的额外维，这里仍整体减均值/除方差（与 vanilla_policy 的简化一致）
        return ((obs - self.mean_queue) / self.std_queue).float()

    # ---------- API ----------
    def pre_train(self):
        if self.args.behavior_cloning:
            self._behavior_cloning()

    def learn(self):
        self.print("Start training (single-env, batched, vanilla)")
        for epoch in range(self.args.total_epochs):
            t0 = time.time()
            traj = self._rollout(self.train_env, self.args.episode_steps, self.args.train_batch)

            # GAE
            adv, ret = compute_gae(traj['rew'], traj['done'],
                                   traj['val'][:-1], traj['val'][1:],
                                   self.args.gamma, self.args.gae_lambda)
            if self.args.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # rollout returns 统计（rescale_v 用）
            with torch.no_grad():
                self.returns_mean = ret.mean()
                self.returns_std  = ret.std().clamp_min(1e-6)

            # flatten
            T, B = self.args.episode_steps, self.args.train_batch
            obs_f  = traj['obs'][:-1].reshape(T*B, self.args.obs_dim)
            act_f  = traj['act'].reshape(T*B, self.args.S, self.args.Q)
            oldlp  = traj['logp'].reshape(T*B)
            adv_f  = adv.reshape(T*B)
            ret_f  = ret.reshape(T*B)

            approx_kl_running = 0.0
            idx = torch.randperm(T*B, device=obs_f.device)
            mb = self.args.minibatch_size

            for _ in range(self.args.ppo_epochs):
                for start in range(0, T*B, mb):
                    mb_idx = idx[start:start+mb]
                    o = obs_f[mb_idx]
                    a = act_f[mb_idx]
                    old_logp = oldlp[mb_idx]
                    adv_mb = adv_f[mb_idx]
                    ret_mb = ret_f[mb_idx]

                    # 前向：标准化后进网络
                    std_o = self._standardize_queues(o)
                    logits, v_pred = self.policy(std_o)

                    # value 反标尺（与 vanilla_policy 的 rescale_v 一致）
                    if self.args.rescale_value:
                        v_pred = v_pred * self.returns_std + self.returns_mean

                    # vanilla 概率：纯 softmax
                    probs = F.softmax(logits, dim=-1)  # [mb,S,Q]

                    # new_logp（多分类独立求和）
                    new_logp = self._log_prob_of(probs, a)

                    ratio = torch.exp(new_logp - old_logp)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # vanilla 可加熵正则（来自纯 softmax）
                    ent = self._entropy(probs).mean()
                    value_loss = F.mse_loss(v_pred, ret_mb)

                    # policy step
                    self.opt_pi.zero_grad(set_to_none=True)
                    (policy_loss - self.args.ent_coef * ent).backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_pi.step()

                    # value step
                    self.opt_v.zero_grad(set_to_none=True)
                    (self.args.vf_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_v.step()

                    # LR schedule（SB3-style）
                    self._lr_step()

                    # 近似 KL（与 SB3 相同）
                    with torch.no_grad():
                        log_ratio = new_logp - old_logp
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).clamp_min(0).item()
                        approx_kl_running = 0.9 * approx_kl_running + 0.1 * approx_kl
                    if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
                        break
                if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
                    break

            self.ct.get_end_time(time.time())
            print(f'get total time {self.ct.get_total_time():.2f}s')
            self.print(f"[Epoch {epoch+1}/{self.args.total_epochs}] , KL~{approx_kl_running:.5f}")

            if (epoch + 1) % self.args.eval_every == 0:
                self.evaluate()
                # self.print(f"  Eval: return mean {mean_r:.4f} ± {std_r:.4f}")

    # ---------- internals ----------
    @torch.no_grad()
    def _rollout(self, env: EnvBase, T: int, B: int) -> Dict[str, torch.Tensor]:
        device = torch.device(self.args.device)
        td = env.reset()
        obs = torch.zeros(T+1, B, self.args.obs_dim, device=device)
        act = torch.zeros(T,   B, self.args.S, self.args.Q, device=device)
        logp= torch.zeros(T,   B, device=device)
        rew = torch.zeros(T,   B, device=device)
        done= torch.zeros(T,   B, device=device)
        val = torch.zeros(T+1, B, device=device)

        obs[0] = td['obs'].to(device)
        for t in range(T):
            std_o_t = self._standardize_queues(obs[t])
            logits, v = self.policy(std_o_t)

            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean

            # vanilla 概率：纯 softmax
            probs = F.softmax(logits, dim=-1)  # [B,S,Q]

            # 随机/贪心
            if self.args.randomize:
                a, lp = self._sample_and_logp(probs)
            else:
                a, lp = self._argmax_and_logp(probs)

            val[t] = v
            act[t] = a
            logp[t] = lp

            out = env.step(TensorDict({"action": a.to(env.device)}, batch_size=[B]))
            nxt = out["next"]
            obs[t + 1] = nxt["obs"].to(device)
            rew[t] = nxt["reward"].reshape(B).to(device)
            done[t] = nxt["done"].reshape(B).to(device)

        std_o_last = self._standardize_queues(obs[-1])
        _, v_last = self.policy(std_o_last)
        if self.args.rescale_value:
            v_last = v_last * self.returns_std + self.returns_mean
        val[-1] = v_last
        return {"obs": obs, "act": act, "logp": logp, "rew": rew, "done": done, "val": val}

    def _sample_and_logp(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, Q = probs.shape
        a = torch.zeros(B, S, Q, device=probs.device)
        logp = torch.zeros(B, device=probs.device)
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            idx = cat.sample()                      # [B]
            a[torch.arange(B), s, idx] = 1.0
            logp += cat.log_prob(idx)
        return a, logp

    def _argmax_and_logp(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, Q = probs.shape
        idx = probs.argmax(dim=-1)                  # [B,S]
        a = F.one_hot(idx, num_classes=Q).float()
        lp = torch.zeros(B, device=probs.device)
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            lp += cat.log_prob(idx[:, s])
        return a, lp

    def _log_prob_of(self, probs: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        idxs = one_hot.argmax(dim=-1)               # [B,S]
        B, S, _ = one_hot.shape
        logp = torch.zeros(B, device=probs.device)
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            logp += cat.log_prob(idxs[:, s])
        return logp

    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        B, S, Q = probs.shape
        ent = 0.0
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            ent = ent + cat.entropy()
        return ent / S

    def _lr_step(self):
        progress = self.update_idx / max(1, self.total_updates - 1)
        progress_remaining = 1.0 - progress
        for pg in self.opt_pi.param_groups:
            pg["lr"] = cosine_with_warmup_sb3_style(self.args.lr_policy, self.args.min_lr_policy,
                                                    progress_remaining, self.args.warmup)
        for pg in self.opt_v.param_groups:
            pg["lr"] = cosine_with_warmup_sb3_style(self.args.lr_value, self.args.min_lr_value,
                                                    progress_remaining, self.args.warmup)
        self.update_idx += 1

    def _estimate_total_updates(self) -> int:
        TnB = self.args.episode_steps * self.args.train_batch
        mb = max(1, self.args.minibatch_size)
        steps_per_epoch = math.ceil(TnB / mb) * self.args.ppo_epochs
        return max(1, steps_per_epoch * self.args.total_epochs)

    # ---------- Behavior Cloning (vanilla) ----------
    # ---------- Behavior Cloning (vanilla) ----------
    def _behavior_cloning(self):
        self.print("[BC] start (vanilla)")
        dataset = BCDVanilla(self.args.bc_samples, self.args.S, self.args.Q, time_f=self.args.time_f)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.test_batch, shuffle=True)
        opt = torch.optim.Adam(self.policy.pi.parameters(), lr=self.args.bc_lr)
        device = torch.device(self.args.device)
        self.policy.train()

        for i, (obs, target) in enumerate(loader):
            # obs: [B, Q(+1)]，target: [B, S, Q]（由 DataLoader 自动堆叠）
            obs = obs.to(device)
            target = target.to(device)

            std_obs = self._standardize_queues(obs)  # [B, obs_dim]
            logits, _ = self.policy(std_obs)  # [B, S, Q]
            pred = F.softmax(logits, dim=-1)  # [B, S, Q]

            # 与 pred 对齐 target 的形状
            if target.dim() == 2:
                # 兼容极端情况下的 [S, Q]
                target_b = target.unsqueeze(0).expand(pred.shape[0], -1, -1)
            elif target.dim() == 3:
                # 正常情况：DataLoader 已堆叠成 [B, S, Q]
                target_b = target
            else:
                raise RuntimeError(f"Unexpected BC target shape: {tuple(target.shape)}")

            loss = F.mse_loss(pred, target_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            opt.step()

        self.print("[BC] done")

    # # ---------- Evaluation ----------
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        device = torch.device(self.args.device)
        td = self.eval_env.reset()
        B, Q = self.args.test_batch, self.args.Q

        total_r = torch.zeros(B, device=device)

        # —— 时间积分累积量
        time_weight_queue_len = torch.zeros(B, Q, device=device)  # ∑(next_queues * dt)
        time_weight_cost = torch.zeros(B, device=device)  # ∑(cost * dt)
        time_now = torch.zeros(B, device=device)  # ∑ dt

        for _ in range(self.args.eval_T):
            # 标准化仅用于策略前向；统计一律用“右端点 next”
            obs_raw = td["obs"].to(device)  # [B, obs_dim]
            std_obs = self._standardize_queues(obs_raw)  # [B, obs_dim]

            # 策略动作（vanilla：softmax）
            logits, v = self.policy(std_obs)
            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean
            probs = F.softmax(logits, dim=-1)  # [B,S,Q]
            if self.args.randomize:
                a, _ = self._sample_and_logp(probs)
            else:
                a, _ = self._argmax_and_logp(probs)

            # 环境前进一步（右端点统计）
            out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
            nxt = out["next"]

            # 奖励累计（保持原有返回）
            r_t = nxt["reward"].reshape(B).to(device)
            total_r += r_t

            # 步长 Δt：优先 event_time；其次 time 差；最后退化为 1
            if "event_time" in nxt.keys():
                dt = nxt["event_time"].reshape(B).to(device)
            elif "time" in nxt.keys() and "time" in td.keys():
                dt = (nxt["time"].reshape(B).to(device) - td["time"].reshape(B).to(device)).clamp_min(0)
            else:
                dt = torch.ones(B, device=device)

            # 右端点队列（next）
            if "queues" in nxt.keys():
                queues_next = nxt["queues"][:, :Q].to(device)  # [B,Q]
            else:
                queues_next = nxt["obs"][:, :Q].to(device)  # [B,Q] 回退

            # 右端点 cost（若无 cost 则用 -reward）
            if "cost" in nxt.keys():
                c_t = nxt["cost"].reshape(B).to(device)
            else:
                c_t = -r_t

            # —— 时间积分累计
            time_weight_queue_len += queues_next * dt.view(B, 1)  # [B,Q]
            time_weight_cost += c_t * dt  # [B]
            time_now += dt  # [B]

            td = nxt

        # —— 每并行环境的“每队列时间平均长度”：[B,Q]
        qlen_per_env = time_weight_queue_len / time_now.view(B, 1).clamp_min(1e-12)
        # 与上面口径一致：对 Q 取均值得到“每环境的平均队列长度”：[B]
        qlen_overall_per_env = qlen_per_env.mean(dim=1)

        # 跨 B 的统计
        q_mean = qlen_overall_per_env.mean().item()
        q_std = qlen_overall_per_env.std(unbiased=True)
        q_se = (q_std / math.sqrt(B)).item()


        # 保持原有的 return 统计/返回
        # ret_mean = total_r.mean().item()
        # ret_std = total_r.std(unbiased=True).item()
        self.print(f"Eval (B={B}): queue length mean (overall): {q_mean:.4f}")
        self.print(f"se (overall): {q_se:.4f}")
        # self.print(
        #     f"  Eval (B={B}): queue length mean (overall): {q_mean:.4f}, SE {q_se:.4f} | "
        #     f"time-avg cost mean {cost_mean:.4f}, SE {cost_se:.4f}"
        # )
        # return ret_mean, ret_std

    # @torch.no_grad()
    # def evaluate(self) -> Tuple[float, float]:
    #     device = torch.device(self.args.device)
    #     td = self.eval_env.reset()
    #     B = self.args.test_batch
    #
    #     total_r = torch.zeros(B, device=device)
    #
    #     # 统计序列（按时间堆叠后展平成 N=T*B）
    #     qlens_list = []   # 每步每样本的 queue length
    #     costs_list = []   # 每步每样本的 cost
    #
    #     for _ in range(self.args.eval_T):
    #         # —— 用原始 obs 统计队列长度；用标准化 obs 进网络
    #         obs_raw = td["obs"].to(device)                 # [B, obs_dim]
    #         std_obs = self._standardize_queues(obs_raw)    # [B, obs_dim]
    #
    #         # queue length：对 obs 的前 Q 维求和
    #         Q = self.args.Q
    #         queues = obs_raw[:, :Q]
    #         qlen_t = queues.sum(dim=-1)                    # [B]
    #         qlens_list.append(qlen_t)
    #
    #         # 策略动作（vanilla：纯 softmax）
    #         logits, v = self.policy(std_obs)
    #         if self.args.rescale_value:
    #             v = v * self.returns_std + self.returns_mean
    #         probs = F.softmax(logits, dim=-1)              # [B,S,Q]
    #
    #         if self.args.randomize:
    #             a, _ = self._sample_and_logp(probs)
    #         else:
    #             a, _ = self._argmax_and_logp(probs)
    #
    #         # 环境一步并记录 reward / cost
    #         out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
    #         nxt = out["next"]
    #         r_t = nxt["reward"].reshape(B).to(device)
    #         total_r += r_t
    #
    #         # cost：若环境提供 "cost" 则优先使用；否则采用常见约定 cost = -reward
    #         if "cost" in nxt.keys():
    #             c_t = nxt["cost"].reshape(B).to(device)
    #         else:
    #             c_t = -r_t
    #         costs_list.append(c_t)
    #
    #         td = nxt
    #
    #     # —— 统计：对所有 T×B 样本求均值与标准误（SE = std / sqrt(N)）
    #     T = self.args.eval_T
    #     N = max(1, T * B)
    #
    #     qlens = torch.stack(qlens_list, dim=0).reshape(-1)   # [N]
    #     costs = torch.stack(costs_list, dim=0).reshape(-1)   # [N]
    #
    #     q_mean = qlens.mean().item()
    #     q_se   = (qlens.std(unbiased=True) / math.sqrt(N)).item()
    #
    #     cost_mean = costs.mean().item()
    #     cost_se   = (costs.std(unbiased=True) / math.sqrt(N)).item()
    #
    #     # 保持原有的 return 统计/返回
    #     ret_mean = total_r.mean().item()
    #     ret_std  = total_r.std(unbiased=True).item()
    #
    #     self.print(f"  Eval (T={T}, B={B}): "
    #                f"queue mean {q_mean:.4f}, SE {q_se:.4f} | "
    #                f"cost mean {cost_mean:.4f}, SE {cost_se:.4f}")
    #
    #     return ret_mean, ret_std

