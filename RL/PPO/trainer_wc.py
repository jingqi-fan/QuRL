# A_revised.py — TorchRL(0.6) 单环境-多batch，计算逻辑贴近你的实现
import math, time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Categorical, OneHotCategorical
from torchrl.envs import EnvBase


# ==========================
# Utilities
# ==========================
def cosine_with_warmup_sb3_style(initial_lr: float, min_lr: float, progress_remaining: float, warmup: float) -> float:
    """
    SB3风格的进度：progress_remaining 从 1.0 线性衰减到 0.0
    - 在 warmup 阶段(靠近1.0的一段) 线性升至 initial_lr
    - 之后余弦衰减到 min_lr
    """
    pr = max(0.0, min(1.0, progress_remaining))
    # warmup 区：pr ∈ (1-warmup, 1] —— 注意 warmup 是比例
    if pr > (1.0 - warmup):
        warmup_progress = (1.0 - pr) / max(1e-12, warmup)
        return min_lr + (initial_lr - min_lr) * warmup_progress
    # 余弦段：把 pr 映射到 [0,1]
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
    def forward(self, x):
        return self.net(x)


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
    """Shapes: reward/done/value/next_value -> [T,B]. Return adv, ret both [T,B]."""
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
# 行为克隆数据（按你的概率构造规则）
# ==========================
class BCD(torch.utils.data.Dataset):
    """
    生成随机 obs（队列长度），并用“WC_Policy 风格”的规则构造 teacher 概率：
      p = softmax(base) * network
      p = min(p, obs)
      若全零行 → 回退到 network
      行归一化
    """
    def __init__(self, num_samples: int, network: torch.Tensor, time_f: bool = False):
        net = network if isinstance(network, torch.Tensor) else torch.tensor(network, dtype=torch.float32)
        assert net.dim() == 2, f"network must be 2D (s,q), got {net.shape}"
        self.network = net.float()
        self.s, self.q = self.network.shape
        self.num_samples = num_samples
        self.time_f = time_f

    def __len__(self): return self.num_samples

    def __getitem__(self, idx):
        # obs_raw: [q] or [q+1]（最后一维可作为 time 特征；若 time_f=True，动作分布只看前 q）
        queues = torch.randint(0, 101, (self.q,), dtype=torch.float32)
        if self.time_f:
            obs = torch.cat([queues, torch.randint(0, 101, (1,), dtype=torch.float32)], dim=0)
        else:
            obs = queues

        base = F.softmax(queues, dim=-1)              # [q]
        p = base.unsqueeze(0).repeat(self.s, 1)       # [s,q]
        p = p * self.network
        p = torch.minimum(p, queues.unsqueeze(0).repeat(self.s, 1))
        zero_row = torch.all(p == 0, dim=1, keepdim=True).float()
        p = p + zero_row * self.network
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
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
    warmup: float  # 比例（0~1）

    # training
    total_epochs: int

    # extras
    normalize_advantage: bool
    rescale_value: bool      # == your rescale_v
    behavior_cloning: bool
    randomize: bool          # == WC_Policy.randomize
    time_f: bool             # == WC_Policy.time_f

    # eval
    eval_every: int
    eval_T: int
    bc_samples: int = 1000
    bc_lr: float = 3e-4


class PPOTrainerTorchRL:
    def __init__(self,
                 train_env: EnvBase,
                 eval_env: EnvBase,
                 args: PPOArgs,
                 network_mask: torch.Tensor,
                 print_fn: Callable[[str], None] = print,
                 ct=None):
        self.train_env = train_env
        self.eval_env = eval_env
        self.args = args
        self.network_mask = network_mask.float()
        self.print = print_fn
        self.ct = ct

        self.device = torch.device(args.device)
        self.policy = ActorCritic(args.obs_dim, args.S, args.Q, args.hidden).to(self.device)

        # 优化器
        self.opt_pi = torch.optim.Adam(self.policy.pi.parameters(), lr=args.lr_policy)
        self.opt_v = torch.optim.Adam(self.policy.v.parameters(), lr=args.lr_value)

        self.total_updates = self._estimate_total_updates()
        self.update_idx = 0

        self.returns_mean = torch.tensor(0.0, device=self.device)
        self.returns_std = torch.tensor(1.0, device=self.device)

    # ---------- API ----------
    def pre_train(self):
        if self.args.behavior_cloning:
            self._behavior_cloning()

    # def learn(self):
    #     self.print("Start training (single-env, batched)")
    #     for epoch in range(self.args.total_epochs):
    #         t0 = time.time()
    #         traj = self._rollout(self.train_env, self.args.episode_steps, self.args.train_batch)
    #
    #         # SB3 等价 GAE
    #         adv, ret = compute_gae(traj['rew'], traj['done'],
    #                                traj['val'][:-1], traj['val'][1:],
    #                                self.args.gamma, self.args.gae_lambda)
    #         if self.args.normalize_advantage:
    #             adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    #
    #         # 用本轮 rollout 的 returns 统计来做 value 重标尺（与你一致）
    #         with torch.no_grad():
    #             self.returns_mean = ret.mean()
    #             self.returns_std  = ret.std().clamp_min(1e-6)
    #
    #         # flatten to [T*B]
    #         T, B = self.args.episode_steps, self.args.train_batch
    #         obs_f  = traj['obs'][:-1].reshape(T*B, self.args.obs_dim)
    #         act_f  = traj['act'].reshape(T*B, self.args.S, self.args.Q)
    #         oldlp  = traj['logp'].reshape(T*B)
    #         adv_f  = adv.reshape(T*B)
    #         ret_f  = ret.reshape(T*B)
    #
    #         # PPO updates（与 SB3 对齐：approx_kl、entropy、进度/调度）
    #         approx_kl_running = 0.0
    #         idx = torch.randperm(T*B, device=obs_f.device)
    #         mb = self.args.minibatch_size
    #
    #         for _ in range(self.args.ppo_epochs):
    #             for start in range(0, T*B, mb):
    #                 mb_idx = idx[start:start+mb]
    #                 o = obs_f[mb_idx]
    #                 a = act_f[mb_idx]
    #                 old_logp = oldlp[mb_idx]
    #                 adv_mb = adv_f[mb_idx]
    #                 ret_mb = ret_f[mb_idx]
    #
    #                 logits, v_pred = self.policy(o)
    #
    #                 # value 反标尺（与你的 WC_policy.rescale_v 一样）
    #                 if self.args.rescale_value:
    #                     v_pred = v_pred * self.returns_std + self.returns_mean
    #
    #                 # --- 分布：softmax → *network → min(·, obsQ) → 零行回退 → 归一化
    #                 probs = self._wc_probs_from_logits_obs(logits, o)
    #
    #                 # new_logp（多分类独立的和）
    #                 new_logp = self._log_prob_of(probs, a)
    #
    #                 ratio = torch.exp(new_logp - old_logp)
    #                 surr1 = ratio * adv_mb
    #                 surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * adv_mb
    #                 policy_loss = -torch.min(surr1, surr2).mean()
    #
    #                 # 熵（按 S 求和后取均值）
    #                 ent = self._entropy(probs).mean()
    #
    #                 value_loss = F.mse_loss(v_pred, ret_mb)
    #
    #                 # -- policy step
    #                 self.opt_pi.zero_grad(set_to_none=True)
    #                 (policy_loss - self.args.ent_coef * ent).backward(retain_graph=True)
    #                 nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
    #                 self.opt_pi.step()
    #
    #                 # -- value step
    #                 self.opt_v.zero_grad(set_to_none=True)
    #                 (self.args.vf_coef * value_loss).backward()
    #                 nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
    #                 self.opt_v.step()
    #
    #                 # LR schedule（SB3风格 progress_remaining）
    #                 self._lr_step()
    #
    #                 # KL 近似（SB3公式）
    #                 with torch.no_grad():
    #                     log_ratio = new_logp - old_logp
    #                     approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).clamp_min(0).item()
    #                     approx_kl_running = 0.9 * approx_kl_running + 0.1 * approx_kl
    #                 if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
    #                     break
    #             if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
    #                 break
    #
    #         self.ct.get_end_time(time.time())
    #         print(f'get total time {self.ct.get_total_time():.2f}s')
    #         self.print(f"[Epoch {epoch+1}/{self.args.total_epochs}] , KL~{approx_kl_running:.5f}")
    #
    #         if (epoch + 1) % self.args.eval_every == 0:
    #             self.evaluate()
    #             # self.print(f"  Eval: return mean {mean_r:.4f} ± {std_r:.4f}")

    # # ---------- internals ----------
    # @torch.no_grad()
    # def _rollout(self, env: EnvBase, T: int, B: int) -> Dict[str, torch.Tensor]:
    #     device = torch.device(self.args.device)
    #     td = env.reset()
    #     obs = torch.zeros(T+1, B, self.args.obs_dim, device=device)
    #     act = torch.zeros(T,   B, self.args.S, self.args.Q, device=device)
    #     logp= torch.zeros(T,   B, device=device)
    #     rew = torch.zeros(T,   B, device=device)
    #     done= torch.zeros(T,   B, device=device)
    #     val = torch.zeros(T+1, B, device=device)
    #
    #     obs[0] = td['obs'].to(device)
    #     for t in range(T):
    #         logits, v = self.policy(obs[t])
    #
    #         # value 反标尺
    #         if self.args.rescale_value:
    #             v = v * self.returns_std + self.returns_mean
    #
    #         # 概率
    #         probs = self._wc_probs_from_logits_obs(logits, obs[t])
    #
    #         # 采样/贪心（与 WC_Policy.randomize 一致）
    #         if self.args.randomize:
    #             dist = OneHotCategorical(probs=probs)  # 一次性 over [B,S,Q] → 需按 S 拆分，简化用逐S采样：
    #             # OneHotCategorical 只支持最后一维，这里手动逐S采样：
    #             a, lp = self._sample_and_logp(probs)
    #         else:
    #             a, lp = self._argmax_and_logp(probs)
    #
    #         val[t] = v
    #         act[t] = a
    #         logp[t] = lp
    #
    #         out = env.step(TensorDict({"action": a.to(env.device)}, batch_size=[B]))
    #         nxt = out["next"]
    #         obs[t + 1] = nxt["obs"].to(device)
    #         rew[t] = nxt["reward"].reshape(B).to(device)
    #         done[t] = nxt["done"].reshape(B).to(device)
    #
    #     _, v_last = self.policy(obs[-1])
    #     if self.args.rescale_value:
    #         v_last = v_last * self.returns_std + self.returns_mean
    #     val[-1] = v_last
    #     return {"obs": obs, "act": act, "logp": logp, "rew": rew, "done": done, "val": val}

    # ---------- Learn ----------
    def learn(self):
        self.print("Start training (TorchRL GPU-optimized)")
        for epoch in range(self.args.total_epochs):
            t0 = time.time()

            traj = self._rollout(self.train_env, self.args.episode_steps, self.args.train_batch)
            adv, ret = compute_gae(traj["rew"], traj["done"],
                                   traj["val"][:-1], traj["val"][1:],
                                   self.args.gamma, self.args.gae_lambda)
            if self.args.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            with torch.no_grad():
                self.returns_mean = ret.mean()
                self.returns_std = ret.std().clamp_min(1e-6)

            T, B = self.args.episode_steps, self.args.train_batch
            obs_f = traj["obs"][:-1].reshape(T * B, self.args.obs_dim)
            act_f = traj["act"].reshape(T * B, self.args.S, self.args.Q)
            oldlp = traj["logp"].reshape(T * B)
            adv_f = adv.reshape(T * B)
            ret_f = ret.reshape(T * B)

            idx = torch.randperm(T * B, device=self.device)
            mb = self.args.minibatch_size

            approx_kl_running = 0.0
            for _ in range(self.args.ppo_epochs):
                for start in range(0, T * B, mb):
                    mb_idx = idx[start:start + mb]
                    o = obs_f[mb_idx]
                    a = act_f[mb_idx]
                    old_logp = oldlp[mb_idx]
                    adv_mb = adv_f[mb_idx]
                    ret_mb = ret_f[mb_idx]

                    logits, v_pred = self.policy(o)
                    if self.args.rescale_value:
                        v_pred = v_pred * self.returns_std + self.returns_mean

                    probs = self._wc_probs_from_logits_obs(logits, o)
                    new_logp = self._log_prob_of(probs, a)

                    ratio = torch.exp(new_logp - old_logp)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    ent = self._entropy(probs).mean()
                    value_loss = F.mse_loss(v_pred, ret_mb)

                    self.opt_pi.zero_grad(set_to_none=True)
                    (policy_loss - self.args.ent_coef * ent).backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_pi.step()

                    self.opt_v.zero_grad(set_to_none=True)
                    (self.args.vf_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_v.step()

                    self._lr_step()

                    with torch.no_grad():
                        log_ratio = new_logp - old_logp
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).clamp_min(0).item()
                        approx_kl_running = 0.9 * approx_kl_running + 0.1 * approx_kl

                if self.args.target_kl and approx_kl_running > 1.5 * self.args.target_kl:
                    break

            self.ct.get_end_time(time.time())
            print(f"get total time {self.ct.get_total_time():.2f}s")
            self.print(f"[Epoch {epoch + 1}/{self.args.total_epochs}] KL~{approx_kl_running:.5f}")

            if (epoch + 1) % self.args.eval_every == 0:
                self.evaluate()


    # ---------- Rollout ----------
    @torch.no_grad()
    def _rollout(self, env: EnvBase, T: int, B: int) -> Dict[str, torch.Tensor]:
        """
        完全 GPU 版 rollout：
        - 环境应已在 CUDA 上运行 (env.device == 'cuda:0')
        - 不再重复 .to(device)
        """
        device = self.device
        td = env.reset()

        # 确保环境返回的数据与 policy 在同一 device
        assert td.device == device or str(td.device) == str(device), \
            f"Env and model on different devices: env={td.device}, model={device}"

        obs = torch.zeros(T + 1, B, self.args.obs_dim, device=device)
        act = torch.zeros(T, B, self.args.S, self.args.Q, device=device)
        logp = torch.zeros(T, B, device=device)
        rew = torch.zeros(T, B, device=device)
        done = torch.zeros(T, B, device=device)
        val = torch.zeros(T + 1, B, device=device)

        obs[0] = td["obs"]  # 不再 .to(device)

        for t in range(T):
            logits, v = self.policy(obs[t])
            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean

            probs = self._wc_probs_from_logits_obs(logits, obs[t])
            if self.args.randomize:
                a, lp = self._sample_and_logp(probs)
            else:
                a, lp = self._argmax_and_logp(probs)

            val[t] = v
            act[t] = a
            logp[t] = lp

            # GPU TensorDict step，无需拷贝
            td = env.step(TensorDict({"action": a}, batch_size=[B]))
            nxt = td["next"]

            obs[t + 1] = nxt["obs"]
            rew[t] = nxt["reward"].reshape(B)
            done[t] = nxt["done"].reshape(B)

        _, v_last = self.policy(obs[-1])
        if self.args.rescale_value:
            v_last = v_last * self.returns_std + self.returns_mean
        val[-1] = v_last

        return {"obs": obs, "act": act, "logp": logp, "rew": rew, "done": done, "val": val}

    def _wc_probs_from_logits_obs(self, logits: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
        与 WC_Policy 一致的概率构造：
          action = logits.view(B,S,Q) → softmax(dim=-1)
          * network_mask
          min(·, queues)     (若 time_f=True，queues = obs[..., :Q])
          若一行全 0 → 回退 network_mask
          行归一化
        """
        B, S, Q = logits.shape
        probs = F.softmax(logits, dim=-1)                  # [B,S,Q]
        probs = probs * self.network_mask.to(probs.device) # 掩码
        # 取 obs 的队列部分
        if self.args.time_f:
            queues = obs[:, :Q]
        else:
            queues = obs[:, :Q]  # 若 obs 仅 Q 维，这里等价
        probs = torch.minimum(probs, queues.view(B, 1, Q).repeat(1, S, 1))
        zero_mask = torch.all(probs == 0, dim=-1, keepdim=True).float()
        probs = probs + zero_mask * self.network_mask.to(probs.device)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _sample_and_logp(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        独立 S 个 Categorical 的采样与 log_prob 和（与你代码一致）
        """
        B, S, Q = probs.shape
        a = torch.zeros(B, S, Q, device=probs.device)
        logp = torch.zeros(B, device=probs.device)
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            idx = cat.sample()            # [B]
            a[torch.arange(B), s, idx] = 1.0
            logp += cat.log_prob(idx)
        return a, logp

    def _argmax_and_logp(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        贪心动作与对应 log_prob（按当前分布取 argmax）
        """
        B, S, Q = probs.shape
        idx = probs.argmax(dim=-1)  # [B,S]
        a = F.one_hot(idx, num_classes=Q).float()
        # log_prob of chosen (deterministic) actions under probs
        lp = torch.zeros(B, device=probs.device)
        for s in range(S):
            cat = Categorical(probs=probs[:, s, :])
            lp += cat.log_prob(idx[:, s])
        return a, lp

    def _log_prob_of(self, probs: torch.Tensor, one_hot: torch.Tensor) -> torch.Tensor:
        idxs = one_hot.argmax(dim=-1)  # [B,S]
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
        return ent / S  # 平均每个 S 的熵

    def _lr_step(self):
        # SB3风格：progress_remaining = 1 - progress
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

    # ---------- Behavior Cloning ----------
    def _behavior_cloning(self):
        self.print("[BC] start")
        assert isinstance(self.network_mask, torch.Tensor) and self.network_mask.dim() == 2, \
            "BC requires a [S,Q] network mask"
        dataset = BCD(self.args.bc_samples, self.network_mask, time_f=self.args.time_f)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.test_batch, shuffle=True)
        opt = torch.optim.Adam(self.policy.pi.parameters(), lr=self.args.bc_lr)
        device = torch.device(self.args.device)
        self.policy.train()
        for i, (obs, target) in enumerate(loader):
            obs = obs.to(device)                    # [B, q(+1)]
            target = target.to(device)              # [S,Q]
            logits, _ = self.policy(obs)            # [B,S,Q]
            pred = self._wc_probs_from_logits_obs(logits, obs)  # 用与你一致的构造拿到概率
            target_b = target.unsqueeze(0).expand(pred.shape[0], -1, -1)
            loss = F.mse_loss(pred, target_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            opt.step()
        self.print("[BC] done")

    # ---------- Evaluation ----------
    @torch.no_grad()
    def evaluate(self):
        """
          - 使用 next.queues × next.event_time 做时间积分（右连续）
          - 先得到每队列的时间平均，再对 Q 求均值（每队列平均长度）
          - 跨 B 计算 mean/std/se
        """
        device = torch.device(self.args.device)
        td = self.eval_env.reset()
        B, Q = self.args.test_batch, self.args.Q

        total_r = torch.zeros(B, device=device)

        time_weight_queue_len = torch.zeros(B, Q, device=device)  # [B,Q]
        time_now = torch.zeros(B, device=device)  # [B]

        for _ in range(self.args.eval_T):
            obs = td["obs"].to(device)  # [B, obs_dim]

            # 策略动作（与训练/原评估一致）
            logits, v = self.policy(obs)
            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean
            probs = self._wc_probs_from_logits_obs(logits, obs)
            if self.args.randomize:
                a, _ = self._sample_and_logp(probs)
            else:
                a, _ = self._argmax_and_logp(probs)

            # 环境前进一步（右端点统计）
            out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
            nxt = out["next"]

            # 累计奖励（保持原来的返回/打印习惯）
            r_t = nxt["reward"].reshape(B).to(device)
            total_r += r_t

            # 右端点：使用 next 的队列
            if "queues" in nxt.keys():
                queues_next = nxt["queues"][:, :Q].to(device)  # [B,Q]
            else:
                # 若环境未显式提供 queues，就从 next.obs 里取前 Q 维
                queues_next = nxt["obs"][:, :Q].to(device)

            # 步长 Δt：优先 next.event_time，其次用 time 差，最后退化为 1
            if "event_time" in nxt.keys():
                dt = nxt["event_time"].reshape(B).to(device)  # [B]
            elif "time" in nxt.keys() and "time" in td.keys():
                dt = (nxt["time"].reshape(B).to(device) - td["time"].reshape(B).to(device)).clamp_min(0)
            else:
                dt = torch.ones(B, device=device)

            # 时间积分：与之前 test_epoch 完全一致的形式
            time_weight_queue_len += queues_next * dt.view(B, 1)  # [B,Q]
            time_now += dt  # [B]

            td = nxt

        # 每个并行环境、每个队列的时间平均队列长度：[B,Q]
        qlen_per_env = time_weight_queue_len / time_now.view(B, 1).clamp_min(1e-12)

        # 与“之前”一致的口径：对 Q 求均值得到“每队列平均长度”，再跨 B 做统计
        qlen_overall_per_env = qlen_per_env.mean(dim=1)  # [B] 对每个queue求平均
        qlen_mean = qlen_overall_per_env.mean()
        qlen_std = qlen_overall_per_env.std(unbiased=True)
        qlen_se = qlen_std / math.sqrt(B)

        self.print(f"Eval (B={B}): queue length mean (overall): {qlen_mean.item():.4f}")
        # self.print(f"std (overall): {qlen_std.item():.4f}")
        self.print(f"se (overall): {qlen_se.item():.4f}")

        # 若你仍想保持原函数返回 (mean return, std return)：
        # ret_mean = total_r.mean().item()
        # ret_std = total_r.std(unbiased=True).item()
        # return ret_mean, ret_std

    # @torch.no_grad()
    # def evaluate(self) -> Tuple[float, float]:
    #     device = torch.device(self.args.device)
    #     td = self.eval_env.reset()
    #     B = self.args.test_batch
    #
    #     total_r = torch.zeros(B, device=device)
    #
    #     # 收集用于统计的序列
    #     qlens_list = []   # 每步的每个样本的 queue length，shape 累积成 [T, B]
    #     costs_list = []   # 同上，cost
    #
    #     for _ in range(self.args.eval_T):
    #         obs_dev = td["obs"].to(device)  # [B, obs_dim]
    #
    #         # --- 队列长度：对 obs 的前 Q 维求和
    #         Q = self.args.Q
    #         queues = obs_dev[:, :Q]
    #         qlen_t = queues.sum(dim=-1)  # [B]
    #         qlens_list.append(qlen_t)
    #
    #         # --- 策略动作
    #         logits, v = self.policy(obs_dev)
    #         if self.args.rescale_value:
    #             v = v * self.returns_std + self.returns_mean
    #         probs = self._wc_probs_from_logits_obs(logits, obs_dev)
    #         if self.args.randomize:
    #             a, _ = self._sample_and_logp(probs)
    #         else:
    #             a, _ = self._argmax_and_logp(probs)
    #
    #         # --- 环境前进一步并记录 reward / cost
    #         out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
    #         nxt = out["next"]
    #         r_t = nxt["reward"].reshape(B).to(device)
    #         total_r += r_t
    #
    #         # cost 优先找显式 "cost"，否则假定 cost = -reward
    #         if "cost" in nxt.keys():
    #             c_t = nxt["cost"].reshape(B).to(device)
    #         else:
    #             c_t = -r_t
    #         costs_list.append(c_t)
    #
    #         td = nxt
    #
    #     # --- 统计：对所有 T×B 样本取 mean 和标准误（SE = std / sqrt(N)）
    #     T = self.args.eval_T
    #     N = T * B
    #
    #     qlens = torch.stack(qlens_list, dim=0).reshape(N)      # [N]
    #     costs = torch.stack(costs_list, dim=0).reshape(N)      # [N]
    #
    #     q_mean = qlens.mean().item()
    #     # 使用无偏标准差做标准误：std(unbiased=True)/sqrt(N)
    #     q_se = (qlens.std(unbiased=True) / math.sqrt(max(1, N))).item()
    #
    #     cost_mean = costs.mean().item()
    #     cost_se = (costs.std(unbiased=True) / math.sqrt(max(1, N))).item()
    #
    #     # 原来的 return 统计
    #     ret_mean = total_r.mean().item()
    #     ret_std  = total_r.std(unbiased=True).item()
    #
    #     # 这里打印新增统计信息；返回值保持原样 (ret_mean, ret_std)
    #     self.print(f"  Eval (T={T}, B={B}): "
    #                f"queue mean {q_mean:.4f}, SE {q_se:.4f} | "
    #                f"cost mean {cost_mean:.4f}, SE {cost_se:.4f}")
    #
    #     return ret_mean, ret_std
