import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Categorical
# from torchrl.data import TensorDict
from torchrl.envs import EnvBase

# ==========================
# Utilities
# ==========================

def cosine_with_warmup(initial_lr: float, min_lr: float, warmup: float, progress: float) -> float:
    """Cosine decay with a linear warmup.
    progress: 0->1 over the whole training (update-wise).
    """
    progress = max(0.0, min(1.0, progress))
    if progress < warmup:
        w = progress / max(1e-12, warmup)
        return min_lr + (initial_lr - min_lr) * w
    t = (progress - warmup) / max(1e-12, (1.0 - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (initial_lr - min_lr) * cos


class MaskedMultiCategorical:
    """S independent Categoricals over Q, with optional mask.
    logits: [B, S, Q], mask: [B, S, Q] in {0,1}.
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, float("-inf"))
        self.logits = logits
        self.S = logits.shape[1]
        self.Q = logits.shape[2]
        self._cats = [Categorical(logits=logits[:, s, :]) for s in range(self.S)]

    def sample_one_hot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = [cat.sample() for cat in self._cats]             # list of [B]
        logps = [cat.log_prob(i) for cat, i in zip(self._cats, idxs)]
        B, Q = idxs[0].shape[0], self.Q
        a = torch.zeros(B, self.S, Q, device=idxs[0].device)
        for s, i in enumerate(idxs):
            a[torch.arange(B), s, i] = 1.0
        logp = torch.stack(logps, dim=0).sum(dim=0)             # [B]
        return a, logp

    def log_prob_of(self, one_hot: torch.Tensor) -> torch.Tensor:
        idxs = one_hot.argmax(dim=-1)  # [B,S]
        logps = [cat.log_prob(idxs[:, s]) for s, cat in enumerate(self._cats)]
        return torch.stack(logps, dim=0).sum(dim=0)


# ==========================
# Networks (keep SB3-style shapes)
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
        self.pi = MLP(obs_dim, hidden, S * Q)     # logits
        self.v  = MLP(obs_dim, hidden, 1)         # scalar value

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        logits = self.pi(obs).view(B, self.S, self.Q)
        value  = self.v(obs).squeeze(-1)
        return logits, value


# ==========================
# GAE (SB3-equivalent)
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
# Behavior Cloning dataset (same as你的BC逻辑)
# ==========================
class BCD(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, network: torch.Tensor):
        net = network if isinstance(network, torch.Tensor) else torch.tensor(network, dtype=torch.float32)
        assert net.dim() == 2, f"network must be 2D (s,q), got {net.shape}"
        self.network = net.float()
        self.s, self.q = self.network.shape
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = torch.randint(0, 101, (self.q,), dtype=torch.float32)
        base = F.softmax(obs, dim=-1)
        action_probs = base.unsqueeze(0).repeat(self.s, 1)  # [s,q]
        action_probs = action_probs * self.network
        pos_mask = (obs > 0).float().unsqueeze(0).repeat(self.s, 1)
        action_probs = action_probs * pos_mask
        row_sum = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        zero_row = (row_sum <= 1e-12).float()
        action_probs = action_probs + zero_row * self.network
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return obs, action_probs


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
    warmup: float

    # training
    total_epochs: int

    # extras
    normalize_advantage: bool
    rescale_value: bool
    behavior_cloning: bool

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
                 network_mask: Optional[torch.Tensor] = None,  # [S,Q] or [B,S,Q]
                 print_fn: Callable[[str], None] = print):
        self.train_env = train_env
        self.eval_env = eval_env
        self.args = args
        self.network_mask = network_mask  # 用于动作可行性屏蔽
        self.print = print_fn

        device = torch.device(args.device)
        self.policy = ActorCritic(args.obs_dim, args.S, args.Q, args.hidden).to(device)

        # separate optimizers like SB3 custom trainer
        self.opt_pi = torch.optim.Adam(self.policy.pi.parameters(), lr=args.lr_policy)
        self.opt_v  = torch.optim.Adam(self.policy.v.parameters(),  lr=args.lr_value)

        # schedulers will be applied manually via cosine_with_warmup per update
        self.total_updates = self._estimate_total_updates()
        self.update_idx = 0

        # running stats for value rescale (to mimic your WC_policy behavior)
        self.ret_mean = torch.tensor(0.0, device=device)
        self.ret_std  = torch.tensor(1.0, device=device)

    # ---------- API ----------
    def pre_train(self):
        if self.args.behavior_cloning:
            self._behavior_cloning()

    def learn(self):
        self.print("Start training (single-env, batched)")
        for epoch in range(self.args.total_epochs):
            t0 = time.time()
            traj = self._rollout(self.train_env, self.args.episode_steps, self.args.train_batch)
            adv, ret = compute_gae(traj['rew'], traj['done'], traj['val'][:-1], traj['val'][1:], self.args.gamma, self.args.gae_lambda)
            if self.args.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            # update running stats of returns
            with torch.no_grad():
                self.ret_mean = 0.9 * self.ret_mean + 0.1 * ret.mean()
                self.ret_std  = 0.9 * self.ret_std  + 0.1 * ret.std().clamp_min(1e-6)

            # flatten to [T*B]
            T, B = self.args.episode_steps, self.args.train_batch
            obs_f  = traj['obs'][:-1].reshape(T*B, self.args.obs_dim)
            act_f  = traj['act'].reshape(T*B, self.args.S, self.args.Q)
            oldlp  = traj['logp'].reshape(T*B)
            adv_f  = adv.reshape(T*B)
            ret_f  = ret.reshape(T*B)

            # PPO updates (separate optimizers)
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

                    logits, v_pred = self.policy(o)
                    dist = MaskedMultiCategorical(logits, mask=self._mask_for_batch(logits, mb_idx))
                    new_logp = dist.log_prob_of(a)

                    ratio = torch.exp(new_logp - old_logp)
                    surr1 = ratio * adv_mb
                    surr2 = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * adv_mb
                    policy_loss = -torch.min(surr1, surr2).mean()

                    ent_bonus = 0.0
                    if self.args.ent_coef != 0.0:
                        ent_per_s = [Categorical(logits=logits[:, s, :]).entropy() for s in range(self.args.S)]
                        ent_bonus = torch.stack(ent_per_s, dim=0).sum(dim=0).mean()

                    value_loss = F.mse_loss(v_pred, ret_mb)

                    # separate updates
                    self.opt_pi.zero_grad(set_to_none=True)
                    (policy_loss - self.args.ent_coef * (ent_bonus if isinstance(ent_bonus, torch.Tensor) else 0.0)).backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_pi.step()

                    self.opt_v.zero_grad(set_to_none=True)
                    (self.args.vf_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_v.step()

                    # LR schedule step
                    self._lr_step()

                    # KL early stop approx (SB3-style)
                    with torch.no_grad():
                        kl = (old_logp - new_logp).mean().clamp_min(0).item()
                        approx_kl_running = 0.9 * approx_kl_running + 0.1 * kl
                    if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
                        break
                if self.args.target_kl is not None and approx_kl_running > 1.5 * self.args.target_kl:
                    break

            t1 = time.time()
            self.print(f"[Epoch {epoch+1}/{self.args.total_epochs}] time={t1-t0:.2f}s, KL~{approx_kl_running:.5f}")

            # periodic eval
            if (epoch + 1) % self.args.eval_every == 0:
                mean_r, std_r = self.evaluate()
                self.print(f"  Eval: return mean {mean_r:.4f} ± {std_r:.4f}")

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
            logits, v = self.policy(obs[t])
            dist = MaskedMultiCategorical(logits, mask=self._mask_for_batch(logits))
            a, lp = dist.sample_one_hot()
            val[t] = v
            act[t] = a
            logp[t] = lp

            # td = env.step(TensorDict({"action": a.to(env.device)}, batch_size=[B]))
            # rew[t] = td.get("reward").view(B).to(device)
            # done[t]= td.get("done").view(B).to(device)
            # obs[t+1] = td.get("obs").to(device)

            td = env.step(TensorDict({"action": a.to(env.device)}, batch_size=[B]))
            nxt = td["next"]  # 统一用 next
            obs[t + 1] = nxt["obs"].to(device)
            rew[t] = nxt["reward"].reshape(B).to(device)
            done[t] = nxt["done"].reshape(B).to(device)

        _, v_last = self.policy(obs[-1])
        val[-1] = v_last
        return {"obs": obs, "act": act, "logp": logp, "rew": rew, "done": done, "val": val}

    def _mask_for_batch(self, logits: torch.Tensor, mb_idx: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.network_mask is None:
            return None

        mask = self.network_mask
        # 统一到正确的 device & dtype
        mask = mask.to(logits.device)

        if mask.dim() == 2:
            # [S,Q] -> 直接广播到 [mb, S, Q]
            return mask.unsqueeze(0).expand(logits.shape[0], -1, -1)

        elif mask.dim() == 3:
            # [B, S, Q] 的掩码
            B = mask.shape[0]

            if mb_idx is None:
                # 没给索引，就按当前 logits 的 batch 长度构造 0..mb-1 的索引，并取模到 B
                mb = logits.shape[0]
                idx_b = torch.arange(mb, device=logits.device) % B
            else:
                # 展平后的索引 -> 映射回 batch 维
                idx_b = (mb_idx % B).to(mask.device)

            return mask.index_select(0, idx_b)  # [mb, S, Q]

        else:
            raise ValueError(f"network_mask must be [S,Q] or [B,S,Q], got shape {tuple(mask.shape)}")

    def _lr_step(self):
        progress = self.update_idx / max(1, self.total_updates - 1)
        for pg in self.opt_pi.param_groups:
            pg["lr"] = cosine_with_warmup(self.args.lr_policy, self.args.min_lr_policy, self.args.warmup, progress)
        for pg in self.opt_v.param_groups:
            pg["lr"] = cosine_with_warmup(self.args.lr_value, self.args.min_lr_value, self.args.warmup, progress)
        self.update_idx += 1

    def _estimate_total_updates(self) -> int:
        TnB = self.args.episode_steps * self.args.train_batch
        mb = max(1, self.args.minibatch_size)
        steps_per_epoch = math.ceil(TnB / mb) * self.args.ppo_epochs
        return max(1, steps_per_epoch * self.args.total_epochs)

    # ---------- Behavior Cloning ----------
    def _behavior_cloning(self):
        self.print("[BC] start")
        # try to fetch network mask from eval_env if not given
        net = None
        if self.network_mask is not None:
            net = self.network_mask
        elif hasattr(self.eval_env, 'network'):
            net = self.eval_env.network
            if isinstance(net, torch.Tensor) and net.dim() == 3:
                net = net[0]
        assert isinstance(net, torch.Tensor) and net.dim() == 2, "BC requires a [S,Q] network mask"

        dataset = BCD(self.args.bc_samples, net)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.test_batch, shuffle=True)
        opt = torch.optim.Adam(self.policy.pi.parameters(), lr=self.args.bc_lr)
        device = torch.device(self.args.device)
        self.policy.train()
        for i, (obs, target) in enumerate(loader):
            obs = obs.to(device)
            target = target.to(device)  # [S,Q]
            logits, _ = self.policy(obs)
            # logits: [B,S,Q]  target needs [B,S,Q]
            target_b = target.unsqueeze(0).expand(logits.shape[0], -1, -1)
            # MSE between softmax(logits) and target
            pred = F.softmax(logits, dim=-1)
            loss = F.mse_loss(pred, target_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
            opt.step()
        self.print("[BC] done")

    # ---------- Evaluation ----------
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        device = torch.device(self.args.device)
        td = self.eval_env.reset()  # 这里顶层有 'obs'
        B = self.args.test_batch
        total_r = torch.zeros(B, device=device)

        for _ in range(self.args.eval_T):
            # 用当前 obs 走策略
            logits, _ = self.policy(td["obs"].to(device))
            dist = MaskedMultiCategorical(logits, mask=self._mask_for_batch(logits))
            a, _ = dist.sample_one_hot()

            # 环境一步，返回的内容在 td["next"] 里
            out = self.eval_env.step(
                TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B])
            )
            nxt = out["next"]

            # 累积奖励；保证形状是 [B]
            total_r += nxt["reward"].reshape(B).to(device)

            # 把 next 当作下一步的当前状态（包含 'obs'）
            td = nxt

        return total_r.mean().item(), total_r.std(unbiased=True).item()

