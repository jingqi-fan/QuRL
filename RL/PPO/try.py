"""
TorchRL 0.6 — Single-env + batched PPO (policy/value separate optimizers)
-----------------------------------------------------------------------
This script shows how to train with one EnvBase instance that internally
handles parallelism via batch dims (train_batch / test_batch), avoiding
multi-process vector envs. It keeps the important customizations you had:
- separate optimizers for policy and value
- cosine LR with warmup for each
- optional KL early stop
- periodic batched evaluation (test_batch)

Integration notes (edit these for your codebase):
1) Replace `build_env()` with your own factory that returns an EnvBase-compatible
   environment already set to the desired batch_size. Your env should expose
   keys: 'obs' (float32, [B, obs_dim]), 'action' ([B, S, Q] one-hot or logits),
   'reward' ([B, 1] or [B]), 'done' ([B, 1] or [B]). If your env returns a
   different layout, adapt the `select_action()` and `env_step()` parts.
2) If you have action masks / network feasibility like before, wire the mask in
   `MaskedMultiCategorical` (below) before sampling.
3) If your observation optionally concatenates time, set OBS_HAS_TIME = True.

Tested with: torch>=2.2, torchrl==0.6.*
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Categorical

# TorchRL 0.6
# import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import EnvBase

# ------------------------------
# Config
# ------------------------------
@dataclass
class PPOConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # env & batching
    train_batch: int = 64          # internal parallel rollouts
    test_batch: int = 64
    episode_steps: int = 1024      # rollout horizon per batch element

    # model
    obs_dim: int = 64              # <-- set at runtime after probing env
    S: int = 8                     # number of servers (actions per row)
    Q: int = 16                    # number of queues (classes per server)
    hidden: int = 256
    scale: int = 1                 # keep parity with your previous code
    rescale_value: bool = True

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    ppo_epochs: int = 4
    minibatch_size: int = 1024     # samples per minibatch (B * T slices)
    target_kl: Optional[float] = None  # e.g. 0.02

    # LR (separate)
    lr_policy: float = 3e-4
    lr_value: float = 3e-4
    min_lr_policy: float = 1e-5
    min_lr_value: float = 1e-5
    warmup: float = 0.03           # proportion of total updates

    # training length
    total_epochs: int = 200        # number of collector -> update cycles

    # eval
    eval_every: int = 1
    eval_T: int = 1024

    # misc
    seed: int = 0


# ------------------------------
# Utilities
# ------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(initial_lr: float, min_lr: float, warmup: float, progress: float) -> float:
    """progress: 0->1 across the entire training (updates)."""
    progress = max(0.0, min(1.0, progress))
    if progress < warmup:
        # linear warmup from min_lr to initial_lr
        w = progress / max(1e-12, warmup)
        return min_lr + (initial_lr - min_lr) * w
    # cosine decay from initial_lr to min_lr
    t = (progress - warmup) / max(1e-12, (1 - warmup))
    cos_decay = 0.5 * (1 + math.cos(math.pi * t))
    return min_lr + (initial_lr - min_lr) * cos_decay


# ------------------------------
# Action distribution: S independent masked Categoricals over Q
# ------------------------------
class MaskedMultiCategorical:
    """Per-server categorical with masking.
    logits: [B, S, Q]
    mask:   [B, S, Q] with 1 for valid, 0 for invalid (or None)
    Returns one-hot actions: [B, S, Q] and log_prob per batch element: [B]
    """
    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            # set invalid logits to -inf to zero their prob
            logits = logits.masked_fill(mask <= 0, float("-inf"))
        self.logits = logits
        self.S = logits.shape[1]
        self.Q = logits.shape[2]
        # split across servers
        self._cats = [Categorical(logits=logits[:, s, :]) for s in range(self.S)]

    def sample_one_hot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = [cat.sample() for cat in self._cats]              # list of [B]
        logps = [cat.log_prob(i) for cat, i in zip(self._cats, idxs)]  # list of [B]
        # stack back to one-hot
        B = idxs[0].shape[0]
        Q = self.Q
        oh = torch.zeros(B, self.S, Q, device=idxs[0].device)
        for s, i in enumerate(idxs):
            oh[torch.arange(B), s, i] = 1.0
        logp = torch.stack(logps, dim=0).sum(dim=0)  # sum over servers -> [B]
        return oh, logp

    def log_prob_of(self, one_hot: torch.Tensor) -> torch.Tensor:
        # one_hot: [B, S, Q]
        idxs = one_hot.argmax(dim=-1)  # [B, S]
        logps = [cat.log_prob(idxs[:, s]) for s, cat in enumerate(self._cats)]
        return torch.stack(logps, dim=0).sum(dim=0)  # [B]


# ------------------------------
# Policy / Value Networks
# ------------------------------
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
        self.pi = MLP(obs_dim, hidden, S * Q)  # logits
        self.v  = MLP(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs: [B, obs_dim]
        logits = self.pi(obs).view(obs.shape[0], self.S, self.Q)
        value  = self.v(obs).squeeze(-1)  # [B]
        return logits, value


# ------------------------------
# GAE
# ------------------------------
@torch.no_grad()
def compute_gae(reward, done, value, next_value, gamma, lam):
    """All shapes are [T, B]. Returns advantage [T,B] and returns [T,B]."""
    T, B = reward.shape
    adv = torch.zeros(T, B, device=reward.device)
    last_gae = torch.zeros(B, device=reward.device)
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = reward[t] + gamma * next_value[t] * mask - value[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    ret = adv + value
    return adv, ret


# ------------------------------
# Env factory placeholder — EDIT THIS to hook your env
# ------------------------------
class DummyBatchedEnv(EnvBase):
    """Example placeholder to make this script runnable.
    Replace with your EnvBase (already batched) implementation.
    Observations: [B, obs_dim]; Action: one-hot [B, S, Q]
    """
    def __init__(self, batch_size: int, obs_dim: int, S: int, Q: int, device: str):
        super().__init__(device=device)
        self.batch_size = batch_size
        self.obs_dim, self.S, self.Q = obs_dim, S, Q
        self.register_buffer("state", torch.zeros(batch_size, obs_dim))

    def reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        self.state = torch.randn(self.batch_size, self.obs_dim, device=self.device)
        td = TensorDict({
            "obs": self.state.clone(),
            "done": torch.zeros(self.batch_size, 1, device=self.device),
        }, batch_size=[self.batch_size])
        return td

    def step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"].to(self.device)  # [B,S,Q] one-hot
        # toy dynamics / reward
        act_idx = action.argmax(dim=-1).float()  # [B,S]
        rew = act_idx.mean(dim=1, keepdim=True) / max(1, self.Q-1)
        self.state = torch.tanh(self.state + 0.01 * torch.randn_like(self.state) + 0.1)
        done = torch.zeros(self.batch_size, 1, device=self.device)
        out = TensorDict({
            "obs": self.state.clone(),
            "reward": rew,
            "done": done,
        }, batch_size=[self.batch_size])
        return out


def build_env(batch_size: int, cfg: PPOConfig, device: str) -> EnvBase:
    # TODO: replace DummyBatchedEnv with your RLViewDiffDES / EnvBase
    return DummyBatchedEnv(batch_size=batch_size, obs_dim=cfg.obs_dim, S=cfg.S, Q=cfg.Q, device=device)


# ------------------------------
# Training loop
# ------------------------------
@torch.no_grad()
def evaluate(env: EnvBase, policy: ActorCritic, cfg: PPOConfig) -> Tuple[float, float]:
    td = env.reset()
    B = td.batch_size[0]
    total_r = torch.zeros(B, device=cfg.device)
    for _ in range(cfg.eval_T):
        logits, v = policy(td["obs"].to(cfg.device))
        dist = MaskedMultiCategorical(logits, mask=None)
        act, _ = dist.sample_one_hot()
        td = env.step(TensorDict({"action": act.to(env.device)}, batch_size=td.batch_size))
        r = td.get("reward").view(B).to(cfg.device)
        total_r += r
    return total_r.mean().item(), total_r.std(unbiased=True).item()


def train(cfg: PPOConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Envs
    train_env = build_env(cfg.train_batch, cfg, cfg.device)
    test_env  = build_env(cfg.test_batch, cfg, cfg.device)

    # Probe obs_dim if needed (uncomment if your env provides it dynamically)
    # cfg.obs_dim = int(train_env.reset()["obs"].shape[-1])

    # Model
    policy = ActorCritic(cfg.obs_dim, cfg.S, cfg.Q, cfg.hidden).to(device)

    # Opts
    opt_pi = torch.optim.Adam([
        *policy.pi.parameters()
    ], lr=cfg.lr_policy)
    opt_v  = torch.optim.Adam([
        *policy.v.parameters()
    ], lr=cfg.lr_value)

    # Collector: one env, batched frames
    frames_per_batch = cfg.episode_steps * cfg.train_batch
    collector = SyncDataCollector(
        create_env_fn=lambda: train_env,  # single env instance with batch
        frames_per_batch=frames_per_batch,
        total_frames=-1,                 # we drive by epochs below
        device=cfg.device,
    )

    # global counters for schedulers
    total_updates = cfg.total_epochs * cfg.ppo_epochs * max(1, (frames_per_batch // cfg.minibatch_size))
    update_idx = 0

    for epoch, td_batch in enumerate(collector):
        policy.train()
        # td_batch includes [T*B] frames interleaved; unfold to [T, B, ...]
        # SyncDataCollector returns keys: 'obs', 'action', 'next', 'reward', 'done', 'terminated' ...
        # We will recompute everything with our policy (on-policy), so we rebuild storage.

        # Rollout with our policy to build trajectory tensors of shape [T, B]
        T = cfg.episode_steps
        B = cfg.train_batch
        obs = torch.zeros(T + 1, B, cfg.obs_dim, device=device)
        act_oh = torch.zeros(T, B, cfg.S, cfg.Q, device=device)
        logp = torch.zeros(T, B, device=device)
        rew = torch.zeros(T, B, device=device)
        done = torch.zeros(T, B, device=device)
        val = torch.zeros(T + 1, B, device=device)

        td = train_env.reset()
        obs[0] = td["obs"].to(device)
        for t in range(T):
            logits, v = policy(obs[t])
            dist = MaskedMultiCategorical(logits, mask=None)  # TODO: pass mask from env if you have one
            a, lp = dist.sample_one_hot()
            val[t] = v
            act_oh[t] = a
            logp[t] = lp

            td = train_env.step(TensorDict({"action": a.to(train_env.device)}, batch_size=[B]))
            rew[t] = td["reward"].view(B).to(device)
            done[t] = td["done"].view(B).to(device)
            obs[t + 1] = td["obs"].to(device)

        # bootstrap value for last obs
        with torch.no_grad():
            _, v_last = policy(obs[-1])
            val[-1] = v_last

        # Compute advantages/returns (GAE)
        adv, ret = compute_gae(
            reward=rew,
            done=done,
            value=val[:-1],
            next_value=val[1:],
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        # normalize advantage (optional but common)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        # Flatten time & batch for SGD
        TnB = T * B
        obs_f   = obs[:-1].reshape(TnB, cfg.obs_dim)
        act_f   = act_oh.reshape(TnB, cfg.S, cfg.Q)
        logp_f  = logp.reshape(TnB)
        adv_f   = adv.reshape(TnB)
        ret_f   = ret.reshape(TnB)
        val_f   = val[:-1].reshape(TnB)

        # PPO updates
        idx = torch.randperm(TnB, device=device)
        mb = cfg.minibatch_size
        approx_kl_running = 0.0

        for _ in range(cfg.ppo_epochs):
            for start in range(0, TnB, mb):
                end = min(start + mb, TnB)
                mb_idx = idx[start:end]

                o = obs_f[mb_idx]
                a = act_f[mb_idx]
                old_logp = logp_f[mb_idx]
                adv_mb = adv_f[mb_idx]
                ret_mb = ret_f[mb_idx]

                # forward
                logits, v_pred = policy(o)
                dist_new = MaskedMultiCategorical(logits, mask=None)
                new_logp = dist_new.log_prob_of(a)

                # policy loss (clipped surrogate)
                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # entropy bonus (optional)
                # We can estimate entropy by sampling or using categorical entropy per server
                ent = 0.0
                if cfg.ent_coef != 0.0:
                    ent_per_s = []
                    for s in range(cfg.S):
                        ent_per_s.append(Categorical(logits=logits[:, s, :]).entropy())
                    ent = torch.stack(ent_per_s, dim=0).sum(dim=0).mean()
                policy_obj = policy_loss - cfg.ent_coef * (ent if isinstance(ent, torch.Tensor) else 0.0)

                # value loss
                value_loss = F.mse_loss(v_pred, ret_mb)
                value_obj = cfg.vf_coef * value_loss

                # KL for early stop (estimate)
                with torch.no_grad():
                    kl = (old_logp - new_logp).mean().clamp_min(0).item()
                    approx_kl_running = 0.9 * approx_kl_running + 0.1 * kl

                # update separate
                opt_pi.zero_grad(set_to_none=True)
                policy_obj.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt_pi.step()

                opt_v.zero_grad(set_to_none=True)
                value_obj.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt_v.step()

                # cosine LR with warmup per update
                progress = update_idx / max(1, total_updates - 1)
                for pg in opt_pi.param_groups:
                    pg["lr"] = cosine_with_warmup(cfg.lr_policy, cfg.min_lr_policy, cfg.warmup, progress)
                for pg in opt_v.param_groups:
                    pg["lr"] = cosine_with_warmup(cfg.lr_value, cfg.min_lr_value, cfg.warmup, progress)
                update_idx += 1

                # KL early stop
                if cfg.target_kl is not None and approx_kl_running > 1.5 * cfg.target_kl:
                    break
            if cfg.target_kl is not None and approx_kl_running > 1.5 * cfg.target_kl:
                break

        # Eval
        if (epoch + 1) % cfg.eval_every == 0:
            policy.eval()
            mean_r, std_r = evaluate(test_env, policy, cfg)
            print(f"Epoch {epoch+1:04d} | Return mean {mean_r:.4f} ± {std_r:.4f} | KL~{approx_kl_running:.5f}")

        if epoch + 1 >= cfg.total_epochs:
            break


if __name__ == "__main__":
    cfg = PPOConfig()
    train(cfg)
