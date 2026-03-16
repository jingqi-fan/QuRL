import math, time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Categorical, OneHotCategorical
from torchrl.envs import EnvBase


def cosine_with_warmup_sb3_style(initial_lr: float, min_lr: float, progress_remaining: float, warmup: float) -> float:
    pr = max(0.0, min(1.0, progress_remaining))
    if pr > (1.0 - warmup):
        warmup_progress = (1.0 - pr) / max(1e-12, warmup)
        return min_lr + (initial_lr - min_lr) * warmup_progress
    adj = (pr - (1.0 - warmup)) / max(1e-12, (1.0 - warmup))
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * adj))
    return min_lr + (initial_lr - min_lr) * cos_decay


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
        self.v = MLP(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor):
        B = obs.shape[0]
        logits = self.pi(obs).view(B, self.S, self.Q)
        value = self.v(obs).squeeze(-1)
        return logits, value


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


class BCD(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, network: torch.Tensor, time_f: bool = False):
        net = network if isinstance(network, torch.Tensor) else torch.tensor(network, dtype=torch.float32)
        assert net.dim() == 2, f"network must be 2D (s,q), got {net.shape}"
        self.network = net.float()
        self.s, self.q = self.network.shape
        self.num_samples = num_samples
        self.time_f = time_f

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        queues = torch.randint(0, 101, (self.q,), dtype=torch.float32)
        if self.time_f:
            obs = torch.cat([queues, torch.randint(0, 101, (1,), dtype=torch.float32)], dim=0)
        else:
            obs = queues

        base = F.softmax(queues, dim=-1)  # [q]
        p = base.unsqueeze(0).repeat(self.s, 1)  # [s,q]
        p = p * self.network
        p = torch.minimum(p, queues.unsqueeze(0).repeat(self.s, 1))
        zero_row = torch.all(p == 0, dim=1, keepdim=True).float()
        p = p + zero_row * self.network
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return obs, p


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

    # algorithms
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
    warmup: float  # Ratio (0~1)

    # training
    total_epochs: int

    # extras
    normalize_advantage: bool
    rescale_value: bool  # == your rescale_v
    behavior_cloning: bool
    randomize: bool  # == WC_Policy.randomize
    time_f: bool  # == WC_Policy.time_f

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

        # Optimizers
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

    # ---------- Learn ----------
    def learn(self):
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

                    with torch.no_grad():
                        log_ratio = new_logp - old_logp
                        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).clamp_min(0).item()
                        approx_kl_running = 0.9 * approx_kl_running + 0.1 * approx_kl

                    # If KL is too large: decrease current lr by 1e-4 (not below min_lr), zero gradients and skip this update
                    # if self.args.target_kl and approx_kl_running > 1.5 * self.args.target_kl:
                    if approx_kl > 1.5 * self.args.target_kl:

                        # Decrease policy lr
                        for pg in self.opt_pi.param_groups:
                            old_lr = pg["lr"]
                            new_lr = max(old_lr - 1e-4, self.args.min_lr_policy)
                            pg["lr"] = new_lr
                        # # Decrease value lr
                        # for pg in self.opt_v.param_groups:
                        #     old_lr = pg["lr"]
                        #     new_lr = max(old_lr - 1e-4, self.args.min_lr_value)
                        #     pg["lr"] = new_lr

                        # Skip this minibatch: do not perform backward/step, do not advance scheduler
                        self.opt_pi.zero_grad(set_to_none=True)
                        self.opt_v.zero_grad(set_to_none=True)
                        # self.print(
                        #     f"[AutoLR] KL {approx_kl_running:.4g} > {1.5 * self.args.target_kl:.4g} → "
                        #     f"decrease lr by 1e-4 → "
                        #     f"π:{self.opt_pi.param_groups[0]['lr']:.6f}, "
                        #     f"V:{self.opt_v.param_groups[0]['lr']:.6f}. Skip minibatch."
                        # )
                        continue

                        # ===== Normal update path (KL within threshold) =====
                    total_loss = (policy_loss - self.args.ent_coef * ent) + (self.args.vf_coef * value_loss)
                    self.opt_pi.zero_grad(set_to_none=True)
                    self.opt_v.zero_grad(set_to_none=True)
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.opt_pi.step()
                    self.opt_v.step()

                    # Progress-based LR schedule (warmup -> cosine), remains unchanged
                    self._lr_step()

                # Keep original outer KL early stopping (optional)
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
        device = self.device
        td = env.reset()
        obs = torch.zeros(T + 1, B, self.args.obs_dim, device=device)
        act = torch.zeros(T, B, self.args.S, self.args.Q, device=device)
        logp = torch.zeros(T, B, device=device)
        rew = torch.zeros(T, B, device=device)
        done = torch.zeros(T, B, device=device)
        val = torch.zeros(T + 1, B, device=device)

        obs[0] = td["obs"]

        for t in range(T):
            logits, v = self.policy(obs[t])
            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean

            probs = self._wc_probs_from_logits_obs(logits, obs[t])
            if self.args.randomize:
                a, lp = self._sample_and_logp_vec(probs)
            else:
                a, lp = self._argmax_and_logp(probs)

            val[t] = v
            act[t] = a
            logp[t] = lp

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
        B, S, Q = logits.shape
        probs = F.softmax(logits, dim=-1)  # [B,S,Q]
        probs = probs * self.network_mask.to(probs.device)
        if self.args.time_f:
            queues = obs[:, :Q]
        else:
            queues = obs[:, :Q]  # If obs is only Q-dimensional, this is equivalent
        probs = torch.minimum(probs, queues.view(B, 1, Q).repeat(1, S, 1))
        zero_mask = torch.all(probs == 0, dim=-1, keepdim=True).float()
        probs = probs + zero_mask * self.network_mask.to(probs.device)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _sample_and_logp_vec(self, probs):  # probs: [B,S,Q]
        B, S, Q = probs.shape
        flat = probs.reshape(B * S, Q)
        idx = torch.multinomial(flat, 1).squeeze(-1)  # [B*S]
        a = F.one_hot(idx, num_classes=Q).float().reshape(B, S, Q)
        # logp = sum_s log p_s(a_s)
        logp = torch.log(flat.gather(1, idx.view(-1, 1)).squeeze(1).clamp_min(1e-12)) \
            .view(B, S).sum(dim=1)  # [B]
        return a, logp

    def _argmax_and_logp(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

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
        return ent / S  # Average entropy per S

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
            obs = obs.to(device)  # [B, q(+1)]
            target = target.to(device)  # [S,Q]
            logits, _ = self.policy(obs)  # [B,S,Q]
            pred = self._wc_probs_from_logits_obs(logits, obs)  # Get probabilities using a consistent construction
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

        device = torch.device(self.args.device)
        td = self.eval_env.reset()
        B, Q = self.args.test_batch, self.args.Q

        total_r = torch.zeros(B, device=device)

        time_weight_queue_len = torch.zeros(B, Q, device=device)  # [B,Q]
        time_now = torch.zeros(B, device=device)  # [B]

        for _ in range(self.args.eval_T):
            obs = td["obs"].to(device)  # [B, obs_dim]

            logits, v = self.policy(obs)
            if self.args.rescale_value:
                v = v * self.returns_std + self.returns_mean
            probs = self._wc_probs_from_logits_obs(logits, obs)
            if self.args.randomize:
                a, _ = self._sample_and_logp_vec(probs)
            else:
                a, _ = self._argmax_and_logp(probs)

            # Environment step forward (right endpoint statistics)
            out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
            nxt = out["next"]

            # Accumulate reward (keep original return/print conventions)
            r_t = nxt["reward"].reshape(B).to(device)
            total_r += r_t

            # Right endpoint: use next queues
            queues_next = nxt["obs"][:, :Q].to(device)
            dt = nxt["event_time"].reshape(B).to(device)  # [B]

            # Time integral: exactly the same form as the previous test_epoch
            time_weight_queue_len += queues_next * dt.view(B, 1)  # [B,Q]
            time_now += dt  # [B]

            td = nxt

        # Time-averaged queue length per parallel env and per queue: [B,Q]
        qlen_per_env = time_weight_queue_len / time_now.view(B, 1).clamp_min(1e-12)

        # Consistent approach with "before": average over Q to get "average length per queue", then compute statistics across B
        qlen_overall_per_env = qlen_per_env.mean(dim=1)  # [B] Average over each queue

        qlen_mean = qlen_overall_per_env.mean()
        qlen_std = qlen_overall_per_env.std(unbiased=True)
        qlen_se = qlen_std / math.sqrt(B)

        self.print(f"Eval (B={B}): queue length mean (overall): {qlen_mean.item():.4f}")
        self.print(f"se (overall): {qlen_se.item():.4f}")