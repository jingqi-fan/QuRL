import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from typing import Any, Dict, Optional, Tuple, Union

Tensor = torch.Tensor


def build_mlp(in_dim: int, hidden: list[int], out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class Vanilla_Policy(nn.Module):
    """
    TorchRL-friendly Vanilla Policy (不依赖 stable-baselines3).
    输入: obs (B, q)
    输出: forward -> action(one-hot, B,q), values(B,1), log_prob(B,)
    """

    def __init__(
        self,
        *,
        network: Tensor,
        randomize: bool = False,
        time_f: bool = False,
        scale: int,
        rescale_v: bool,
        alpha,
        mu: Tensor,
        D: Tensor,
        net_arch: Dict[str, list[int]],
        activation_fn=nn.Tanh,
        **kwargs: Any,
    ):
        super().__init__()
        self.randomize = randomize
        self.time_f = time_f
        self.rescale_v = rescale_v

        # 网络维度 (s, q)
        self.q = network.size(1)
        self.s = network.size(0)

        # rollout统计
        self.mean_queue_length = 0.0
        self.std_queue_length = 1.0
        self.returns_mean = 0.0
        self.returns_std = 1.0

        # 注册 buffer，保证 .to(device) 时搬迁
        self.register_buffer("network", network.float())
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("mu", mu.float().unsqueeze(0))
        self.register_buffer("D", D.float())

        # 策略/价值网络
        pi_hidden = net_arch.get("pi", [])
        vf_hidden = net_arch.get("vf", [])
        self.policy_mlp = build_mlp(self.q, pi_hidden, self.q, activation=activation_fn)
        self.value_mlp = build_mlp(self.q, vf_hidden, 1, activation=activation_fn)

    # ====== 工具函数 ======
    def update_mean_std(self, mean_queue_length: float, std_queue_length: float):
        self.mean_queue_length = float(mean_queue_length)
        self.std_queue_length = float(std_queue_length)

    def standardize_queues(self, queues: Tensor) -> Tensor:
        return ((queues - self.mean_queue_length) / (self.std_queue_length + 1e-8)).float()

    def update_rollout_stats(self, returns_mean: float, returns_std: float):
        self.returns_mean = float(returns_mean)
        self.returns_std = float(returns_std)

    def rescale_values(self, values: Tensor) -> Tensor:
        return values * self.returns_std + self.returns_mean

    # ====== 前向 ======
    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)

        logits = self.policy_mlp(input_obs)           # (B, q)
        action_probs = F.softmax(logits, dim=-1)      # (B, q)
        values = self.value_mlp(input_obs)            # (B, 1)
        if self.rescale_v:
            values = self.rescale_values(values)

        if self.randomize and not deterministic:
            dist = OneHotCategorical(probs=action_probs)
            action = dist.sample()                    # (B, q)
        else:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=self.q).float()

        selected_probs_sum = (action * action_probs).sum(dim=-1).clamp_min(1e-8)
        log_prob = torch.log(selected_probs_sum)      # (B,)

        return action, values, log_prob

    def get_prob_act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        logits = self.policy_mlp(input_obs)
        action_probs = F.softmax(logits, dim=-1)

        if self.randomize and not deterministic:
            action = OneHotCategorical(probs=action_probs).sample()
        else:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=self.q).float()
        return action, action_probs

    @torch.no_grad()
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray], Tensor],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(observation, np.ndarray):
            obs_tensor = torch.from_numpy(observation).to(next(self.parameters()).device).float()
        elif isinstance(observation, torch.Tensor):
            obs_tensor = observation.to(next(self.parameters()).device).float()
        else:
            raise ValueError("predict() expects numpy array or tensor.")
        return self.get_prob_act(obs_tensor, deterministic)

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        logits = self.policy_mlp(input_obs)
        action_probs = F.softmax(logits, dim=-1)

        actions = actions.view(-1, self.q).float()
        selected_probs_sum = (actions * action_probs).sum(dim=-1).clamp_min(1e-8)
        log_prob = torch.log(selected_probs_sum)
        entropy = None
        return log_prob, entropy

    def evaluate_values(self, obs: Tensor) -> Tensor:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        values = self.value_mlp(input_obs)
        return values

    def predict_values(self, obs: Tensor) -> Tensor:
        obs = obs.view(-1, self.q)
        input_obs = self.standardize_queues(obs)
        values = self.value_mlp(input_obs)
        if self.rescale_v:
            values = self.rescale_values(values)
        return values

    # ====== AMP 逻辑保留 ======
    @torch.no_grad()
    def AMP(self, obs: Tensor, action_repeat: int) -> Tensor:
        x = self.standardize_queues(obs)
        B = x.size(0)

        logits = self.policy_mlp(x)
        action_probs = F.softmax(logits, dim=-1)

        pr = action_probs.repeat_interleave(action_repeat, 0)
        action = OneHotCategorical(probs=pr).sample()

        pmu = action * self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1)
        pmu_flat = pmu  # (A*B, q)

        prob_transitions = torch.hstack((self.alpha.unsqueeze(0).repeat(B * action_repeat, 1), pmu_flat))
        prob_transitions /= prob_transitions.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        prob_transitions = torch.stack(torch.chunk(prob_transitions, action_repeat, dim=0), dim=0).mean(dim=0)
        prob_transitions = prob_transitions.view(B * (2 * self.q), 1)

        Px = obs.unsqueeze(1) + self.D.unsqueeze(0).repeat(B, 1, 1)
        Px = torch.relu(Px).view(B * (2 * self.q), self.q)

        Pfx = (prob_transitions * self.predict_values(Px)).view(B, 2 * self.q, 1).sum(dim=1)
        return Pfx

    @torch.no_grad()
    def predict_next_states(self, obs: Tensor, action_repeat: int) -> Tensor:
        obs = obs.view(-1, self.q)
        return self.AMP(obs, action_repeat)
