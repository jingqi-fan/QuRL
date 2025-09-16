import numpy as np
import torch
from tensordict import TensorDictBase
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from torch.distributions import OneHotCategorical

Tensor = torch.Tensor


def build_mlp(in_dim: int, hidden: list[int], out_dim: int, activation=nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class WC_Policy(nn.Module):
    """
    纯 PyTorch 版本（TorchRL 友好）的 WC_Policy。
    - 输入: obs (B, q)
    - 输出: forward -> action(one-hot, B,s,q), values(B,1), log_prob(B,)
    - 另外提供: predict, evaluate_actions, evaluate_values, predict_values, AMP, predict_next_states
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

        # 常量/环境参数
        self.randomize = randomize
        self.time_f = time_f
        self.rescale_v = rescale_v

        # 网络拓扑 (s, q)
        self.q = network.size(1)
        self.s = network.size(0)

        # 训练期统计量
        self.mean_queue_length = 0.0
        self.std_queue_length = 1.0
        self.returns_mean = 0.0
        self.returns_std = 1.0

        # 注册为 buffer，确保 .to(device) 时一起搬运到 GPU，且不参与梯度
        self.register_buffer("network", network.float())                  # (s, q)
        # alpha 可能传标量或向量；统一成 (q,) 向量
        alpha = torch.as_tensor(alpha, dtype=torch.float32)
        if alpha.ndim == 0:
            alpha = alpha.repeat(self.q)
        assert alpha.numel() == self.q, f"alpha size must be q={self.q}, got {alpha.numel()}"
        self.register_buffer("alpha", alpha)                              # (q,)
        self.register_buffer("mu", mu.float().unsqueeze(0))               # (1, s, q)
        self.register_buffer("D", D.float())                              # (2q, q)

        # 策略/价值网络（内部 value 主要给 AMP / predict_values 用）
        pi_hidden = net_arch.get("pi", [])
        vf_hidden = net_arch.get("vf", [])

        self.policy_mlp = build_mlp(self.q, pi_hidden, self.s * self.q, activation=activation_fn)
        self.value_mlp = build_mlp(self.q, vf_hidden, 1, activation=activation_fn)

        # 打印检查
        # print(f"alpha: {self.alpha.shape}, mu: {self.mu.shape}, D: {self.D.shape}, network: {self.network.shape}")

    # ====== 统计量维护 ======
    def update_mean_std(self, mean_queue_length: float, std_queue_length: float):
        self.mean_queue_length = float(mean_queue_length)
        self.std_queue_length = float(std_queue_length)

    def forward_td(self, queues: torch.Tensor, time: torch.Tensor | None = None, deterministic: bool = False):
        """
        TorchRL 适配：输入是 env 的观测里的 queues（可选还有 time），
        输出 (action, sample_log_prob)。这里我们忽略 time（你的模型按 q 维建的）。
        """
        action, values, log_prob = self.forward(queues, deterministic=deterministic)
        # ClipPPOLoss 默认读取 "sample_log_prob"
        return action, log_prob

    def standardize_queues(self, queues: Tensor) -> Tensor:
        # 标准化，不反传统计量
        standardization = (queues - self.mean_queue_length) / (self.std_queue_length + 1e-8)
        _ = standardization.detach()  # 与原实现保持一致（虽未赋值，但不影响结果）
        return standardization.float()

    # ==== 在 WC_Policy 里已有的工具函数基础上新增一个保留 batch 维的取 obs ====
    def _to_obs_tensor_preserve_batch(self, obs):
        """把输入统一为张量，形状 (*B, D)，保留原始 batch 维，不做 flatten。"""
        if isinstance(obs, TensorDictBase):
            queues = obs.get("queues")
            t = obs.get("time", None)
            use_time = getattr(self, "time_f", False) or getattr(self, "use_time", False)
            if use_time and t is not None:
                obs = torch.cat([queues, t], dim=-1)  # (*B, q+1)
            else:
                obs = queues  # (*B, q)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, D)
        return obs  # (*B, D)

    # ==== 一个“只算 value”的方法（供 GAE / Critic 使用），保持 (*B,1) ====
    def value_only(self, obs):
        obs = self._to_obs_tensor_preserve_batch(obs)  # (*B, q) 或 (*B, q+1)
        input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))  # (*B, q)
        v = self.value_mlp(input_obs)  # (*B, 1)
        if self.rescale_v:
            v = self.rescale_values(v)
        return v  # (*B, 1)

    def update_rollout_stats(self, returns_mean: float, returns_std: float):
        self.returns_mean = float(returns_mean)
        self.returns_std = float(returns_std)

    def rescale_values(self, values: Tensor) -> Tensor:
        return values * self.returns_std + self.returns_mean

    # ====== 基础计算 ======
    def _policy_logits(self, obs: Tensor) -> Tensor:
        """
        obs: (B, q)
        return: logits (B, s, q)
        """
        logits = self.policy_mlp(obs)                    # (B, s*q)
        return logits.view(-1, self.s, self.q)

    def _masked_action_probs(self, obs_raw: Tensor, logits: Tensor) -> Tensor:
        """
        obs_raw: (B, q) 未裁切的原 obs（用于可服务约束）
        logits: (B, s, q)
        return: 归一化后的概率 (B, s, q)，已做 network & 可服务队列掩码
        """
        # 在这里做了 work conserving (wc)
        action_probs = F.softmax(logits, dim=-1)         # (B, s, q)
        action_probs = action_probs * self.network       # 按网络拓扑掩码
        # 受队列可服务数量限制: prob <= obs
        action_probs = torch.minimum(action_probs, obs_raw.unsqueeze(1).repeat(1, self.s, 1))
        # 若某行全 0，用 network 作为回退（与原实现一致）
        zero_mask = torch.all(action_probs == 0, dim=2).unsqueeze(-1).repeat(1, 1, self.q)
        action_probs = action_probs + zero_mask * self.network
        # 归一化
        denom = torch.sum(action_probs, dim=-1, keepdim=True).clamp_min(1e-12)
        action_probs = action_probs / denom
        return action_probs

    def _maybe_time_feature_crop(self, obs: Tensor) -> Tensor:
        # 与原实现保持一致：若 time_f=True，丢弃最后一列
        return obs[:, :-1] if self.time_f and obs.size(1) > 0 else obs

    def _ensure_obs_tensor(self, obs):
        if isinstance(obs, TensorDictBase):
            queues = obs.get("queues")
            t = obs.get("time", None)
            if (getattr(self, "time_f", False) or getattr(self, "use_time", False)) and t is not None:
                obs = torch.cat([queues, t], dim=-1)  # [..., q+1]
            else:
                obs = queues  # [..., q]
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (1, D)
        return obs.reshape(obs.shape[0], -1)  # (B, D)

    # ====== 前向（给 PPO 用） ======
    def forward(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        返回:
          - action: (B, s, q) one-hot
          - values: (B, 1)
          - log_prob: (B,)
        """
        # # obs 形状整理
        # obs = obs.view(-1, self.q)
        # obs_for_mask = obs  # 用于可服务掩码

        # 1) 统一输入
        obs = self._ensure_obs_tensor(obs)  # (B, q) 或 (B, q+1)
        # 掩码只看前 q 列（原始队列，不含 time 特征）
        obs_for_mask = obs[..., : self.q]

        input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))

        logits = self._policy_logits(input_obs)          # (B, s, q)
        action_probs = self._masked_action_probs(obs_for_mask, logits)

        # 价值
        values = self.value_mlp(input_obs)               # (B, 1)
        if self.rescale_v:
            values = self.rescale_values(values)

        # 采样/贪心
        if self.randomize and not deterministic:
            dist = OneHotCategorical(probs=action_probs)
            action = dist.sample()                       # (B, s, q) one-hot
        else:
            action_indices = torch.argmax(action_probs, dim=-1)       # (B, s)
            action = F.one_hot(action_indices, num_classes=self.q).float()

        # log_prob（对所有 server 维度求和）
        selected_probs_sum = (action * action_probs).sum(dim=-1).clamp_min(1e-8)  # (B, s)
        log_prob = torch.log(selected_probs_sum).sum(dim=1)                        # (B,)

        return action, values, log_prob

    # # ====== 概率+动作（用于 predict） ======
    # def get_prob_act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor]:
    #     obs = obs.view(-1, self.q)
    #     obs_for_mask = obs
    #     input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
    #
    #     logits = self._policy_logits(input_obs)
    #     action_probs = self._masked_action_probs(obs_for_mask, logits)
    #
    #     if self.randomize and not deterministic:
    #         action = OneHotCategorical(probs=action_probs).sample()
    #     else:
    #         action_indices = torch.argmax(action_probs, dim=-1)
    #         action = F.one_hot(action_indices, num_classes=self.q).float()
    #
    #     return action, action_probs
    #
    # # ====== 与原 SB3 风格一致的 API ======
    # @torch.no_grad()
    # def predict(
    #     self,
    #     observation: Union[np.ndarray, Dict[str, np.ndarray], Tensor],
    #     state: Optional[Tuple[np.ndarray, ...]] = None,
    #     episode_start: Optional[np.ndarray] = None,
    #     deterministic: bool = False,
    # ) -> Tuple[Tensor, Tensor]:
    #     """
    #     返回:
    #       - action(one-hot): (B, s, q)
    #       - action_probs: (B, s, q)
    #     """
    #     if isinstance(observation, dict):
    #         # 若外部传 dict，请自行在上层转换；这里简单拼接/选择不支持
    #         raise ValueError("predict() expects a tensor/ndarray observation of shape (B, q).")
    #     if isinstance(observation, np.ndarray):
    #         obs_tensor = torch.from_numpy(observation).to(next(self.parameters()).device).float()
    #     else:
    #         obs_tensor = observation.to(next(self.parameters()).device).float()
    #
    #     action, action_probs = self.get_prob_act(obs_tensor, deterministic)
    #     return action, action_probs
    #
    # def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    #     """
    #     obs: (B, q)
    #     actions: (B, s, q) one-hot
    #     return:
    #       - log_prob: (B,)
    #       - entropy: None（保持兼容）
    #     """
    #     obs = obs.view(-1, self.q)
    #     obs_for_mask = obs
    #     input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
    #
    #     logits = self._policy_logits(input_obs)
    #     action_probs = self._masked_action_probs(obs_for_mask, logits)
    #
    #     actions = actions.view(-1, self.s, self.q).float()
    #     selected_probs_sum = (actions * action_probs).sum(dim=-1).clamp_min(1e-8)  # (B, s)
    #     log_prob = torch.log(selected_probs_sum).sum(dim=1)                          # (B,)
    #     entropy = None
    #     return log_prob, entropy
    #
    # def evaluate_values(self, obs: Tensor) -> Tensor:
    #     obs = obs.view(-1, self.q)
    #     input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
    #     values = self.value_mlp(input_obs)
    #     return values
    #
    # # ====== 仅供 AMP / 规划等用 ======
    # def predict_values(self, obs: Tensor) -> Tensor:
    #     obs = obs.view(-1, self.q)
    #     input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
    #     values = self.value_mlp(input_obs)
    #     if self.rescale_v:
    #         values = self.rescale_values(values)
    #     return values
    #
    # @torch.no_grad()
    # def AMP(self, obs: Tensor, action_repeat: int) -> Tensor:
    #     """
    #     近似鞅过程评估（与原实现对齐）
    #     obs: (B, q) 非负队列长度
    #     return: (B, 1)
    #     """
    #     x = self.standardize_queues(obs)
    #     B = x.size(0)
    #
    #     # 计算策略概率
    #     logits = self._policy_logits(x)
    #     action_probs = self._masked_action_probs(obs, logits)
    #
    #     # 重复采样
    #     pr = action_probs.repeat_interleave(action_repeat, dim=0)  # (A*B, s, q)
    #     action = OneHotCategorical(probs=pr).sample()              # (A*B, s, q)
    #
    #     # 乘以 mu
    #     pmu = action * self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1, 1)  # (A*B, s, q)
    #     pmu_flat = pmu.sum(dim=1)                                               # (A*B, q)
    #
    #     # 拼 arrival rates 并归一化
    #     prob_transitions = torch.hstack((self.alpha.unsqueeze(0).repeat(B * action_repeat, 1), pmu_flat))  # (A*B, 2q)
    #     prob_transitions = prob_transitions / prob_transitions.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    #
    #     # 对 A 次采样求均值（蒙特卡洛近似）
    #     prob_transitions = torch.stack(torch.chunk(prob_transitions, action_repeat, dim=0), dim=0).mean(dim=0)  # (B, 2q)
    #     prob_transitions = prob_transitions.view(B * (2 * self.q), 1)  # (B*2q, 1)
    #
    #     # 构造转移后的状态集合
    #     Px = obs.unsqueeze(1) + self.D.unsqueeze(0).repeat(B, 1, 1)  # (B, 2q, q)
    #     Px = torch.relu(Px).view(B * (2 * self.q), self.q)           # (B*2q, q)
    #
    #     # 评估期望值
    #     Pfx = (prob_transitions * self.predict_values(Px)).view(B, 2 * self.q, 1).sum(dim=1)  # (B, 1)
    #     return Pfx
    #
    # @torch.no_grad()
    # def predict_next_states(self, obs: Tensor, action_repeat: int) -> Tensor:
    #     obs = obs.view(-1, self.q)
    #     AMP_values = self.AMP(obs, action_repeat)
    #     return AMP_values

    def get_prob_act(self, obs, deterministic: bool = False):
        obs = self._to_obs_tensor_preserve_batch(obs)  # (*B, D)
        obs_for_mask = obs[..., : self.q]
        input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
        logits = self._policy_logits(input_obs)  # (*B, s, q)
        action_probs = self._masked_action_probs(obs_for_mask, logits)
        if self.randomize and not deterministic:
            action = OneHotCategorical(probs=action_probs).sample()
        else:
            action_idx = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_idx, num_classes=self.q).float()
        return action, action_probs

    def evaluate_actions(self, obs, actions):
        obs = self._to_obs_tensor_preserve_batch(obs)  # (*B, D)
        obs_for_mask = obs[..., : self.q]
        input_obs = self.standardize_queues(self._maybe_time_feature_crop(obs))
        logits = self._policy_logits(input_obs)  # (*B, s, q)
        action_probs = self._masked_action_probs(obs_for_mask, logits)
        actions = actions.view(*actions.shape[:-2], self.s, self.q).float()  # 保留前导维
        selected = (actions * action_probs).sum(dim=-1).clamp_min(1e-8)  # (*B, s)
        log_prob = torch.log(selected).sum(dim=-1)  # (*B,)
        entropy = None
        return log_prob, entropy

    def evaluate_values(self, obs):
        return self.value_only(obs)  # (*B, 1)

    def predict_values(self, obs):
        v = self.value_only(obs)
        return v  # (*B, 1)

    @torch.no_grad()
    def AMP(self, obs, action_repeat: int):
        obs = self._to_obs_tensor_preserve_batch(obs)  # (B, q)
        x = self.standardize_queues(obs)
        B = x.size(0)
        logits = self._policy_logits(x)
        action_probs = self._masked_action_probs(obs[..., : self.q], logits)
        pr = action_probs.repeat_interleave(action_repeat, dim=0)  # (A*B, s, q)
        action = OneHotCategorical(probs=pr).sample()  # (A*B, s, q)
        pmu = action * self.mu[0].unsqueeze(0).repeat(B * action_repeat, 1, 1)
        pmu_flat = pmu.sum(dim=1)  # (A*B, q)
        prob_transitions = torch.hstack((self.alpha.unsqueeze(0).repeat(B * action_repeat, 1), pmu_flat))
        prob_transitions = prob_transitions / prob_transitions.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        prob_transitions = torch.stack(torch.chunk(prob_transitions, action_repeat, dim=0), dim=0).mean(
            dim=0)  # (B, 2q)
        prob_transitions = prob_transitions.view(B * (2 * self.q), 1)
        Px = obs.unsqueeze(1) + self.D.unsqueeze(0).repeat(B, 1, 1)  # (B, 2q, q)
        Px = torch.relu(Px).view(B * (2 * self.q), self.q)  # (B*2q, q)
        Pfx = (prob_transitions * self.predict_values(Px)).view(B, 2 * self.q, 1).sum(dim=1)  # (B, 1)
        return Pfx

