# RL/models/continuous.py
import numpy as np
import torch
from torch import nn
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torch.distributions import Normal
from torchrl.data.tensor_specs import DiscreteTensorSpec  # 仅用于断言/提示


class _SplitLocScale(nn.Module):
    """backbone(x) -> (loc, scale)，loc/scale 形状与 action_shape 一致。"""
    def __init__(self, backbone: nn.Module, action_shape):
        super().__init__()
        self.backbone = backbone
        self.action_shape = tuple(action_shape)
        self.flat_dim = int(np.prod(self.action_shape))
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        # x: (..., obs_dim)
        out = self.backbone(x)  # (..., 2*flat_dim)
        loc_flat, raw_flat = out[..., :self.flat_dim], out[..., self.flat_dim:]
        loc   = loc_flat.view(*loc_flat.shape[:-1], *self.action_shape)
        scale = self.softplus(raw_flat).view(*raw_flat.shape[:-1], *self.action_shape) + 1e-5
        # 返回张量序列/元组，TensorDictModule 会按 out_keys 写入
        return loc, scale


def build_continuous_actor_critic(
    obs_dim: int,
    action_spec,                 # torchrl 连续动作 spec：Bounded(...)
    hidden_sizes=(64, 64),
    in_key="obs",
    activation=nn.Tanh,
):
    # 断言连续动作
    assert not isinstance(action_spec, DiscreteTensorSpec), "当前构建器仅支持连续动作"
    act_shape = tuple(action_spec.shape)
    flat_dim = int(np.prod(act_shape))

    # === Actor ===
    actor_backbone = MLP(
        in_features=obs_dim,
        out_features=2 * flat_dim,            # 输出 loc + scale
        depth=len(hidden_sizes),
        num_cells=hidden_sizes,
        activation_class=activation,
    )
    split = _SplitLocScale(actor_backbone, act_shape)

    # 注意：in_keys=['obs'] => forward 接收 obs 张量；out_keys 写回到 TD
    actor_td = TensorDictModule(split, in_keys=[in_key], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=actor_td,
        in_keys=["loc", "scale"],             # 从 TD 取出分布参数
        spec=action_spec,
        distribution_class=Normal,            # 按需可用 Independent(Normal, 1)，但 ProbabilisticActor 会处理
        distribution_kwargs={"validate_args": False},
        return_log_prob=True,                 # 需要 sample_log_prob 给 PPO
        # default_interaction_mode="random",  # 由 ExplorationType 控制
    )

    # === Critic ===
    critic_backbone = MLP(
        in_features=obs_dim,
        out_features=1,
        depth=len(hidden_sizes),
        num_cells=hidden_sizes,
        activation_class=activation,
    )
    critic = ValueOperator(module=critic_backbone, in_keys=[in_key])  # 写入 'state_value'

    return actor, critic
