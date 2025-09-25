# RL/models/continuous.py
import numpy as np
import torch
from torch import nn
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from torch.distributions import Normal
from torchrl.data.tensor_specs import DiscreteTensorSpec  # 仅用于断言/提示


class _SplitLocScale(nn.Module):
    """将 backbone 输出的向量切分为 loc/scale，并 reshape 回 action_shape。"""
    def __init__(self, backbone: nn.Module, action_shape):
        super().__init__()
        self.backbone = backbone
        self.action_shape = tuple(action_shape)
        self.flat_dim = int(np.prod(self.action_shape))
        self.softplus = nn.Softplus()
    def forward(self, td):
        x = td.get("obs")
        out = self.backbone(x)  # (..., 2*flat_dim)
        loc, raw = out[..., :self.flat_dim], out[..., self.flat_dim:]
        loc   = loc.view(*loc.shape[:-1], *self.action_shape)
        scale = self.softplus(raw).view(*raw.shape[:-1], *self.action_shape) + 1e-5
        td.set("loc", loc); td.set("scale", scale)
        return td

def build_continuous_actor_critic(
    obs_dim: int,
    action_spec,                 # torchrl spec，连续：Bounded(...)
    hidden_sizes=(64, 64),
    in_key="obs",
    activation=nn.Tanh,
):
    # 断言连续动作
    assert not isinstance(action_spec, DiscreteTensorSpec), "当前构建器仅支持连续动作"
    act_shape = tuple(action_spec.shape)
    flat_dim = int(np.prod(act_shape))

    # Actor backbone：输出 2*flat_dim（loc+scale）
    actor_backbone = MLP(
        in_features=obs_dim,
        out_features=2 * flat_dim,
        depth=len(hidden_sizes),
        num_cells=hidden_sizes,
        activation_class=activation,
    )
    split = _SplitLocScale(actor_backbone, act_shape)

    actor_td = TensorDictModule(split, in_keys=[in_key], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=actor_td,
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=Normal,
        distribution_kwargs={"validate_args": False},
        return_log_prob=True,
        # default_interaction_mode="random",
    )

    # Critic
    critic_backbone = MLP(
        in_features=obs_dim,
        out_features=1,
        depth=len(hidden_sizes),
        num_cells=hidden_sizes,
        activation_class=activation,
    )
    critic = ValueOperator(module=critic_backbone, in_keys=[in_key])

    return actor, critic
