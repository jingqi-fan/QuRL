import torch
from torch import nn
from tensordict import TensorDict

# 轻量Sequential：依次对 tensordict 调用子模块
class TDSequential(nn.Module):
    def __init__(self, *mods):
        super().__init__()
        self.mods = nn.ModuleList(mods)

    def forward(self, td: TensorDict) -> TensorDict:
        for m in self.mods:
            td = m(td)
        return td

# 把普通 nn.Module 变成 “in_keys -> out_key” 的 tensordict 模块
class TDLinearHead(nn.Module):
    def __init__(self, net: nn.Module, in_key: str, out_key: str):
        super().__init__()
        self.net = net
        self.in_key = in_key
        self.out_key = out_key

    def forward(self, td: TensorDict) -> TensorDict:
        x = td.get(self.in_key)
        y = self.net(x)
        td.set_(self.out_key, y)
        return td

# 产生 state-independent 的 scale（可训练 log_std）
class StateIndependentScale(nn.Module):
    def __init__(self, in_key: str = "loc", out_key: str = "scale",
                 init_scale: float = 0.5, min_scale: float = 1e-4, max_scale: float = 1e3):
        super().__init__()
        self.in_key = in_key
        self.out_key = out_key
        self.log_std = nn.Parameter(torch.log(torch.tensor(init_scale, dtype=torch.float32)))
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, td: TensorDict) -> TensorDict:
        loc = td.get(self.in_key)
        scale = self.log_std.exp().clamp_(min=self.min_scale, max=self.max_scale).expand_as(loc)
        td.set_(self.out_key, scale)
        return td
