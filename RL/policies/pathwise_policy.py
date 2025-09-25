from torch import nn

class Pathwise_Policy(nn.Module):
    """
    TorchRL-friendly Vanilla Policy (不依赖 stable-baselines3).
    输入: obs (B, q)
    输出: forward -> action(one-hot, B,q), values(B,1), log_prob(B,)
    """
    ...