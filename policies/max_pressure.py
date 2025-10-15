import torch

class MaxPressurePolicy:
    def __init__(self, queue_event_options):
        self.queue_event_options = queue_event_options

    # def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
    #     q = repeated_network.shape[-1]
    #     A = self.queue_event_options[q:]
    #     return repeated_mu * (torch.sum(-A.unsqueeze(0).repeat(repeated_queue.shape[0], 1, 1) * repeated_queue[:, 0].unsqueeze(2).repeat(1, 1, q) * repeated_h[:, 0].unsqueeze(2).repeat(1, 1, q), 2)).unsqueeze(1).repeat(1, repeated_network.shape[1], 1)

    def test_forward(self, step, batch_queue, batch_time,
                     repeated_queue, repeated_network, repeated_mu, repeated_h):
        q = repeated_network.shape[-1]
        device = repeated_queue.device
        dtype = repeated_queue.dtype

        # 提取离开事件的 Q×Q 子矩阵，并搬到相同 device
        A = torch.as_tensor(self.queue_event_options[q:, :], dtype=dtype, device=device)

        # [B, n]
        Q0 = repeated_queue[:, 0]
        H0 = repeated_h[:, 0]

        # 用广播代替 repeat（高效且无 device 冲突）
        pressure = -(A.unsqueeze(0) * Q0.unsqueeze(2) * H0.unsqueeze(2)).sum(dim=2)  # [B, n]
        out = (repeated_mu.to(device) * pressure)[:, None, :].expand(-1, repeated_network.shape[1], -1)
        return out
