import torch

class MaxPressurePolicy:
    def __init__(self, queue_event_options):
        self.queue_event_options = queue_event_options

    # def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
    #     q = repeated_network.shape[-1]
    #     A = self.queue_event_options[q:]
    #     return repeated_mu * (torch.sum(-A.unsqueeze(0).repeat(repeated_queue.shape[0], 1, 1) * repeated_queue[:, 0].unsqueeze(2).repeat(1, 1, q) * repeated_h[:, 0].unsqueeze(2).repeat(1, 1, q), 2)).unsqueeze(1).repeat(1, repeated_network.shape[1], 1)

        # 仅修复 device/dtype，不改变任何原有计算逻辑与张量形状

    def test_forward(self, step, batch_queue, batch_time,
                     repeated_queue, repeated_network, repeated_mu, repeated_h):
        q = repeated_network.shape[-1]
        device = repeated_queue.device
        dtype = repeated_queue.dtype

        # A 放到与 repeated_queue 相同的 device/dtype
        A = torch.as_tensor(self.queue_event_options[q:], dtype=dtype, device=device)

        # 参与运算的输入也放到同一 device（保持你原本的 repeat 逻辑）
        rq0 = repeated_queue[:, 0].to(device)
        rh0 = repeated_h[:, 0].to(device)
        rmu = repeated_mu.to(device)

        # 原表达式（未改动数学逻辑，仅保证 device 一致）
        out = rmu * (
            torch.sum(
                -A.unsqueeze(0).repeat(repeated_queue.shape[0], 1, 1)
                * rq0.unsqueeze(2).repeat(1, 1, q)
                * rh0.unsqueeze(2).repeat(1, 1, q),
                2
            )
        ).unsqueeze(1).repeat(1, repeated_network.shape[1], 1)

        return out
