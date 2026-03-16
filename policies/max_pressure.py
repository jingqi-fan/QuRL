import torch

class MaxPressurePolicy:
    def __init__(self, queue_event_options):
        self.queue_event_options = queue_event_options

    def test_forward(self, step, batch_queue, batch_time,
                     repeated_queue, repeated_network, repeated_mu, repeated_h):
        q = repeated_network.shape[-1]
        device = repeated_queue.device
        dtype = repeated_queue.dtype

        A = torch.as_tensor(self.queue_event_options[q:], dtype=dtype, device=device)
        rq0 = repeated_queue[:, 0].to(device)
        rh0 = repeated_h[:, 0].to(device)
        rmu = repeated_mu.to(device)

        out = rmu * (
            torch.sum(
                -A.unsqueeze(0).repeat(repeated_queue.shape[0], 1, 1)
                * rq0.unsqueeze(2).repeat(1, 1, q)
                * rh0.unsqueeze(2).repeat(1, 1, q),
                2
            )
        ).unsqueeze(1).repeat(1, repeated_network.shape[1], 1)

        return out
