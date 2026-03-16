import numpy as np
import scipy.optimize as opt
import scipy.sparse as sparse
import torch
import torch.nn.functional as F


def match_constraint_mat(num_s, num_q, f_fluid=False):
    # rhs = np.append(data['s'],-data['d'])

    # form A
    # column (i,j)=n*i+j has two nonzeroes:
    #    1 at row i with rhs supply(i)
    #    1 at row N+j with rhs demand(j)
    N = num_s
    M = num_q
    NZ = 2 * N * M
    irow = np.zeros(NZ, dtype=int)
    jcol = np.zeros(NZ, dtype=int)
    value = np.zeros(NZ)
    for i in range(N):
        for j in range(M):
            k = M * i + j
            k1 = 2 * k
            k2 = k1 + 1
            irow[k1] = i
            jcol[k1] = k
            value[k1] = 1.0
            if not f_fluid:
                irow[k2] = N + j
                jcol[k2] = k
                value[k2] = 1.0

    A = sparse.coo_matrix((value, (irow, jcol)))

    return A


def linear_assignment(values, servers, jobs):
    s, q = values.shape
    A = match_constraint_mat(s, q).toarray()
    c = np.reshape(-values, s * q)
    b = np.append(servers, jobs)

    res = opt.linprog(c=c, A_ub=A, b_ub=b, method='highs-ds')
    X = np.reshape(res.x, (s, q))

    X = np.rint(X)[:s - 1, :q - 1].tolist()

    return torch.tensor(X)


def linear_assignment_batch(values: torch.Tensor,
                            s_bar: torch.Tensor,
                            q_bar: torch.Tensor) -> torch.Tensor:
    """
    values: [B, S, Q] Cost/utility matrix (minimizing using -v as per your original logic)
    s_bar : [B, S]    Server capacities
    q_bar : [B, Q]    Job demands
    return: [B, S-1, Q-1]
    """
    assert values.ndim == 3, "values must be [B,S,Q]"
    B, S, Q = values.size()

    # Pre-generate the constraint matrix (only depends on S, Q)
    A = match_constraint_mat(S, Q).toarray()

    actions = []
    for b_idx in range(B):
        # --- Move everything to CPU then convert to numpy ---
        v = values[b_idx].detach().cpu().numpy()  # [S,Q]
        servers = s_bar[b_idx].detach().cpu().numpy()  # [S]
        jobs = q_bar[b_idx].detach().cpu().numpy()  # [Q]

        # Linear programming: minimize c^T x, subject to A_ub x <= b_ub, x >= 0
        c = np.reshape(-v, S * Q)
        bvec = np.append(servers, jobs)

        # Set bounds to non-negative
        res = opt.linprog(c=c, A_ub=A, b_ub=bvec, bounds=(0, None), method='highs-ds')

        if not res.success:
            raise RuntimeError(f"[linear_assignment_batch] linprog failed at batch {b_idx}: {res.message}")

        X = np.reshape(res.x, (S, Q))  # Back to [S, Q]
        X = np.rint(X)[:S - 1, :Q - 1]
        actions.append(torch.from_numpy(X))

    # Stack back into a batch, and return to the same device / dtype as input
    out = torch.stack(actions, dim=0)  # [B, S-1, Q-1]
    out = out.to(device=values.device, dtype=values.dtype)
    return out


def pad(vals, queues, network,
        device='cpu', compliance=True):
    # setup mu bar
    batch = network.size()[0]
    s = network.size()[1]
    q = network.size()[2]

    free_servers = torch.ones((batch, s)).to(device)

    pad_q = -torch.ones((batch, 1, q)).to(device)
    pad_s = -torch.ones((batch, s + 1, 1)).to(device)

    if compliance:
        vals = vals * network - 1 * (network == 0.).to(device)

    v = torch.cat((vals, pad_q), 1)
    v = torch.cat((v, pad_s), 2)

    excess_server = F.relu(s - torch.sum(queues, dim=1)).unsqueeze(1).to(device)
    q_bar = torch.hstack((queues, excess_server)).to(device)

    excess_queues = F.relu(torch.sum(queues, dim=1) - s).unsqueeze(1).to(device)
    s_bar = torch.hstack((free_servers, excess_queues)).to(device)

    return v, s_bar, q_bar


def pad_pool(vals, queues, network, server_pool_size,
             device='cpu', compliance=True):
    """
    vals:   [B,S,Q]  Allocation scores/values (larger is better)
    queues: [B,Q]    Current number of jobs in each queue (demand)
    network:[B,S,Q]  0/1 connectivity mask (or [S,Q] then expanded to [B,S,Q] outside)
    server_pool_size: [S] Parallel capacity for each server (float)
    return:
      v:     [B,S+1,Q+1]  Padded value matrix (will be converted to cost in Sinkhorn)
      s_bar: [B,S+1]      Server-side capacity (the last dimension is the virtual server capacity for "excess queues")
      q_bar: [B,Q+1]      Queue-side capacity (the last dimension is the virtual queue capacity for "excess servers")
    """
    device = torch.device(device)
    vals = vals.to(device).float()
    queues = queues.to(device).float()
    # network can be [S,Q] or [B,S,Q]; unify to [B,S,Q]
    if network.dim() == 2:
        network = network.unsqueeze(0).expand(vals.size(0), -1, -1)
    network = network.to(device).float()
    server_pool_size = server_pool_size.to(device).float()  # Key info: Ensure it is float

    B, S, Q = vals.shape

    # Total server-side capacity s (scalar tensor, keeping device)
    s_total = server_pool_size.sum()  # shape: [] (scalar tensor on device)

    # Server capacity row vector per batch [B,S]
    free_servers = server_pool_size.view(1, S).expand(B, -1)  # [B,S]

    # Pad rows/cols (-1e9 is safer than -1 to avoid selecting invalid edges)
    big_neg = -1e9
    pad_q = torch.full((B, 1, Q), big_neg, device=device)  # A row for the "virtual queue"
    pad_s = torch.full((B, S + 1, 1), big_neg, device=device)  # A column for the "virtual server"

    # Compliance mask (assign large negative values to invalid edges to prevent selection)
    if compliance:
        vals = vals * network + big_neg * (network == 0)

    # v: First add a row, then add a column -> [B,S+1,Q+1]
    v = torch.cat((vals, pad_q), dim=1)
    v = torch.cat((v, pad_s), dim=2)

    # q_bar (queue-side capacity) = [queues, excess_server]
    # excess_server = max(0, s_total - sum_q) ; sum_q: [B]
    excess_server = F.relu(s_total - queues.sum(dim=1)).unsqueeze(1)  # [B,1]
    q_bar = torch.hstack((queues, excess_server))  # [B,Q+1]

    # s_bar (server-side capacity) = [free_servers, excess_queues]
    # excess_queues = max(0, sum_q - s_total)
    excess_queues = F.relu(queues.sum(dim=1) - s_total).unsqueeze(1)  # [B,1]
    s_bar = torch.hstack((free_servers, excess_queues))  # [B,S+1]

    return v, s_bar, q_bar


class Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, a, b, num_iter, temp, eps=1e-6, back_temp=None, device='cpu'):
        """
        c:   [B,M,N]  Cost matrix (you will pass -v in)
        a:   [B,M]    Row marginals (server-side capacity)
        b:   [B,N]    Column marginals (queue-side capacity)
        """
        device = torch.device(device)
        c = c.to(device).float()
        a = a.to(device).float()
        b = b.to(device).float()

        # log-domain Sinkhorn, better stability
        log_p = -c / temp  # [B,M,N]
        log_a = torch.log(a.clamp_min(eps)).unsqueeze(2)  # [B,M,1]
        log_b = torch.log(b.clamp_min(eps)).unsqueeze(1)  # [B,1,N]

        for _ in range(num_iter):
            log_p -= (torch.logsumexp(log_p, dim=1, keepdim=True) - log_b)  # Column normalization -> b
            log_p -= (torch.logsumexp(log_p, dim=2, keepdim=True) - log_a)  # Row normalization -> a

        p = torch.exp(log_p)  # [B,M,N]
        # Save p, row/col sums for backward
        ctx.save_for_backward(p, p.sum(dim=2), p.sum(dim=1))
        ctx.temp = temp
        ctx.back_temp = back_temp
        ctx.device = device
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors  # a=[B,M], b=[B,N]
        B, M, N = p.shape
        device = ctx.device

        a = a.clamp_min(1e-1)
        b = b.clamp_min(1e-1)

        scale = -1.0 / (ctx.back_temp if ctx.back_temp is not None else ctx.temp)
        grad_p = grad_p.to(device) * (scale * p)

        # Block matrix K_b (add a little to the diagonal for numerical stability)
        K_b = torch.cat((
            torch.cat((torch.diag_embed(a), p), dim=2),
            torch.cat((p.transpose(1, 2), torch.diag_embed(b)), dim=2)
        ), dim=1)[:, :-1, :-1]  # [B, M+N-1, M+N-1]

        I = torch.eye(K_b.size(1), device=device)
        K_b = K_b + 1e-2 * I.unsqueeze(0)  # stability

        t_b = torch.cat((
            grad_p.sum(dim=2),  # [B,M]
            grad_p[:, :, :-1].sum(dim=1)  # [B,N-1]
        ), dim=1).unsqueeze(2)  # [B, M+N-1, 1]

        grad_ab_b = torch.linalg.solve(K_b, t_b)  # [B, M+N-1, 1]
        grad_a_b = grad_ab_b[:, :M, :]
        grad_b_b = torch.cat(
            (grad_ab_b[:, M:, :], torch.zeros((B, 1, 1), dtype=torch.float32, device=device)),
            dim=1
        )

        U = grad_a_b + grad_b_b.transpose(1, 2)  # [B,M,1] + [B,1,N] -> broadcast

        grad_p = grad_p - p * U  # chain rule correction
        return grad_p, None, None, None, None, None, None, None
