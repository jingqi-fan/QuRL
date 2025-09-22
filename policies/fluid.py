#!/user/ewc2119/.conda/envs/CheWorld/bin/python3
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import os
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import json
import torch.distributions.one_hot_categorical as one_hot_sample
import yaml

import cvxpy as cvx
from multiprocessing import Pool

# Optimize over plans
def FluidSolver(obs, H, s, q, h, mu, queue_event_options, delta_t = 1):
    
    # eval batch is either 1 or equal to dq.batch
    queues, time = obs
    queues = queues.detach().numpy()

    h = h.numpy()
    A = queue_event_options[q:].numpy()
    mu_flat = mu.numpy().flatten()

    start_time = time[0].detach().numpy()
    N = round(H / delta_t)

    x = cvx.Variable((N, q))
    u = cvx.Variable((N, s * q))

    dynamic_constraints = [x[0] == queues]
    nonneg_constraints = [x[0] >= 0]
    action_feasibility = [u[0] >= 0]
    action_feasibility += [cvx.sum(cvx.reshape(u[0], (s, q), 'C'), axis = 1) <= np.ones(s)]
    action_feasibility += [cvx.sum(cvx.reshape(u[0], (s, q), 'C'), axis = 0) <= x[0]]

    for k in range(1, N):
        time = start_time + k * delta_t
        dynamic_constraints += [x[k] == x[k-1] + delta_t * (torch.ones(q) - cvx.sum(cvx.reshape(cvx.multiply(mu_flat, u[k-1]), (s, q), 'C') @ (-A), axis = 0))]
        nonneg_constraints += [x[k] >= 0]
        
        action_feasibility += [u[k] >= 0]
        action_feasibility += [cvx.sum(cvx.reshape(u[k], (s, q), 'C'), axis = 1) <= np.ones(s)]
        action_feasibility += [cvx.sum(cvx.reshape(u[k], (s, q), 'C'), axis = 0) <= np.ones(q)]

    constraints = dynamic_constraints + nonneg_constraints + action_feasibility
    holding_cost = cvx.sum((x @ h) * delta_t) / H
    objective = cvx.Minimize(holding_cost)

    prob = cvx.Problem(objective, constraints)
    min_cost = prob.solve()
    min_plan = torch.tensor(u.value.reshape(N, s, q, order = 'C'))

    return min_plan, min_cost

def solve_fluid(obs, H, s, q, h, mu, queue_event_options, delta_t):
        return FluidSolver(obs, H, s, q, h, mu, queue_event_options, delta_t)



class FluidPolicy:
    def __init__(self, queue_event_options):
        self.freq = 200
        self.delta_t =10
        self.H = 1000
        self.queue_event_options = queue_event_options
    
    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        s = repeated_network.shape[-2]
        q = repeated_network.shape[-1]
        h = repeated_h[0, 0]
        mu = repeated_mu[0]

        if step % self.freq == 0:
            self.start_time = batch_time.detach()

            batch_obs = [(batch_queue[idx], batch_time[idx]) for idx in range(len(batch_time))]

            with Pool(10) as pool:
                results = pool.starmap(solve_fluid, [(obs, self.H, s, q, h, mu, self.queue_event_options, self.delta_t) for obs in batch_obs])

            self.plan, self.cost = zip(*results)
            self.plan = torch.stack(self.plan)

        # batch_k = ((batch_time - self.start_time) / self.delta_t).int()
        # # print(batch_k)
        # # print(self.plan.shape)
        # pr = F.relu(torch.stack([self.plan[batch_idx][batch_k[batch_idx]] for batch_idx in range(len(batch_time))]))
        # pr = pr[:,0,:,:]

        # 1) 计算 elapsed 和索引（用 floor 更明确），并转 long
        elapsed = (batch_time - self.start_time) / self.delta_t
        batch_k = torch.floor(elapsed).to(torch.long)

        # 2) 计划长度
        N = self.plan.shape[1]

        # 3) 如果任何样本已经打到或超过右边界，立刻重解一次（按当前 batch）
        if (batch_k >= N - 1).any():
            # 重新求解（和你 step%freq==0 那块相同的流程）
            batch_obs = [(batch_queue[idx], batch_time[idx]) for idx in range(len(batch_time))]
            # ... 调用 solve_fluid 并得到 self.plan ...
            self.start_time = batch_time.detach()  # reset 起始时间
            # 重新计算索引
            elapsed = (batch_time - self.start_time) / self.delta_t
            batch_k = torch.floor(elapsed).to(torch.long)

        # 4) 夹紧索引，确保万无一失
        batch_k = torch.clamp(batch_k, 0, N - 1)

        # 5) 取对应计划片段
        pr = F.relu(torch.stack([self.plan[bi, batch_k[bi]] for bi in range(len(batch_time))]))
        pr = pr[:, 0, :, :]

        return pr

