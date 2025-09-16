import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import trange
import numpy as np
from torchrl.envs import ParallelEnv


class BCD(Dataset):
    def __init__(self, num_samples, network):
        self.num_samples = num_samples
        if isinstance(network, torch.Tensor):
            self.network = network.float()
        else:
            self.network = torch.tensor(network, dtype=torch.float32)
        assert self.network.dim() == 2, f"network must be 2D (s,q), got {self.network.shape}"
        self.s, self.q = self.network.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        obs = torch.randint(0, 101, (self.q,), dtype=torch.float32)
        base = F.softmax(obs, dim=-1)             # (q,)
        action_probs = base.unsqueeze(0).repeat(self.s, 1)  # (s,q)
        action_probs = action_probs * self.network
        pos_mask = (obs > 0).float().unsqueeze(0).repeat(self.s, 1)
        action_probs = action_probs * pos_mask
        row_sum = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        zero_row = (row_sum <= 1e-12).float()
        action_probs = action_probs + zero_row * self.network
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return obs, action_probs


class ParallelEvalTorchRL:
    def __init__(self, actor, make_env_fn, eval_freq, eval_t, test_seed,
                 test_batch, device, bc=False, per_iter_normal_obs=False,
                 verbose=1):
        self.actor = actor
        self.make_env_fn = make_env_fn
        self.eval_freq = eval_freq
        self.eval_t = eval_t
        self.test_seed = test_seed
        self.test_batch = test_batch
        self.device = device
        self.bc = bc
        self.per_iter_normal_obs = per_iter_normal_obs
        self.verbose = verbose
        self.iter = 0

        # 构造 test 环境
        env_fns = [lambda s=seed: self.make_env_fn(s)
                   for seed in range(test_seed, test_seed + test_batch)]
        self.eval_env = ParallelEnv(test_batch, env_fns).to(device)

    def behavior_cloning(self, network):
        print(f'---------------------behavior_cloning---------------------')
        dataset = BCD(num_samples=100000, network=network)
        loader = DataLoader(dataset, batch_size=self.test_batch, shuffle=True)
        optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        for obs, target in loader:
            obs, target = obs.to(self.device), target.to(self.device)
            optim.zero_grad()
            dist = self.actor(obs)
            # 假设 actor 输出 dist，支持 log_prob/probs
            probs = dist.probs
            loss = F.mse_loss(probs, target)
            loss.backward()
            optim.step()

    @torch.no_grad()
    def eval(self):
        self.iter += 1
        if self.verbose:
            print(f'iter: {self.iter}')

        td = self.eval_env.reset()
        total_rewards = torch.zeros(self.test_batch, device=self.device)
        total_time = torch.zeros(self.test_batch, device=self.device)
        time_weighted_qlen = None

        for _ in trange(self.eval_t):
            dist = self.actor(td)
            td.set("action", dist.sample())
            td = self.eval_env.step(td)

            reward = td["next", "reward"].squeeze(-1)
            total_rewards += reward

            queues = td["next", "queues"]
            dt = td["next", "time"].squeeze(-1)
            if time_weighted_qlen is None:
                time_weighted_qlen = torch.zeros(
                    self.test_batch, queues.shape[-1], device=self.device
                )
            time_weighted_qlen += queues * dt.unsqueeze(-1)
            total_time += dt
            td = td["next"]

        avg_reward = total_rewards.mean().item()
        qlen_per_env = time_weighted_qlen / total_time.unsqueeze(-1)
        avg_qlen_per_queue = qlen_per_env.mean(0)
        overall_ql_mean = qlen_per_env.mean(1).mean().item()
        overall_ql_std = qlen_per_env.mean(1).std(unbiased=True).item()
        overall_ql_se = overall_ql_std / np.sqrt(self.test_batch)

        if self.verbose:
            print("------ TorchRL Eval ------")
            print(f"avg reward: {avg_reward:.4f}")
            print(f"avg queue length per q: {avg_qlen_per_queue.cpu().numpy().round(4).tolist()}")
            print(f"overall avg queue length: {overall_ql_mean:.6f}")
            print(f"overall queue length SE:  {overall_ql_se:.6f}")

        q_mean = avg_qlen_per_queue.mean()
        q_std = avg_qlen_per_queue.std(unbiased=True)
        return q_mean, q_std, avg_reward, overall_ql_mean, overall_ql_se
