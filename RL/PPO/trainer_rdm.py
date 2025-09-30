# random_trainer.py
import math, time
import torch
from tensordict import TensorDict
from torch.distributions import Categorical
from torchrl.envs import EnvBase

class RandomPolicyTrainer:
    def __init__(self, train_env: EnvBase, eval_env: EnvBase, args, ct=None, print_fn=print):
        self.train_env = train_env
        self.eval_env = eval_env
        self.args = args
        self.ct = ct
        self.print = print_fn
        self.S, self.Q = args.S, args.Q

    def pre_train(self):
        # 随机策略，不需要 BC
        pass

    def learn(self):
        self.print("Start training with RANDOM policy (no learning)")
        for epoch in range(self.args.total_epochs):
            self.ct.get_end_time(time.time())
            print(f'get total time {self.ct.get_total_time():.2f}s')
            # 直接跑评估
            mean_r, std_r = self.evaluate()
            self.print(f"[Epoch {epoch+1}/{self.args.total_epochs}] (Random) "
                       f"Eval: return mean {mean_r:.4f} ± {std_r:.4f}")

    @torch.no_grad()
    def evaluate(self):
        device = torch.device(self.args.device)
        td = self.eval_env.reset()
        B = self.args.test_batch

        total_r = torch.zeros(B, device=device)
        qlens_list, costs_list = [], []

        for _ in range(self.args.eval_T):
            obs_dev = td["obs"].to(device)
            # 队列长度
            queues = obs_dev[:, :self.Q]
            qlens_list.append(queues.sum(dim=-1))

            # 随机动作：每个 S 独立均匀采样
            a = torch.zeros(B, self.S, self.Q, device=device)
            for s in range(self.S):
                cat = Categorical(probs=torch.ones(B, self.Q, device=device) / self.Q)
                idx = cat.sample()
                a[torch.arange(B), s, idx] = 1.0

            # 环境一步
            out = self.eval_env.step(TensorDict({"action": a.to(self.eval_env.device)}, batch_size=[B]))
            nxt = out["next"]
            r_t = nxt["reward"].reshape(B).to(device)
            total_r += r_t

            if "cost" in nxt.keys():
                c_t = nxt["cost"].reshape(B).to(device)
            else:
                c_t = -r_t
            costs_list.append(c_t)
            td = nxt

        # 统计
        T, N = self.args.eval_T, self.args.eval_T * B
        qlens = torch.stack(qlens_list, dim=0).reshape(N)
        costs = torch.stack(costs_list, dim=0).reshape(N)

        q_mean = qlens.mean().item()
        q_se = (qlens.std(unbiased=True) / math.sqrt(max(1, N))).item()
        cost_mean = costs.mean().item()
        cost_se = (costs.std(unbiased=True) / math.sqrt(max(1, N))).item()

        ret_mean = total_r.mean().item()
        ret_std  = total_r.std(unbiased=True).item()

        self.print(f"  Eval (T={T}, B={B}): "
                   f"queue mean {q_mean:.4f}, SE {q_se:.4f} | "
                   f"cost mean {cost_mean:.4f}, SE {cost_se:.4f}")

        return ret_mean, ret_std
