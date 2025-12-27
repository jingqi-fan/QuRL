# trainer_vectorized.py (Multi-Env Object Version - Identical Output)
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import math
from tensordict import TensorDict
from datetime import datetime

from utils.switchplot import create_plot_dir, create_loss_dir
import utils.routing as rt
from main.env import BatchedDiffDES


class Trainer:
    def __init__(self, model_config, env_config, policy, optimizer,
                 draw_service, draw_inter_arrivals, experiment_name, draw_due_date=None):
        self.model_config = model_config
        self.env_config = env_config
        self.policy = policy
        self.optimizer = optimizer
        self.draw_service = draw_service
        self.draw_inter_arrivals = draw_inter_arrivals
        self.draw_due_date = draw_due_date

        self.test_loss = []
        self.fig_dir = create_plot_dir(self.model_config, self.env_config, experiment_name=experiment_name)
        self.loss_dir = create_loss_dir(self.model_config, self.env_config, experiment_name=experiment_name)
        self.experiment_name = experiment_name
        self.device = torch.device(self.model_config['env']['device'])

    def _make_envs(self, batch_size, seed_base):
        envs = []
        for i in range(batch_size):
            env = BatchedDiffDES(
                self.env_config['network'],
                self.env_config['mu'],
                torch.tensor(self.env_config['h']).float(),
                queue_event_options=self.env_config.get('queue_event_options', None),
                queue_event_options2=self.env_config.get('queue_event_options2', None),
                default_B=1,
                temp=self.model_config['env']['env_temp'],
                seed=seed_base + i,
                device=self.device,
                draw_service=self.draw_service,
                draw_inter_arrivals=self.draw_inter_arrivals,
                reentrant=self.env_config.get('reentrant', 0)
            )
            envs.append(env)
        return envs

    # ------------------------------ 训练 ------------------------------ #
    def train_epoch(self):
        B_train = self.model_config['opt']['train_batch']
        envs = self._make_envs(B_train, self.model_config['env']['train_seed'])
        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]

        self.optimizer.zero_grad()
        back_outs, nn_back_ins = [], []

        def action_hook(grad):
            back_outs.append(grad.detach().cpu().tolist())

        def priority_hook(grad):
            nn_back_ins.append(grad.detach().cpu().tolist())

        total_cost = torch.zeros((B_train, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_train, self.env_config['network'].shape[-1]), device=self.device)

        S, Q = envs[0].S, envs[0].Q
        shared_network = envs[0].network
        shared_h = envs[0].h.unsqueeze(0).expand(1, S, Q)
        shared_mu = envs[0].mu.view(1, S, Q)

        for _ in trange(self.env_config['train_T'], disable=True, leave=False):
            queues = torch.cat([td["queues"] for td in td_list], dim=0)
            time = torch.cat([td["time"] for td in td_list], dim=0)

            pr = self.policy.train_forward(queues, time, shared_network, shared_h, shared_mu)
            pr.register_hook(priority_hook)

            if self.model_config['policy']['train_policy'] == 'sinkhorn':
                lex = torch.zeros(B_train, S, Q, device=self.device)
                v, s_bar, q_bar = rt.pad_pool(2 * pr + lex, queues.detach(), network=shared_network,
                                              device=self.device, server_pool_size=self.env_config['server_pool_size'])
                pr = rt.Sinkhorn.apply(-v, s_bar, q_bar,
                                       self.model_config['policy']['sinkhorn']['num_iter'],
                                       self.model_config['policy']['sinkhorn']['temp'],
                                       self.model_config['policy']['sinkhorn']['eps'],
                                       self.model_config['policy']['sinkhorn']['back_temp'],
                                       self.device)[:, :S, :Q]
            elif self.model_config['policy']['train_policy'] == 'softmax':
                pr = F.softmax(pr, dim=-1) * shared_network.unsqueeze(0)
                pr = torch.minimum(pr, queues.unsqueeze(1).expand(-1, S, -1)).clamp_min(1e-4)
                pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

            action = pr
            action.register_hook(action_hook)

            new_td_list = []
            for i in range(B_train):
                out = envs[i].step(TensorDict({"action": action[i:i + 1]}, batch_size=[1]))
                total_cost[i] += out["cost"].squeeze(0)
                time_weight_queue_len[i] += (out["queues"] * out["event_time"]).squeeze(0)
                new_td_list.append(out.select("queues", "time"))
            td_list = new_td_list

        loss = torch.mean(total_cost / self.env_config['train_T'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(),
                                       max_norm=self.model_config['opt']['grad_clip_norm'])
        self.optimizer.step()

        current_time = torch.cat([td["time"] for td in td_list], dim=0)
        train_cost_per_env = (total_cost / current_time).squeeze(-1)
        twql_per_env = (time_weight_queue_len / current_time)
        print(f"train cost mean: {train_cost_per_env.mean().item():.6f}")
        print(f"train time-weighted mean queue len per q: {twql_per_env.mean(dim=0).tolist()}")

    # ------------------------------ 测试 ------------------------------ #
    def test_epoch(self, epoch):
        B_test = self.model_config['opt']['test_batch']
        envs = self._make_envs(B_test, self.model_config['env']['test_seed'])

        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]
        total_cost = torch.zeros((B_test, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_test, envs[0].Q), device=self.device)

        S, Q = envs[0].S, envs[0].Q
        pr_history = []

        with torch.no_grad():
            pbar = trange(self.env_config['test_T'],
                          desc=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} - {self.experiment_name}",
                          disable=True, leave=False)

            for step in pbar:
                # 聚合状态 [B, Q]
                queues = torch.cat([td["queues"] for td in td_list], dim=0)
                time = torch.cat([td["time"] for td in td_list], dim=0)

                # 构造 Policy 输入
                repeated_queue = queues.unsqueeze(1).expand(-1, S, -1)
                repeated_network = envs[0].network.unsqueeze(0).expand(B_test, -1, -1)
                repeated_mu = envs[0].mu.view(1, S, Q).expand(B_test, -1, -1)
                repeated_h = envs[0].h.view(1, 1, Q).expand(B_test, S, -1)

                pr = self.policy.test_forward(step, queues, time, repeated_queue, repeated_network, repeated_mu,
                                              repeated_h)

                if self.model_config['policy']['test_policy'] == 'sinkhorn':
                    v, s_bar, q_bar = rt.pad_pool(2 * pr, queues, network=envs[0].network, device=self.device,
                                                  server_pool_size=self.env_config['server_pool_size'])
                    pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, self.model_config['policy']['sinkhorn']['num_iter'],
                                           self.model_config['policy']['sinkhorn']['temp'],
                                           self.model_config['policy']['sinkhorn']['eps'],
                                           self.model_config['policy']['sinkhorn']['back_temp'], self.device)[:, :S, :Q]
                elif self.model_config['policy']['test_policy'] == 'linear_assigment':
                    v, s_bar, q_bar = rt.pad_pool(2 * pr, queues, network=envs[0].network, device=self.device,
                                                  server_pool_size=self.env_config['server_pool_size'])
                    pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                elif self.model_config['policy']['test_policy'] == 'softmax':
                    pr = F.softmax(pr, dim=-1) * repeated_network
                    pr = torch.minimum(pr, queues.unsqueeze(1).expand(-1, S, -1)).clamp_min(1e-4)
                    pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

                pr_history.append(pr.detach().cpu())
                action = torch.round(pr)

                # 分发步进并更新
                new_td_list = []
                for i in range(B_test):
                    out = envs[i].step(TensorDict({"action": action[i:i + 1]}, batch_size=[1]))
                    # 注意：ParallelEnv 模式下 step 返回的是 next 嵌套结构，这里模拟的是单环境 step
                    total_cost[i] += out["next", "reward"]
                    time_weight_queue_len[i] += (out["next", "queues"] * out["next", "event_time"]).squeeze(0)
                    new_td_list.append(out["next"].select("queues", "time"))
                td_list = new_td_list

        # 保存历史记录
        np.save("pr_history.npy", torch.stack(pr_history).numpy())

        # -------- 汇总测试指标 (与原代码逻辑完全一致) --------
        time_now = torch.cat([td["time"] for td in td_list], dim=0)  # [B, 1]
        cost_per_env = (total_cost / time_now).squeeze(-1)  # [B]
        test_cost_mean = cost_per_env.mean()
        test_cost_std = cost_per_env.std(unbiased=True)
        test_cost_se = test_cost_std / math.sqrt(B_test)

        qlen_per_env = (time_weight_queue_len / time_now)  # [B, Q]
        qlen_overall_per_env = qlen_per_env.mean(dim=1)  # [B]

        qlen_mean = qlen_overall_per_env.mean()
        qlen_std = qlen_overall_per_env.std(unbiased=True)
        qlen_se = qlen_std / math.sqrt(B_test)

        print(f'------------------------test result------------------------')
        print(f"experiment: {self.experiment_name}")
        print(f"queue length mean (overall): {qlen_mean.item():.4f}")
        print(f"queue length std  (overall): {qlen_std.item():.4f}")
        print(f"queue length se   (overall): {qlen_se.item():.4f}")
        print(f"test cost mean: {test_cost_mean.item():.4f}")
        print(f"test cost std : {test_cost_std.item():.4f}")
        print(f"test cost se  : {test_cost_se.item():.4f}")