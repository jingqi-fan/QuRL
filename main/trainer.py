# trainer_vectorized.py
from tqdm import trange
import torch
import torch.nn.functional as F
import math
from tensordict import TensorDict
from datetime import datetime

from configs.scripts.s_cmu import draw_due_date
# 你自己的工具（保持不变）
from utils.switchplot import create_plot_dir, create_loss_dir
# 如果你用到了 rt.*（pad_pool / Sinkhorn / linear_assignment_batch），请确保导入：
# import your_runtime_lib as rt

# from main.env import BatchedDiffDES
from main.env_s import BatchedDiffDES
import utils.routing as rt

class Trainer:
    """
    向量化版 Trainer：单实例 + 批量并行（default_B = batch_size）
    适配最新的 BatchedDiffDES（TorchRL EnvBase 版，step 返回 TensorDict，含 cost/event_time）。
    """

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

    # ------------------------------ 训练 ------------------------------ #
    def train_epoch(self):
        B_train = self.model_config['opt']['train_batch']  # 并行轨迹数
        # 单实例 batched 环境
        dq = BatchedDiffDES(
            self.env_config['network'],
            self.env_config['mu'],
            torch.tensor(self.env_config['h']).float(),
            queue_event_options=self.env_config.get('queue_event_options', None),
            queue_event_options2=self.env_config.get('queue_event_options2', None),
            default_B=B_train,
            temp=self.model_config['env']['env_temp'],
            seed=self.model_config['env']['train_seed'],
            device=self.device,
            draw_service=self.draw_service,
            draw_inter_arrivals=self.draw_inter_arrivals,
            draw_due_date=self.draw_due_date,
            reentrant=self.env_config.get('reentrant', 0)
        )

        # reset（用 gen_params 指定 batch 维）
        td = dq.reset(dq.gen_params(batch_size=[B_train]))  # td['queues']: [B,Q], td['time']: [B,1]

        self.optimizer.zero_grad()

        # 可选：记录梯度（保持你原本的 hooks）
        back_outs = []
        def action_hook(grad):
            back_outs.append(grad.detach().cpu().tolist())

        nn_back_ins = []
        def priority_hook(grad):
            nn_back_ins.append(grad.detach().cpu().tolist())

        total_cost = torch.zeros((B_train, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_train, self.env_config['network'].shape[-1]), device=self.device)

        S = dq.S
        Q = dq.Q

        for _ in trange(self.env_config['train_T'], disable=True, leave=False):
            queues = td["queues"]  # [B,Q]
            time   = td["time"]    # [B,1]

            # 注意：保持你 policy 接口一致
            pr = self.policy.train_forward(
                queues, time,
                dq.network,                                 # [S,Q]
                dq.h.unsqueeze(0).expand(1, S, Q),          # [1,S,Q]（与你原代码接口一致）
                dq.mu.view(1, S, Q)                         # [1,S,Q]
            )
            pr.register_hook(priority_hook)

            # 如果你训练阶段有“server pools”，可保留 repeat_interleave（原代码是 1）
            pr = pr.repeat_interleave(1, dim=1)

            # ---- 你的策略分支（按需保留/修改） ----
            if self.model_config['policy']['train_policy'] == 'sinkhorn':
                # 需要 rt.pad_pool / rt.Sinkhorn；若你的工程里叫别的名字，请自行改动
                lex = torch.zeros(B_train, S, Q, device=self.device)
                v, s_bar, q_bar = rt.pad_pool(
                    2 * pr + lex, queues.detach(),
                    network=dq.network, device=self.device,
                    server_pool_size=self.env_config['server_pool_size']
                )
                pr = rt.Sinkhorn.apply(
                    -v, s_bar, q_bar,
                    self.model_config['policy']['sinkhorn']['num_iter'],
                    self.model_config['policy']['sinkhorn']['temp'],
                    self.model_config['policy']['sinkhorn']['eps'],
                    self.model_config['policy']['sinkhorn']['back_temp'],
                    self.model_config['env']['device']
                )[:, :S, :Q]

            elif self.model_config['policy']['train_policy'] == 'softmax':
                pr = F.softmax(pr, dim=-1) * dq.network.unsqueeze(0)       # [B,S,Q]
                pr = torch.minimum(pr, queues.unsqueeze(1).expand(-1, S, -1)).clamp_min(1e-4)
                pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

            # 最终的动作（环境里会按你“离散名额分配”逻辑使用 action）
            action = pr
            action.register_hook(action_hook)

            # TorchRL step：传入 TensorDict
            out = dq.step(TensorDict({"action": action}, batch_size=[B_train]))
            # 统计
            total_cost += out["cost"]                              # [B,1]
            time_weight_queue_len += out["queues"] * out["event_time"]  # [B,Q]
            # 下一步
            td = out.select("queues", "time", "params")

        # 反传 + 优化
        loss = torch.mean(total_cost / self.env_config['train_T'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(),
                                       max_norm=self.model_config['opt']['grad_clip_norm'])
        self.optimizer.step()

        # 打印训练期指标
        current_time = td["time"]  # [B,1]
        train_cost_per_env = (total_cost / current_time).squeeze(-1)  # [B]
        twql_per_env = (time_weight_queue_len / current_time)         # [B,Q]

        print(f"train cost mean: {train_cost_per_env.mean().item():.6f}")
        print(f"train time-weighted mean queue len per q: {twql_per_env.mean(dim=0).tolist()}")

        if self.model_config['env'].get('print_grads', False) and len(back_outs) > 0 and len(nn_back_ins) > 0:
            action_grads = torch.tensor(back_outs).sum(0).mean(0)
            pri_grads = torch.tensor(nn_back_ins).sum(0).mean(0)
            print("Action Grads (mean over steps):", action_grads)
            print("Priority Grads (mean over steps):", pri_grads)

    # ------------------------------ 测试 ------------------------------ #
    def test_epoch(self, epoch):
        B_test = self.model_config['opt']['test_batch']

        dq = BatchedDiffDES(
            self.env_config['network'],
            self.env_config['mu'],
            torch.tensor(self.env_config['h']).float(),
            queue_event_options=self.env_config.get('queue_event_options', None),
            queue_event_options2=self.env_config.get('queue_event_options2', None),
            default_B=B_test,
            temp=self.model_config['env']['env_temp'],
            seed=self.model_config['env']['test_seed'],
            device=self.device,
            draw_service=self.draw_service,
            draw_inter_arrivals=self.draw_inter_arrivals,
            draw_due_date=draw_due_date,
            reentrant=self.env_config.get('reentrant', 0)
        )

        td = dq.reset(dq.gen_params(batch_size=[B_test]))

        total_cost = torch.zeros((B_test, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_test, self.env_config['network'].shape[-1]), device=self.device)

        S = dq.S
        Q = dq.Q

        with torch.no_grad():
            pbar = trange(self.env_config['test_T'],
                          desc=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} - {self.experiment_name}",
                          disable=True,  # <<< 强制关闭
                          leave=False)
            for step in pbar:
                queues = td["queues"]   # [B,Q]
                time   = td["time"]     # [B,1]

                # 组装“重复版”输入，保持你原来的 policy 接口
                repeated_queue   = queues.unsqueeze(1).expand(-1, S, -1)          # [B,S,Q]
                repeated_network = dq.network.unsqueeze(0).expand(B_test, -1, -1) # [B,S,Q]
                repeated_mu      = dq.mu.view(1, S, Q).expand(B_test, -1, -1)     # [B,S,Q]
                repeated_h       = dq.h.view(1, 1, Q).expand(B_test, S, -1)       # [B,S,Q]

                pr = self.policy.test_forward(
                    step, queues, time,
                    repeated_queue, repeated_network, repeated_mu, repeated_h
                )

                pr = pr.repeat_interleave(1, dim=1)  # 如果你需要保持一致；否则可以去掉

                # ---- 测试策略分支（与你原代码一致） ----
                if self.model_config['policy']['test_policy'] == 'sinkhorn':
                    lex = torch.zeros(B_test, S, Q, device=self.device)
                    v, s_bar, q_bar = rt.pad_pool(
                        2 * pr + lex, queues.detach(),
                        network=dq.network, device=self.device,
                        server_pool_size=self.env_config['server_pool_size']
                    )
                    pr = rt.Sinkhorn.apply(
                        -v, s_bar, q_bar,
                        self.model_config['policy']['sinkhorn']['num_iter'],
                        self.model_config['policy']['sinkhorn']['temp'],
                        self.model_config['policy']['sinkhorn']['eps'],
                        self.model_config['policy']['sinkhorn']['back_temp'],
                        self.model_config['env']['device']
                    )[:, :S, :Q]

                elif self.model_config['policy']['test_policy'] == 'linear_assigment':
                    lex = torch.zeros(B_test, S, Q, device=self.device)
                    v, s_bar, q_bar = rt.pad_pool(
                        2 * pr + lex, queues.detach(),
                        network=dq.network, device=self.device,
                        server_pool_size=self.env_config['server_pool_size']
                    )
                    pr = rt.linear_assignment_batch(v, s_bar, q_bar)

                elif self.model_config['policy']['test_policy'] == 'softmax':
                    pr = F.softmax(pr, dim=-1) * repeated_network
                    pr = torch.minimum(pr, queues.unsqueeze(1).expand(-1, S, -1)).clamp_min(1e-4)
                    pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

                # 测试时你原来做的是四舍五入为整数名额
                action = torch.round(pr)

                out = dq.step(TensorDict({"action": action}, batch_size=[B_test]))
                # 奖励 / 代价
                total_cost += out["next", "reward"]
                time_weight_queue_len += out["next", "queues"] * out["next", "event_time"]
                td = out["next"].select("queues", "time")

        # -------- 汇总测试指标 --------
        time_now = td["time"]  # [B,1]
        cost_per_env = (total_cost / time_now).squeeze(-1)  # [B]
        test_cost_mean = cost_per_env.mean()
        test_cost_std  = cost_per_env.std(unbiased=True)
        test_cost_se   = test_cost_std / math.sqrt(B_test)

        qlen_per_env = (time_weight_queue_len / time_now)   # [B,Q]
        qlen_overall_per_env = qlen_per_env.mean(dim=1)     # [B]

        # qlen_overall_per_env = qlen_per_env.sum(dim=1)

        qlen_mean = qlen_overall_per_env.mean()
        qlen_std  = qlen_overall_per_env.std(unbiased=True)
        qlen_se   = qlen_std / math.sqrt(B_test)

        print(f'------------------------test result------------------------')
        print(f"experiment: {self.experiment_name}")
        print(f"queue length mean (overall): {qlen_mean.item():.4f}")
        print(f"queue length std  (overall): {qlen_std.item():.4f}")
        print(f"queue length se   (overall): {qlen_se.item():.4f}")
        print(f"test cost mean: {test_cost_mean.item():.4f}")
        print(f"test cost std : {test_cost_std.item():.4f}")
        print(f"test cost se  : {test_cost_se.item():.4f}")
