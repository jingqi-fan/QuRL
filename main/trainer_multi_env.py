# trainer_vectorized.py
import numpy as np
from tqdm import trange
import torch
import torch.nn.functional as F
import math
from tensordict import TensorDict
from datetime import datetime

# 你自己的工具（保持不变）
from utils.switchplot import create_plot_dir, create_loss_dir
# 如果你用到了 rt.*（pad_pool / Sinkhorn / linear_assignment_batch），请确保导入：
# import your_runtime_lib as rt

from main.env import BatchedDiffDES
# from main.env_s import BatchedDiffDES
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
        # 并行数：通过创建多个 env 来做“并行 batch”，不使用多进程
        B_train = self.model_config["opt"]["train_batch"]

        # 把环境相关张量准备好（替代 dq.xxx）
        network = self.env_config["network"].to(self.device)  # [S,Q]
        mu = self.env_config["mu"].to(self.device)  # [S,Q]
        h = torch.tensor(self.env_config["h"], device=self.device).float()  # [S,Q] 或可广播形状
        S, Q = network.shape

        # 创建多个 env，每个 env reset 的 batch_size 都是 [1]
        envs = self._make_envs(B_train, self.model_config["env"]["train_seed"])
        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]

        # stack 成一个 batched TensorDict：batch_size=[B_train]
        # 注意：如果你 tensordict 版本不支持 torch.stack，可改成 TensorDict.stack(td_list, dim=0)
        td = torch.stack(td_list, dim=0)

        self.optimizer.zero_grad()

        # 可选：记录梯度（保持你原本的 hooks）
        back_outs = []

        def action_hook(grad):
            back_outs.append(grad.detach().cpu().tolist())

        nn_back_ins = []

        def priority_hook(grad):
            nn_back_ins.append(grad.detach().cpu().tolist())

        total_cost = torch.zeros((B_train, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_train, Q), device=self.device)

        for _ in trange(self.env_config["train_T"], disable=True, leave=False):
            queues = td["queues"]  # [B,Q]
            time = td["time"]  # [B,1]

            # policy 前向：一次性对 B_train 个 env 计算
            pr = self.policy.train_forward(
                queues,
                time,
                network,  # [S,Q]
                h.unsqueeze(0).expand(1, S, Q),  # [1,S,Q]
                mu.unsqueeze(0),  # [1,S,Q]
            )
            pr.register_hook(priority_hook)

            # 如果训练阶段有“server pools”，你原代码 repeat_interleave(1, dim=1) 实际无影响，这里保留
            pr = pr.repeat_interleave(1, dim=1)

            # ---- 策略分支（按需保留/修改） ----
            if self.model_config["policy"]["train_policy"] == "sinkhorn":
                lex = torch.zeros(B_train, S, Q, device=self.device)
                v, s_bar, q_bar = rt.pad_pool(
                    2 * pr + lex,
                    queues.detach(),
                    network=network,
                    device=self.device,
                    server_pool_size=self.env_config["server_pool_size"],
                )
                pr = rt.Sinkhorn.apply(
                    -v,
                    s_bar,
                    q_bar,
                    self.model_config["policy"]["sinkhorn"]["num_iter"],
                    self.model_config["policy"]["sinkhorn"]["temp"],
                    self.model_config["policy"]["sinkhorn"]["eps"],
                    self.model_config["policy"]["sinkhorn"]["back_temp"],
                    self.model_config["env"]["device"],
                )[:, :S, :Q]

            elif self.model_config["policy"]["train_policy"] == "softmax":
                pr = F.softmax(pr, dim=-1) * network.unsqueeze(0)  # [B,S,Q]
                pr = torch.minimum(
                    pr, queues.unsqueeze(1).expand(-1, S, -1)
                ).clamp_min(1e-4)
                pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

            # 最终动作
            action = pr
            action.register_hook(action_hook)

            # 关键：不多进程。逐 env step，但 action 是 batched 一次算好的
            out_list = []
            for i, env in enumerate(envs):
                action_i = action[i: i + 1]  # [1,S,Q]
                out_i = env.step(
                    TensorDict({"action": action_i}, batch_size=[1])
                )
                out_list.append(out_i)

            # stack 回 batched TensorDict
            out = torch.stack(out_list, dim=0)  # batch_size=[B_train]

            # 统计
            total_cost += out["cost"]  # [B,1]
            time_weight_queue_len += out["queues"] * out["event_time"]  # [B,Q]

            # 下一步
            td = out.select("queues", "time", "params")

        # 反传 + 优化
        loss = torch.mean(total_cost / self.env_config["train_T"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.network.parameters(),
            max_norm=self.model_config["opt"]["grad_clip_norm"],
        )
        self.optimizer.step()

        # 打印训练期指标
        current_time = td["time"]  # [B,1]
        train_cost_per_env = (total_cost / current_time).squeeze(-1)  # [B]
        twql_per_env = (time_weight_queue_len / current_time)  # [B,Q]

        print(f"train cost mean: {train_cost_per_env.mean().item():.6f}")
        print(f"train time-weighted mean queue len per q: {twql_per_env.mean(dim=0).tolist()}")

        if (
                self.model_config["env"].get("print_grads", False)
                and len(back_outs) > 0
                and len(nn_back_ins) > 0
        ):
            action_grads = torch.tensor(back_outs).sum(0).mean(0)
            pri_grads = torch.tensor(nn_back_ins).sum(0).mean(0)
            print("Action Grads (mean over steps):", action_grads)
            print("Priority Grads (mean over steps):", pri_grads)

    # ------------------------------ 测试 ------------------------------ #
    def test_epoch(self, epoch):
        B_test = self.model_config["opt"]["test_batch"]

        # 环境参数（替代 dq.xxx）
        network = self.env_config["network"].to(self.device)  # [S,Q]
        mu = self.env_config["mu"].to(self.device)  # [S,Q]
        h = torch.tensor(self.env_config["h"], device=self.device).float()
        S, Q = network.shape

        # ---- helper：把 queues/time/reward/event_time 规整成稳定形状 ----
        def _to_BQ(x: torch.Tensor) -> torch.Tensor:
            # 目标 [B,Q]
            # 允许 [B,Q], [B,1,Q], [B,1,1,Q] 等（中间全是 1）
            while x.dim() > 2:
                # squeeze 中间的 singleton 维
                squeezed = False
                for d in range(1, x.dim() - 1):
                    if x.size(d) == 1:
                        x = x.squeeze(d)
                        squeezed = True
                        break
                if not squeezed:
                    break
            if x.dim() != 2:
                raise RuntimeError(f"queues cannot be normalized to [B,Q], got {tuple(x.shape)}")
            return x

        def _to_B1(x: torch.Tensor) -> torch.Tensor:
            # 目标 [B,1]
            # 允许 [B,1], [B,1,1], [B,1,1,1]...
            while x.dim() > 2:
                squeezed = False
                for d in range(1, x.dim() - 1):
                    if x.size(d) == 1:
                        x = x.squeeze(d)
                        squeezed = True
                        break
                if not squeezed:
                    break
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if x.dim() != 2 or x.size(-1) != 1:
                raise RuntimeError(f"tensor cannot be normalized to [B,1], got {tuple(x.shape)}")
            return x

        # 多 env（不多进程），每个 env 内 batch=1
        envs = self._make_envs(B_test, self.model_config["env"]["test_seed"])
        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]

        # 将 td_list 组织成 batched td（只用于读 queues/time）
        try:
            td = torch.stack(td_list, dim=0)
        except Exception:
            td = TensorDict.stack(td_list, dim=0)

        total_cost = torch.zeros((B_test, 1), device=self.device)
        time_weight_queue_len = torch.zeros((B_test, Q), device=self.device)

        with torch.no_grad():
            pbar = trange(
                self.env_config["test_T"],
                desc=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} - {self.experiment_name}",
                disable=True,
                leave=False,
            )

            for step in pbar:
                queues = _to_BQ(td["queues"])  # [B,Q]
                time = _to_B1(td["time"])  # [B,1]

                repeated_queue = queues.unsqueeze(1).expand(-1, S, -1)  # [B,S,Q]
                repeated_network = network.unsqueeze(0).expand(B_test, -1, -1)  # [B,S,Q]
                repeated_mu = mu.unsqueeze(0).expand(B_test, -1, -1)  # [B,S,Q]

                # repeated_h：保持你原来 dq.h.view(1,1,Q).expand(B,S,-1) 的形状
                if h.dim() == 1:  # [Q]
                    h_q = h
                elif h.dim() == 2:  # [S,Q] 或 [1,Q]
                    h_q = h[0] if h.shape[0] != 1 else h.squeeze(0)
                else:
                    h_q = h.view(-1)[-Q:]
                repeated_h = h_q.view(1, 1, Q).expand(B_test, S, -1)  # [B,S,Q]

                pr = self.policy.test_forward(
                    step, queues, time,
                    repeated_queue, repeated_network, repeated_mu, repeated_h
                )
                pr = pr.repeat_interleave(1, dim=1)  # 保持一致

                # ---- 测试策略分支（不改你的逻辑） ----
                if self.model_config["policy"]["test_policy"] == "sinkhorn":
                    lex = torch.zeros(B_test, S, Q, device=self.device)
                    v, s_bar, q_bar = rt.pad_pool(
                        2 * pr + lex,
                        queues.detach(),
                        network=network,
                        device=self.device,
                        server_pool_size=self.env_config["server_pool_size"],
                    )
                    pr = rt.Sinkhorn.apply(
                        -v, s_bar, q_bar,
                        self.model_config["policy"]["sinkhorn"]["num_iter"],
                        self.model_config["policy"]["sinkhorn"]["temp"],
                        self.model_config["policy"]["sinkhorn"]["eps"],
                        self.model_config["policy"]["sinkhorn"]["back_temp"],
                        self.model_config["env"]["device"],
                    )[:, :S, :Q]

                elif self.model_config["policy"]["test_policy"] == "linear_assigment":
                    lex = torch.zeros(B_test, S, Q, device=self.device)
                    v, s_bar, q_bar = rt.pad_pool(
                        2 * pr + lex,
                        queues.detach(),
                        network=network,
                        device=self.device,
                        server_pool_size=self.env_config["server_pool_size"],
                    )
                    pr = rt.linear_assignment_batch(v, s_bar, q_bar)

                elif self.model_config["policy"]["test_policy"] == "softmax":
                    pr = F.softmax(pr, dim=-1) * repeated_network
                    pr = torch.minimum(
                        pr, queues.unsqueeze(1).expand(-1, S, -1)
                    ).clamp_min(1e-4)
                    pr = pr / (pr.sum(dim=-1, keepdim=True) + 1e-8)

                action = torch.round(pr)  # 测试时四舍五入为整数名额

                # --------- 关键修复：逐 env 提取并规整张量，再 cat ---------
                reward_list = []
                q_next_list = []
                t_next_list = []
                et_list = []

                for i, env in enumerate(envs):
                    action_i = action[i:i + 1]  # [1,S,Q]
                    out_i = env.step(TensorDict({"action": action_i}, batch_size=[1]))

                    # 兼容 TorchRL ("next", ...) 或直接输出
                    if "next" in out_i.keys():
                        nxt = out_i["next"]
                        reward_i = nxt.get("reward", None)
                        if reward_i is None:
                            raise KeyError("test_epoch: out['next'] 中没有 reward")
                        queues_i = nxt["queues"]
                        time_i = nxt["time"]
                        event_time_i = nxt["event_time"]
                    else:
                        # 与 train_epoch 一致的字段命名（如果你的 env 是 cost，就当 reward 用）
                        if "reward" in out_i.keys():
                            reward_i = out_i["reward"]
                        elif "cost" in out_i.keys():
                            reward_i = out_i["cost"]
                        else:
                            raise KeyError("test_epoch: out 中未找到 reward/cost 字段")
                        queues_i = out_i["queues"]
                        time_i = out_i["time"]
                        event_time_i = out_i["event_time"]

                    # 规整形状：reward/time/event_time -> [1,1]，queues -> [1,Q]
                    reward_list.append(_to_B1(reward_i))  # [1,1]
                    q_next_list.append(_to_BQ(queues_i))  # [1,Q]
                    t_next_list.append(_to_B1(time_i))  # [1,1]
                    et_list.append(_to_B1(event_time_i))  # [1,1]

                reward = torch.cat(reward_list, dim=0)  # [B,1]
                queues_next = torch.cat(q_next_list, dim=0)  # [B,Q]
                time_next = torch.cat(t_next_list, dim=0)  # [B,1]
                event_time = torch.cat(et_list, dim=0)  # [B,1]

                # 统计（不会再出现 [B,B,1]）
                total_cost += reward  # [B,1]
                time_weight_queue_len += queues_next * event_time  # [B,Q]

                # 更新 td（只需要 queues/time）
                td = TensorDict({"queues": queues_next, "time": time_next}, batch_size=[B_test])

        # -------- 汇总测试指标（输出保持不变） --------
        time_now = time_next  # [B,1]
        cost_per_env = (total_cost / time_now).squeeze(-1)  # [B]
        test_cost_mean = cost_per_env.mean()
        test_cost_std = cost_per_env.std(unbiased=True)
        test_cost_se = test_cost_std / math.sqrt(B_test)

        qlen_per_env = (time_weight_queue_len / time_now)  # [B,Q]
        qlen_overall_per_env = qlen_per_env.mean(dim=1)  # [B]

        qlen_mean = qlen_overall_per_env.mean()
        qlen_std = qlen_overall_per_env.std(unbiased=True)
        qlen_se = qlen_std / math.sqrt(B_test)

        print(f"------------------------test result------------------------")
        print(f"experiment: {self.experiment_name}")
        print(f"queue length mean (overall): {qlen_mean.item():.4f}")
        print(f"queue length std  (overall): {qlen_std.item():.4f}")
        print(f"queue length se   (overall): {qlen_se.item():.4f}")
        print(f"test cost mean: {test_cost_mean.item():.4f}")
        print(f"test cost std : {test_cost_std.item():.4f}")
        print(f"test cost se  : {test_cost_se.item():.4f}")