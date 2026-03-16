from tqdm import trange
import torch
import torch.nn.functional as F
import math
from tensordict import TensorDict
from datetime import datetime

from main.env import BatchedDiffDES
import utils.routing as rt

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
        # self.fig_dir = create_plot_dir(self.model_config, self.env_config, experiment_name=experiment_name)
        # self.loss_dir = create_loss_dir(self.model_config, self.env_config, experiment_name=experiment_name)
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


    # ------------------------------ Train ------------------------------ #
    def train_epoch(self):
        # Parallel count: Use multiple envs to do "parallel batching" instead of multiprocessing
        B_train = self.model_config["opt"]["train_batch"]

        # Prepare environment-related tensors (replacing dq.xxx)
        network = self.env_config["network"].to(self.device)  # [S,Q]
        mu = self.env_config["mu"].to(self.device)  # [S,Q]
        h = torch.tensor(self.env_config["h"], device=self.device).float()  # [S,Q]
        S, Q = network.shape

        # Create multiple envs, the batch_size for each env reset is [1]
        envs = self._make_envs(B_train, self.model_config["env"]["train_seed"])
        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]

        td = torch.stack(td_list, dim=0)

        self.optimizer.zero_grad()

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

            # Policy forward pass: Compute for B_train envs at once
            pr = self.policy.train_forward(
                queues,
                time,
                network,  # [S,Q]
                h.unsqueeze(0).expand(1, S, Q),  # [1,S,Q]
                mu.unsqueeze(0),  # [1,S,Q]
            )
            pr.register_hook(priority_hook)

            pr = pr.repeat_interleave(1, dim=1)

            # ---- Policy branch (keep/modify as needed) ----
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

            # Final action
            action = pr
            action.register_hook(action_hook)

            out_list = []
            for i, env in enumerate(envs):
                action_i = action[i: i + 1]  # [1,S,Q]
                out_i = env.step(
                    TensorDict({"action": action_i}, batch_size=[1])
                )
                out_list.append(out_i)

            out = torch.stack(out_list, dim=0)

            # Statistics
            total_cost += out["cost"]  # [B,1]
            time_weight_queue_len += out["queues"] * out["event_time"]  # [B,Q]

            # Next step
            td = out.select("queues", "time", "params")

        loss = torch.mean(total_cost / self.env_config["train_T"])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.network.parameters(),
            max_norm=self.model_config["opt"]["grad_clip_norm"],
        )
        self.optimizer.step()

        # Print training metrics
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

    # ------------------------------ Test ------------------------------ #
    def test_epoch(self, epoch):
        B_test = self.model_config["opt"]["test_batch"]

        # Environment parameters (replacing dq.xxx)
        network = self.env_config["network"].to(self.device)  # [S,Q]
        mu = self.env_config["mu"].to(self.device)  # [S,Q]
        h = torch.tensor(self.env_config["h"], device=self.device).float()
        S, Q = network.shape

        # ---- helper: Normalize queues/time/reward/event_time into stable shapes ----
        def _to_BQ(x: torch.Tensor) -> torch.Tensor:
            # Target [B,Q]
            # Allow [B,Q], [B,1,Q], [B,1,1,Q], etc. (1s in the middle)
            while x.dim() > 2:
                # squeeze the middle singleton dimensions
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
            # Target [B,1]
            # Allow [B,1], [B,1,1], [B,1,1,1]...
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

        envs = self._make_envs(B_test, self.model_config["env"]["test_seed"])
        td_list = [env.reset(env.gen_params(batch_size=[1])) for env in envs]

        # Organize td_list into a batched td (only used to read queues/time)
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

                # repeated_h: Keep the original shape of dq.h.view(1,1,Q).expand(B,S,-1)
                if h.dim() == 1:  # [Q]
                    h_q = h
                elif h.dim() == 2:  # [S,Q] or [1,Q]
                    h_q = h[0] if h.shape[0] != 1 else h.squeeze(0)
                else:
                    h_q = h.view(-1)[-Q:]
                repeated_h = h_q.view(1, 1, Q).expand(B_test, S, -1)  # [B,S,Q]

                pr = self.policy.test_forward(
                    step, queues, time,
                    repeated_queue, repeated_network, repeated_mu, repeated_h
                )
                pr = pr.repeat_interleave(1, dim=1)  # Keep consistent

                # ---- Test policy branch ----
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

                action = torch.round(pr)

                # --------- Extract and normalize tensors per env, then cat ---------
                reward_list = []
                q_next_list = []
                t_next_list = []
                et_list = []

                for i, env in enumerate(envs):
                    action_i = action[i:i + 1]  # [1,S,Q]
                    out_i = env.step(TensorDict({"action": action_i}, batch_size=[1]))

                    if "next" in out_i.keys():
                        nxt = out_i["next"]
                        reward_i = nxt.get("reward", None)
                        if reward_i is None:
                            raise KeyError("test_epoch: No reward in out['next']")
                        queues_i = nxt["queues"]
                        time_i = nxt["time"]
                        event_time_i = nxt["event_time"]
                    else:
                        # Consistent field naming with train_epoch (if your env uses cost, treat it as reward)
                        if "reward" in out_i.keys():
                            reward_i = out_i["reward"]
                        elif "cost" in out_i.keys():
                            reward_i = out_i["cost"]
                        else:
                            raise KeyError("test_epoch: reward/cost field not found in out")
                        queues_i = out_i["queues"]
                        time_i = out_i["time"]
                        event_time_i = out_i["event_time"]

                    # Normalize shapes: reward/time/event_time -> [1,1], queues -> [1,Q]
                    reward_list.append(_to_B1(reward_i))  # [1,1]
                    q_next_list.append(_to_BQ(queues_i))  # [1,Q]
                    t_next_list.append(_to_B1(time_i))  # [1,1]
                    et_list.append(_to_B1(event_time_i))  # [1,1]

                reward = torch.cat(reward_list, dim=0)  # [B,1]
                queues_next = torch.cat(q_next_list, dim=0)  # [B,Q]
                time_next = torch.cat(t_next_list, dim=0)  # [B,1]
                event_time = torch.cat(et_list, dim=0)  # [B,1]

                # Statistics (will no longer output [B,B,1])
                total_cost += reward  # [B,1]
                time_weight_queue_len += queues_next * event_time  # [B,Q]

                # Update td (only need queues/time)
                td = TensorDict({"queues": queues_next, "time": time_next}, batch_size=[B_test])

        # -------- Summarize test metrics --------
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