import sys
# sys.path.append('.../')
from stable_baselines3 import PPO
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import trange
from typing import NamedTuple
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import Dataset, DataLoader



class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor


# class BCD(Dataset):
#     def __init__(self, num_samples, network):
#         self.num_samples = num_samples
#         self.network = network
#         self.s = self.network.shape[0]
#         self.q = self.network.shape[1]
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         # Generate random input data
#         input_data = np.random.randint(0, 101, self.q)
#         obs = torch.tensor(input_data)
#         action_probs = F.softmax(torch.tensor(input_data).float(), dim=-1)
#         action_probs = action_probs * self.network
#         action_probs = torch.minimum(action_probs, obs.unsqueeze(0).repeat(1, self.s, 1))
#         zero_mask = torch.all(action_probs == 0, dim=2).reshape(-1, self.s, 1).repeat(1, 1, self.q)
#         action_probs = action_probs + zero_mask * self.network
#         action_probs = action_probs / torch.sum(action_probs, dim=-1).reshape(-1, self.s, 1)
#         output_data = action_probs
#         input_tensor = torch.tensor(input_data, dtype=torch.float32).squeeze()
#         output_tensor = torch.tensor(output_data, dtype=torch.float32).squeeze()
#
#         return input_tensor, output_tensor

class BCD(Dataset):
    def __init__(self, num_samples, network):
        self.num_samples = num_samples
        # network 必须是 (s, q)
        if isinstance(network, torch.Tensor):
            self.network = network.float()
        else:
            self.network = torch.tensor(network, dtype=torch.float32)
        assert self.network.dim() == 2, f"network must be 2D (s,q), got {self.network.shape}"
        self.s, self.q = self.network.shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # obs: (q,)
        obs = torch.randint(0, 101, (self.q,), dtype=torch.float32)

        # teacher target: (s, q)
        base = F.softmax(obs, dim=-1)             # (q,)
        action_probs = base.unsqueeze(0).repeat(self.s, 1)  # (s,q)
        action_probs = action_probs * self.network          # 可行性掩码

        # 只允许分配到正队列（可选）
        pos_mask = (obs > 0).float().unsqueeze(0).repeat(self.s, 1)
        action_probs = action_probs * pos_mask

        # 全零行回退
        row_sum = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        zero_row = (row_sum <= 1e-12).float()
        action_probs = action_probs + zero_row * self.network

        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # (s,q)

        return obs, action_probs  # 不要 squeeze！



class parallel_eval(BaseCallback):
    def __init__(self, model, eval_env, eval_freq, eval_t, test_policy, test_seed, init_test_queues, test_batch, device, num_pool, time_f, policy_name, per_iter_normal_obs, env_config_name, bc, randomize = True, 
                 verbose=1):
        super(parallel_eval, self).__init__(verbose)
        self.model = model
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_t = eval_t
        print('eval_t', eval_t)
        self.test_policy = test_policy
        self.test_seed = test_seed
        self.init_test_queues = init_test_queues
        self.test_batch = test_batch
        self.device = device
        self.num_pool = num_pool
        self.randomize = randomize
        self.time_f = time_f
        self.policy_name = policy_name
        self.test_costs = []  
        self.final_costs = []
        self.per_iter_normal_obs = per_iter_normal_obs
        self.env_config_name = env_config_name
        self.bc = bc
        print(f'eval env config name: {self.env_config_name}')
        self.iter = 0

        self.lex_batch = []
        self.obs_batch = []
        self.state_batch = []
        self.total_cost_batch = []
        self.time_weight_queue_len_batch = []




    
    def behavior_cloning(self):

        print(f'---------------------behavior_cloning---------------------')
        if hasattr(self.model.policy, "log_std"):
            self.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.log_std},
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        else:
            self.model.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        # print(f'network: {self.eval_env[0].network}, shape: {self.eval_env[0].network[0].shape}')
        BCD_dataset = BCD(num_samples = 100000, network = self.eval_env[0].network[0])
        BCD_loader = DataLoader(BCD_dataset, batch_size = self.test_batch, shuffle = True)

        for i, (obs, target) in enumerate(BCD_loader):
            self.optimizer_policy.zero_grad()
            action, action_probs = self.model.policy.get_prob_act(obs)
            loss = F.mse_loss(action_probs, target)
            loss.backward()
            self.optimizer_policy.step()

    def pre_train_eval(self):
        print('pre_train_eval')
        if self.bc:
            self.behavior_cloning()
        # self.behavior_cloning()
        q_mean, q_std, t_mean, t_max, t_min, t_std = self.eval()

        self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
        print(f'------- pre train eval -------')
        print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
        print(f"std_queue_length: {self.model.policy.std_queue_length}")

        return True

    def _on_step(self) -> bool:
        if self.per_iter_normal_obs:
            if (self.n_calls) % self.eval_freq == 0:
                q_mean, q_std, t_mean, t_max, t_min, t_std = self.eval()

                self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
                print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
                print(f"std_queue_length: {self.model.policy.std_queue_length}")
        else:
            if (self.n_calls) % self.eval_freq == 0:
                self.eval()
                print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
                print(f"std_queue_length: {self.model.policy.std_queue_length}")

                se_queue_length = self.model.policy.std_queue_length / torch.sqrt(torch.tensor(float(self.test_batch)))
                print(f'se_queue_length: {se_queue_length}')

        return True
    

    def eval(self):
        self.iter += 1
        print(f'iter: {self.iter}')

        lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.construct_batch()
        test_dq_batch = self.eval_env

        with torch.no_grad():
            for tt in trange(self.eval_t):
                # print(tt)
                # print(f'---------------------')
                # print(f'obs:')
                # for obs in obs_batch:
                #     print(f'{obs}')


                batch_queue = torch.cat([obs[0] for obs in obs_batch], dim = 0).reshape(self.test_batch,-1)
                # batch_queue = torch.cat([
                #     obs if obs.dim() == 2 else obs.unsqueeze(0)
                #     for obs in obs_batch
                # ], dim=0)  # (B, q)

                # print(f'batch_queue: {batch_queue}')

                raw_actions, probs = self.model.predict(batch_queue)
                action = torch.tensor(raw_actions).float().to(self.device)
                
                for test_dq_idx in range(len(test_dq_batch)):
                    # step_time_start = time.time()
                    _, _, _, _, info = test_dq_batch[test_dq_idx].step(action[test_dq_idx])
                    # step_time_end = time.time()
                    # print(f'step time: {step_time_end - step_time_start}')
                    obs_batch[test_dq_idx], state_batch[test_dq_idx], cost, event_time  = info['obs'], info['state'], info['cost'], info['event_time']
                    total_cost_batch[test_dq_idx] = total_cost_batch[test_dq_idx] + cost
                    time_weight_queue_len_batch[test_dq_idx] = time_weight_queue_len_batch[test_dq_idx] + info['queues'] * info['event_time']

        # Test cost metrics
        # pdb.set_trace()
        test_cost_batch = [total_cost_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]
        test_cost = torch.mean(torch.concat(test_cost_batch))
        test_std = torch.std(torch.concat(test_cost_batch))
        test_queue_len = torch.mean(torch.concat([time_weight_queue_len_batch[test_dq_idx] / state_batch[test_dq_idx].time for test_dq_idx in range(len(test_dq_batch))]), dim = 0)
        test_queue_len = [float(_item) for _item in test_queue_len.to('cpu').detach().numpy().tolist()]
        print(f'------------------------------test---------------------------------')
        print(f"queue lengths: \t{test_queue_len}")

        # 先得到每个并行环境的时间加权平均队长 [B, q]
        qlen_per_env = torch.concat(
            [time_weight_queue_len_batch[i] / state_batch[i].time for i in range(len(test_dq_batch))],
            dim=0
        )  # shape [B, q]
        # 跨 batch 取均值/标准差/标准误（逐队列）
        # qlen_mean = qlen_per_env.mean(dim=0)  # [q]
        # qlen_std = qlen_per_env.std(dim=0, unbiased=True)  # [q]
        # n = torch.tensor(qlen_per_env.shape[0], dtype=qlen_per_env.dtype, device=qlen_per_env.device)
        # qlen_se = qlen_std / torch.sqrt(n)
        # 如果你还想给“整体平均队长”的标量及其SE（可选）
        # overall_ql_mean = qlen_mean.mean()
        # overall_ql_se = qlen_se.mean()
        # print(f"overall avg queue length:       \t{overall_ql_mean.item():.6f}  (SE {overall_ql_se.item():.6f})")

        # ---- 关键修改：总体队长均值 + 总体SE ----
        # 每个环境先算整体平均队长 (把 q 个队列均值)
        overall_ql_per_env = qlen_per_env.mean(dim=1)  # [B]
        overall_ql_mean = overall_ql_per_env.mean()  # 标量
        overall_ql_std = overall_ql_per_env.std(unbiased=True)
        overall_ql_se = overall_ql_std / np.sqrt(len(test_dq_batch))

        print(f"overall avg queue length: {overall_ql_mean.item():.6f}")
        print(f"overall queue length SE:  {overall_ql_se.item():.6f}")


        print(f"test cost: \t{test_cost}")
        print(f"test cost std: \t{test_std}")

        test_se = test_std / np.sqrt(len(test_dq_batch))
        print(f"len test dq batch {len(test_dq_batch)}, test cost se: \t{test_se}")


        test_queue_len = torch.tensor(test_queue_len)

        q_mean = torch.mean(test_queue_len)
        # q_std = torch.std(test_queue_len)

        if test_queue_len.numel() > 1:
            q_std = torch.std(test_queue_len)
        else:
            q_std = torch.tensor(0.0)

        t_mean = torch.mean(state_batch[0].time)
        t_max = torch.max(state_batch[0].time)
        t_min = torch.min(state_batch[0].time)
        t_std = torch.std(state_batch[0].time)
        
        return q_mean, q_std, t_mean, t_max, t_min, t_std

    def construct_batch(self):
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []


        for dq_idx in range(self.test_batch):

            dq = self.eval_env[dq_idx]
            lex = torch.zeros(dq.batch, dq.s, dq.q)
            obs, state = dq.reset(seed = dq.seed)
            obs = torch.tensor(obs).to(self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)  # (q,) -> (1, q)
            total_cost = torch.tensor([[0.]])
            time_weight_queue_len = torch.tensor([[0.]])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

        
        return lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch
