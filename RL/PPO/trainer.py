# RL/trainers/ppo_trainer.py
import torch
from torch import optim
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE
from RL.models.continuous import build_continuous_actor_critic
from torchrl.envs.utils import ExplorationType, set_exploration_type


class PPOTrainerTorchRL:
    def __init__(self, train_env, eval_env, obs_dim, act_spec, config, device="cpu"):
        self.device = torch.device(device)
        self.train_env = train_env
        self.eval_env  = eval_env
        self.cfg = config

        # actor / critic
        self.actor, self.critic = build_continuous_actor_critic(
            obs_dim=obs_dim,
            action_spec=act_spec,
            hidden_sizes=self.cfg["hidden_sizes"],
            in_key="obs",
        )
        self.actor.to(self.device); self.critic.to(self.device)

        # advantage / loss / opt
        self.adv = GAE(gamma=self.cfg["gamma"], lmbda=self.cfg["gae_lambda"],
                       value_network=self.critic, average_gae=True)
        self.loss = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.cfg["clip_epsilon"],
            entropy_coef=self.cfg["ent_coef"],
            critic_coef=self.cfg["vf_coef"],
            normalize_advantage=True,
        )
        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=self.cfg["lr"])
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=self.cfg["lr"])

        self.collector = SyncDataCollector(
            self.train_env, self.actor,
            frames_per_batch=self.cfg["frames_per_batch"],
            total_frames=self.cfg["total_frames"],
            device=self.device,
        )

    def train(self):
        ppo_epochs = self.cfg["ppo_epochs"]
        minibatch_sz = self.cfg["minibatch_size"]

        for batch in self.collector:
            # batch: TensorDict with batch_size = [T, B]  (T=frames_per_batch, B=内部batch)
            with torch.no_grad():
                self.adv(batch)  # 写入 advantage、(可选) value_target 等

            # 展平成单一批维，再打乱
            td = batch.reshape(-1).shuffle()  # 形状从 [T,B,...] -> [T*B, ...]

            for _ in range(ppo_epochs):
                for i in range(0, td.numel(), minibatch_sz):
                    sub = td[i:i + minibatch_sz]  # [minibatch, ...]
                    losses = self.loss(sub)  # 需要的键：action / sample_log_prob / advantage / state_value / done / reward
                    self.opt_actor.zero_grad(set_to_none=True)
                    self.opt_critic.zero_grad(set_to_none=True)
                    losses["loss_objective"].backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_actor.step()
                    self.opt_critic.step()

    @torch.no_grad()
    def evaluate(self, n_steps=200):
        if self.eval_env is None: return
        with set_exploration_type(ExplorationType.MODE):
            td = self.eval_env.reset()
            ret = 0.0
            for _ in range(n_steps):
                td = self.actor(td)
                td = self.eval_env.step(td)
                ret += float(td.get("reward").mean().cpu())
            print(f"[eval] avg reward over {n_steps}: {ret / n_steps:.3f}")
