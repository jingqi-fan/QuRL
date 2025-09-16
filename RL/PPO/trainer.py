# trainer_torchrl.py
import time, math, numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from RL.utils.cosine import cosine_lr_schedule  # 你已有cosine lr函数可复用



class TorchRLPPOTrainer:
    def __init__(
        self,
        actor,
        value,
        env,
        lr_policy,
        lr_value,
        min_lr_policy,
        min_lr_value,
        clip_range,
        ent_coef,
        vf_coef,
        gamma,
        gae_lambda,
        ppo_epochs,
        num_epochs,
        episode_steps,
        actors,
        device,
        target_kl=None,
        ct=None,
    ):
        self.actor = actor
        self.value = value
        self.env = env
        self.device = device

        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.min_lr_policy = min_lr_policy
        self.min_lr_value = min_lr_value
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.num_epochs = num_epochs
        self.episode_steps = episode_steps
        self.actors = actors
        self.target_kl = target_kl
        self.ct = ct

        # 优化器
        self.optim_policy = Adam(self.actor.parameters(), lr=lr_policy)
        self.optim_value = Adam(self.value.parameters(), lr=lr_value)

        # 损失
        self.advantage = GAE(
            gamma=gamma,
            lmbda=gae_lambda,
            value_network=value,
            average_gae=True,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=actor,
            critic=value,
            clip_epsilon=clip_range,
            entropy_coef=ent_coef,
            critic_coef=vf_coef,
            normalize_advantage=True,
        ).to(device)

        # 采样器
        self.collector = SyncDataCollector(
            env,
            policy=actor,
            frames_per_batch=episode_steps * actors,
            total_frames=num_epochs * episode_steps * actors,
            device=device,
            storing_device=device,
            reset_at_each_iter=False,
            split_trajs=True,
        )

        # buffer
        storage = LazyTensorStorage(episode_steps * actors, device=device)
        self.rb = TensorDictReplayBuffer(storage=storage)

    def learn(self):
        global_frames = 0
        for epoch, tensordict_data in enumerate(self.collector):
            frames = tensordict_data.numel()
            global_frames += frames

            # GAE
            with torch.no_grad():
                tensordict_data = self.advantage(tensordict_data)
            self.rb.extend(tensordict_data)

            # PPO inner loop
            for _ in range(self.ppo_epochs):
                batch = self.rb.sample(len(self.rb))
                self.loss_module.update_sampled_log_prob(batch)

                # actor update
                self.optim_policy.zero_grad()
                loss_pi = self.loss_module.actor_loss(batch)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optim_policy.step()

                # value update
                self.optim_value.zero_grad()
                loss_v = self.loss_module.critic_loss(batch)
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
                self.optim_value.step()

                # KL early stop
                if self.target_kl is not None:
                    with torch.no_grad():
                        approx_kl = self.loss_module.approx_kl(batch).mean().item()
                    if approx_kl > 1.5 * self.target_kl:
                        print(f"[epoch {epoch}] early stop PPO, KL={approx_kl:.4f}")
                        break

            self.rb.empty()

            # 余弦学习率
            progress = 1.0 - (epoch + 1) / float(self.num_epochs)
            for pg in self.optim_policy.param_groups:
                pg["lr"] = cosine_lr_schedule(self.lr_policy, self.min_lr_policy, progress)
            for pg in self.optim_value.param_groups:
                pg["lr"] = cosine_lr_schedule(self.lr_value, self.min_lr_value, progress)

            # 日志
            ep_reward = tensordict_data.get(("next", "reward")).mean().item()
            ep_len = frames / self.actors
            print(f"[{epoch+1:04d}/{self.num_epochs}] frames={global_frames} | "
                  f"ep_len≈{ep_len:.0f} | rew={ep_reward:.4f} | "
                  f"Lpi={loss_pi.item():.4f} | Lv={loss_v.item():.4f} | "
                  f"lr_pi={self.optim_policy.param_groups[0]['lr']:.2e} "
                  f"lr_v={self.optim_value.param_groups[0]['lr']:.2e}")

            if epoch + 1 >= self.num_epochs:
                break

        print("Training finished.")
        return self.actor, self.value
