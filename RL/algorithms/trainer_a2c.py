import math
import torch.nn.functional as F
import torch
from torch import nn

from RL.algorithms.trainer_vanilla import PPOTrainerTorchRL_Vanilla, compute_gae


class A2CTrainerTorchRL_Vanilla(PPOTrainerTorchRL_Vanilla):
    """
    A2C /  simplified from PPOTrainerTorchRL_Vanilla
    """

    def learn(self):
        self.print("Start training (single-env, batched, vanilla A2C/Actor-Critic)")
        for epoch in range(self.args.total_epochs):
            traj = self._rollout(self.train_env, self.args.episode_steps, self.args.train_batch)

            # advantage/return
            adv, ret = compute_gae(
                traj['rew'], traj['done'],
                traj['val'][:-1], traj['val'][1:],
                self.args.gamma, self.args.gae_lambda
            )
            if self.args.normalize_advantage:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            with torch.no_grad():
                self.returns_mean = ret.mean()
                self.returns_std = ret.std().clamp_min(1e-6)

            # flatten
            T, B = self.args.episode_steps, self.args.train_batch
            obs_f = traj['obs'][:-1].reshape(T * B, self.args.obs_dim)
            act_f = traj['act'].reshape(T * B, self.args.S, self.args.Q)
            adv_f = adv.reshape(T * B)
            ret_f = ret.reshape(T * B)

            mb = max(1, self.args.minibatch_size)
            idx = torch.randperm(T * B, device=obs_f.device)

            for start in range(0, T * B, mb):
                mb_idx = idx[start:start + mb]
                o = obs_f[mb_idx]
                a = act_f[mb_idx]
                adv_mb = adv_f[mb_idx].detach()
                ret_mb = ret_f[mb_idx]

                std_o = self._standardize_queues(o)
                logits, v_pred = self.policy(std_o)

                if self.args.rescale_value:
                    v_pred = v_pred * self.returns_std + self.returns_mean

                new_logp = self._log_prob_of_logits(logits, a)
                ent = self._entropy_from_logits(logits).mean()

                # A2C actor loss (no ratio/clip)
                policy_loss = -(new_logp * adv_mb).mean()
                value_loss = F.mse_loss(v_pred, ret_mb)

                # policy step
                self.opt_pi.zero_grad(set_to_none=True)
                (policy_loss - self.args.ent_coef * ent).backward()
                nn.utils.clip_grad_norm_(self.policy.pi.parameters(), self.args.max_grad_norm)
                self.opt_pi.step()

                # value step
                self.opt_v.zero_grad(set_to_none=True)
                (self.args.vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.v.parameters(), self.args.max_grad_norm)
                self.opt_v.step()

                self._lr_step()

            self.print(f"[Epoch {epoch + 1}/{self.args.total_epochs}] done (A2C)")

            if (epoch + 1) % self.args.eval_every == 0:
                self.evaluate()

    def _estimate_total_updates(self) -> int:
        TnB = self.args.episode_steps * self.args.train_batch
        mb = max(1, self.args.minibatch_size)
        steps_per_epoch = math.ceil(TnB / mb)
        return max(1, steps_per_epoch * self.args.total_epochs)