import logging

import torch
import wandb
from tqdm import tqdm

from agents.base_agent import BaseAgent
from agents.utils.ema import ExponentialMovingAverage
from trainers.base_trainer import BaseTrainer

log = logging.getLogger(__name__)


class AWRTrainer(BaseTrainer):
    """Trainer for advantage-weighted regression."""

    def main(self, agent):
        """Run the AWR training pipeline."""

        agent.set_scaler(self.scaler)

        if self.if_use_ema:
            self.ema_helper = ExponentialMovingAverage(
                agent.parameters(), self.decay_ema, self.device
            )

        if agent.use_lr_scheduler:
            self.optimizer, self.scheduler = agent.configure_optimizers()
        else:
            self.optimizer = agent.configure_optimizers()

        for num_epoch in tqdm(range(self.epoch)):
            epoch_metrics = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "value_mean": 0.0,
                "return_mean": 0.0,
                "advantage_mean": 0.0,
                "weight_mean": 0.0,
                "weight_max": 0.0,
            }
            num_batches = 0

            for data in self.train_dataloader:
                if len(data) != 4:
                    raise ValueError(
                        "AWRTrainer expects dataset batches of "
                        "(obs_dict, action, mask, returns)."
                    )

                obs_dict, action, mask, returns = data
                obs_dict = self.prepare_obs_dict(obs_dict)

                action = self.scaler.scale_output(action)
                action = action[:, self.obs_seq_len - 1:, :].contiguous()

                mask = mask.to(self.device)
                mask = mask[:, self.obs_seq_len - 1:].contiguous()

                returns = returns.to(self.device)
                returns = returns[:, self.obs_seq_len - 1:].contiguous()

                _, batch_metrics = self.train_one_step(
                    agent, obs_dict, action, returns, mask
                )

                for key in epoch_metrics:
                    epoch_metrics[key] += batch_metrics[key].item()
                num_batches += 1

            epoch_metrics = {
                key: value / max(num_batches, 1)
                for key, value in epoch_metrics.items()
            }

            wandb.log(
                {
                    "train_loss": epoch_metrics["loss"],
                    "train_policy_loss": epoch_metrics["policy_loss"],
                    "train_value_loss": epoch_metrics["value_loss"],
                    "train_value_mean": epoch_metrics["value_mean"],
                    "train_return_mean": epoch_metrics["return_mean"],
                    "train_advantage_mean": epoch_metrics["advantage_mean"],
                    "train_weight_mean": epoch_metrics["weight_mean"],
                    "train_weight_max": epoch_metrics["weight_max"],
                }
            )
            log.info(
                "Epoch {}: mean train loss is {}".format(
                    num_epoch, epoch_metrics["loss"]
                )
            )

        log.info("training done")

        if self.if_use_ema:
            self.ema_helper.store(agent.parameters())
            self.ema_helper.copy_to(agent.parameters())

        agent.store_model_weights(agent.working_dir, sv_name="last_model")

    def prepare_obs_dict(self, obs_dict):
        for key in obs_dict.keys():
            if key == "lang":
                continue

            obs_dict[key] = obs_dict[key].to(self.device)

            if "rgb" not in key and "image" not in key:
                continue
            obs_dict[key] = obs_dict[key][:, :self.obs_seq_len].contiguous()

        return obs_dict

    def train_one_step(self, agent: BaseAgent, obs_dict, action, returns, mask):
        """Run a single AWR training step."""
        agent.train()

        loss, info = agent(
            obs_dict,
            action,
            returns=returns,
            mask=mask,
            return_info=True,
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        if agent.use_lr_scheduler:
            self.scheduler.step()

        if self.if_use_ema:
            self.ema_helper.update(agent.parameters())

        return loss.detach(), info
