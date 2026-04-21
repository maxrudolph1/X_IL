import logging

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class AWR_Agent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        obs_seq_len: int,
        act_seq_len: int,
        cam_names: list[str],
        beta: float = 1.0,
        weight_clip: float = 20.0,
        value_coef: float = 1.0,
        value_hidden_dim: int = 256,
        goal_dim: int = 512,
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            obs_seq_len=obs_seq_len,
            act_seq_len=act_seq_len,
            cam_names=cam_names,
        )

        if beta <= 0:
            raise ValueError("beta must be positive")
        if weight_clip <= 0:
            raise ValueError("weight_clip must be positive")

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.beta = beta
        self.weight_clip = weight_clip
        self.value_coef = value_coef

        self.eval_model_name = "eval_best_awr.pth"
        self.last_model_name = "last_awr.pth"

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

        value_input_dim = latent_dim + goal_dim
        self.value_head = nn.Sequential(
            nn.LayerNorm(value_input_dim),
            nn.Linear(value_input_dim, value_hidden_dim),
            nn.GELU(),
            nn.Linear(value_hidden_dim, 1),
        ).to(device)

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    def compute_values(self, perceptual_emb, latent_goal):
        perceptual_context = perceptual_emb.mean(dim=1)
        goal_context = latent_goal.mean(dim=1)
        value_input = torch.cat([perceptual_context, goal_context], dim=-1)
        return self.value_head(value_input)

    def compute_loss(self, pred, actions, values, returns, mask=None):
        target_returns = returns[:, :1].to(values.dtype)
        advantages = target_returns - values.detach()
        weights = torch.exp(advantages / self.beta).clamp(max=self.weight_clip)

        action_loss = (pred - actions).pow(2).mean(dim=-1)
        if mask is not None:
            mask = mask.to(action_loss.dtype)
            action_loss = action_loss * mask
            policy_loss = (action_loss * weights).sum() / mask.sum().clamp_min(1.0)
        else:
            policy_loss = (action_loss * weights).mean()

        value_loss = F.mse_loss(values, target_returns)
        loss = policy_loss + self.value_coef * value_loss

        info = {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "value_mean": values.detach().mean(),
            "return_mean": target_returns.detach().mean(),
            "advantage_mean": advantages.detach().mean(),
            "weight_mean": weights.detach().mean(),
            "weight_max": weights.detach().max(),
        }

        return loss, info

    def forward(self, obs_dict, actions=None, returns=None, mask=None, return_info=False):
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        pred = self.model(
            perceptual_emb,
            latent_goal,
        )

        if self.training and actions is not None:
            if returns is None:
                raise ValueError("AWR training requires returns")

            values = self.compute_values(perceptual_emb, latent_goal)
            loss, info = self.compute_loss(pred, actions, values, returns, mask)

            if return_info:
                return loss, info

            return loss

        return pred
