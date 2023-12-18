"""Agents using PG-family algorithms."""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import Agent


class DDPGAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.critic_target = deepcopy(self.critic)
        self.actor_target = deepcopy(self.actor)
        self.tau = 0.005  # Soft update parameter

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Parameters
        ----------
        local_model : nn.Module
            The main network whose weights are being copied.
        target_model : nn.Module
            The target network which is being updated.
        tau : float
            The interpolation parameter.

        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )

    def learn(self, batch):
        super().learn(batch)

        # Soft update the target networks
        self.soft_update(self.actor, self.target_actor, self.tau)
        if self.critic is not None:
            self.soft_update(self.critic, self.target_critic, self.tau)

    def compute_actor_loss(self, states: Tensor) -> Tensor:
        """Compute the actor loss for the DDPGAgent.

        Parameters
        ----------
        states : Tensor
            The states encountered in the batch.

        Returns
        -------
        Tensor
            The computed actor loss.

        """
        # Actor loss in DDPG is based on the policy's performance, evaluated by the critic
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()
        return actor_loss

    def compute_critic_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute the critic loss for the DDPGAgent.

        Parameters
        ----------
        states : Tensor
            The states encountered in the batch.
        actions : Tensor
            The actions taken in the batch.
        rewards : Tensor
            The rewards received in the batch.
        next_states : Tensor
            The subsequent states in the batch.
        dones : Tensor
            Boolean flags indicating if an episode is done after each step.

        Returns
        -------
        Tensor
            The computed critic loss.

        """
        # Critic loss in DDPG is Mean Squared Error between the estimated Q-values and the target Q-values
        current_q_values = self.critic(states, actions)
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions).detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = F.mse_loss(current_q_values, expected_q_values)
        return critic_loss


class PPOAgent(Agent):
    def __init__(self, *args, clip_param: float, **kwargs):
        """Initialize the PPOAgent.

        Parameters
        ----------
        clip_param : float
            Clipping parameter for the PPO loss function.
        *args, **kwargs :
            Additional arguments passed to the base Agent class.

        """
        super(PPOAgent, self).__init__(*args, **kwargs)
        self.clip_param = clip_param

    def compute_actor_loss(
        self,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute the actor loss for the PPOAgent.

        Parameters
        ----------
        states : Tensor
            The states encountered in the batch.
        actions : Tensor
            The actions taken in the batch.
        old_log_probs : Tensor
            The log probabilities of the actions from the old policy.
        rewards : Tensor
            The rewards received in the batch.
        next_states : Tensor
            The subsequent states in the batch.
        dones : Tensor
            Boolean flags indicating if an episode is done after each step.

        Returns
        -------
        Tensor
            The computed actor loss.

        """
        new_log_probs, state_values = self.actor(states)
        advantages = (
            rewards
            + self.gamma * self.critic(next_states) * (1 - dones)
            - state_values
        )

        # Compute the ratio between new and old probabilities
        ratios = torch.exp(new_log_probs - old_log_probs)

        # Clipped function
        surr1 = ratios * advantages
        surr2 = (
            torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param)
            * advantages
        )

        # Actor loss
        actor_loss = -torch.min(surr1, surr2).mean()
        return actor_loss

    def compute_critic_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute the critic loss for the PPOAgent.

        Parameters
        ----------
        states : Tensor
            The states encountered in the batch.
        actions : Tensor
            The actions taken in the batch.
        rewards : Tensor
            The rewards received in the batch.
        next_states : Tensor
            The subsequent states in the batch.
        dones : Tensor
            Boolean flags indicating if an episode is done after each step.

        Returns
        -------
        Tensor
            The computed critic loss.

        """
        # Current V(s) estimates
        current_v = self.critic(states).squeeze()

        # Target V(s) estimates
        target_v = rewards + self.gamma * self.critic(next_states).squeeze() * (
            1 - dones
        )

        # Critic loss as Mean Squared Error
        critic_loss = nn.MSELoss()(current_v, target_v)
        return critic_loss
