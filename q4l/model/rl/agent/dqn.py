# TODO: Q-learning agents under construction
import torch.nn as nn
from torch import Tensor

from .base import Agent


class DQNAgent(Agent):
    def compute_loss(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Compute the loss for the DQNAgent.

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
            The computed loss.

        """
        # DQN loss is the Mean Squared Error between the estimated Q-values and the target Q-values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )
        next_q_values = self.target_q_network(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(current_q_values, expected_q_values)
        return loss
