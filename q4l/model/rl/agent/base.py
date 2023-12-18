from abc import abstractmethod
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from gym import Env
from torch import Tensor

from ....config import ModuleConfig
from ....data.dataset import Q4LDataModule
from ....utils.misc import create_instance


class Agent(nn.Module):
    def __init__(
        self,
        data: Q4LDataModule,
        actor: ModuleConfig,
        critic: Optional[ModuleConfig],
        actor_optimizer: ModuleConfig,
        critic_optimizer: Optional[ModuleConfig],
        factor_dim: int,
        gamma: float = 0.99,
        **kwargs,
    ):
        """Initialize the Agent.

        Parameters
        ----------
        actor : ModuleConfig
            Neural network model for the actor.
        critic : ModuleConfig
            Neural network model for the critic. Can be None for actor-only methods.
        actor_optimizer : ModuleConfig
            Optimizer for updating the actor model.
        critic_optimizer : ModuleConfig
            Optimizer for updating the critic model. Can be None for actor-only methods.
        gamma: float
            Discount factor for future rewards.

        """
        super(Agent, self).__init__()
        self.data = data
        self.ticker_list = data.ticker_list
        self.state_dim = factor_dim
        self.action_dim = 1
        self.actor = create_instance(
            actor, input_dim=factor_dim, ticker_list=self.ticker_list
        )

        critic_kwargs = {
            "factor_dim": factor_dim,
            "window_size": data.cfg.data.sampler.x_window,
            "action_dim": 1,
            "num_stocks": len(self.ticker_list),
        }
        self.critic = (
            create_instance(critic, **critic_kwargs)
            if critic is not None
            else None
        )
        self.actor_optimizer = create_instance(
            actor_optimizer, params=self.actor.parameters()
        )
        self.critic_optimizer = (
            create_instance(critic_optimizer, params=self.critic.parameters())
            if self.critic is not None
            else None
        )
        self.gamma = gamma

    def get_optimizer(self):
        """A helper class for the outer LightningModule to get the optimizer."""
        return self.actor_optimizer, self.critic_optimizer

    def explore_env(self, env: Env, num_steps: int) -> List[Tuple[Any, ...]]:
        """Explore the environment for a number of steps and collect
        experiences.

        Parameters
        ----------
        env : Environment
            The environment to interact with.
        num_steps : int
            Number of steps to explore.

        Returns
        -------
        List[Tuple[Any, ...]]
            List of trajectories (experiences) collected from the environment.

        """
        with torch.no_grad():
            trajectories = []
            state = env.reset()
            for _ in range(num_steps):
                action = self.select_action(state)
                next_state, reward, _, done, info = env.step(action)
                trajectories.append((state, action, reward, next_state, done))
                state = next_state if not torch.any(done) else env.reset()
            # Stack them on the first dimension
            trajectories = list(zip(*trajectories))
            trajectories = [torch.stack(t, dim=0) for t in trajectories]
        return trajectories

    def learn(
        self, batch: List[Tuple[Tensor, Tensor, float, Tensor, bool]]
    ) -> None:
        """Perform a learning step based on a batch of experiences.

        Parameters
        ----------
        batch : List[Tuple[Tensor, Tensor, float, Tensor, bool]]
            A batch of experiences from the replay buffer.

        """
        states, actions, rewards, next_states, dones = zip(*batch)

        actor_loss = self.compute_actor_loss(
            states, actions, rewards, next_states, dones
        )
        critic_loss = (
            self.compute_critic_loss(
                states, actions, rewards, next_states, dones
            )
            if self.critic is not None
            else 0
        )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.critic is not None:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

    def select_action(self, state: Any) -> Any:
        """Select an action based on the current state. This method should be
        implemented in subclasses.

        Parameters
        ----------
        state : Any
            The current state from the environment.

        Returns
        -------
        Any
            The selected action.

        """
        return self.actor(state)

    @abstractmethod
    def compute_actor_loss(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError

    @abstractmethod
    def compute_critic_loss(self, states, actions, rewards, next_states, dones):
        raise NotImplementedError
