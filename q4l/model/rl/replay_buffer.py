from collections import namedtuple
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

ReplayItem = namedtuple(
    "ReplayItem", ["state", "action", "reward", "next_state", "done"]
)


class SumTree:
    # Define your SumTree implementation here, used for PER
    pass


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        num_stocks: int,
        state_dim: int,
        action_dim: int,
        device: int = 0,
        num_seqs: int = 4,
    ):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.add_size = 0
        self.add_item = None
        self.max_size = max_size
        self.num_seqs = num_seqs
        self.device = device

        self.states = torch.empty(
            (max_size, num_seqs, num_stocks, state_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.actions = torch.empty(
            (max_size, num_seqs, action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rewards = torch.empty(
            (max_size, num_seqs), dtype=torch.float32, device=self.device
        )
        self.undones = torch.empty(
            (max_size, num_seqs), dtype=torch.float32, device=self.device
        )

    def update(self, items: Tuple[torch.Tensor, ...]):
        states, actions, rewards, next_states, undones = items
        self.add_size = rewards.shape[0]

        p = self.p + self.add_size  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p : p] = states
            self.actions[self.p : p] = actions
            self.rewards[self.p : p] = rewards
            self.undones[self.p : p] = undones

        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        Tuple[torch.Tensor, ...]
            A tuple containing states, actions, rewards, next_states, dones, indices, and weights.

        """
        indices = np.random.choice(self.cur_size, batch_size, replace=False)
        seqs = np.random.choice(self.num_seqs, batch_size, replace=False)

        states = self.states[indices, seqs]
        actions = self.actions[indices, seqs]
        rewards = self.rewards[indices, seqs]
        next_states = self.states[
            (indices + 1) % self.cur_size, seqs
        ]  # Assuming next state follows current state
        dones = self.undones[indices, seqs]

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the number of experiences stored in the buffer.

        Returns
        -------
        int
            The current number of experiences in the buffer.

        """
        return len(self.buffer)


class ReplayBufferDataset(IterableDataset):
    """A dataset to sample batches of experiences from a given replay buffer.

    Attributes
    ----------
    replay_buffer : ReplayBuffer
        The replay buffer from which experiences are sampled.
    batch_size : int
        The size of the batch to sample.

    Methods
    -------
    __iter__():
        Yields batches of experiences from the replay buffer.

    """

    def __init__(
        self, replay_buffer: ReplayBuffer, batch_size: int, max_steps: int
    ):
        """
        Parameters
        ----------
        replay_buffer : ReplayBuffer
            The replay buffer from which experiences are sampled.
        batch_size : int
            The size of the batch to sample.
        """
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.max_steps = max_steps

    def __iter__(self):
        """
        Yields
        ------
        Tuple[np.ndarray, ...]
            A tuple containing batches of states (shape `(B, N)`),
            actions (shape `(B,)` for discrete or `(B, M)` for continuous),
            rewards (shape `(B,)`), next_states (shape `(B, N)`),
            and dones (shape `(B,)`), where `B` is the batch size.
        """
        counter = 0
        while counter < self.max_steps:
            counter += 1
            yield self.replay_buffer.sample(self.batch_size)


def get_data_loader(replay_buffer: ReplayBuffer, batch_size: int) -> DataLoader:
    """Creates a DataLoader-like sampler for the given replay buffer.

    Parameters
    ----------
    replay_buffer : ReplayBuffer
        The replay buffer from which experiences are sampled.
    batch_size : int
        The size of the batch to sample.

    Returns
    -------
    DataLoader
        The DataLoader-like sampler yielding batches from the replay buffer.

    """
    dataset = ReplayBufferDataset(replay_buffer, batch_size)
    dataloader = DataLoader(
        dataset, batch_size=None, num_workers=0, pin_memory=True
    )
    return dataloader
