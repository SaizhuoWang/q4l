from typing import Any, Dict, Iterator, List, Tuple

import torch
from gymnasium import Env
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from ...config import ExperimentConfig, ModuleConfig
from ...data.dataset import Q4LDataModule
from ...data.sampler import TCDataSampler
from ...utils.misc import create_instance
from ..base import QuantModel
from .agent import Agent
from .replay_buffer import ReplayBuffer


class RLTrainingDataset(IterableDataset):
    """A custom IterableDataset for reinforcement learning training.

    This dataset handles the interaction with an RL environment, maintains a replay buffer, and provides
    trajectories for training the agent.

    Parameters
    ----------
    data : Any
        The raw data or configurations to create the environment.
    agent : Agent
        The agent to be trained.
    replay_buffer : ReplayBuffer
        The replay buffer for storing and sampling experiences.
    buffer_size : int
        The maximum size of the replay buffer.
    batch_size : int
        The number of experiences to sample from the buffer for each training step.
    max_steps : int
        The maximum number of steps to simulate in the environment for each episode.

    """

    def __init__(
        self,
        data: TCDataSampler,
        env_config: ModuleConfig,
        agent: Agent,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        max_steps: int,
        device: str = "cpu",
    ):
        self.agent = agent
        self.env = create_instance(env_config, data=data, device=device)
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.state = self.env.reset()

    def __iter__(self) -> Iterator[List[Tuple[Tensor, ...]]]:
        return self

    def __next__(self) -> List[Tuple[Tensor, ...]]:
        traj = self.agent.explore_env(self.env, self.max_steps)
        self.replay_buffer.update(traj)
        sample = self.replay_buffer.sample(self.batch_size)
        return sample


class RLEvalDataset(IterableDataset):
    """This is a `tricky` dataset class that essentially performs inference at
    each `__iter__` step.

    This is designed to be in cope with pytorch lightning's style.

    """

    def __init__(
        self,
        data: TCDataSampler,
        env_config: ModuleConfig,
        agent: Agent,
        max_steps: int = 1e10,
        device: str = "cpu",
        **kwargs,  # Garbage bin
    ):
        self.agent: Agent = agent
        self.env: Env = create_instance(env_config, data=data, device=device)
        self.max_steps = max_steps
        self.current_step = 0
        self.state = self.env.reset()

    def __iter__(self):
        return self

    def __next__(self):
        action = self.agent.select_action(self.state)
        next_state, reward, _, done, _ = self.env.step(action)

        self.current_step += 1
        if torch.any(done):
            self.current_step = 0
            self.state = self.env.reset()
            raise StopIteration

        self.state = next_state
        return self.state, reward, done


def get_rl_dataloader(env, agent, max_steps, batch_size=1):
    dataset = RLEvalDataset(env, agent, max_steps)
    return DataLoader(dataset, batch_size=batch_size)


class RLModel(QuantModel):
    """A model template for tasks involving reinforcement learning. This model
    incorporates:

        - An `Environment` object for the agent to interact with.
        - A `ReplayBuffer` object for the agent to store and sample
          experiences.
        - An `Agent` object for the agent to learn from the environment.
    The neural networks are contained within the `Agent` object.

    A training step typically contains the following steps:
        1. Collect data (trajectory) from the environment
           for some (1/N/etc.) steps.
        2. Update replay buffer and sample a batch of data from it.
        3. Use the sampled data to update the agent.

    In (online) RL setting, the data sampling process is different from that
    of supervised learning. Specifically, a batch of data samples is sampled
    from the replay buffer, and upon each sample iteration, the replay buffer
    also gets updated with new data from the environment. This is different
    from the static data sampling process in supervised learning, where the
    dataset is fixed and does not change during training.

    """

    def __init__(
        self, data: Q4LDataModule, config: ExperimentConfig, device, **kwargs
    ):
        """Create a reinforcement learning model. Must contain an env, a replay
        buffer and an agent. These three components must be present in the input
        config.

        Parameters
        ----------
        config : ExperimentConfig
            The configuration object for this experiment.

        """
        self.qdevice = device
        super().__init__(config, data)
        # Initialize the replay buffer (maybe pre-fill it with some data)
        self.init_replay_buffer()
        self.automatic_optimization = False

    # Initialization
    def init_replay_buffer(self):
        pass

    def build_model_arch(self):
        # Instantiate model components
        model_component_config = self.config.model.components
        try_kwargs = {"data": self.data}
        # self.env: Env = create_instance(
        #     model_component_config.env, try_kwargs=try_kwargs
        # )
        num_stocks = len(self.data.ticker_list)
        factor_dim = self.config.model.input_size
        self.env_config = model_component_config.env
        self.replay_buffer: ReplayBuffer = create_instance(
            model_component_config.replay_buffer,
            try_kwargs=try_kwargs,
            device=self.qdevice,
            num_stocks=num_stocks,
            state_dim=factor_dim + 4,
            action_dim=num_stocks,
        )

        # Instantiate the agent
        self.agent: Agent = create_instance(
            model_component_config.agent,
            factor_dim=self.config.model.input_size,
            try_kwargs=try_kwargs,
        )

    def _get_dataloader(self, mode: str) -> DataLoader:
        dataset_class = RLTrainingDataset if mode == "train" else RLEvalDataset
        segment = "test" if mode == "predict" else mode
        dataset = dataset_class(
            data=self.data.prepare(segment, return_sampler=True),
            env_config=self.env_config,
            agent=self.agent,
            replay_buffer=self.replay_buffer,
            batch_size=self.config.model.basic_info.batch_size,  # Adjust as necessary
            max_steps=self.config.model.basic_info.max_step,
            device=self.qdevice,
        )
        return DataLoader(
            dataset=dataset,
            batch_size=None,  # Batch size is handled internally by the dataset
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("valid")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self._get_dataloader("predict")

    # Training step
    def configure_optimizers(self) -> Any:
        return self.agent.get_optimizer()

    def on_train_epoch_start(self) -> None:
        return super().on_train_epoch_start()

    def training_step(self, batch: Dict, batch_idx: int):
        # Training logic for a single step, using the batch of experiences
        # batch contains trajectories sampled from the replay buffer
        loss = self.agent.learn(batch)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.validation_trajectory: List[Any] = []
        self.validation_rewards: List[float] = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        state, reward, done = batch
        self.validation_trajectory.append((state, reward))
        self.validation_rewards.append(reward)

        if torch.any(done):
            self.log(
                "val_reward",
                sum(self.validation_rewards),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.validation_rewards = []

    def on_validation_epoch_end(self) -> None:
        # Summarize and log validation metrics
        total_reward = sum(self.validation_rewards)
        self.log(
            "val_total_reward",
            total_reward,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_test_epoch_start(self) -> None:
        self.test_trajectory: List[Any] = []
        self.test_rewards: List[float] = []

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> None:
        state, reward, done = batch
        self.test_trajectory.append((state, reward))
        self.test_rewards.append(reward)

        if done:
            self.log(
                "test_reward",
                sum(self.test_rewards),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.test_rewards = []

    def on_test_epoch_end(self) -> None:
        # Summarize and log test metrics
        total_reward = sum(
            [sum(episode_rewards) for episode_rewards in self.test_rewards]
        )
        self.log(
            "test_total_reward",
            total_reward,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
