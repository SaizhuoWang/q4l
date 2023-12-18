import typing as tp
from dataclasses import dataclass

import numpy as np
import torch

from q4l.config import ModuleConfig
from q4l.utils.misc import create_instance

from ....data.sampler import TCDataSampler


@dataclass
class PortfolioState:
    day: torch.IntTensor  # Current day (index)
    amount: torch.Tensor  # Cash in hand
    shares: torch.Tensor  # Stock shares in hand
    closes: torch.Tensor  # Close prices of stocks
    factors: torch.Tensor  # Factors of stocks
    mask: torch.Tensor  # Mask of tradable stocks


class PortfolioEnv:
    """Portfolio Vector Environment for Reinforcement Learning in Stock Trading.
    Supports vectorized simulation, i.e. simulate multiple trading environments
    at once.

    Attributes
    ----------
    device : torch.device
        Device to run computations on, either CPU or CUDA.
    num_envs : int
        Number of parallel environments.
    masks, closes, factors : torch.Tensor
        Masks, closes, and factors from the stock data.
    max_stock, initial_amount : float
        Maximum stocks and initial amount for the environment.
    buy_cost_rate, sell_cost_rate : float
        Buy and sell cost rate considering slippage.
    max_step : int
        Maximum steps allowed in the environment.
    day, day_end, mask, total_asset : various types
        Environment runtime variables.
    amount, shares, num_shares : various types
        Amount and shares related attributes.
    state_dim, action_dim, if_discrete : various types
        Environment information.

    """

    def __init__(
        self,
        data: TCDataSampler,
        initial_amount=1e7,
        max_share=10000,
        slippage=1e-3,
        max_step=60,
        if_random_reset=False,
        device: str = "cpu",
        num_envs: int = 4,
        gamma: float = 0.99,
        reward: ModuleConfig = None,
    ):
        """Initialize the PortfolioVecEnv environment.

        Parameters
        ----------
        initial_amount : float, optional
            Initial amount of money.
        max_stock : int, optional
            Maximum number of stocks.
        slippage : float, optional
            Slippage rate.
        max_step : int, optional
            Maximum number of steps.
        if_random_reset : bool, optional
            Whether to reset the environment randomly.
        gpu_id : int, optional
            GPU ID to use for computation.
        num_envs : int, optional
            Number of parallel environments.

        """
        # Device configuration
        self.device = device
        self.num_envs = num_envs
        self.gamma = gamma

        # Load data
        self.data = data
        # masks: is stock tradable (T, N)
        # closes: close price   (T, N)
        # factors: stock alpha factors  (T, N, W, D)
        masks, closes, factors = self._extract_data(data)

        self.masks, self.closes, self.factors = (
            torch.tensor(masks, dtype=torch.float32, device=self.device),
            torch.tensor(closes, dtype=torch.float32, device=self.device),
            factors,
        )

        # Initialization parameters
        self.max_share, self.initial_amount = max_share, initial_amount
        self.buy_cost_rate, self.sell_cost_rate = (
            1.0 + slippage,
            1.0 - slippage,
        )
        self.max_step = factors.shape[0]
        assert max_step <= self.closes.shape[0]

        # Runtime attributes (will be set in reset)
        self.day, self.day_end, self.mask, self.total_asset = (
            None,
            None,
            None,
            None,
        )
        self.if_random_reset = if_random_reset
        self.cumulative_returns = None

        # Amount and shares
        self.amount, self.shares = None, None
        self.num_stocks = self.closes.shape[1]

        # Environment info
        self.env_name = "PortfolioVecEnv-v0"
        self.window_size = self.factors.shape[2]
        self.factor_dim = self.factors.shape[3]
        self.state_dim = self.num_stocks * (
            self.factor_dim * self.window_size + 2
        )  # +2 for amount and share dimensions
        self.action_dim = self.num_stocks
        self.if_discrete = False
        self.reward = create_instance(reward)

    def _extract_data(
        self, data: TCDataSampler
    ) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        valid_mask = data.valid_mask
        close = data.data["ohlcvp"]["ohlcvp_close"].unstack().values
        factor = data.x
        return valid_mask, close, factor

    def _get_state(self) -> PortfolioState:
        """Retrieve the current state of the environment as a dictionary."""
        # Preparing the tensors for factors, amount, and shares
        factors_tensor = (
            torch.from_numpy(self.factors[None, self.day, :, :])
            .to(self.device)
            .repeat(self.num_envs, 1, 1, 1)
        )
        amount_tensor = (
            self.amount[:, None, None].repeat(1, self.num_stocks, 1) * 2**-16
        )
        shares_tensor = self.shares[:, :, None] * 2**-9

        # Creating the state dictionary
        state = {
            "closes": self.closes[self.day],
            "factors": factors_tensor,
            "amount": amount_tensor,
            "shares": shares_tensor,
            "mask": self.mask[None, :, None],
        }
        # Concat the tensors and return the state
        state_tensor = torch.cat(
            [
                state["factors"].flatten(-2, -1),
                state["closes"].unsqueeze(0).unsqueeze(2).repeat(4, 1, 1),
                state["amount"],
                state["shares"],
                state["mask"].repeat(4, 1, 1),
            ],
            dim=-1,
        )
        return state_tensor
        # return PortfolioState(
        #     day=self.day,
        #     amount=state["amount"],
        #     shares=state["shares"],
        #     closes=state["closes"],
        #     factors=state["factors"],
        #     mask=state["mask"],
        # )

        # return PortfolioState(**state)

    def _get_total_asset(self):
        """Retrieve the total asset value for the current state.

        Returns
        -------
        torch.Tensor
            Total asset tensor.

        """
        return (self.closes[self.day] * self.shares).sum(dim=1) + self.amount

    def _process_buying(self, stock_action):
        """Process buying stocks based on given stock action.

        Parameters
        ----------
        stock_action : torch.Tensor
            Stock action tensor.

        """
        buy_id0, buy_id1 = torch.where(stock_action > 0)
        part_close = self.closes[self.day, buy_id1]

        if buy_id1.shape[0] > 0:
            part_amount = self.amount[buy_id0]
            part_stock = stock_action[buy_id0, buy_id1]

            stock_delta = torch.min(
                part_stock,
                torch.div(part_amount, part_close, rounding_mode="floor"),
            )
            stock_delta = stock_delta.clip(0, None)
            delta_amount = part_close * stock_delta * self.buy_cost_rate

            self.amount -= torch.bincount(
                buy_id0, weights=delta_amount, minlength=self.num_envs
            )
            self.shares[buy_id0, buy_id1] += stock_delta

    def _process_selling(self, stock_action):
        """Process selling stocks based on given stock action.

        Parameters
        ----------
        stock_action : torch.Tensor
            Stock action tensor.

        """
        sell_id0, sell_id1 = torch.where((stock_action < 0) & (self.shares > 0))
        part_close = self.closes[self.day, sell_id1]

        if sell_id1.shape[0] > 0:
            part_shares = self.shares[sell_id0, sell_id1]
            part_stock = stock_action[sell_id0, sell_id1]

            stock_delta = torch.min(-part_stock, part_shares)
            delta_amount = part_close * stock_delta * self.sell_cost_rate

            self.amount += torch.bincount(
                sell_id0, weights=delta_amount, minlength=self.num_envs
            )
            self.shares[sell_id0, sell_id1] -= stock_delta

    def reset(self) -> tp.Dict:
        """Reset the environment.

        Returns
        -------
        tuple
            A tuple containing the state and additional info.

        """
        if self.if_random_reset:
            self.day = torch.randint(
                0, self.closes.shape[0] - self.max_step - 1, size=()
            ).item()
        else:
            self.day = 0

        # Setting environment runtime attributes
        self.day_end = self.day + self.max_step - 1
        self.mask = self.masks[self.day]
        self.amount = (
            torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            + self.initial_amount
        )
        self.shares = torch.zeros(
            (self.num_envs, self.num_stocks),
            dtype=torch.float32,
            device=self.device,
        )

        # Random initialization if necessary
        if self.if_random_reset:
            self.amount *= (
                torch.rand(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
                * 0.10
                + 0.95
            )
            self.shares += torch.randint(
                0,
                int(self.max_share),
                size=(self.num_envs, self.num_stocks),
                device=self.device,
            )
            self.shares[:, self.mask == 0] = 0.0

        self.total_asset = self._get_total_asset()
        state = self._get_state()
        self.rewards = []
        return state

    def step(self, action: torch.Tensor) -> tp.Tuple[torch.Tensor, ...]:
        """Take a step using the given action.

        Parameters
        ----------
        action : torch.Tensor
            Action tensor.

        Returns
        -------
        tuple
            A tuple containing the next state, reward, terminal flag,
            and additional info.

        """
        self.day += 1

        # Apply mask and filter negligible actions
        action = action.clone()
        action[:, self.mask == 0] = 0

        # Convert action to integer representation of stock actions
        stock_action = (action * self.max_share).to(torch.int32)

        # Process buying stocks
        self._process_buying(stock_action)

        # Process selling stocks
        self._process_selling(stock_action)

        # Update mask and state
        self.mask = self.masks[self.day]
        state = self._get_state()

        # Calculate reward and check for environment termination
        total_asset = self._get_total_asset()
        if self.reward is None:
            reward = (total_asset - self.total_asset) * 2**-15
        else:
            reward = self.reward(state, action)
        self.rewards.append(reward)
        self.total_asset = total_asset
        truncate = self.day == self.day_end

        # Reset environment if necessary
        if truncate:
            self.cumulative_returns = self.total_asset / self.initial_amount
            state = self.reset()
            reward += torch.stack(self.rewards).mean(dim=0) * (
                1.0 / (1.0 - self.gamma)
            )

        terminal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )  # Always False
        truncate = torch.tensor(
            truncate, dtype=torch.bool, device=self.device
        ).expand(self.num_envs)
        return state, reward, terminal, truncate, {"mask": self.mask}
