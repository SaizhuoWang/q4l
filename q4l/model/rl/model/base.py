import typing as tp
from abc import abstractmethod

import torch
import torch.nn as nn


class PortfolioActorModel(nn.Module):
    def __init__(self, ticker_list: tp.List, input_dim: int, **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.ticker_list: tp.List = ticker_list

    @abstractmethod
    def actual_forward(self, **kwargs):
        raise NotImplementedError

    def align_prediction(
        self, prediction: torch.Tensor, cross_section_ticker_list: tp.List
    ) -> torch.Tensor:
        """Given today's prediction, align it to the total tickers.

        Parameters
        ----------
        prediction : torch.Tensor
            The predicted positions of shape (N_p,)
        cross_section_ticker_list : tp.List
            The list of tickers, which is the cross-section of the tickers

        Returns
        -------
        torch.Tensor
            The up-sampled tickers with shape (N,), with zero-padding

        """
        prediction = prediction.squeeze()
        output = torch.zeros(len(self.ticker_list))
        cross_section_ticker_index = torch.Tensor(
            [
                self.ticker_list.index(ticker)
                for ticker in cross_section_ticker_list
            ]
        )
        output[cross_section_ticker_index] = prediction
        return output

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Since the batch data passed in may not contain all the tickers, we
        may forward predict (the `actual_forward` method) on the given batch
        data, and then in the final prediction (the output positions) up-sample
        the predicted positions to the total tickers."""
        state = torch.nan_to_num(state)
        prediction = self.actual_forward(state)
        # cross_section_ticker_list = state["label"]
        # actual_prediction = self.align_prediction(
        #     prediction, cross_section_ticker_list
        # )
        return prediction


class PortfolioCriticModel(nn.Module):
    def __init__(
        self,
        factor_dim: int,
        window_size: int,
        action_dim: int,
        num_stocks: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.factor_dim = factor_dim
        self.window_size = window_size
        self.action_dim = action_dim
        self.num_stocks = num_stocks

    @abstractmethod
    def actual_forward(self, state, action):
        raise NotImplementedError

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        value = self.actual_forward(state, action)
        return value

    # No need for align_prediction as the critic evaluates the value,
    # not aligning actions.


class MLPCriticModel(PortfolioCriticModel):
    def __init__(
        self,
        state_bottleneck_dim: tp.Optional[int] = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.state_bottleneck_dim = (
            state_bottleneck_dim
            if state_bottleneck_dim is not None
            else self.factor_dim // 8  # TODO: magic number for shrinkage
        )
        self.dropout = dropout

        self.info_compression_mlp = nn.Sequential(
            nn.Linear(
                self.factor_dim * self.window_size + 2,
                self.state_bottleneck_dim * 4,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.state_bottleneck_dim * 4, self.state_bottleneck_dim),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(
                self.num_stocks * (self.state_bottleneck_dim + self.action_dim),
                self.state_bottleneck_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.state_bottleneck_dim, 1),
        )

    def actual_forward(self, state: torch.Tensor, action: torch.Tensor):
        """Compute the actual Q-value of a given state-action pair.

        Parameters
        ----------
        state : torch.Tensor
            Shape (B, num_stocks, num_factors)
        action : torch.Tensor
            Shape (B, num_stocks)

        Returns
        -------
        torch.Tensor
            Shape (B, 1)

        """
        batch_size, num_stocks, num_factors = state.shape
        compressed_state = self.info_compression_mlp(state)
        mlp_input = torch.cat([compressed_state, action.unsqueeze(-1)], dim=-1)
        mlp_input = mlp_input.reshape(batch_size, -1)
        value = self.output_mlp(mlp_input)
        return value


class RNNCriticModel(PortfolioCriticModel):
    def __init__(
        self, hidden_dim: int, num_layers: int, dropout: float = 0.0, **kwargs
    ) -> None:
        """
        State: (N, state_dim) ->
        """
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.mlp_input_dim = self.num_stocks * (
            self.hidden_dim + self.action_dim + 2
        )

        self.rnn_encoder = nn.LSTM(
            input_size=self.factor_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.readout_head = nn.Sequential(
            nn.Linear(
                in_features=self.mlp_input_dim,
                out_features=self.mlp_input_dim // 4,
            ),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.mlp_input_dim // 4, out_features=1),
        )

    def actual_forward(self, state: torch.Tensor, action: torch.Tensor):
        """First use RNN to encode the factor window, then use MLP to readout
        the Q-value.

        Parameters
        ----------
        state : torch.Tensor
            Shape (B, N, factor_dim * window_size + 2)
        action : torch.Tensor
            Shape (B, N, num_stocks)

        Returns
        -------
        torch.Tensor
            Shape (B, 1)

        """
        batch_size, num_stocks, state_dim = state.shape
        factor_window = state[:, :, :-2].reshape(
            batch_size, num_stocks, self.window_size, self.factor_dim
        )
        state_residual = state[:, :, -2:]

        # Flatten first 2 dims and split it back after RNN
        rnn_output, _ = self.rnn_encoder(
            factor_window.reshape(-1, self.window_size, self.factor_dim)
        )
        rnn_output = rnn_output[:, -1, :].reshape(
            batch_size, num_stocks, self.hidden_dim
        )

        # Concatenate the state residual and action
        mlp_input = torch.cat([rnn_output, state_residual, action], dim=-1)
        mlp_input = mlp_input.reshape(batch_size, -1)
        value = self.readout_head(mlp_input)
        return value


class Reward:
    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, env_snapshot: tp.Dict):
        raise NotImplementedError
