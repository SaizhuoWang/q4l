import torch
import torch.nn as nn

from . import PortfolioActorModel, Reward


class EIIELSTMModel(PortfolioActorModel):
    def __init__(self, hidden_dim, num_layers=2, **kwargs):
        super().__init__(**kwargs)
        self.lstm = nn.LSTM(
            self.input_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )
        self.head = nn.Linear(hidden_dim + 4, 1)
        self.cash_bias = nn.Parameter(torch.rand(1, 1))

    def actual_forward(self, state) -> torch.Tensor:
        """First use RNN to encode the factor window, then use MLP to
        readout."""
        factor_state = state[:, :, :-4]
        num_env, num_stocks, _ = factor_state.shape
        factor_state = factor_state.reshape(-1, factor_state.shape[-1])
        factor_state = factor_state.reshape(
            factor_state.shape[0], -1, self.input_dim
        )
        emb, _ = self.lstm(factor_state)
        emb = emb[:, -1, :]
        emb = emb.reshape(num_env, num_stocks, -1)
        emb = torch.cat([emb, state[:, :, -4:]], dim=2)
        pred = self.head(emb).squeeze()
        # pred = torch.cat(
        #     [self.cash_bias.repeat(pred.shape[0], 1), pred], dim=1
        # )  # add cash bias
        position = torch.softmax(pred, dim=1)
        return position


class EIIEReward(Reward):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.prev_asset = 0
        self.new = True

    def __call__(self, state, action):
        this_asset = state[..., -3]
        if self.new:
            self.prev_asset = this_asset
            return 0.0
        self.new = False
        asset = this_asset
        ret_rate = (asset - self.prev_asset) / self.prev_asset
        self.prev_asset = asset
        return ret_rate
