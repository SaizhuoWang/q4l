from typing import Dict, Tuple

import torch
import torch.nn as nn

from . import PortfolioActorModel, Reward


class AlphaStockModel(PortfolioActorModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        G: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, num_layers=num_layers
        )
        self.att_w1 = nn.Linear(hidden_dim, hidden_dim)
        self.att_w2 = nn.Linear(hidden_dim, hidden_dim)
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)

        # Stock-wise self-attention block with readout layer
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=1)
        self.readout_head = nn.Sequential(nn.Linear(hidden_dim, 1, bias=True))

    def portfolio_generator(
        self, stock_scores: torch.Tensor, G: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the portfolio based on stock scores.

        Parameters
        ----------
        stock_scores : torch.Tensor
            The scores for each stock generated by the model.
        G : int
            The threshold for selecting the top and bottom stocks.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The investment proportion for stocks in the b+ and b- portfolios.

        """
        # Sort the scores in descending order and get the sorted indices
        sorted_scores, sorted_indices = torch.sort(
            stock_scores, descending=True
        )

        # Calculate investment proportion for b+
        exp_scores = torch.exp(sorted_scores[:G])
        b_plus = exp_scores / torch.sum(exp_scores)

        # Calculate investment proportion for b-
        exp_neg_scores = torch.exp(1 - sorted_scores[-G:])
        b_minus = exp_neg_scores / torch.sum(exp_neg_scores)

        # Initialize full investment proportion tensors with zeros
        b_plus_full = torch.zeros_like(stock_scores)
        b_minus_full = torch.zeros_like(stock_scores)

        # Place the calculated proportions back to their original positions based on sorted_indices
        b_plus_full[sorted_indices[:G]] = b_plus
        b_minus_full[sorted_indices[-G:]] = b_minus

        return b_plus_full, b_minus_full

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        emb, _ = self.lstm(state)
        emb_last = emb[:, -1, :].unsqueeze(1)

        # Time-wise attention
        attn_score = self.att_w(
            torch.tanh(self.att_w1(emb) + self.att_w2(emb_last))
        ).squeeze()
        attn_score = torch.softmax(attn_score, dim=1)
        final_emb = torch.sum(
            attn_score.unsqueeze(2) * emb, dim=1
        )  # (num_stocks, hidden_dim)

        # Stock-wise self-attention
        stock_wise_attn_emb = self.self_attn(final_emb, final_emb, final_emb)
        stock_score = self.readout_head(
            stock_wise_attn_emb
        ).squeeze()  # (num_stocks,)

        # Portfolio Generator part
        b_plus, b_minus = self.portfolio_generator(stock_score, G)

        return b_plus, b_minus


class AlphaStockReward(Reward):
    """This reward computes the T-day Sharpe ratio, so it maintains a buffer of
    T days of returns and then computes the Sharpe ratio over that window."""

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.window = torch.nan * torch.ones(self.window_size)
        self.prev_asset = 0
        self.new = True

    def __call__(self, env_snapshot: Dict):
        # Push return to window
        ret_rate = (env_snapshot["asset"] - self.prev_asset) / self.prev_asset
        self.prev_asset = env_snapshot["asset"]
        if self.new:
            self.window[0] = ret_rate
            self.new = False
        else:
            self.window[1:] = self.window[:-1]
            self.window[0] = ret_rate
        # Compute sharpe ratio considering NaN as invalid samples
        sharpe_ratio = torch.nanmean(self.window) / torch.nanstd(self.window)
        return sharpe_ratio
