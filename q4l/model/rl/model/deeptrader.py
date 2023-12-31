from typing import Tuple

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

from ...zoo.temporal.tcn import TemporalConvNet
from . import PortfolioActorModel


# NOTE: Very preliminary implementation, need further polishing
class DeepTraderASU(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size, dropout
        )
        hidden_size = num_channels[-1]

        # TODO: Compute `t`, `n` based on TCN config
        t = 1
        n = 10

        self.w1 = nn.Linear(t, 1, bias=False)
        self.w2 = nn.Linear(t, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, 1, bias=False)
        self.bs = nn.Parameter(torch.rand(n))
        self.vs = nn.Linear(n, n, bias=False)

        self.fc = nn.Linear(hidden_size, 1, bias=True)

        self.gcn = GraphConv(
            in_feats=num_channels[-1], out_feats=num_channels[-1]
        )

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

    def forward(self, x, graph):
        emb = self.tcn(x)  # (B, C, T)

        sa_x = (
            torch.mm(self.w2(self.w1(emb)), self.w3(emb.transpose(1, 2)))
            + self.bs
        )
        sa_s = self.vs(torch.sigmoid(sa_s))

        g_emb = self.gcn(graph, x)

        sa_ag = torch.mm(sa_s, g_emb)
        s = torch.sigmoid(self.fc(sa_ag))
        action = self.portfolio_generator(s, 10)

        return action


class DeepTraderModel(PortfolioActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actual_forward(self, **kwargs):
        return super().actual_forward(**kwargs)
