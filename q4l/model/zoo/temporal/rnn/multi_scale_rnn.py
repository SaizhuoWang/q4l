# Hierarchical Multiscale Recurrent Neural Networks, ICLR 2017
# https://arxiv.org/abs/1609.01704

import math
from typing import Dict, List

import torch
import torch.nn as nn


def get_lcm(a: int, b: int) -> int:
    return int(round(a * b / math.gcd(a, b)))


class MultiScaleRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        scales: List[int],
    ):
        super(MultiScaleRNN, self).__init__()
        self.scales = scales
        self.input_size = input_size
        self.seq_len = seq_len
        self._init_multi_scale_rnn(num_layers, hidden_size, dropout)

    def _init_multi_scale_rnn(
        self,
        num_layers: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        if len(self.scales) == 1:
            lcm = self.scales[0]
        else:
            lcm = get_lcm(*self.scales[:2])
            for scale in self.scales[2:]:
                lcm = get_lcm(lcm, scale)
        if self.seq_len % lcm != 0:
            raise ValueError(f"time steps should be divisible by {lcm}")
        self.multi_scale_rnns = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in self.scales
            ]
        )

    def _extract(self, net: torch.Tensor, gap: int) -> torch.Tensor:
        indices = torch.tensor(list(range(0, self.seq_len, gap)) + [-1])
        return net[:, indices]

    def _get_nets(self, net: torch.Tensor) -> List[torch.Tensor]:
        nets = []
        for i, scale in enumerate(sorted(self.scales, reverse=True)):
            nets.append(self.multi_scale_rnns[i](self._extract(net, scale))[0])
        return nets

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        x = x.squeeze()
        nets = self._get_nets(x)
        emb = torch.mean(torch.stack([net[:, -1] for net in nets]), dim=0)
        return {"emb": emb}
