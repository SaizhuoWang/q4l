# Temporal Convolutional Network (TCN)
# Code referred from: https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
# Modified by Saizhuo Wang and GPT-4

from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Input shape: (Batch, Channels, Length)
        return x[
            :, :, : -self.chomp_size
        ].contiguous()  # Output shape: (Batch, Channels, Length - chomp_size)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # Input shape: (Batch, Length, Channels)
        # x = x.transpose(1, 2)  # Shape => (Batch, Channels, Length)
        out = self.net(x)  # Shape: (Batch, n_outputs, Length)
        res = (
            x if self.downsample is None else self.downsample(x)
        )  # Shape: (Batch, n_outputs, Length)
        return self.relu(out + res)  # Output shape: (Batch, n_outputs, Length)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(
        self, input: Dict[str, torch.Tensor]
    ) -> Dict[
        str, torch.Tensor
    ]:  # Input dict with 'x' shape: (Batch, Length, Channels)
        x = input["x"].squeeze(0)
        x = x.transpose(1, 2)  # Shape => (Batch, Channels, Length)
        emb = self.network(x)  # Shape: (Batch, num_channels[-1], Length)
        emb = emb.transpose(1, 2)[:, -1, :]
        return {
            "emb": emb
        }  # Output dict with 'emb' shape: (Batch, num_channels[-1], Length)
