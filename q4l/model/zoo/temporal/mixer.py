# Code referred from: https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
# Modified by Saizhuo Wang and GPT-4


import typing as tp

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce

from .base import BaseTemporalEncoder


class TimeMixingBlock(nn.Module):
    def __init__(
        self, window_size: int, expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=window_size, out_features=window_size
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_features=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, (B, T, D)

        Returns
        -------
        torch.Tensor, (B, T, D)
        """
        x_transpose = x.transpose(1, 2)
        x_norm = self.norm(x_transpose)
        x_proj = self.linear(x_norm)
        x_act = self.activation(x_proj)
        x_dropout = self.dropout(x_act)
        return x + x_dropout


class AlphaMixingBlock(nn.Module):
    def __init__(
        self, input_size: int, expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=input_size,
            out_features=int(input_size * expansion_factor),
        )
        self.activation = nn.ReLU()
        self.linear_2 = nn.Linear(
            in_features=int(input_size * expansion_factor),
            out_features=input_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_features=input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, (B, T, D)

        Returns
        -------
        torch.Tensor, (B, T, D)
        """
        x_norm = self.norm(x)
        x_proj_1 = self.linear_1(x_norm)
        x_act = self.activation(x_proj_1)
        x_dropout_1 = self.dropout(x_act)
        x_proj_2 = self.linear_2(x_dropout_1)
        x_dropout_2 = self.dropout(x_proj_2)
        return x + x_dropout_2


class MixerBlock(nn.Module):
    def __init__(
        self, input_size: int, expansion_factor: float, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=input_size,
            out_features=int(input_size * expansion_factor),
        )
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(
            in_features=int(input_size * expansion_factor),
            out_features=input_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, (B, T, D)

        Returns
        -------
        torch.Tensor, (B, T, D)
        """
        x_norm = self.norm(x)
        x_proj_1 = self.linear_1(x_norm)
        x_act = self.activation(x_proj_1)
        x_dropout_1 = self.dropout(x_act)
        x_proj_2 = self.linear_2(x_dropout_1)
        x_dropout_2 = self.dropout(x_proj_2)
        return x + x_dropout_2


class MLPMixer(BaseTemporalEncoder):
    def __init__(
        self,
        input_size: int,
        window_size: int,
        hidden_size: int,
        num_layers: int,
        mixer_version: str = "original",
        expansion_coeff_factor: float = 4,
        expansion_coeff_time: float = 4,
        dropout: float = 0.0,
    ) -> None:
        """MLP-Mixer time-series encoder. Two versions available:

        Parameters
        ----------
        input_size : int
            Number of factors in the input data.
        window_size : int
            Number of time steps in the input data.
        output_dim : int
            Output embedding dimension.
        num_layers : int
            Number of mixer layers.
        mixer_version : str, optional
            Mixer implementation ("original" or "tsmixer"), by default "original"
        expansion_coeff_factor : int, optional
            Hidden dim expansion factor for factors, by default 4
        expansion_coeff_time : float, optional
            Hidden dim expansion factor for time, by default 0.5
        dropout : float, optional
            Dropout probability, by default 0.0

        """
        super().__init__()
        if mixer_version == "original":
            self.mixer_layers = [
                nn.Sequential(
                    Rearrange("b t d -> b d t"),
                    MixerBlock(
                        input_size=window_size,
                        expansion_factor=expansion_coeff_time,
                        dropout=dropout,
                    ),
                    Rearrange("b d t -> b t d"),
                    MixerBlock(
                        input_size=input_size,
                        expansion_factor=expansion_coeff_factor,
                        dropout=dropout,
                    ),
                )
                for _ in range(num_layers)
            ]
        elif mixer_version == "tsmixer":
            self.mixer_layers = [
                nn.Sequential(
                    Rearrange("b t d -> b d t"),
                    TimeMixingBlock(
                        window_size=window_size,
                        expansion_factor=expansion_coeff_time,
                        dropout=dropout,
                    ),
                    Rearrange("b d t -> b t d"),
                    AlphaMixingBlock(
                        input_size=input_size,
                        expansion_factor=expansion_coeff_factor,
                        dropout=dropout,
                    ),
                )
                for _ in range(num_layers)
            ]
        else:
            raise ValueError(
                f"Invalid mixer version {mixer_version}, must be one of ['original', 'tsmixer']"
            )
        self.mixer_encoder = nn.Sequential(*self.mixer_layers)
        self.pooling_layer = Reduce("b t d -> b d", reduction="mean")
        self.readout_layer = nn.Linear(
            in_features=input_size, out_features=hidden_size
        )

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        mixer_emb = self.mixer_encoder(x)
        pooling_emb = self.pooling_layer(mixer_emb)
        output_emb = self.readout_layer(pooling_emb)
        return {
            "emb": output_emb,
        }
