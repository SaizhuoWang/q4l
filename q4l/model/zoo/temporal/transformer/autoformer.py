# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


from typing import Any

import torch
import torch.nn as nn

from ..base import BaseTemporalEncoder
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from .layers.embedding import DataEmbedding_wo_temporal


class Model(nn.Module):
    """Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity."""

    def __init__(
        self,
        output_attention: bool,
        moving_avg: int,
        input_size: int,
        hidden_size: int,
        dropout: float,
        factor: float,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        activation: str = "relu",
        **kwargs: Any,
    ):
        """Initialize the Model.

        Parameters
        ----------
        seq_len : int
            Length of the input sequence.
        label_len : int
            Length of the labeled sequence.
        pred_len : int
            Length of the predicted sequence.
        output_attention : bool
            Flag indicating whether to output attention weights.
        moving_avg : int
            Size of the moving average.
        enc_in : int
            Dimensionality of the input embedding.
        d_model : int
            Dimensionality of the model.
        dropout : float
            Dropout rate.
        factor : float
            Factor for auto-correlation calculation.
        n_heads : int
            Number of attention heads.
        d_ff : int
            Dimensionality of the feed-forward layer.
        e_layers : int
            Number of encoder layers.
        activation : str
            Activation function name.

        """
        super(Model, self).__init__()
        self.output_attention = output_attention

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp(kernel_size[0])
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_temporal(
            input_size,
            hidden_size,
            dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        hidden_size,
                        n_heads,
                    ),
                    hidden_size,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(hidden_size),
        )

    def forward(
        self,
        x_enc,
        enc_self_mask=None,
    ):
        # enc
        x_enc += 1e-10
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = torch.mean(enc_out, dim=1)
        if torch.any(torch.isnan(enc_out)):
            torch.nan_to_num(enc_out, nan=0.0)
        return enc_out


class AutoFormer(BaseTemporalEncoder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = Model(**kwargs)
