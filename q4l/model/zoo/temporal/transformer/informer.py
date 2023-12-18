# masking.py

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseTemporalEncoder
from .layers.embedding import DataEmbedding_wo_temporal
from .layers.SelfAttention_Family import AttentionLayer, ProbAttention
from .layers.Transformer_EncDec import ConvLayer, Encoder, EncoderLayer


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = (
            torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        )
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index,
            :,
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LocalMask:
    def __init__(self, B, L, S, device="cpu"):
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)
            self._mask2 = ~torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=-self.len
            ).to(device)
            self._mask = self._mask1 + self._mask2

    @property
    def mask(self):
        return self._mask


class Model(nn.Module):
    """Informer with Propspare attention in O(LlogL) complexity."""

    def __init__(
        self,
        output_attention: bool,
        input_size: int,
        hidden_size: int,
        dropout: float,
        factor: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        distil: bool,
        activation: str = "relu",
        **kwargs: Any,
    ):
        super(Model, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_wo_temporal(
            input_size,
            hidden_size,
            dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
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
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(hidden_size) for _ in range(e_layers - 1)]
            if distil
            else None,
            norm_layer=torch.nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        x_enc,
        enc_self_mask=None,
    ):
        x_enc += 1e-10

        if torch.any(torch.isnan(x_enc)):
            print(f"Detected NaN in x_enc")

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = torch.mean(enc_out, dim=1, keepdim=False)
        if torch.any(torch.isnan(enc_out)):
            torch.nan_to_num(enc_out, nan=1e-10)
        return enc_out


class Informer(BaseTemporalEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = Model(**kwargs)
