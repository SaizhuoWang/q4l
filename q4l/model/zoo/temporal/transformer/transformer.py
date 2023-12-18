from typing import Any

import torch
import torch.nn as nn
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer

from q4l.model.zoo.temporal.base import BaseTemporalEncoder

from .layers.embedding import DataEmbedding_wo_temporal


class Model(nn.Module):
    """Vanilla Transformer with O(L^2) complexity."""

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_wo_temporal(
            configs.enc_in,
            configs.d_model,
            # configs.embed,
            # configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding_wo_temporal(
            configs.dec_in,
            configs.d_model,
            # configs.embed,
            # configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(
        self,
        x_enc,
        # x_mark_enc,
        # x_dec,
        # x_mark_dec,
        enc_self_mask=None,
        # dec_self_mask=None,
        # dec_enc_mask=None,
    ):

        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = torch.mean(enc_out, dim=1)
        return enc_out

        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(
        #     dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask
        # )

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len :, :], attns
        # else:
        #     return dec_out[:, -self.pred_len :, :]


class VanillaTransformer(BaseTemporalEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model = Model()
