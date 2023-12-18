from typing import Any, List, Union

import torch
import torch.nn as nn

from q4l.model.zoo.temporal.base import BaseTemporalEncoder

from .layers.AutoCorrelation import AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
    series_decomp_multi,
)
from .layers.embedding import DataEmbedding_wo_temporal
from .layers.FourierCorrelation import FourierBlock
from .layers.MultiWaveletCorrelation import MultiWaveletTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    """FEDformer performs the attention mechanism on frequency domain and
    achieved O(N) complexity."""

    def __init__(
        self,
        version: str,
        mode_select: str,
        modes: int,
        seq_len: int,
        moving_avg: Union[int, List[int]],
        input_size: int,
        hidden_size: int,
        dropout: float,
        L: int,
        base: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        activation: str = "relu",
        **kwargs: Any,
    ) -> None:
        super(Model, self).__init__()

        self.version = version
        self.mode_select = mode_select
        self.modes = modes
        self.seq_len = seq_len

        # Decomp
        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_temporal(
            input_size, hidden_size, dropout
        )
        self.dec_embedding = DataEmbedding_wo_temporal(
            input_size, hidden_size, dropout
        )

        if version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=hidden_size, L=L, base=base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=hidden_size,
                out_channels=hidden_size,
                seq_len=self.seq_len,
                modes=modes,
                mode_select_method=mode_select,
            )
        # Encoder
        enc_modes = int(min(modes, seq_len // 2))
        print("enc_modes: {}".format(enc_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, hidden_size, n_heads
                    ),
                    hidden_size,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
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
        enc_out = self.enc_embedding(
            x_enc,
        )
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = torch.mean(enc_out, dim=1)
        if torch.any(torch.isnan(enc_out)):
            torch.nan_to_num(enc_out, nan=0.0)
        return enc_out


class FEDFormer(BaseTemporalEncoder):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.model = Model(**kwargs)
