import typing as tp

import torch
import torch.nn as nn

from ....base import EMBEDDING_KEY, INPUT_KEY


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        self.keep_seq_len = False
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.activation = nn.ReLU()
        self.keep_seq = False

    def forward(
        self, batch: tp.Dict[str, torch.Tensor]
    ) -> tp.Dict[str, torch.Tensor]:
        x = batch[INPUT_KEY].squeeze()
        # ticker_symbols = [item[1] for item in batch["label"][0]]
        emb, _ = self.lstm(
            x
        )  # lstm forward returns: output(batch_size, hidden_size)
        # emb = emb[:, -1, :]
        if self.keep_seq:
            emb = emb
        else:
            emb = emb[:, -1, :]
        emb = self.activation(emb)
        return {EMBEDDING_KEY: emb}
