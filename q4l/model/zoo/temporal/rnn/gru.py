import typing as tp

import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.activation = nn.ReLU()
        self.keep_seq = False

    def forward(self, batch: tp.Dict) -> tp.Dict:
        x = batch["x"].squeeze(0)
        emb, _ = self.gru(x)  # See lstm.py for explanation
        if self.keep_seq:
            emb = emb
        else:
            emb = emb[:, -1, :]
        emb = self.activation(emb)
        return {"emb": emb}
