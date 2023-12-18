import typing as tp

import torch
import torch.nn as nn

from ..base import BaseTemporalEncoder


class ALSTMModel(BaseTemporalEncoder):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        rnn_type="GRU",
    ):
        super().__init__()
        self.hid_size = hidden_size
        self.input_size = d_feat
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn_layer = num_layers
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module(
            "fc_in",
            nn.Linear(in_features=self.input_size, out_features=self.hid_size),
        )
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hid_size,
            hidden_size=self.hid_size,
            num_layers=self.rnn_layer,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(
            in_features=self.hid_size * 2, out_features=self.hid_size
        )
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(
                in_features=self.hid_size, out_features=int(self.hid_size / 2)
            ),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(
                in_features=int(self.hid_size / 2), out_features=1, bias=False
            ),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs: torch.Tensor):
        rnn_out, _ = self.rnn(
            self.net(inputs)
        )  # [batch, seq_len, num_directions * hidden_size]
        attention_score = self.att_net(rnn_out)  # [batch, seq_len, 1]
        out_att = torch.mul(rnn_out, attention_score)
        out_att = torch.sum(out_att, dim=1)
        out = self.fc_out(torch.cat((rnn_out[:, -1, :], out_att), dim=1))
        return out


class ALSTM(BaseTemporalEncoder):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: str = "GRU",
    ) -> None:
        super().__init__()
        self.model = ALSTMModel(
            d_feat=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
        )

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        emb = self.model(x)
        return {"emb": emb}
