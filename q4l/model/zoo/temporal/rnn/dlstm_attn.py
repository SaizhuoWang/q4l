"""From "Huynh et al., Efficient Integration of Multi-Order Dynamics and
Internal Dynamics in Stock Movement Prediction", WSDM 2023 Official code:

https://github.com/thanhtrunghuynh93/estimate.

"""

import typing as tp

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from .....data.dataset import Q4LDataModule
from ....base import EMBEDDING_KEY, INPUT_KEY


class DLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        emb_dim: int,
        bottleneck_dim: int,
        rnn_hidden_dim: int,
        num_total_stocks: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        # Model hyperparams
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.bottleneck_dim = bottleneck_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_total_stocks = num_total_stocks
        self.bias = bias

        # Model parameters
        self.stock_memory_embedding = nn.Embedding(num_total_stocks, emb_dim)
        self.rnn_param_readout_model = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(
                bottleneck_dim,
                4 * (input_size + rnn_hidden_dim) * rnn_hidden_dim,
            ),
        )
        self.lstm_bias = nn.Parameter(data=torch.rand(4 * rnn_hidden_dim))

    def forward(
        self, x: torch.Tensor, index: torch.IntTensor, hx=None
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """A step of LSTM cell.

        Parameters
        ----------
        x : torch.Tensor
            (num_stocks, input_size), input features
        index: torch.Tensor
            (num_stocks, ), index w.r.t. the whole stock list
        hx: torch.Tensor, optional
            Hidden state, by default None

        Returns
        -------
        tp.Tuple[torch.Tensor, torch.Tensor]
            (hx, cx)

        """
        if hx is None:
            hx = Variable(x.new_zeros(x.size(0), self.rnn_hidden_dim))
            hx = (hx, hx)
        hx, cx = hx
        gates = self._dynamic_weight_fwd(inputs=x, index=index, state=hx)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t
        hy = o_t * torch.tanh(cy)

        return (hy, cy)

    def _dynamic_weight_fwd(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        state: torch.Tensor,
    ):
        """TBD.

        Parameters
        ----------
        inputs : torch.Tensor
            (num_stocks, input_size)
        index : torch.Tensor
            (num_stocks, )
        state : torch.Tensor
            (num_stocks, hidden_size)

        Returns
        -------
        _type_
            _description_

        """
        num_stocks, input_size = inputs.shape
        # Perform stock-specific memory-based dynamic weight generation
        memory = self.stock_memory_embedding(index)
        weights = self.rnn_param_readout_model(
            memory
        )  # (num_stocks, input_dim * rnn_hidden_dim * 4)
        weights = weights.reshape(
            num_stocks, (input_size + self.rnn_hidden_dim), -1
        )  # (num_stocks, input_dim + rnn_hidden_dim, rnn_hidden_dim * 4)
        data = torch.cat(
            [inputs, state], dim=-1
        )  # (num_stocks, input_dim + rnn_hidden_dim)
        value = torch.sigmoid(
            torch.matmul(data.unsqueeze(1), weights).squeeze()
            + self.lstm_bias.unsqueeze(0)
        )
        return value


class DLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        emb_dim,
        bottleneck_dim,
        num_stocks,
        num_layers=1,
        bias=True,
        output_size=1,
    ):
        super(DLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.num_nodes = num_stocks
        self.output_size = output_size

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(
            DLSTMCell(
                input_size=self.input_size,
                emb_dim=emb_dim,
                bottleneck_dim=bottleneck_dim,
                rnn_hidden_dim=hidden_size,
                num_total_stocks=num_stocks,
                bias=bias,
            )
        )
        for l in range(1, self.num_layers):
            self.rnn_cell_list.append(
                DLSTMCell(
                    input_size=self.input_size,
                    emb_dim=emb_dim,
                    bottleneck_dim=bottleneck_dim,
                    rnn_hidden_dim=hidden_size,
                    num_total_stocks=num_stocks,
                    bias=bias,
                )
            )

    def forward(self, input, index, hx=None):
        device = input.device
        if hx is None:
            h0 = Variable(
                torch.zeros(
                    (self.num_layers, input.size(0), self.hidden_size),
                    device=device,
                )
            )
        else:
            h0 = hx

        outs = []

        hidden = list()
        for layer in range(self.num_layers):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                        input[:, t, :],
                        index,
                        (hidden[layer][0], hidden[layer][1]),
                    )
                else:
                    hidden_l = self.rnn_cell_list[layer](
                        hidden[layer - 1][0],
                        index,
                        (hidden[layer][0], hidden[layer][1]),
                    )

                hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        out = torch.stack(outs, axis=1).to(device)

        return out, hidden


def get_subsequent_mask(seq):
    """For masking out the subsequent info."""

    sz_b, len_s = seq.size(1), seq.size(2)
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8),
        diagonal=1,
    )
    subsequent_mask = (
        subsequent_mask.bool().bool().unsqueeze(0).expand(sz_b, -1, -1)
    )

    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class TemporalAttention(nn.Module):
    """Temporal Attention module."""

    def __init__(self, n_head, rnn_unit, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(rnn_unit, n_head * d_k)
        self.w_ks = nn.Linear(rnn_unit, n_head * d_k)
        self.w_vs = nn.Linear(rnn_unit, n_head * d_v)
        nn.init.normal_(
            self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k))
        )
        nn.init.normal_(
            self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_k))
        )
        nn.init.normal_(
            self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (rnn_unit + d_v))
        )

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5)
        )
        self.layer_norm = nn.LayerNorm(rnn_unit)

        self.fc = nn.Linear(n_head * d_v, rnn_unit)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        # output = output[:, -1, :]
        return output, attn


class DLSTM_ATTN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        data: Q4LDataModule,
        rnn_hidden_unit=8,
        emb_dim: int = 64,
        bottleneck_dim: int = 32,
        n_head=4,
        d_k=8,
        d_v=8,
        drop_prob=0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_stocks = len(data.ticker_list)
        self.rnn_hidden_unit = rnn_hidden_unit
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.rnn_hidden_unit = rnn_hidden_unit
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.ticker_list = data.ticker_list

        self.lstm1 = DLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            emb_dim=emb_dim,
            bottleneck_dim=bottleneck_dim,
            num_stocks=self.num_stocks,
        )
        # TODO: make clear ln_1, lstm2, temp_attn
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.lstm2 = DLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            emb_dim=emb_dim,
            bottleneck_dim=bottleneck_dim,
            num_stocks=self.num_stocks,
        )
        self.temp_attn = TemporalAttention(
            self.n_head,
            self.hidden_size,
            self.d_k,
            self.d_v,
            dropout=self.drop_prob,
        )
        self.dropout = nn.Dropout(self.drop_prob)

        self.embedding = nn.Linear(input_size, hidden_size)

    def forward(
        self, batch: tp.Dict[str, torch.Tensor]
    ) -> tp.Dict[str, torch.Tensor]:
        inputs = batch[
            INPUT_KEY
        ].squeeze()  # (batch_size, num_stock, seq_len, input_size)
        device = inputs.device
        num_stock = inputs.shape[0]
        seq_len = inputs.shape[1]

        slf_attn_mask = get_subsequent_mask(batch[INPUT_KEY]).bool()
        ticker_codes = [item[1] for item in batch["label"][0]]
        ticker_idx = torch.tensor(
            [self.ticker_list.index(x) for x in ticker_codes]
        ).to(device)

        output, _ = self.lstm1(inputs, ticker_idx)
        output = self.ln_1(output)
        enc_output, _ = self.lstm2(output, ticker_idx)
        enc_output, enc_slf_attn = self.temp_attn(
            enc_output, enc_output, enc_output, mask=slf_attn_mask.bool()
        )

        enc_output = torch.reshape(
            enc_output, (-1, seq_len, num_stock, self.rnn_hidden_unit)
        )  # [batch_size, seq_len, self.num_stock, self.rnn_hidden_unit]
        enc_output = self.dropout(enc_output)
        emb = enc_output.permute(2, 1, 0, 3).reshape(num_stock, seq_len, -1)[
            :, -1, :
        ]
        return {EMBEDDING_KEY: emb}
