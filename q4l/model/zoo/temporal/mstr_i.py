import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....data.dataset import Q4LDataModule
from ...base import EMBEDDING_KEY


def Self_Attention(x):
    Q = K = V = x
    d_k = Q.size(-1)

    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V)
    output = x + attn_output

    return output


class DilatedCausalConvolution(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, causal=True
    ):
        super(DilatedCausalConvolution, self).__init__()
        self.causal = causal
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation if causal else 0
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

    def forward(self, x):
        convolved = self.conv(x)
        if self.causal:
            # Remove the padding from the end to maintain the causality
            return (
                convolved[:, :, : -self.padding]
                if self.padding > 0
                else convolved
            )
        else:
            return convolved


class GatedLinearUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedLinearUnit, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv1(x) * torch.sigmoid(self.conv2(x))


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = DilatedCausalConvolution(
            n_inputs, n_outputs, kernel_size, dilation
        )
        self.glu1 = GatedLinearUnit(n_outputs, n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        self.conv1.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.glu1(out)
        out = self.dropout1(out)
        res = self.downsample(x) if self.downsample is not None else x
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.hidden_states = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size, dilation, dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        self.hidden_states = []
        out = x
        for i, layer in enumerate(self.network):
            out = layer(out)
            self.hidden_states.append(out)
        return out, self.hidden_states


class Hawkes_Attention(torch.nn.Module):
    def __init__(self, dimensions, num_stocks):
        super().__init__()

        self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = torch.nn.Linear(
            dimensions * 2, dimensions, bias=False
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = torch.nn.Parameter(torch.FloatTensor(num_stocks, 1, 1))
        self.ab = torch.nn.Parameter(torch.FloatTensor(num_stocks, 1, 1))
        nn.init.uniform_(self.ae, -0.1, 0.1)
        nn.init.uniform_(self.ab, -0.1, 0.1)

    def forward(self, query, context, index):
        device = query.device
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        ae = self.ae[index]
        ab = self.ab[index]

        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(
            query, context.transpose(1, 2).contiguous()
        )

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(
            batch_size * output_len, query_len
        )
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(
            batch_size, output_len, query_len
        )

        mix = attention_weights * (context.permute(0, 2, 1))

        delta_t = (
            torch.flip(torch.arange(0, query_len), [0])
            .type(torch.float32)
            .to(device)
        )
        delta_t = delta_t.repeat(batch_size, 1).reshape(
            batch_size, 1, query_len
        )
        bt = torch.exp(-1 * ab * delta_t)
        term_2 = F.relu(ae * mix * bt)
        mix = torch.sum(term_2 + mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(
            batch_size, output_len, dimensions
        )
        output = self.tanh(output)

        return output, attention_weights


class MSTR_I(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        kernel_size,
        dropout,
        num_channels,
        data: Q4LDataModule,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.data = data
        self.ticker_list = data.ticker_list
        self.num_stocks = len(self.ticker_list)

        # TCN+GLU
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size, dropout
        )

        # Hawkes_attention
        self.hawkes = Hawkes_Attention(
            dimensions=hidden_size,
            num_stocks=self.num_stocks,
        )
        self.embedding_table = nn.Embedding(
            len(self.ticker_list), self.hidden_size
        )
        nn.init.uniform_(self.embedding_table.weight, -0.1, 0.1)
        self.stock_query_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze(0)
        x.shape[0]
        device = x.device
        [item[1] for item in input_data["label"][0]]
        ticker_codes = [item[1] for item in input_data["label"][0]]
        ticker_idx = torch.tensor(
            [self.ticker_list.index(x) for x in ticker_codes]
        ).to(device)

        # Self-Attention
        x = Self_Attention(x)  # (batch_size, time_steps, input_size)

        # Dilated causal convolution & GLU
        x = x.permute(0, 2, 1)
        output, hidden_states = self.tcn(x)  # (batch_size, hidden, time_steps)

        emb_transformed = []  # list to store transformed embeddings
        for i in hidden_states:
            emb_transformed.append(self.linear(i))
        emb = emb_transformed

        return {EMBEDDING_KEY: emb}
