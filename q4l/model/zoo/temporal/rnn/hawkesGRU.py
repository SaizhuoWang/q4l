import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .....data.dataset import Q4LDataModule
from ....base import EMBEDDING_KEY


class HawkesGRU(nn.Module):
    def __init__(self, input_size, hidden_size, data: Q4LDataModule) -> None:
        super(HawkesGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        print(input_size)
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

        self.linear_in = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = torch.nn.Linear(
            hidden_size * 2, hidden_size, bias=False
        )
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.data = data
        self.ticker_list = data.ticker_list
        self.num_tickers = len(self.ticker_list)
        self.ae = nn.Parameter(torch.FloatTensor(self.num_tickers, 1, 1))
        self.ab = nn.Parameter(torch.FloatTensor(self.num_tickers, 1, 1))

    def forward(self, input_data: tp.Dict) -> tp.Dict:
        x = input_data["x"].squeeze()
        batch_size = x.shape[0]
        device = x.device
        stock_labels = input_data["label"]
        ticker_indices = (
            torch.Tensor(
                [self.ticker_list.index(x[1]) for x in stock_labels[0]]
            )
            .long()
            .to(device)
        )
        ae = self.ae[ticker_indices]
        ab = self.ab[ticker_indices]
        context, query = self.gru(
            x
        )  # (batch_size, seq_len,  hidden_size)   (1, batch_size, hidden_size)
        query = query.reshape(batch_size, 1, self.hidden_size)  # (1284, 1,100)

        # Hawkes Attention
        batch_size, output_len, dim = query.size()  # 1284, 1, 100
        query_len = context.size(1)  # 1284

        query_flat = query.reshape(batch_size * output_len, dim)
        query_flat = self.linear_in(query_flat)
        query = query_flat.reshape(batch_size, output_len, dim)  # 1284, 1, 100

        attention_out = torch.bmm(query, context.transpose(1, 2).contiguous())
        attention_out = attention_out.view(batch_size * output_len, query_len)
        att_weight = self.softmax(attention_out)
        att_weight = att_weight.view(batch_size, output_len, query_len)

        ## Function3
        mix = att_weight * (context.permute(0, 2, 1))
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
        combined = combined.view(batch_size * output_len, 2 * dim)
        output = self.linear_out(combined).view(batch_size, output_len, dim)
        output = output[:, -1, :]
        emb = self.tanh(output)

        return {EMBEDDING_KEY: emb}
