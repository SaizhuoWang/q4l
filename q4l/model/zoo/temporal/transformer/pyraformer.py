# Reference: https://github.com/cure-lab/LTSF-Linear/blob/main/Pyraformer/pyraformer/Layers.py

# Modules.py
import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear

from q4l.model.zoo.temporal.base import BaseTemporalEncoder
from q4l.model.zoo.temporal.transformer.layers.embedding import (
    DataEmbedding_wo_temporal,
)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention."""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module."""

    def __init__(
        self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


def get_mask(input_size, window_size, inner_size):
    """Get the attention mask of PAM-Naive."""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (
                i - start
            ) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (
                    i - start + 1
                ) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(all_sizes, window_size):
    """Gather features from PAM's pyramid sequences."""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes))

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(
                inner_layer_idx // window_size[j - 1], all_sizes[j] - 1
            )
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][: input_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][: all_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """Get the index of the key that a given query needs to attend to."""
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = (
        torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1
    )

    for i in range(input_size):
        mask[i, 0:window_size] = (
            i + torch.arange(window_size) - window_size // 2
        )
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size + i, 0:window_size] = (
            input_size + i + torch.arange(window_size) - window_size // 2
        )
        mask[input_size + i, mask[input_size + i] < input_size] = -1
        mask[input_size + i, mask[input_size + i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size + i, window_size : (window_size + stride)] = (
                torch.arange(stride) + i * stride
            )
        else:
            mask[input_size + i, window_size : (window_size + second_last)] = (
                torch.arange(second_last) + i * stride
            )

        mask[input_size + i, -1] = i // stride + third_start
        mask[input_size + i, mask[input_size + i] > fourth_start - 1] = (
            fourth_start - 1
        )
    for i in range(third_length):
        mask[third_start + i, 0:window_size] = (
            third_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[third_start + i, mask[third_start + i] < third_start] = -1
        mask[third_start + i, mask[third_start + i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start + i, window_size : (window_size + stride)] = (
                input_size + torch.arange(stride) + i * stride
            )
        else:
            mask[third_start + i, window_size : (window_size + third_last)] = (
                input_size + torch.arange(third_last) + i * stride
            )

        mask[third_start + i, -1] = i // stride + fourth_start
        mask[third_start + i, mask[third_start + i] > full_length - 1] = (
            full_length - 1
        )
    for i in range(fourth_length):
        mask[fourth_start + i, 0:window_size] = (
            fourth_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[fourth_start + i, mask[fourth_start + i] < fourth_start] = -1
        mask[fourth_start + i, mask[fourth_start + i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start + i, window_size : (window_size + stride)] = (
                third_start + torch.arange(stride) + i * stride
            )
        else:
            mask[
                fourth_start + i, window_size : (window_size + fourth_last)
            ] = (third_start + torch.arange(fourth_last) + i * stride)

    return mask


def get_k_q(q_k_mask):
    """Get the index of the query that can attend to the given key."""
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]

    return k_q_mask


class EncoderLayer(nn.Module):
    """Compose with two layers."""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        normalize_before=True,
        use_tvm=False,
    ):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask
            )

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=window_size,
            stride=window_size,
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM."""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                ]
            )
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)
        self.up = Linear(d_inner, d_model)
        self.down = Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)

        return all_inputs


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        d_model: int,
        window_size: List[int],
        n_layer: int,
        d_inner_hid: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float,
        input_size: int,
        d_bottleneck: int,
        inner_size: int,
        seq_len: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.mask, self.all_size = get_mask(seq_len, window_size, inner_size)
        self.indexes = refer_points(self.all_size, window_size)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    d_inner_hid,
                    n_head,
                    d_k,
                    d_v,
                    dropout=dropout,
                    normalize_before=False,
                )
                for i in range(n_layer)
            ]
        )
        self.enc_embedding = DataEmbedding_wo_temporal(
            input_size, d_model, dropout
        )
        self.conv_layers = Bottleneck_Construct(
            d_model, window_size, d_bottleneck
        )

        self.hier_downsample = nn.Linear(len(self.all_size) * d_model, d_model)

    def forward(self, x_enc):
        seq_enc = self.enc_embedding(x_enc)
        seq_enc = self.conv_layers(seq_enc)
        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)
        indexes = self.indexes.repeat(
            seq_enc.size(0), 1, 1, seq_enc.size(2)
        ).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)
        seq_enc = self.hier_downsample(seq_enc)
        return seq_enc


class Model(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        # The above code is declaring a variable named "d_model" in Python.
        input_size: int,
        hidden_size: int,
        window_size: List[int],
        n_layer: int,
        d_inner_hid: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float,
        d_bottleneck: int,
        inner_size: int,
        seq_len: int,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model=hidden_size,
            window_size=window_size,
            n_layer=n_layer,
            d_inner_hid=d_inner_hid,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            input_size=input_size,
            d_bottleneck=d_bottleneck,
            inner_size=inner_size,
            seq_len=seq_len,
        )

    def forward(self, x_enc):
        """Return the hidden representations and predictions.

        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.

        """
        enc_output = self.encoder(x_enc)
        enc_output = torch.mean(enc_output, dim=1)
        return enc_output


class PyraFormer(BaseTemporalEncoder):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.model = Model(**kwargs)
