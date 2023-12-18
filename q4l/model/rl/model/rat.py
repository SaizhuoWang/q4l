import torch
import torch.nn as nn
from torch import Tensor

from . import PortfolioActorModel, Reward


class CABlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, tau: int, dropout: float = 0.1
    ) -> None:
        self.num_heads = num_heads
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size, num_tokens, emb_dim = query.size()
        # For Q and K, they are padded and used for other purposes
        padding_zeros = torch.zeros(
            (batch_size, self.tau, emb_dim), device=query.device
        )
        windowed_key = torch.cat([padding_zeros, key], dim=1).unfold(
            1, self.tau, 1
        )  # (B, T, L, E)
        windowed_value = torch.cat([padding_zeros, value], dim=1).unfold(
            1, self.tau, 1
        )  # (B, T, L, E)
        windowed_query = query.unsqueeze(2)  # (B, T, 1, E)
        # First, on the L dimension, do the multi-head attention with projection
        k = (
            self.proj(windowed_key)
            .view(
                batch_size, num_tokens, self.tau, self.num_heads, self.head_dim
            )
            .permute(0, 3, 1, 2, 4)
        )  # (B, H, T, L, E/H)
        v = (
            self.proj(windowed_value)
            .view(
                batch_size, num_tokens, self.tau, self.num_heads, self.head_dim
            )
            .permute(0, 3, 1, 2, 4)
        )  # (B, H, T, L, E/H)
        q = (
            self.proj(windowed_query)
            .view(batch_size, num_tokens, 1, self.num_heads, self.head_dim)
            .permute(0, 3, 1, 2, 4)
        )  # (B, H, T, 1, E/H)
        # Do attention on the L dimension
        attn_score_context = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scaling, dim=-1
        )  # (B, H, T, 1, L)
        # Aggregate context on the L dimension
        weighted_sum = torch.matmul(attn_score_context, v)  # (B, H, T, 1, E/H)
        # Concatenate the heads
        weighted_sum = (
            weighted_sum.permute(0, 2, 1, 3, 4)
            .squeeze()
            .reshape(batch_size, num_tokens, emb_dim)
        )
        return weighted_sum


class ContextAttentionBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, tau: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim**-0.5

        self.q_transform = CABlock(embed_dim, num_heads, tau, dropout)
        self.k_transform = CABlock(embed_dim, num_heads, tau, dropout)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.tau = tau

    def forward(self, query, key, value) -> Tensor:
        # query, key, value = x, x, x
        batch_size, num_tokens, emb_dim = query.size()
        q = self.q_transform(query, key, value)  # (B, T, E)
        k = self.k_transform(key, query, value)  # (B, T, E)
        v = self.v_proj(value)  # (B, T, E)
        # Multi-head attention on the T dimension
        q = q.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (B, H, T, E/H)
        k = k.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (B, H, T, E/H)
        v = v.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).permute(
            0, 2, 1, 3
        )  # (B, H, T, E/H)
        attn_score = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scaling, dim=-1
        )  # (B, H, T, T)
        weighted_sum = torch.matmul(attn_score, v)  # (B, H, T, E/H)
        # Concat back the heads
        weighted_sum = weighted_sum.permute(0, 2, 1, 3).reshape(
            batch_size, num_tokens, emb_dim
        )  # (B, T, E)
        return weighted_sum


class RATEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        tau: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(window_size, embed_dim))
        self.seq_attn = ContextAttentionBlock(
            embed_dim, num_heads, tau, dropout
        )
        self.rel_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        x = x + self.pos_enc.unsqueeze(0)
        x1 = self.seq_attn(x, x, x)
        x2 = self.rel_attn(x1, x1, x1)
        x = self.layer_norm_1(x + x2)
        x3 = self.mlp(x)
        x = self.layer_norm_2(x + x3)
        return x


class RATDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        tau: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(window_size, embed_dim))
        self.seq_attn_1 = ContextAttentionBlock(
            embed_dim, num_heads, tau, dropout
        )
        self.seq_attn_2 = ContextAttentionBlock(
            embed_dim, num_heads, tau, dropout
        )
        self.rel_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.layer_norm_3 = nn.LayerNorm(embed_dim)

    def forward(self, enc_output, dec_input):
        x = dec_input + self.pos_enc.unsqueeze(0)
        x1 = self.seq_attn_1(x, x, x)
        x2 = self.rel_attn(x1, x1, x1)
        x = self.layer_norm_1(x + x2)
        x3 = self.seq_attn_2(x, enc_output, enc_output)
        x = self.layer_norm_2(x + x3)
        x4 = self.mlp(x)
        x = self.layer_norm_3(x + x4)


class DecisionLayer(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.initial_layer = nn.Linear(emb_dim + 1, 1)
        self.shortsale_layer = nn.Linear(emb_dim + 1, 1)
        self.reinvest_layer = nn.Linear(emb_dim + 1, 1)
        self.prev_layer = 0

    def forward(self, emb):
        init = torch.softmax(self.initial_layer(emb), dim=-1)
        shortsale = torch.softmax(self.shortsale_layer(emb), dim=-1)
        reinvest = torch.softmax(self.reinvest_layer(emb), dim=-1)
        action = init - shortsale + reinvest
        # TODO: Add previous action here to enforce continuity
        # ...

        return action


class RelationAwareTransformer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, tau: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = RATEncoderLayer(embed_dim, num_heads, tau, dropout)
        self.decoder = RATDecoderLayer(embed_dim, num_heads, tau, dropout)
        self.decision_layer = DecisionLayer(embed_dim)

    def forward(self, x: Tensor):
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output, x)
        action = self.decision_layer(dec_output)
        return action


class RATModel(PortfolioActorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def actual_forward(self, **kwargs):
        return super().actual_forward(**kwargs)


class RATReward(Reward):
    def __init__(self, cost_rate: float = 0.0):
        super().__init__()
        self.cost_rate = cost_rate
        self.reset()

    def reset(self):
        self.prev_asset = 0
        self.new = True

    def __call__(self, state, action):
        this_asset = state[..., -3]
        if self.new:
            self.prev_asset = this_asset
            return 0.0
        self.new = False
        asset = this_asset
        ret_rate = (asset - self.prev_asset) / self.prev_asset
        self.prev_asset = asset
        return ret_rate * (1 - self.cost_rate)
