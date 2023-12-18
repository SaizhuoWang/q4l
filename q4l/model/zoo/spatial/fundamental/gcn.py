import typing as tp

import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv

from .....data.graph import StockGraph


class GCN(nn.Module):
    def __init__(
        self,
        node_emb_dim: int,
        stock_kg: StockGraph,
        num_heads: int = 4,
        feat_drop: float = 0.5,
        attn_drop: float = 0.5,
        negative_slope: float = 0.2,
        residual: bool = True,
    ):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.stock_kg = stock_kg

        self.gcnconv = GraphConv(
            in_feats=node_emb_dim,
            out_feats=node_emb_dim,
            allow_zero_in_degree=True,
        )

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        graph.ndata["node_emb"] = temporal_info["emb"]
        homogeneous_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])
        feat_emb = self.gcnconv(homogeneous_graph, temporal_info["emb"])
        return feat_emb
