import typing as tp

import dgl
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import RelGraphConv

from .....data.graph import StockGraph


class RGCN(nn.Module):
    def __init__(
        self,
        node_emb_dim: int,
        stock_kg: StockGraph,
        regularizer: str = "basis",
        num_bases: int = 4,
        bias: bool = True,
        activation: tp.Callable = None,
        self_loop: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.stock_kg = stock_kg

        self.gcnconv = RelGraphConv(
            in_feat=node_emb_dim,
            out_feat=node_emb_dim,
            num_rels=stock_kg.num_edge_types,
            regularizer=regularizer,
            num_bases=num_bases,
            bias=bias,
            activation=activation,
            self_loop=self_loop,
            dropout=dropout,
        )

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        graph.ndata["node_emb"] = temporal_info["emb"]
        homogeneous_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])
        feat_emb = self.gcnconv(
            homogeneous_graph,
            temporal_info["emb"],
            etypes=homogeneous_graph.edata["_TYPE"],
        )
        return feat_emb
