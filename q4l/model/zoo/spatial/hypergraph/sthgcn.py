import typing as tp

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from torch_geometric import nn as nnn
from torch_geometric import utils


class TimeBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size))
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size))
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, (1, kernel_size))

    def forward(self, X):
        X = X.unsqueeze(0).unsqueeze(2)
        X = X.permute(
            0, 3, 1, 2
        )  # (batch_size, num_features=in_channels, num_nodes, timesteps)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        out = out.permute(0, 2, 3, 1)
        out = out.squeeze()
        return out


class STGCNBlock(nn.Module):
    def __init__(self, node_emb_dim):
        super().__init__()
        self.time_block1 = TimeBlock(node_emb_dim)
        self.hypergraph_conv = nnn.HypergraphConv(
            node_emb_dim,
            node_emb_dim,
            use_attention=False,
            heads=4,
            concat=False,
            negative_slope=0.2,
            dropout=0.5,
            bias=True,
        )
        self.time_block2 = TimeBlock(node_emb_dim)

    def forward(self, X, hyp_input, hyperedge_attr):
        X = self.time_block1(X)
        X = self.hypergraph_conv(X, hyp_input, hyperedge_attr=hyperedge_attr)
        X = self.time_block2(X)
        return X


class STHGCN(nn.Module):
    def __init__(
        self,
        node_emb_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.node_emb_dim = node_emb_dim

        self.sthgcn_block1 = STGCNBlock(node_emb_dim)
        self.sthgcn_block2 = STGCNBlock(node_emb_dim)
        self.time_block = TimeBlock(node_emb_dim)

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        graph.ndata["node_emb"] = temporal_info["emb"]
        device = graph.device
        graph = graph.to("cpu")
        num_nodes = sum(
            [
                graph.num_nodes(ntype)
                for ntype in graph.ntypes
            ]
        )
        num_edge_types = len(graph.canonical_etypes)
        inci_matrix = np.zeros((num_nodes, num_edge_types))
        offset = 0
        node_type_to_offset = {}

        for ntype in graph.ntypes:
            node_type_to_offset[ntype] = offset
            offset += graph.num_nodes(ntype)

        for idx, etype in enumerate(graph.canonical_etypes):
            src, dst = graph.edges(etype=etype)
            src_offset = node_type_to_offset[etype[0]]
            dst_offset = node_type_to_offset[etype[2]]
            inci_matrix[src + src_offset, idx] = 1
            inci_matrix[dst + dst_offset, idx] = 1

        inci_sparse = sparse.coo_matrix(inci_matrix)
        incidence_edge = utils.from_scipy_sparse_matrix(inci_sparse)
        hyp_input = incidence_edge[0].to(device)
        hyperedge_attr = torch.randn((num_edge_types, self.node_emb_dim)).to(
            device
        )

        x = self.sthgcn_block1(
            temporal_info["emb"], hyp_input, hyperedge_attr=hyperedge_attr
        )
        x = self.sthgcn_block2(x, hyp_input, hyperedge_attr=hyperedge_attr)
        x = self.time_block(x)
        return x
