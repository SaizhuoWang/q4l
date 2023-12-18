import typing as tp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from scipy import sparse
from torch_geometric import nn as nnn
from torch_geometric import utils


class STHAN(nn.Module):
    def __init__(self, node_emb_dim: int, **kwargs):
        super().__init__()
        self.node_emb_dim = node_emb_dim

        self.hatt1 = nnn.HypergraphConv(
            node_emb_dim,
            node_emb_dim,
            use_attention=True,
            heads=4,
            concat=False,
            negative_slope=0.2,
            dropout=0.5,
            bias=True,
        )
        self.hatt2 = nnn.HypergraphConv(
            node_emb_dim,
            node_emb_dim,
            use_attention=True,
            heads=1,
            concat=False,
            negative_slope=0.2,
            dropout=0.5,
            bias=True,
        )
        self.linear = torch.nn.Linear(node_emb_dim, 1)

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        graph.ndata["node_emb"] = temporal_info["emb"]
        device = graph.device
        graph = graph.to("cpu")
        num_nodes = sum([graph.num_nodes(ntype) for ntype in graph.ntypes])
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

        h = F.leaky_relu(
            self.hatt1(
                temporal_info["emb"], hyp_input, hyperedge_attr=hyperedge_attr
            ),
            0.2,
        )
        updated_h = F.leaky_relu(
            self.hatt2(h, hyp_input, hyperedge_attr=hyperedge_attr), 0.2
        )
        return updated_h
