import typing as tp

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from dgl import DGLGraph
from dgl.nn.pytorch import TypedLinear
from dgl.udf import EdgeBatch, NodeBatch
from torch_scatter.composite import scatter_softmax

from .....data.graph import StockGraph


class HATS(nn.Module):
    """Implementation of the model proposed in the paper:

    `HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction`.
    URL: http://arxiv.org/abs/1908.07999
    GPU memory usage: 5131MiB

    """

    model_name = "HATS"

    def __init__(
        self,
        node_emb_dim: int,
        rel_emb_dim: int,
        stock_kg: StockGraph,
        num_bases: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()
        self.stock_kg = stock_kg
        self.node_emb_dim = node_emb_dim
        self.rel_emb_dim = rel_emb_dim

        hyperedge_rels = 0
        for k, v in stock_kg.hyperedge_incimatrix_dict.items():
            hyperedge_rels += v.shape[-1]

        self.num_rels = stock_kg.num_edge_types + hyperedge_rels
        self.edge_embeddings = nn.Parameter(
            torch.Tensor(self.num_rels, rel_emb_dim)
        )
        self.W_s = TypedLinear(
            in_size=node_emb_dim * 2 + rel_emb_dim,
            out_size=1,
            num_types=self.num_rels,
            regularizer="basis",
            num_bases=num_bases,
        )
        self.b_s = nn.Parameter(torch.Tensor(self.num_rels, 1))
        self.W_r = TypedLinear(
            in_size=node_emb_dim * 2 + rel_emb_dim,
            out_size=1,
            num_types=self.num_rels,
            regularizer="basis",
            num_bases=num_bases,
        )
        self.b_r = nn.Parameter(torch.Tensor(self.num_rels, 1))

        nn.init.xavier_normal_(self.edge_embeddings)
        nn.init.xavier_normal_(self.b_s)
        nn.init.xavier_normal_(self.b_r)

    def homo_message(self, edges: EdgeBatch):
        etype_index = edges.data["_TYPE"]
        edge_embedding = self.edge_embeddings[etype_index]
        src_emb = edges.src["node_emb"]
        dst_emb = edges.dst["node_emb"]  # (num_edges, node_emb_dim)
        emb_cat = torch.cat([src_emb, dst_emb, edge_embedding], dim=-1)
        etype_index = etype_index.unsqueeze(1)
        return {
            "etype_index": etype_index,
            "emb_cat": emb_cat,
            "dst_msg": dst_emb,
        }

    def homo_reduce(self, nodes: NodeBatch):
        # Collect node mailbox
        emb_cat = nodes.mailbox[
            "emb_cat"
        ]  # (num_nodes, num_edges, node_emb_dim * 2 + rel_emb_dim)
        etype_index = nodes.mailbox["etype_index"]  # (num_nodes, num_edges, 1)
        dst_msg = nodes.mailbox[
            "dst_msg"
        ]  # (num_nodes, num_edges, node_emb_dim)
        num_nodes, num_edges = emb_cat.shape[:2]

        # Equation (3.1)
        aggr_score = self.W_s(
            emb_cat.flatten(0, 1), etype_index.flatten()
        ).view(num_nodes, num_edges, 1) + self.b_s[etype_index.flatten()].view(
            num_nodes, num_edges, 1
        )  # (num_nodes, num_edges, 1)

        # Equation (3.2)
        aggr_score = scatter_softmax(
            aggr_score, etype_index, dim=1
        )  # (num_nodes, num_edges, 1)

        # Equation (3.3)
        aggr_msg = torch_scatter.scatter_add(
            aggr_score * dst_msg,
            index=etype_index,
            dim=1,
            dim_size=self.num_rels,
        )  # (num_nodes, num_rels, node_emb_dim)

        # Make rel_msg_cat <- \hat{x}^{rm}_i
        edge_count = torch_scatter.scatter_add(
            torch.ones_like(aggr_score),
            index=etype_index,
            dim=1,
            dim_size=self.num_rels,
        )  # (num_nodes, num_rels, 1)
        node_emb_expanded = (
            nodes.data["node_emb"].unsqueeze(1).expand(-1, self.num_rels, -1)
        )  # (num_nodes, num_rels, node_emb_dim)
        rel_emb_expanded = self.edge_embeddings.unsqueeze(0).expand(
            num_nodes, -1, -1
        )  # (num_nodes, num_rels, rel_emb_dim)
        rel_msg_cat = torch.cat(
            [node_emb_expanded, aggr_msg, rel_emb_expanded], dim=-1
        )  # (num_nodes, num_rels, node_emb_dim * 2 + rel_emb_dim)

        # Equation (3.4)
        r_weight = (
            self.W_r.get_weight()
        )  # (num_rels, node_emb_dim * 2 + rel_emb_dim, 1)
        rel_msg_score = torch.matmul(
            rel_msg_cat.unsqueeze(-2), r_weight
        ).squeeze(-1) + self.b_r.unsqueeze(
            0
        )  # (num_nodes, num_rels, 1)

        # Equation (3.5)
        rel_msg_score = F.softmax(
            torch.where(edge_count > 0, rel_msg_score, -1e10), dim=1
        )  # (num_nodes, num_rels, 1)

        # Equation (3.6)
        new_node_emb = torch.sum(
            rel_msg_score * aggr_msg, dim=1
        )  # (num_nodes, node_emb_dim)

        # Equation (3.7)
        return {"node_emb": new_node_emb + nodes.data["node_emb"]}

    def forward(
        self, graph: DGLGraph, temporal_info: tp.Dict, **kwargs
    ) -> torch.Tensor:
        graph.ndata["node_emb"] = temporal_info["emb"]
        homo_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])
        homo_graph.update_all(
            message_func=self.homo_message, reduce_func=self.homo_reduce
        )
        updated_h = homo_graph.ndata["node_emb"]
        return updated_h
