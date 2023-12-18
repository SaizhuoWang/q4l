import typing as tp

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import TypedLinear
from dgl.udf import EdgeBatch, NodeBatch

from .base import StockGraph


class RSR(nn.Module):
    def __init__(
        self,
        node_emb_dim: int,
        rel_emb_dim: int,
        stock_kg: StockGraph,
        num_bases: int = 100,
        modeling_mode: str = "implicit",
    ):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.rel_emb_dim = rel_emb_dim
        self.num_bases = num_bases
        self.stock_kg = stock_kg

        self.modeling_mode = modeling_mode
        if self.modeling_mode not in ["implicit", "explicit"]:
            raise ValueError(
                f"modeling_mode should be either implicit or explicit, got {self.modeling_mode}"
            )

        self.input_dim = (
            2 * self.node_emb_dim + self.rel_emb_dim
            if self.modeling_mode == "implicit"
            else self.rel_emb_dim
        )
        self.w = nn.Parameter(torch.Tensor(self.input_dim, 1))
        self.b = nn.Parameter(torch.rand(1))
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        hyperedge_rels = 0
        for k, v in stock_kg.hyperedge_incimatrix_dict.items():
            hyperedge_rels += v.shape[-1]

        self.num_rels = self.stock_kg.num_edge_types + hyperedge_rels
        self.edge_embeddings = nn.Parameter(
            torch.Tensor(self.num_rels, rel_emb_dim)
        )

        nn.init.xavier_normal_(self.w)
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

        # Init all parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def homo_message(self, edges: EdgeBatch):
        etype_index = edges.data["_TYPE"]  # (num_edges,)
        edge_embedding = self.edge_embeddings[
            etype_index
        ]  # (num_edges, rel_emb_dim)
        src_emb = edges.src["node_emb"]  # (num_edges, node_emb_dim)
        dst_emb = edges.dst["node_emb"]  # (num_edges, node_emb_dim)
        degree = edges.dst["degree"]  # (num_edges, 1)
        if self.modeling_mode == "implicit":
            msg_score = torch.cat(
                [src_emb, dst_emb, edge_embedding], dim=-1
            )  # (num_edges, 2 * node_emb_dim + rel_emb_dim)
            msg_score = torch.mm(msg_score, self.w) + self.b  # (num_edges, 1)
        else:
            msg_score = (src_emb * dst_emb).sum(-1, keepdim=True) * (
                torch.mm(edge_embedding, self.w) + self.b
            )  # (num_edges, 1)
        msg_score = self.activation(msg_score)
        return {"score": msg_score, "degree": degree, "e_j": dst_emb}

    def homo_reduce(self, nodes: NodeBatch):
        score = nodes.mailbox["score"]  # (num_nodes, num_neighbors, 1)
        degree = nodes.mailbox["degree"]  # (num_nodes, num_neighbors, 1)
        e_j = nodes.mailbox["e_j"]  # (num_nodes, num_neighbors, node_emb_dim)

        score = F.softmax(score, dim=1)  # (num_nodes, num_neighbors, 1)
        score = score / degree  # (num_nodes, num_neighbors, 1)
        aggr_msg = torch.sum(score * e_j, dim=1)  # (num_nodes, node_emb_dim,)
        return {"node_emb": aggr_msg}

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        # NOTE: We convert to homogeneous graph to speed up computation
        graph.ndata["node_emb"] = temporal_info["emb"]
        homogeneous_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])

        # Compute node out-degree and put it to node features named 'degree'
        degree = homogeneous_graph.out_degrees().float().clamp(min=1)
        homogeneous_graph.ndata["degree"] = degree.unsqueeze(1)
        homogeneous_graph.update_all(
            message_func=self.homo_message, reduce_func=self.homo_reduce
        )

        updated_h = homogeneous_graph.ndata["node_emb"]
        return updated_h
