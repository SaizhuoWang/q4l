import typing as tp

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from .base import StockGraph


class Attention(torch.nn.Module):
    def __init__(self, dimensions, dimensions2):
        super(Attention, self).__init__()
        self.dimensions2 = dimensions2

        self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)
        self.linear_out = torch.nn.Linear(
            dimensions * 2, dimensions2, bias=False
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        self.ae = None
        self.ab = None

    def forward(self, query, context):
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)
        if self.ae is None or self.ae.shape[0] != batch_size:
            self.ae = torch.nn.Parameter(
                torch.FloatTensor(batch_size, 1, 1)
            ).to(query.device)
            self.ab = torch.nn.Parameter(
                torch.FloatTensor(batch_size, 1, 1)
            ).to(query.device)

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
            .to("cuda")
        )
        delta_t = delta_t.repeat(batch_size, 1).reshape(
            batch_size, 1, query_len
        )
        bt = torch.exp(-1 * self.ab * delta_t)
        term_2 = F.relu(self.ae * mix * bt)
        mix = torch.sum(term_2 + mix, -1).unsqueeze(1)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        output = self.linear_out(combined).view(
            batch_size, output_len, self.dimensions2
        )
        output = self.tanh(output)

        return output, attention_weights


class HATR_I(nn.Module):
    def __init__(self, node_emb_dim: int, stock_kg: StockGraph, K=2, **kwargs):
        super().__init__()

        self.K = K
        self.node_emb_dim = node_emb_dim
        self.stock_kg = stock_kg
        self.conv1_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(node_emb_dim, node_emb_dim))
                for _ in range(K)
            ]
        )
        self.conv2_weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(node_emb_dim, node_emb_dim))
                for _ in range(K)
            ]
        )

        # Time-wise attention
        self.attention_model = Attention(
            dimensions=node_emb_dim, dimensions2=node_emb_dim
        )

        # Scale-wise attention
        self.embedding_table = (
            None  # Assuming `num_stocks` is the total number of unique stocks
        )
        self.tanh = nn.Tanh()

    def diffusion_convolution(self, A_tilde, X, weights):
        P_f = A_tilde / A_tilde.sum(dim=1, keepdim=True)
        P_v = A_tilde.T / A_tilde.sum(dim=0, keepdim=True)

        P_psi_f = P_f.clone()
        P_psi_v = P_v.clone()

        Z_tilde = torch.matmul(P_psi_f, X) @ weights[0]
        for k in range(1, self.K):
            if k < self.K // 2:
                P_psi_f = P_psi_f @ P_f
            else:
                P_psi_v = P_psi_v @ P_v
            Z_tilde += (
                torch.matmul(P_psi_f if k < self.K // 2 else P_psi_v, X)
                @ weights[k]
            )

        return Z_tilde

    def fusion(self, h, h_tilde, theta, W_psi, b_psi):
        # Calculate theta * h and (1 - theta) * h_tilde
        h_weighted = theta * h
        h_tilde_weighted = (1 - theta) * h_tilde

        # Concatenate the weighted representations
        concatenated = torch.cat([h_weighted, h_tilde_weighted], dim=-1)

        # Linear transformation
        output = torch.matmul(concatenated, W_psi) + b_psi

        return output

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:

        h_list = []  # for {h}_{p}^{(l)}
        h_prime_list = temporal_info["emb"]  # for {h}_{p}^{\prime(l)}
        device = temporal_info["emb"][0].device
        # Assuming you have a trained theta value
        theta = torch.tensor([0.5]).to(temporal_info["emb"][0].device)

        # Also assuming W_psi and b_psi are defined and trained
        W_psi = torch.nn.Parameter(
            torch.randn(
                temporal_info["emb"][0].size(-1) * 2,
                temporal_info["emb"][0].size(-1),
            )
        ).to(device)
        b_psi = torch.nn.Parameter(
            torch.randn(temporal_info["emb"][0].size(-1))
        ).to(device)

        for emb in h_prime_list:
            emb = emb.permute(1, 0, 2)
            graph.ndata[
                "node_emb"
            ] = emb  ## ([seq_len, batch_size, input_size]) * n
            homogeneous_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])
            device = homogeneous_graph.device
            A = homogeneous_graph.adj_external(ctx=device).to_dense()

            A_tilde = A + torch.eye(A.size(0), device=device)
            D_tilde_inv_sqrt = torch.diag(torch.pow(A_tilde.sum(dim=1), -0.5))
            A_tilde = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt

            # Introducing the sequence loop
            seq_len = emb.shape[1]
            h_ = []
            for t in range(seq_len):
                h_t = emb[:, t, :]  ## batch_size, input_size
                h_t = self.diffusion_convolution(
                    A_tilde, h_t, self.conv1_weights
                )
                h_t = F.relu(h_t)
                h_t = self.diffusion_convolution(
                    A_tilde, h_t, self.conv2_weights
                )
                h_.append(h_t.unsqueeze(1))

            h = torch.cat(h_, dim=1)
            h_list.append(h)

        # Fusion operation
        output_list = []
        for h, h_tilde in zip(h_list, h_prime_list):
            h = h.permute(1, 0, 2)
            output_list.append(self.fusion(h, h_tilde, theta, W_psi, b_psi))

        # Hawkes Attention
        outputs = []
        for feature in output_list:
            context = feature.permute(1, 0, 2)
            query = context[:, -1, :].unsqueeze(1)  # query: last time step
            output, attention_weights = self.attention_model(query, context)
            output = output[:, -1, :]
            # print(output.shape)
            outputs.append(output)

        # Scale-wise attention
        batch_size = outputs[0].shape[0]
        if (
            self.embedding_table is None
            or self.embedding_table.num_embeddings != batch_size
        ):
            self.embedding_table = nn.Embedding(
                batch_size, self.node_emb_dim
            ).to(device)
            self.embedding_table.weight.data.uniform_(-0.1, 0.1)

        es_list = []
        for idx in range(batch_size):
            es = self.embedding_table(
                torch.tensor([idx], device=device)
            ).squeeze(
                0
            )  # (256,)
            es_list.append(es)
        es_tensor = torch.stack(es_list).squeeze(1)
        qs = F.relu(es_tensor)  # Function 5

        beta_list = []
        h_prime_list = []
        for l in range(len(outputs)):
            hl = outputs[l]
            h_l_prime = F.tanh(
                hl
            )  # Using the last time step from each feature in H
            h_prime_list.append(h_l_prime)
            beta_l = torch.exp(qs.T @ h_l_prime)  # Function 6
            beta_list.append(beta_l)

        beta_denom = sum(beta_list)
        beta_list = [b / beta_denom for b in beta_list]

        Z = torch.zeros_like(outputs[0])
        for o, b in zip(outputs, beta_list):
            Z += o @ b

        # Soft Clustering Regularization
        C = 10  # Intend 10 clusters
        W_p = torch.nn.Parameter(torch.randn(self.node_emb_dim, C)).to(device)

        P = F.softmax(Z @ W_p, dim=1)
        C_matrix = P.T @ Z

        eta = torch.zeros(C, C).to(device)
        for i in range(C):
            for j in range(C):
                eta_ij = torch.sum(torch.mm(P[:, i].unsqueeze(0), A) * P[:, j])
                eta[i, j] = eta_ij

        W_c = torch.nn.Parameter(
            torch.randn(self.node_emb_dim, self.node_emb_dim)
        ).to(device)
        C_prime = torch.relu(eta @ (W_c @ C_matrix.T).T)

        Z_c = P @ C_prime

        return Z_c
