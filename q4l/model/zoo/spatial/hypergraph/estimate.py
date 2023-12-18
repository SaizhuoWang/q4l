"""From "Huynh et al., Efficient Integration of Multi-Order Dynamics and
Internal Dynamics in Stock Movement Prediction", WSDM 2023 Official code:

https://github.com/thanhtrunghuynh93/estimate.

"""


import typing as tp

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

from .....data.dataset import Q4LDataModule


class HWNNLayer(torch.nn.Module):
    """Hypergraph Wavelet Neural Network Layer."""

    def __init__(
        self,
        input_size,
        output_size,
        num_stock,
        K1=2,
        K2=2,
        approx=False,
        upper=True,
        data=None,
    ):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.input_size = input_size
        self.output_size = output_size
        self.num_stock = num_stock
        self.K1 = K1
        self.K2 = K2
        self.approx = approx
        self.upper = upper
        self.s = 1.0
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.input_size, self.output_size)
        )
        self.diagonal_weight_filter = torch.nn.Parameter(
            torch.Tensor(self.num_stock)
        )
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def _convert_incidence_to_adjacency_torch(self, incidence_matrix):
        num_nodes = incidence_matrix.shape[0]
        num_edges = incidence_matrix.shape[1]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_edges):
                if incidence_matrix[i, j]:
                    connected_nodes_indexes = np.where(
                        incidence_matrix[:, j] == 1
                    )
                    adj_matrix[i, connected_nodes_indexes] = 1
        return torch.Tensor(adj_matrix)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, incidence_matrix):
        device = features.device
        indice_matrix = self._convert_incidence_to_adjacency_torch(
            incidence_matrix
        )
        W_e_diag = torch.ones(indice_matrix.size()[1])
        D_e_diag = torch.sum(indice_matrix, 0)
        D_e_diag = D_e_diag.view((D_e_diag.size()[0]))
        D_v_diag = indice_matrix.mm(W_e_diag.view((W_e_diag.size()[0]), 1))
        D_v_diag = D_v_diag.view((D_v_diag.size()[0]))

        Theta = (
            torch.diag(torch.pow(D_v_diag, -0.5))
            @ indice_matrix
            @ torch.diag(W_e_diag)
            @ torch.diag(torch.pow(D_e_diag, -1))
            @ torch.transpose(indice_matrix, 0, 1)
            @ torch.diag(torch.pow(D_v_diag, -0.5))
        )
        Theta_t = torch.transpose(Theta, 0, 1)

        Laplacian = torch.eye(Theta.size()[0]) - Theta
        fourier_e, fourier_v = torch.linalg.eigh(
            Laplacian, UPLO="U" if self.upper else "L"
        )

        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter)

        if self.approx:
            poly = self.par[0] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(1, self.K1):
                Theta_mul = Theta_mul @ Theta
                poly = poly + self.par[ind] * Theta_mul

            poly_t = self.par[self.K1] * torch.eye(self.num_stock)
            Theta_mul = torch.eye(self.num_stock)
            for ind in range(self.K1 + 1, self.K1 + self.K2):
                Theta_mul = Theta_mul @ Theta_t  # 这里也可以使用Theta_transpose
                poly_t = poly_t + self.par[ind] * Theta_mul

            local_fea_1 = (
                poly
                @ diagonal_weight_filter
                @ poly_t
                @ features
                @ self.weight_matrix
            )
        else:
            wavelets = (
                fourier_v
                @ torch.diag(torch.exp(-1.0 * fourier_e * self.s))
                @ torch.transpose(fourier_v, 0, 1)
            )
            wavelets_inverse = wavelets_inv = (
                fourier_v
                @ torch.diag(torch.exp(fourier_e * self.s))
                @ torch.transpose(fourier_v, 0, 1)
            )
            wavelets = wavelets.to(device)
            diagonal_weight_filter = diagonal_weight_filter.to(device)
            wavelets_inverse = wavelets_inverse.to(device)
            self.weight_matrix.data = self.weight_matrix.data.to(device)
            local_fea_1 = (
                wavelets
                @ diagonal_weight_filter
                @ wavelets_inverse
                @ features
                @ self.weight_matrix
            )

        localized_features = local_fea_1
        return localized_features


class ESTIMATE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        data: Q4LDataModule,
        rnn_hidden_unit=8,
        n_head=4,
        drop_prob=0.2,
    ):
        super().__init__()
        self.data = data
        self.num_stocks = len(data.ticker_list)

        self.rnn_hidden_unit = rnn_hidden_unit
        self.hidden_size = hidden_size
        self.par = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.uniform_(self.par, 0, 0.99)
        self.dropout = nn.Dropout(drop_prob)

    def forward(
        self,
        graph: DGLGraph,
        temporal_info: tp.Dict,
    ) -> torch.Tensor:
        enc_output = temporal_info["emb"]
        num_stock = enc_output.shape[1]
        seq_len = enc_output.shape[2]

        ## Hypergraph Inci_matrix
        graph.ndata["node_emb"] = temporal_info["emb"].squeeze(0)[
            :, -1, :
        ]  ## Last time step as feature
        homogeneous_graph = dgl.to_homogeneous(graph, ndata=["node_emb"])
        homogeneous_graph.device
        homogeneous_graph = homogeneous_graph.to("cpu")
        num_nodes = sum(
            [
                homogeneous_graph.num_nodes(ntype)
                for ntype in homogeneous_graph.ntypes
            ]
        )
        num_edge_types = len(homogeneous_graph.canonical_etypes)
        inci_matrix = np.zeros((num_nodes, num_edge_types))
        offset = 0
        node_type_to_offset = {}

        for ntype in homogeneous_graph.ntypes:
            node_type_to_offset[ntype] = offset
            offset += homogeneous_graph.num_nodes(ntype)

        for idx, etype in enumerate(homogeneous_graph.canonical_etypes):
            src, dst = homogeneous_graph.edges(etype=etype)
            src_offset = node_type_to_offset[etype[0]]
            dst_offset = node_type_to_offset[etype[2]]
            inci_matrix[src + src_offset, idx] = 1
            inci_matrix[dst + dst_offset, idx] = 1

        x = enc_output[0].reshape(
            num_stock, seq_len * self.rnn_hidden_unit
        )  # (num_stock, seq_len * rnn_hidden_unit)

        self.convolution_1 = HWNNLayer(
            self.rnn_hidden_unit * seq_len,
            self.rnn_hidden_unit * seq_len,
            num_stock,
            K1=3,
            K2=3,
            approx=False,
            data=inci_matrix,
        )

        self.convolution_2 = HWNNLayer(
            self.rnn_hidden_unit * seq_len,
            self.rnn_hidden_unit * seq_len,
            num_stock,
            K1=3,
            K2=3,
            approx=False,
            data=inci_matrix,
        )

        deep_features_1 = F.leaky_relu(self.convolution_1(x, inci_matrix), 0.1)
        deep_features_1 = self.dropout(deep_features_1)
        deep_features_2 = self.convolution_2(deep_features_1, inci_matrix)
        deep_features_2 = F.leaky_relu(deep_features_2, 0.1)

        deep_features_3 = self.par[0] * deep_features_2

        hyper_output = deep_features_3.reshape(
            1, num_stock, seq_len, self.rnn_hidden_unit
        )

        enc_output = torch.cat((enc_output, hyper_output), dim=3)
        enc_output = enc_output[:, :, -1, :]
        enc_output = enc_output.squeeze(0)

        return enc_output
