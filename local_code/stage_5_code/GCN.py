# File: gcn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def normalize_adjacency(adj_matrix):
    """
    Compute the renormalization trick: D̃^(-1/2) Ã D̃^(-1/2)

    This implements the renormalization from Kipf & Welling (2017):
    - Ã = A + I_N (add self-connections)
    - D̃_ii = Σ_j Ã_ij (degree matrix of augmented adjacency)
    - Return: D̃^(-1/2) Ã D̃^(-1/2)

    Args:
        adj_matrix: Adjacency matrix as scipy sparse matrix or numpy array

    Returns:
        torch.sparse.FloatTensor: Normalized adjacency matrix D̃^(-1/2) Ã D̃^(-1/2)
    """
    # Convert to scipy sparse if needed
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = sp.csr_matrix(adj_matrix)
    elif torch.is_tensor(adj_matrix):
        adj_matrix = sp.csr_matrix(adj_matrix.cpu().numpy())

    # Add self-connections: Ã = A + I
    num_nodes = adj_matrix.shape[0]
    adj_tilde = adj_matrix + sp.eye(num_nodes)

    # Compute degree matrix D̃
    degree_vec = np.array(adj_tilde.sum(axis=1)).flatten()

    # Compute D̃^(-1/2)
    degree_inv_sqrt = np.power(degree_vec, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    degree_inv_sqrt_mat = sp.diags(degree_inv_sqrt)

    # Compute normalized adjacency: D̃^(-1/2) Ã D̃^(-1/2)
    normalized_adj = degree_inv_sqrt_mat @ adj_tilde @ degree_inv_sqrt_mat

    # Convert to PyTorch sparse tensor
    normalized_adj_coo = normalized_adj.tocoo()
    indices = torch.from_numpy(np.vstack((normalized_adj_coo.row, normalized_adj_coo.col))).long()
    values = torch.from_numpy(normalized_adj_coo.data).float()
    shape = normalized_adj_coo.shape

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.

    Architecture:
      Input → GraphConvolution(in_dim → hidden_dim) → ReLU → Dropout → GraphConvolution(hidden_dim → out_dim)

    Args:
        in_dim (int):      Number of input features per node (D).
        hidden_dim (int):  Number of hidden units in the first GCN layer.
        out_dim (int):     Number of output classes (C).
        dropout (float):   Dropout probability applied after the first GCN layer. Default: 0.5.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConvolution(in_dim, hidden_dim, bias=True)
        self.conv2 = GraphConvolution(hidden_dim, out_dim, bias=True)
        self.dropout = dropout

    def forward(self, X: torch.FloatTensor, A_norm: torch.sparse.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass of the two-layer GCN.

        Args:
            X (torch.FloatTensor):        Node feature matrix, shape (N, in_dim).
            A_norm (torch.sparse.FloatTensor): Precomputed normalized adjacency (Â), shape (N, N).

        Returns:
            logits (torch.FloatTensor):   Raw class scores for each node, shape (N, out_dim).
        """
        # First GCN layer + ReLU
        h = self.conv1(X, A_norm)  # → shape: (N, hidden_dim)
        h = F.relu(h)

        # Dropout (only applied during training)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Second GCN layer (produces logits)
        logits = self.conv2(h, A_norm)  # → shape: (N, out_dim)
        return logits
