# File: gcn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    A single Graph Convolutional Network (GCN) layer.
    Applies a linear transformation followed by propagation through a normalized adjacency.

    Z = A_norm · (X W) + b

    Args:
        in_features (int):  Number of input feature dimensions per node.
        out_features (int): Number of output feature dimensions per node.
        bias (bool):        Whether to include a learnable bias term. Default: True.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GCNLayer, self).__init__()
        # Weight matrix W ∈ ℝ^{in_features × out_features}
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights with Xavier/Glorot uniform and bias to zeros.
        """
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.FloatTensor, A_norm: torch.sparse.FloatTensor) -> torch.FloatTensor:
        """a
        Forward pass of a single GCN layer.

        Args:
            X (torch.FloatTensor):      Node feature matrix, shape (N, in_features).
            A_norm (torch.sparse.FloatTensor): Precomputed normalized adjacency (Â), shape (N, N).

        Returns:
            out (torch.FloatTensor):    Transformed node features, shape (N, out_features).
        """
        # Linear transformation: support = X · W  → shape: (N, out_features)
        support = torch.mm(X, self.weight)

        # Propagate through normalized adjacency: out = Â · support
        # Using sparse-dense multiplication: (N×N) × (N×out_features) → (N×out_features)
        out = torch.spmm(A_norm, support)

        # Add bias if it exists
        if self.bias is not None:
            out = out + self.bias

        return out


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network for node classification.

    Architecture:
      Input → GCNLayer(in_dim → hidden_dim) → ReLU → Dropout → GCNLayer(hidden_dim → out_dim)

    Args:
        in_dim (int):      Number of input features per node (D).
        hidden_dim (int):  Number of hidden units in the first GCN layer.
        out_dim (int):     Number of output classes (C).
        dropout (float):   Dropout probability applied after the first GCN layer. Default: 0.5.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim, bias=True)
        self.conv2 = GCNLayer(hidden_dim, out_dim, bias=True)
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
        h = self.conv1(X, A_norm)      # → shape: (N, hidden_dim)
        h = F.relu(h)

        # Dropout (only applied during training)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Second GCN layer (produces logits)
        logits = self.conv2(h, A_norm) # → shape: (N, out_dim)
        return logits
