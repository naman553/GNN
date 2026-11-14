import torch
from torch import nn
import torch.nn.functional as F

def normalize_adjacency(adj):
    """Symmetric GCN normalization of A + I: D^(-1/2)(A+I)D^(-1/2)."""
    n = adj.size(0)
    a_hat = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    degree = a_hat.sum(dim=1).clamp(min=1e-8)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * a_hat * inv_sqrt[None, :]

class DenseGraphConvolution(nn.Module):
    """Small-graph GCN layer, implemented with dense matrix multiplication."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
    def forward(self, x, norm_adj):
        return norm_adj @ self.linear(x)

class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden=64, embedding=32, dropout=.2):
        super().__init__()
        self.gc1 = DenseGraphConvolution(in_features, hidden)
        self.gc2 = DenseGraphConvolution(hidden, embedding)
        self.dropout = dropout
    def forward(self, x, norm_adj):
        h = F.relu(self.gc1(x, norm_adj))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.gc2(h, norm_adj)
