import torch
from torch import nn
import torch.nn.functional as F
from .gcn import GCNEncoder

class EdgeReconstructionGCN(nn.Module):
    def __init__(self, in_features, hidden=64, embedding=32):
        super().__init__()
        self.encoder = GCNEncoder(in_features, hidden, embedding)
    def forward(self, x, norm_adj):
        z = self.encoder(x, norm_adj)
        return z, torch.sigmoid(z @ z.T)

def edge_scores(adj, reconstructed_adj, candidate_edges):
    u, v = candidate_edges[:, 0], candidate_edges[:, 1]
    return (adj[u, v] - reconstructed_adj[u, v]).abs()
