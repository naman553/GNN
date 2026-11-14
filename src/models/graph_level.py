import torch
from torch import nn
import torch.nn.functional as F
from .gcn import normalize_adjacency

class GINStyleEncoder(nn.Module):
    """GIN-inspired graph encoder with learnable MLP message aggregation."""
    def __init__(self, in_features, hidden=32):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.decoder = nn.Linear(hidden, in_features)
    def forward(self, x, adj):
        a = normalize_adjacency(adj)
        h = self.mlp1(x + a @ x)
        h = self.mlp2(h + a @ h)
        return h.mean(dim=0), self.decoder(h)
