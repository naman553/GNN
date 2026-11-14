import torch
from torch import nn
import torch.nn.functional as F
from .gcn import GCNEncoder

class DOMINANT(nn.Module):
    """
    DOMINANT-style architecture:
    shared GCN encoder, adjacency reconstruction decoder, attribute reconstruction decoder.
    Dense adjacency is intended for small/medium educational graphs.
    """
    def __init__(self, in_features, hidden=64, embedding=32, dropout=.2):
        super().__init__()
        self.encoder = GCNEncoder(in_features, hidden, embedding, dropout)
        self.structure_decoder = nn.Linear(embedding, embedding, bias=False)
        self.attribute_decoder = nn.Sequential(
            nn.Linear(embedding, hidden), nn.ReLU(), nn.Linear(hidden, in_features)
        )

    def forward(self, x, norm_adj):
        z = self.encoder(x, norm_adj)
        z_structure = self.structure_decoder(z)
        reconstructed_adj = torch.sigmoid(z_structure @ z_structure.T)
        reconstructed_x = self.attribute_decoder(z)
        return reconstructed_x, reconstructed_adj, z

def dominant_scores(x, adj, x_hat, adj_hat, alpha=.5):
    attr_error = ((x - x_hat) ** 2).mean(dim=1)
    # Mean over each adjacency row. Ignore self-loop diagonal.
    struct_error = ((adj - adj_hat) ** 2).mean(dim=1)
    scores = alpha * struct_error + (1 - alpha) * attr_error
    return scores, attr_error, struct_error
