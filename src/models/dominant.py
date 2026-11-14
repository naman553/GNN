
import torch
from torch import nn

from .gcn import GCNEncoder


class DOMINANT(nn.Module):
    """
    DOMINANT-style architecture:

    - Shared GCN encoder
    - Structure reconstruction decoder
    - Attribute reconstruction decoder
    """

    def __init__(
        self,
        in_features,
        hidden=64,
        embedding=32,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = GCNEncoder(
            in_features,
            hidden,
            embedding,
            dropout
        )

        self.structure_decoder = nn.Linear(
            embedding,
            embedding,
            bias=False
        )

        self.attribute_decoder = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_features)
        )

    def structure_logits(self, z):
        """
        Return raw structure reconstruction logits.
        """

        z_structure = self.structure_decoder(z)

        logits = z_structure @ z_structure.T

        return logits

    def forward(self, x, norm_adj):

        z = self.encoder(x, norm_adj)

        structure_logits = self.structure_logits(z)

        reconstructed_adj = torch.sigmoid(
            structure_logits
        )

        reconstructed_x = self.attribute_decoder(z)

        return reconstructed_x, reconstructed_adj, z


def dominant_scores(
    x,
    adj,
    x_hat,
    adj_hat,
    alpha=0.5
):
    """
    Calculate node anomaly scores.

    Self-connections are excluded from structural scoring.
    """

    if adj.size(0) <= 1:
        raise ValueError(
            "Graph must contain at least two nodes."
        )

    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    # Attribute reconstruction error per node
    attr_error = (
        (x - x_hat) ** 2
    ).mean(dim=1)

    # Structural reconstruction error
    error = (adj - adj_hat) ** 2

    n = adj.size(0)

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=adj.device
    )

    struct_error = (
        error.masked_fill(~mask, 0).sum(dim=1)
        / mask.sum(dim=1).clamp_min(1)
    )

    # Combined anomaly score
    scores = (
        alpha * struct_error
        + (1 - alpha) * attr_error
    )

    return scores, attr_error, struct_error
