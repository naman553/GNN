
import torch
from torch import nn

from .directed_gcn import DirectedGCNEncoder


class DirectedDOMINANT(nn.Module):
    """
    Directed DOMINANT-style architecture.

    Components:
        1. Shared directed GCN encoder
        2. Directed structure reconstruction decoder
        3. Attribute reconstruction decoder

    Convention:
        adj[i, j] = 1 represents edge i -> j.
    """

    def __init__(
        self,
        in_features,
        hidden=64,
        embedding=32,
        dropout=0.2
    ):
        super().__init__()

        self.encoder = DirectedGCNEncoder(
            in_features=in_features,
            hidden=hidden,
            embedding=embedding,
            dropout=dropout
        )

        # Asymmetric bilinear structure decoder.
        # Unlike Z @ Z.T, this can reconstruct
        # different scores for i -> j and j -> i.
        self.structure_weight = nn.Parameter(
            torch.empty(embedding, embedding)
        )

        nn.init.xavier_uniform_(self.structure_weight)

        # Attribute reconstruction decoder.
        self.attribute_decoder = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_features)
        )

    def structure_logits(self, z):
        """
        Calculate directed edge reconstruction logits.

        logits[i, j] represents the score for edge i -> j.
        """

        transformed = z @ self.structure_weight

        return transformed @ z.T

    def forward(self, x, incoming_adj, outgoing_adj):

        # Encode node features using directed neighborhoods.
        z = self.encoder(
            x,
            incoming_adj,
            outgoing_adj
        )

        # Reconstruct directed graph.
        structure_logits = self.structure_logits(z)
        reconstructed_adj = torch.sigmoid(structure_logits)

        # Reconstruct node attributes.
        reconstructed_x = self.attribute_decoder(z)

        return reconstructed_x, reconstructed_adj, z


def directed_dominant_scores(
    x,
    adj,
    x_hat,
    adj_hat,
    alpha=0.5
):
    """
    Calculate node anomaly scores.

    Attribute error:
        Mean squared reconstruction error per node.

    Structural error:
        Average of outgoing and incoming reconstruction errors.

    Self-connections are excluded.
    """

    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("Adjacency must be a square matrix.")

    if adj.size(0) <= 1:
        raise ValueError("Graph must contain at least two nodes.")

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    if x.shape != x_hat.shape:
        raise ValueError("Feature and reconstructed feature shapes differ.")

    if adj.shape != adj_hat.shape:
        raise ValueError("Adjacency and reconstructed adjacency shapes differ.")

    # Attribute reconstruction error.
    attr_error = ((x - x_hat) ** 2).mean(dim=1)

    # Directed structural reconstruction error.
    error = (adj - adj_hat) ** 2

    n = adj.size(0)

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=adj.device
    )

    error = error.masked_fill(~mask, 0)

    # Outgoing reconstruction error:
    # Error for edges originating from each node.
    outgoing_error = error.sum(dim=1) / (n - 1)

    # Incoming reconstruction error:
    # Error for edges pointing towards each node.
    incoming_error = error.sum(dim=0) / (n - 1)

    # A node can be anomalous in either direction.
    struct_error = (outgoing_error + incoming_error) / 2

    # Combined anomaly score.
    scores = (
        alpha * struct_error
        + (1 - alpha) * attr_error
    )

    return scores, attr_error, struct_error
