
import torch
from torch import nn
import torch.nn.functional as F


def normalize_directed_adjacency(adj):
    """
    Normalize incoming and outgoing adjacency separately.

    Convention:
        adj[i, j] = 1 means edge i -> j.

    Incoming aggregation:
        A.T @ X

    Outgoing aggregation:
        A @ X

    Each direction is row-normalized independently.
    """

    if adj.dim() != 2 or adj.size(0) != adj.size(1):
        raise ValueError("Adjacency must be a square matrix.")

    # Remove self-loops from the directional neighborhoods.
    n = adj.size(0)
    adj = adj.clone()
    adj.fill_diagonal_(0)

    # Incoming neighbors of node j are represented
    # by column j in the original adjacency matrix.
    incoming_adj = adj.T

    # Outgoing neighbors of node i are represented
    # by row i in the original adjacency matrix.
    outgoing_adj = adj

    # Row-normalize incoming adjacency.
    incoming_degree = incoming_adj.sum(dim=1).clamp(min=1e-8)
    incoming_norm = incoming_adj / incoming_degree[:, None]

    # Row-normalize outgoing adjacency.
    outgoing_degree = outgoing_adj.sum(dim=1).clamp(min=1e-8)
    outgoing_norm = outgoing_adj / outgoing_degree[:, None]

    return incoming_norm, outgoing_norm


class DirectedGraphConvolution(nn.Module):
    """
    Directed graph convolution with separate transformations
    for incoming, outgoing and self-node information.
    """

    def __init__(self, in_features, out_features):
        super().__init__()

        self.self_linear = nn.Linear(
            in_features, out_features
        )

        self.incoming_linear = nn.Linear(
            in_features, out_features, bias=False
        )

        self.outgoing_linear = nn.Linear(
            in_features, out_features, bias=False
        )

    def forward(self, x, incoming_adj, outgoing_adj):

        # Preserve the node's own information.
        self_part = self.self_linear(x)

        # Aggregate incoming-neighbor information.
        incoming_features = incoming_adj @ x
        incoming_part = self.incoming_linear(incoming_features)

        # Aggregate outgoing-neighbor information.
        outgoing_features = outgoing_adj @ x
        outgoing_part = self.outgoing_linear(outgoing_features)

        return self_part + incoming_part + outgoing_part


class DirectedGCNEncoder(nn.Module):
    """
    Two-layer directed GCN encoder.

    Input:
        x: Node features [N, F]
        incoming_adj: Normalized incoming adjacency [N, N]
        outgoing_adj: Normalized outgoing adjacency [N, N]

    Output:
        Node embeddings [N, embedding]
    """

    def __init__(
        self,
        in_features,
        hidden=64,
        embedding=32,
        dropout=0.2
    ):
        super().__init__()

        self.gc1 = DirectedGraphConvolution(
            in_features, hidden
        )

        self.gc2 = DirectedGraphConvolution(
            hidden, embedding
        )

        self.dropout = dropout

    def forward(self, x, incoming_adj, outgoing_adj):

        h = self.gc1(
            x,
            incoming_adj,
            outgoing_adj
        )

        h = F.relu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training
        )

        z = self.gc2(
            h,
            incoming_adj,
            outgoing_adj
        )

        return z
