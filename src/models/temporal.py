import torch
from torch import nn
from .gcn import GCNEncoder, normalize_adjacency

class GCNGRU(nn.Module):
    """Spatial GCN per snapshot followed by a GRU for each node over time."""
    def __init__(self, in_features, hidden=32):
        super().__init__()
        self.gcn = GCNEncoder(in_features, hidden=64, embedding=hidden)
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, in_features)

    def forward(self, feature_sequence, adjacency_sequence):
        # Inputs: [T, N, F] and [T, N, N].
        spatial=[]
        for t in range(feature_sequence.size(0)):
            norm=normalize_adjacency(adjacency_sequence[t])
            spatial.append(self.gcn(feature_sequence[t],norm))
        # [N,T,H] -> GRU runs independently over each node's temporal sequence.
        sequence=torch.stack(spatial,dim=1)
        temporal,_=self.gru(sequence)
        # Return [T,N,F] so each timestep aligns with the next graph snapshot.
        return self.decoder(temporal).transpose(0,1)
