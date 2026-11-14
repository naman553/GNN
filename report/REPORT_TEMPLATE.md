# Graph Anomaly Detection Using Graph Neural Networks
## Implementation Report

**Student:** [Name]  
**Course:** Data Mining  
**Reference survey:** H. Kim et al., IEEE Access, 2022.

> Replace every bracketed placeholder with your own details and actual experiment outputs. Do not report unexecuted or invented results.

## Abstract
This project implements representative graph neural network approaches for detecting node, edge, subgraph, graph-level and temporal anomalies. The primary node-level model is a DOMINANT-style dual-reconstruction architecture. Additional modules provide controlled comparative demonstrations inspired by the survey taxonomy.

## 1. Introduction
Explain graphs, nodes, edges, graph attributes and why anomaly detection on relational data differs from ordinary tabular anomaly detection.

## 2. Research paper overview
The uploaded paper is a survey. Summarize its organization:
- Static versus dynamic graphs.
- Node, edge, subgraph and graph-level anomalies.
- GCN, GAT, GAE and temporal GNN architectures.
- Challenges: explainability, anomaly identification, class imbalance, heterogeneous graphs and few-shot learning.

## 3. Objectives
1. Implement node anomaly detection using a DOMINANT-style model.
2. Demonstrate edge anomaly scoring through link reconstruction.
3. Demonstrate community/subgraph deviation scoring.
4. Demonstrate graph-level one-class scoring.
5. Demonstrate temporal anomaly scoring with GCN + GRU.
6. Evaluate on controlled datasets with known injected labels.

## 4. Methodology
### 4.1 Graph representation
Define adjacency matrix A, node feature matrix X, and node/edge sets.

### 4.2 Node anomaly detection
Describe GCN encoder, attribute decoder, structure decoder, reconstruction losses and combined anomaly score. State the alpha weighting used.

### 4.3 Edge anomaly detection
Describe link reconstruction and edge surprise scoring.

### 4.4 Subgraph anomaly detection
Describe candidate communities and feature/density deviation score. Clearly state this is a baseline.

### 4.5 Graph-level anomaly detection
Describe graph embeddings and distance from a normal-reference center.

### 4.6 Dynamic anomaly detection
Describe spatial GCN encoding, GRU temporal modeling and prediction residual.

## 5. Experimental setup
- Hardware: [GPU model / CPU]
- OS: [Windows version]
- Python: [version]
- PyTorch: [version]
- Dataset: generated attributed graphs / generated temporal snapshots / generated graph collection
- Random seed: [seed]
- Epochs: [actual epochs]
- Optimizer and learning rate: [actual settings]

## 6. Results
Insert actual metrics from `results/metrics.csv`. Include plots from `results/plots/`.
Discuss ROC-AUC, Average Precision and limitations. For community-level results, note that a tiny number of communities makes ranking metrics unstable.

## 7. Discussion and limitations
- Synthetic anomalies are controlled and may not represent real-world anomalies.
- Edge absence is not automatically proof of an anomalous relationship.
- Community score is a simple baseline, not a reproduction of AnomMAN/HO-GAT.
- Graph-level one-class reference uses known normal graphs in this demonstration.
- Dense adjacency reconstruction is suitable for small graphs, not very large sparse networks.
- The temporal module is a GCN + GRU baseline inspired by dynamic methods, not exact AddGraph reproduction.

## 8. Conclusion
Summarize what was implemented and what the experiments show, based only on actual results.

## References
1. H. Kim, B. S. Lee, W.-Y. Shin, and S. Lim, "Graph Anomaly Detection With Graph Neural Networks: Current Status and Challenges," IEEE Access, vol. 10, 2022, doi: 10.1109/ACCESS.2022.3211306.
2. K. Ding, J. Li, R. Bhanushali, and H. Liu, "Deep Anomaly Detection on Attributed Networks," SIAM International Conference on Data Mining, 2019.
3. T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR, 2017.
