# Graph Anomaly Detection Using Graph Neural Networks – Version 2 Report

**Reference survey:** H. Kim et al., "Graph Anomaly Detection With Graph Neural Networks: Current Status and Challenges," IEEE Access, 2022, DOI: 10.1109/ACCESS.2022.3211306.

> Insert only metrics produced by your own run. Do not copy example or smoke-test values as final results.

## 1. Scope and relationship to the survey
The reference is a survey, not a single proposed algorithm paper. It organizes methods by static/dynamic graph type, anomaly type (node, edge, subgraph, whole graph), and architecture. This project implements a DOMINANT-style node detector and representative educational baselines for the other categories.

## 2. Implemented modules
1. **Node:** GCN encoder with dual attribute/structure reconstruction, following the DOMINANT architecture discussed in the survey. Report alpha and reconstruction loss definitions.
2. **Edge:** GCN link-reconstruction baseline. Train using observed normal links and sampled non-links; evaluate held-out normal links against explicitly injected cross-community anomalous links.
3. **Subgraph:** independent subgraph samples; score feature-distribution and density deviations. Labels are evaluation-only.
4. **Whole graph:** GIN-inspired encoder trained through graph-summary reconstruction on normal training graphs; use reconstruction residual as graph anomaly score.
5. **Dynamic:** GCN + GRU predictive baseline trained on pre-event snapshots; score final-snapshot feature prediction error plus degree change.

## 3. Experimental protocol
- Synthetic data provides known controlled anomaly labels.
- Keep train/test protocols and random seeds fixed and document them.
- Report ROC-AUC and Average Precision; include precision/recall/F1 only when a threshold is selected independently of the test labels.
- Repeat with multiple seeds before making strong claims.
- Discuss limitations of synthetic anomalies and small-scale dense adjacency reconstruction.

## 4. Results
Insert `results_v2/metrics.csv` results and the generated plots. Compare the initial smoke test with the full run, but do not treat the smoke test as final research evidence.

## 5. Limitations
The edge, subgraph, graph-level and dynamic implementations are representative simplified baselines, not exact reproductions of all cited methods. The graph-level model uses graph-summary reconstruction rather than a full OCGIN reproduction. The temporal model is a GCN+GRU predictive baseline rather than exact AddGraph/DynAD reproduction. Synthetic anomalies may not capture naturally occurring anomalies.

## 6. Reference
H. Kim, B. S. Lee, W.-Y. Shin, and S. Lim, "Graph Anomaly Detection With Graph Neural Networks: Current Status and Challenges," IEEE Access, vol. 10, 2022, doi: 10.1109/ACCESS.2022.3211306.
