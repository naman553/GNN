# Graph Anomaly Detection Using Graph Neural Networks

A reproducible educational implementation inspired by the survey:
**H. Kim et al., "Graph Anomaly Detection With Graph Neural Networks: Current Status and Challenges," IEEE Access, 2022.**

This repository is a *comparative implementation project*, not a reproduction of every paper surveyed. It includes:
1. Node anomalies: DOMINANT-style dual reconstruction model (GCN encoder + structure/attribute decoders).
2. Edge anomalies: GCN link reconstruction baseline.
3. Subgraph anomalies: community-level structural/feature deviation baseline.
4. Graph-level anomalies: GIN-style graph embeddings + one-class distance scoring.
5. Dynamic anomalies: GCN + GRU temporal reconstruction baseline.

The DOMINANT implementation follows the architecture described in the survey and the original DOMINANT formulation at a practical educational scale. Check the original paper and repository before claiming bit-for-bit reproduction.

## Requirements
- Windows 10/11
- Python 3.10 or 3.11 recommended
- NVIDIA driver compatible with the CUDA-enabled PyTorch wheel you install
- VS Code + Python extension

## Setup (PowerShell, VS Code terminal)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch with CUDA support using the official selector:
https://pytorch.org/get-started/locally/

Then install the project dependencies:

```powershell
pip install -r requirements.txt
```

Check GPU:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is unavailable, the project automatically falls back to CPU.

## Run all experiments

```powershell
python run_all.py --epochs 80
```

Fast smoke test:

```powershell
python run_all.py --epochs 8 --nodes 180
```

Outputs are written to `results/`:
- `metrics.csv`
- model checkpoints in `results/checkpoints/`
- plots in `results/plots/`
- per-node / per-edge scores in `results/predictions/`

## Run individual modules

```powershell
python -m src.experiments.node_dominant --epochs 80
python -m src.experiments.edge_detection --epochs 80
python -m src.experiments.subgraph_detection
python -m src.experiments.graph_level
python -m src.experiments.dynamic_detection --epochs 80
```

## Dataset notes
The default experiments use generated attributed graphs with known injected anomalies. This is deliberate: it gives us ground-truth labels and makes evaluation reproducible. A real-world Cora loader is provided as an optional extension point; Cora itself does not supply graph-anomaly labels, so anomaly labels must be defined/injected separately.

## Important methodology notes
- The node DOMINANT-style model reconstructs node attributes and adjacency rows.
- The edge baseline scores candidate edges using link reconstruction error.
- The subgraph baseline scores communities using feature and density deviations.
- The graph-level baseline compares graph embeddings to a normal-reference center.
- The dynamic baseline uses GCN spatial encoding and GRU temporal modeling.
- The edge/subgraph/graph/dynamic modules are representative baselines inspired by the survey taxonomy, not exact reproductions of all named methods in the survey.
- Synthetic anomaly injection is useful for controlled experiments but is not equivalent to validation on naturally occurring anomalies.

## Report
See `report/REPORT_TEMPLATE.md`. Replace placeholders with actual outputs from your run. Do not invent results.
