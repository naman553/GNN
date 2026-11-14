# Version 2 – Experimental corrections

This is an update to the initial project. It writes all new outputs to `results_v2/`; the original `results/` baseline is left untouched.

## What changed
- **Node / DOMINANT-style:** preserves dual attribute + structure reconstruction and reports attribute-only / structure-only ROC-AUC ablations.
- **Edge:** trains on a normal graph with held-out normal links, then evaluates those against explicitly injected cross-community edges not seen during training.
- **Subgraph:** uses many independent graph samples. The score uses only observable feature and density deviations; labels are used only for evaluation.
- **Graph-level:** trains a GIN-inspired encoder + graph-summary reconstruction decoder on normal training graphs, then scores reconstruction residual on the graph collection.
- **Dynamic:** trains GCN+GRU to predict next-step node features using only pre-event snapshots; evaluates on a held-out final event snapshot.

## Run
Use the same activated `.venv` and installed dependencies:
```powershell
python run_all.py --epochs 80 --nodes 240
```
Smoke test:
```powershell
python run_all.py --epochs 5 --nodes 100
```

## Interpretation caveats
These are controlled synthetic experiments and representative educational baselines. They are not exact reproductions of every method cited in the survey. Synthetic labels are used only for evaluation in the revised subgraph, graph-level, edge, and dynamic experiments; the model fitting/scoring rules do not directly use those labels. The graph-level and dynamic tasks remain simplified benchmark protocols.
