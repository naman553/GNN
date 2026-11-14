import argparse
from pathlib import Path
import pandas as pd
from src.experiments.node_dominant import run as run_node
from src.experiments.edge_detection import run as run_edge
from src.experiments.subgraph_detection import run as run_subgraph
from src.experiments.graph_level import run as run_graph
from src.experiments.dynamic_detection import run as run_dynamic

def main():
    q=argparse.ArgumentParser()
    q.add_argument("--epochs",type=int,default=80)
    q.add_argument("--nodes",type=int,default=240)
    args=q.parse_args()
    Path("results_v2").mkdir(exist_ok=True)
    results=[
        run_node(args.epochs,args.nodes),
        run_edge(args.epochs,args.nodes),
        run_subgraph(),
        run_graph(epochs=max(args.epochs,40)),
        run_dynamic(args.epochs,max(100,args.nodes//2))
    ]
    df=pd.DataFrame(results); df.to_csv("results_v2/metrics.csv",index=False)
    print("\\n=== VERSION 2 EXPERIMENT SUMMARY ===")
    print(df.to_string(index=False))
    print("\\nSaved all Version 2 outputs under results_v2/")

if __name__=="__main__": main()
