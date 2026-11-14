import numpy as np
import pandas as pd
import networkx as nx
from src.utils.common import seed_everything,ensure_dirs,binary_metrics,save_score_plot

def run(num_communities=120,seed=44):
    seed_everything(seed); ensure_dirs(); rng=np.random.default_rng(seed)
    sizes=rng.integers(18,30,size=num_communities)
    feature_dim=10; graphs=[]; labels=[]
    for idx,size in enumerate(sizes):
        anomalous=int(idx>=int(.8*num_communities))
        p=.18 if not anomalous else .045
        g=nx.erdos_renyi_graph(int(size),p,seed=int(rng.integers(1_000_000)))
        x=rng.normal(0,1,(size,feature_dim))
        if anomalous:
            # Abnormal subgraphs have a shifted feature distribution and sparse internal structure.
            x += 2.0
        density=nx.density(g)
        # Score depends only on observable graph features, never on anomaly labels.
        score=float(np.linalg.norm(x.mean(axis=0))/np.sqrt(feature_dim) + abs(density-.18))
        graphs.append((idx,int(size),density,score)); labels.append(anomalous)
    y=np.asarray(labels,dtype=int); scores=np.asarray([r[3] for r in graphs])
    metrics=binary_metrics(y,scores)
    pd.DataFrame({"subgraph_id":[r[0] for r in graphs],"size":[r[1] for r in graphs],
                  "density":[r[2] for r in graphs],"label":y,"score":scores}).to_csv("results_v2/predictions/subgraph_scores.csv",index=False)
    save_score_plot(y,scores,"subgraph_scores.png","Subgraph anomaly scores")
    print("Subgraph metrics:",metrics)
    return {"task":"subgraph_feature_structure_baseline","seed":seed,**metrics}

if __name__=="__main__": print(run())
