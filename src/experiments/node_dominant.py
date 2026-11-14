import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.data.synthetic import make_attributed_graph
from src.models.gcn import normalize_adjacency
from src.models.dominant import DOMINANT, dominant_scores
from src.utils.common import seed_everything,get_device,ensure_dirs,binary_metrics,save_loss_plot,save_score_plot

def run(epochs=80,num_nodes=260,seed=42):
    seed_everything(seed); ensure_dirs(); data=make_attributed_graph(num_nodes=num_nodes,seed=seed)
    device=get_device(); x=data["x"].to(device); adj=data["adj"].to(device); norm=normalize_adjacency(adj)
    model=DOMINANT(x.size(1)).to(device); opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=5e-4)
    losses=[]; alpha=.5
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        xh,ah,_=model(x,norm)
        la=F.mse_loss(xh,x); ls=F.mse_loss(ah,adj)
        loss=(1-alpha)*la+alpha*ls; loss.backward(); opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        xh,ah,_=model(x,norm)
        score,ae,se=dominant_scores(x,adj,xh,ah,alpha)
        # Ablation scores are computed from the same trained model; no labels enter scoring.
        attr_score=ae.cpu().numpy(); struct_score=se.cpu().numpy(); combined=score.cpu().numpy()
    y=data["labels"]
    result={"task":"node_DOMINANT_style","seed":seed,**binary_metrics(y,combined)}
    result["attribute_only_roc_auc"]=binary_metrics(y,attr_score)["roc_auc"]
    result["structure_only_roc_auc"]=binary_metrics(y,struct_score)["roc_auc"]
    save_loss_plot(losses,"dominant_loss.png","DOMINANT-style reconstruction loss")
    save_score_plot(y,combined,"dominant_scores.png","Node anomaly scores")
    pd.DataFrame({"node_id":np.arange(len(y)),"label":y,"combined_score":combined,
                  "attribute_error":attr_score,"structure_error":struct_score}).to_csv("results_v2/predictions/node_scores.csv",index=False)
    torch.save(model.state_dict(),"results_v2/checkpoints/dominant.pt")
    print(f"DOMINANT-style device={device}, final_loss={losses[-1]:.5f}, metrics={result}")
    return result

if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--epochs",type=int,default=80);q.add_argument("--nodes",type=int,default=260);a=q.parse_args();print(run(a.epochs,a.nodes))
