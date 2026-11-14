import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.data.synthetic import make_temporal_data
from src.models.temporal import GCNGRU
from src.utils.common import seed_everything,get_device,ensure_dirs,binary_metrics,save_loss_plot,save_score_plot

def run(epochs=80,num_nodes=120,seed=46):
    seed_everything(seed); ensure_dirs()
    xs,ads,labels=make_temporal_data(n=num_nodes,seed=seed)
    device=get_device(); xs=xs.to(device); ads=ads.to(device)
    model=GCNGRU(xs.size(-1)).to(device); opt=torch.optim.Adam(model.parameters(),lr=.004,weight_decay=1e-4)
    losses=[]
    # Fit only on pre-event windows; final anomalous snapshot is excluded from training.
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pred=model(xs[:-2],ads[:-2])
        target=xs[1:-1]
        loss=F.mse_loss(pred,target); loss.backward(); opt.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        # Predict the last normal-to-event transition from prior snapshots.
        pred=model(xs[:-1],ads[:-1])
        predicted_last=pred[-1]
        feature_error=((predicted_last-xs[-1])**2).mean(dim=1)
        # Include a local structural surprise term: degree change from the previous snapshot.
        degree_prev=ads[-2].sum(dim=1); degree_now=ads[-1].sum(dim=1)
        structural_error=(degree_now-degree_prev).abs()
        scores=(feature_error/feature_error.mean().clamp(min=1e-8) +
                structural_error/structural_error.mean().clamp(min=1e-8)).cpu().numpy()
    metrics=binary_metrics(labels,scores)
    save_loss_plot(losses,"dynamic_loss.png","GCN + GRU normal-window prediction loss")
    save_score_plot(labels,scores,"dynamic_scores.png","Temporal anomaly scores")
    pd.DataFrame({"node_id":np.arange(num_nodes),"label":labels,"score":scores,
                  "feature_error":feature_error.cpu().numpy(),
                  "degree_change":structural_error.cpu().numpy()}).to_csv("results_v2/predictions/dynamic_scores.csv",index=False)
    torch.save(model.state_dict(),"results_v2/checkpoints/gcn_gru.pt")
    print(f"Dynamic model device={device}, final_loss={losses[-1]:.5f}, metrics={metrics}")
    return {"task":"dynamic_GCN_GRU_predictive_baseline","seed":seed,**metrics,"final_loss":losses[-1]}

if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--epochs",type=int,default=80);q.add_argument("--nodes",type=int,default=120);a=q.parse_args();print(run(a.epochs,a.nodes))
