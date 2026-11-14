import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.models.gcn import normalize_adjacency
from src.models.edge import EdgeReconstructionGCN
from src.utils.common import seed_everything,get_device,ensure_dirs,binary_metrics,save_loss_plot

def run(epochs=80,num_nodes=240,seed=43):
    seed_everything(seed); ensure_dirs(); rng=np.random.default_rng(seed); device=get_device()
    groups=np.arange(num_nodes)%4
    # Build a normal attributed graph.
    a=np.zeros((num_nodes,num_nodes),dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i+1,num_nodes):
            if rng.random()<(.13 if groups[i]==groups[j] else .008): a[i,j]=a[j,i]=1
    # Random node features; graph encoder uses them for link reconstruction.
    x=torch.tensor(rng.normal(size=(num_nodes,12)).astype(np.float32),device=device)
    # Split normal edges: some are withheld for test; model sees only train edges.
    all_edges=np.argwhere(np.triu(a,1)>0); rng.shuffle(all_edges)
    split=max(1,int(.2*len(all_edges))); test_normal=all_edges[:split]; train_edges=all_edges[split:]
    train_a=np.zeros_like(a)
    for u,v in train_edges: train_a[u,v]=train_a[v,u]=1
    # Injected anomalous edges connect different communities and are never included in training adjacency.
    anomalous=[]
    for _ in range(max(20,split)):
        u=int(rng.integers(num_nodes)); candidates=np.where(groups!=groups[u])[0]
        v=int(rng.choice(candidates)); anomalous.append((u,v))
    anomalous=np.unique(np.asarray(anomalous,dtype=int),axis=0)
    # Keep only candidate pairs that are genuinely absent from the training graph.
    anomalous=np.asarray([(u,v) for u,v in anomalous if train_a[u,v]==0],dtype=int)
    train_adj=torch.tensor(train_a,device=device)
    norm=normalize_adjacency(train_adj)
    model=EdgeReconstructionGCN(x.size(1)).to(device); opt=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=5e-4)
    losses=[]
    pos=train_adj>0; pos.fill_diagonal_(False)
    neg=(train_adj==0); neg.fill_diagonal_(False)
    # Balance reconstruction terms to prevent the many absent pairs dominating.
    for _ in range(epochs):
        model.train(); opt.zero_grad(); _,rec=model(x,norm)
        lp=F.binary_cross_entropy(rec[pos],torch.ones_like(rec[pos]))
        ln=F.binary_cross_entropy(rec[neg],torch.zeros_like(rec[neg]))
        loss=lp+ln; loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    # Evaluate unseen normal edges (positive links) against injected cross-community edges.
    candidates=np.vstack([test_normal,anomalous])
    y=np.r_[np.zeros(len(test_normal),dtype=int),np.ones(len(anomalous),dtype=int)]
    model.eval()
    with torch.no_grad():
        _,rec=model(x,norm); uv=torch.tensor(candidates,device=device)
        # High reconstruction residual for an edge absent from the normal training graph => anomalous.
        scores=(1-rec[uv[:,0],uv[:,1]]).cpu().numpy()
    metrics=binary_metrics(y,scores)
    pd.DataFrame({"source":candidates[:,0],"target":candidates[:,1],"label":y,"score":scores}).to_csv("results_v2/predictions/edge_scores.csv",index=False)
    save_loss_plot(losses,"edge_loss.png","Edge link reconstruction loss")
    torch.save(model.state_dict(),"results_v2/checkpoints/edge_gcn.pt")
    print(f"Edge baseline device={device}, final_loss={losses[-1]:.5f}, metrics={metrics}")
    return {"task":"edge_GCN_link_anomaly","seed":seed,**metrics,"final_loss":losses[-1]}

if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--epochs",type=int,default=80);q.add_argument("--nodes",type=int,default=240);a=q.parse_args();print(run(a.epochs,a.nodes))
