import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from src.data.synthetic import make_graph_collection
from src.models.graph_level import GINStyleEncoder
from src.utils.common import seed_everything,get_device,ensure_dirs,binary_metrics,save_loss_plot,save_score_plot

class GraphSummaryDecoder(nn.Module):
    def __init__(self,hidden,feature_dim):
        super().__init__(); self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,feature_dim+2))
    def forward(self,z): return self.net(z)

def graph_summary(x,a):
    # Graph-level target: mean node attributes, edge density and mean degree.
    n=a.shape[0]; density=float(a.sum()/(n*(n-1)+1e-8))
    mean_degree=float(a.sum(axis=1).mean())
    return np.r_[x.mean(axis=0),density,mean_degree].astype(np.float32)

def run(seed=45,epochs=100):
    seed_everything(seed); ensure_dirs(); device=get_device()
    graphs,labels=make_graph_collection(n_graphs=200,seed=seed)
    # Deterministic split: train/validation reference contains only normal graphs.
    normal_ids=np.where(labels==0)[0]; test_ids=np.arange(len(labels))
    train_ids=normal_ids[:int(.7*len(normal_ids))]
    test_ids=np.r_[normal_ids[int(.7*len(normal_ids)):], np.where(labels==1)[0]]
    feature_dim=graphs[0][0].shape[1]
    model=GINStyleEncoder(feature_dim,hidden=32).to(device); decoder=GraphSummaryDecoder(32,feature_dim).to(device)
    opt=torch.optim.Adam(list(model.parameters())+list(decoder.parameters()),lr=.003,weight_decay=1e-4)
    targets=[torch.tensor(graph_summary(x,a),device=device) for x,a in graphs]
    losses=[]
    for _ in range(epochs):
        model.train(); decoder.train(); opt.zero_grad(); batch=[]
        for idx in train_ids:
            x,a=graphs[idx]; xt=torch.tensor(x,device=device); at=torch.tensor(a,device=device)
            z,_=model(xt,at); pred=decoder(z)
            batch.append(F.mse_loss(pred,targets[idx]))
        loss=torch.stack(batch).mean(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
    model.eval(); decoder.eval(); scores=[]
    with torch.no_grad():
        for idx,(x,a) in enumerate(graphs):
            xt=torch.tensor(x,device=device); at=torch.tensor(a,device=device)
            z,_=model(xt,at); pred=decoder(z)
            # Reconstruction residual is the graph anomaly score.
            scores.append(float(F.mse_loss(pred,targets[idx]).cpu()))
    scores=np.asarray(scores)
    test_labels=labels[test_ids]; test_scores=scores[test_ids]
    metrics=binary_metrics(test_labels,test_scores)
    pd.DataFrame({"graph_id":test_ids,"label":test_labels,"score":test_scores}).to_csv("results_v2/predictions/graph_scores.csv",index=False)
    save_loss_plot(losses,"graph_level_loss.png","GIN-inspired graph reconstruction loss")
    save_score_plot(test_labels,test_scores,"graph_level_scores.png","Graph-level anomaly scores")
    torch.save({"encoder":model.state_dict(),"decoder":decoder.state_dict()},"results_v2/checkpoints/graph_level.pt")
    print(f"Graph-level trained model device={device}, final_loss={losses[-1]:.5f}, metrics={metrics}")
    return {"task":"graph_level_GIN_inspired_reconstruction","seed":seed,**metrics,"final_loss":losses[-1]}

if __name__=="__main__": print(run())
