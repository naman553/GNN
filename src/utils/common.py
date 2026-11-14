from pathlib import Path
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dirs():
    for folder in ("results_v2/plots", "results_v2/predictions", "results_v2/checkpoints"):
        Path(folder).mkdir(parents=True, exist_ok=True)

def binary_metrics(labels, scores, threshold=None):
    y=np.asarray(labels,dtype=int); s=np.asarray(scores,dtype=float)
    out={"roc_auc":float(roc_auc_score(y,s)) if len(np.unique(y))>1 else float("nan"),
         "average_precision":float(average_precision_score(y,s)) if len(np.unique(y))>1 else float("nan")}
    if threshold is not None:
        pred=(s>=threshold).astype(int)
        pr,re,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
        out.update(precision=float(pr),recall=float(re),f1=float(f1))
    return out

def save_loss_plot(losses, filename, title):
    ensure_dirs(); plt.figure(figsize=(7,4)); plt.plot(losses)
    plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title(title)
    plt.tight_layout(); plt.savefig(f"results_v2/plots/{filename}",dpi=160); plt.close()

def save_score_plot(labels,scores,filename,title):
    ensure_dirs(); y=np.asarray(labels); s=np.asarray(scores)
    plt.figure(figsize=(7,4))
    plt.hist(s[y==0],bins=25,alpha=.7,label="Normal")
    plt.hist(s[y==1],bins=25,alpha=.7,label="Injected anomaly")
    plt.xlabel("Anomaly score"); plt.ylabel("Count"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(f"results_v2/plots/{filename}",dpi=160); plt.close()
