
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

path = (
    "results_amazon_directed/"
    "directed_seed_42_weight_10.0/"
    "predictions.csv"
)

df = pd.read_csv(path)

labels = df["label"].to_numpy()

attr = df["attribute_score"].to_numpy()
struct = df["structural_score"].to_numpy()

# Standardize each score independently.
attr_z = (attr - attr.mean()) / (attr.std() + 1e-12)
struct_z = (struct - struct.mean()) / (struct.std() + 1e-12)

# Equal-weight normalized fusion.
combined_z = 0.5 * attr_z + 0.5 * struct_z

# Structural-only baseline.
scores = {
    "Original combined": df["combined_score"].to_numpy(),
    "Attribute only": attr,
    "Structural only": struct,
    "Normalized combined": combined_z,
}

print("\n========== SCORE FUSION EXPERIMENT ==========")

for name, score in scores.items():

    auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)

    print(f"\n{name}")
    print(f"ROC-AUC: {auc:.5f}")
    print(f"Average Precision: {ap:.5f}")

print("\n========== NORMALIZED SCORE STATISTICS ==========")

print("Attribute z-score:")
print("Mean:", attr_z.mean())
print("Std:", attr_z.std())

print("\nStructural z-score:")
print("Mean:", struct_z.mean())
print("Std:", struct_z.std())
