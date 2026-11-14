
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# Load saved predictions
path = (
    "results_amazon_directed/"
    "directed_seed_42_weight_10.0/"
    "predictions.csv"
)

df = pd.read_csv(path)

labels = df["label"]

# --------------------------------
# 1. Score distributions
# --------------------------------

print("\n========== SCORE DISTRIBUTIONS ==========")

columns = [
    "combined_score",
    "attribute_score",
    "structural_score",
]

print(
    df.groupby("label")[columns]
    .agg(["mean", "median", "std", "min", "max"])
    .to_string()
)

# --------------------------------
# 2. ROC-AUC and Average Precision
# --------------------------------

print("\n========== EVALUATION METRICS ==========")

for column in columns:

    scores = df[column]

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    reversed_auc = roc_auc_score(labels, -scores)

    print(f"\n{column}")
    print(f"ROC-AUC: {auc:.5f}")
    print(f"Average Precision: {ap:.5f}")
    print(f"Reversed ROC-AUC: {reversed_auc:.5f}")

# --------------------------------
# 3. Score correlations
# --------------------------------

print("\n========== SCORE CORRELATIONS ==========")

print(df[columns].corr().to_string())

# --------------------------------
# 4. Score scale comparison
# --------------------------------

print("\n========== SCORE RANGES ==========")

for column in columns:

    print(
        column,
        "min:", df[column].min(),
        "max:", df[column].max(),
        "std:", df[column].std(),
    )
