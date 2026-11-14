
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data.amazon_fail import load_amazon_fail


SEEDS = [42, 123, 456, 789, 1024]


def evaluate_scores(name, labels, scores):
    """Higher scores must indicate greater anomaly likelihood."""

    scores = np.asarray(scores).reshape(-1)

    return {
        "method": name,
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(labels, scores),
    }


def robust_degree_score(degree):
    """
    Two-sided degree anomaly score.

    Detects both unusually high and unusually low degree.
    Uses log transformation and median absolute deviation.
    """

    log_degree = np.log1p(degree)

    median = np.median(log_degree)
    mad = np.median(np.abs(log_degree - median))

    # Scale MAD; fall back to standard deviation if MAD is zero.
    if mad > 1e-10:
        scale = 1.4826 * mad
    else:
        scale = np.std(log_degree)

    if scale < 1e-10:
        return np.zeros_like(log_degree, dtype=float)

    return np.abs(log_degree - median) / scale


def main():

    print("Loading Amazon Fail dataset...")

    data = load_amazon_fail()

    x = data["x"].numpy()
    adj = data["adj"].numpy()
    labels = data["labels"].numpy()

    print("\nDataset:")
    print("Nodes:", len(labels))
    print("Features:", x.shape[1])
    print("Edges:", int(adj.sum()))
    print("Anomalies:", int(labels.sum()))
    print("Anomaly prevalence:", round(labels.mean(), 5))

    results = []

    # ------------------------------------------------
    # Feature preprocessing
    # ------------------------------------------------

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # ------------------------------------------------
    # 1. Isolation Forest: five seeds
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("ISOLATION FOREST")
    print("=" * 60)

    for seed in SEEDS:

        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )

        model.fit(x_scaled)

        # Lower score_samples means more anomalous.
        scores = -model.score_samples(x_scaled)

        row = evaluate_scores(
            "Isolation Forest",
            labels,
            scores,
        )

        row["seed"] = seed
        results.append(row)

        print(
            f"Seed {seed}: "
            f"ROC-AUC={row['roc_auc']:.5f}, "
            f"AP={row['average_precision']:.5f}"
        )

    # ------------------------------------------------
    # 2. Local Outlier Factor
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("LOCAL OUTLIER FACTOR")
    print("=" * 60)

    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination="auto",
    )

    lof.fit_predict(x_scaled)

    lof_scores = -lof.negative_outlier_factor_

    row = evaluate_scores(
        "Local Outlier Factor",
        labels,
        lof_scores,
    )

    row["seed"] = None
    results.append(row)

    print(
        f"ROC-AUC={row['roc_auc']:.5f}, "
        f"AP={row['average_precision']:.5f}"
    )

    # ------------------------------------------------
    # 3. Directed degree baselines
    # ------------------------------------------------

    print("\n" + "=" * 60)
    print("DIRECTED DEGREE BASELINES")
    print("=" * 60)

    # adj[i, j] = 1 means i -> j.
    out_degree = adj.sum(axis=1)
    in_degree = adj.sum(axis=0)

    in_score = robust_degree_score(in_degree)
    out_score = robust_degree_score(out_degree)

    # Equal contribution from incoming and outgoing degree.
    combined_score = (in_score + out_score) / 2

    degree_methods = {
        "In-degree deviation": in_score,
        "Out-degree deviation": out_score,
        "Combined degree deviation": combined_score,
    }

    for name, scores in degree_methods.items():

        row = evaluate_scores(name, labels, scores)
        row["seed"] = None
        results.append(row)

        print(
            f"\n{name}: "
            f"ROC-AUC={row['roc_auc']:.5f}, "
            f"AP={row['average_precision']:.5f}"
        )

    # ------------------------------------------------
    # 4. Save results
    # ------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    results_dir = project_root / "results_amazon_baselines"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        results_dir / "baseline_results.csv",
        index=False,
    )

    # Aggregate Isolation Forest across five seeds.
    summary = (
        results_df.groupby("method")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
        )
        .reset_index()
    )

    summary.to_csv(
        results_dir / "baseline_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    print(summary.to_string(index=False))

    print("\nSaved results to:", results_dir)


if __name__ == "__main__":
    main()
