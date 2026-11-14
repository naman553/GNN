
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================================================
# CONFIGURATION
# =========================================================

SEEDS = [42, 123, 456, 789, 1024]

EDGE_WEIGHTS = [20.0, 50.0]

K_NEIGHBORS = 5


# =========================================================
# DEGREE FEATURES
# =========================================================

def get_degree_features(adj):

    # adj[i, j] = 1 means i -> j.

    out_degree = adj.sum(axis=1)
    in_degree = adj.sum(axis=0)

    # Log transformation.
    degree_features = np.column_stack([
        np.log1p(in_degree),
        np.log1p(out_degree)
    ])

    return in_degree, out_degree, degree_features


# =========================================================
# DEGREE BASELINE
# =========================================================

def robust_degree_score(degree):

    log_degree = np.log1p(degree)

    median = np.median(log_degree)

    mad = np.median(
        np.abs(log_degree - median)
    )

    scale = 1.4826 * mad

    if scale < 1e-10:
        scale = np.std(log_degree)

    if scale < 1e-10:
        return np.zeros_like(log_degree, dtype=float)

    return np.abs(log_degree - median) / scale


# =========================================================
# MATCH NORMAL NODES TO ANOMALOUS NODES
# =========================================================

def create_degree_matches(
    degree_features,
    labels,
    k=5
):

    """
    For every anomalous node, find its k nearest
    normal nodes in standardized log-degree space.

    Matching uses degree features and labels only
    to identify cases and controls.

    It does not use GNN scores.
    """

    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    if len(normal_indices) < k:
        raise ValueError("Not enough normal nodes for matching.")

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        degree_features
    )

    normal_features = scaled_features[normal_indices]

    anomaly_features = scaled_features[anomaly_indices]

    nearest_neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    nearest_neighbors.fit(normal_features)

    distances, local_indices = nearest_neighbors.kneighbors(
        anomaly_features
    )

    matched_pairs = []

    for i, anomaly_idx in enumerate(anomaly_indices):

        matched_normal_indices = normal_indices[
            local_indices[i]
        ]

        for j, normal_idx in enumerate(matched_normal_indices):

            matched_pairs.append({

                "anomaly_index": int(anomaly_idx),

                "normal_index": int(normal_idx),

                "degree_distance": float(distances[i, j]),

            })

    return pd.DataFrame(matched_pairs)


# =========================================================
# MATCHED PAIRWISE EVALUATION
# =========================================================

def evaluate_matched_scores(
    pairs,
    scores,
    name
):

    anomaly_scores = scores[
        pairs["anomaly_index"].to_numpy()
    ]

    normal_scores = scores[
        pairs["normal_index"].to_numpy()
    ]

    differences = anomaly_scores - normal_scores

    # Pairwise concordance:
    # 1.0 means anomaly score is always higher.
    # 0.5 means random ordering.
    # 0.0 means anomaly score is always lower.

    wins = np.where(
        differences > 0,
        1.0,
        np.where(differences < 0, 0.0, 0.5)
    )

    return {

        f"{name}_matched_accuracy": float(wins.mean()),

        f"{name}_mean_score_difference": float(
            differences.mean()
        ),

        f"{name}_median_score_difference": float(
            np.median(differences)
        )
    }


# =========================================================
# LOAD GNN PREDICTIONS
# =========================================================

def load_gnn_scores(
    results_dir,
    seed,
    weight
):

    folder = (
        results_dir
        / f"directed_seed_{seed}_weight_{weight}"
    )

    prediction_file = folder / "predictions.csv"

    if not prediction_file.exists():

        print("Missing:", prediction_file)

        return None

    df = pd.read_csv(prediction_file)

    if "structural_score" not in df.columns:

        raise ValueError(
            f"structural_score not found in {prediction_file}. "
            f"Available columns: {list(df.columns)}"
        )

    scores = df["structural_score"].to_numpy(
        dtype=float
    )

    if not np.all(np.isfinite(scores)):

        raise ValueError(
            f"Non-finite scores in {prediction_file}"
        )

    return scores


# =========================================================
# STANDARD EVALUATION
# =========================================================

def evaluate_standard(labels, scores):

    return {

        "roc_auc": roc_auc_score(
            labels,
            scores
        ),

        "average_precision": average_precision_score(
            labels,
            scores
        )
    }


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def main():

    print("=" * 65)
    print("DEGREE-MATCHED DIRECTEDDOMINANT EVALUATION")
    print("=" * 65)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    from src.data.amazon_fail import load_amazon_fail

    data = load_amazon_fail()

    adj = data["adj"].numpy()
    labels = data["labels"].numpy()

    print("\nDataset:")
    print("Nodes:", len(labels))
    print("Edges:", int(adj.sum()))
    print("Anomalies:", int(labels.sum()))
    print("Normal nodes:", int((labels == 0).sum()))

    # -----------------------------------------------------
    # Calculate degree features
    # -----------------------------------------------------

    in_degree, out_degree, degree_features = (
        get_degree_features(adj)
    )

    in_degree_score = robust_degree_score(
        in_degree
    )

    # -----------------------------------------------------
    # Create matching pairs ONCE.
    # Matching is independent of GNN configuration.
    # -----------------------------------------------------

    pairs = create_degree_matches(
        degree_features,
        labels,
        k=K_NEIGHBORS
    )

    print("\nMatching information:")
    print("Anomalous nodes:", int(labels.sum()))
    print("Controls per anomaly:", K_NEIGHBORS)
    print("Total matched pairs:", len(pairs))

    print(
        "Unique normal controls:",
        pairs["normal_index"].nunique()
    )

    print(
        "Average degree distance:",
        round(pairs["degree_distance"].mean(), 5)
    )

    # Save matched pairs.
    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "results_degree_matched"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs.to_csv(
        output_dir / "matched_pairs.csv",
        index=False
    )

    # -----------------------------------------------------
    # Evaluate degree baseline
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("IN-DEGREE BASELINE")
    print("=" * 65)

    degree_result = evaluate_matched_scores(
        pairs,
        in_degree_score,
        "in_degree"
    )

    for key, value in degree_result.items():
        print(f"{key}: {value:.5f}")

    # -----------------------------------------------------
    # Evaluate GNN configurations
    # -----------------------------------------------------

    results_dir = project_root / "results_amazon_directed"

    all_results = []

    for weight in EDGE_WEIGHTS:

        print("\n" + "=" * 65)
        print("EDGE WEIGHT:", weight)
        print("=" * 65)

        for seed in SEEDS:

            print(
                f"\nSeed={seed}, Weight={weight}"
            )

            gnn_scores = load_gnn_scores(
                results_dir,
                seed,
                weight
            )

            if gnn_scores is None:
                continue

            if len(gnn_scores) != len(labels):

                raise ValueError(
                    f"Score count mismatch: seed={seed}, "
                    f"weight={weight}"
                )

            # Standard full-dataset evaluation.
            standard = evaluate_standard(
                labels,
                gnn_scores
            )

            # Matched pairwise evaluation.
            matched = evaluate_matched_scores(
                pairs,
                gnn_scores,
                "gnn"
            )

            print("\nStandard metrics:")

            print(
                f"ROC-AUC: {standard['roc_auc']:.5f}"
            )

            print(
                f"AP: {standard['average_precision']:.5f}"
            )

            print("\nDegree-matched metrics:")

            for key, value in matched.items():

                print(f"{key}: {value:.5f}")

            # -------------------------------------------------
            # Save per-configuration matched results
            # -------------------------------------------------

            result = {

                "seed": seed,

                "edge_weight": weight,

                "roc_auc": standard["roc_auc"],

                "average_precision": standard[
                    "average_precision"
                ],

                **matched
            }

            all_results.append(result)

            # Save individual pair comparisons.
            pair_results = pairs.copy()

            pair_results["anomaly_score"] = gnn_scores[
                pair_results["anomaly_index"].to_numpy()
            ]

            pair_results["normal_score"] = gnn_scores[
                pair_results["normal_index"].to_numpy()
            ]

            pair_results["score_difference"] = (
                pair_results["anomaly_score"]
                - pair_results["normal_score"]
            )

            pair_results.to_csv(
                output_dir
                / f"pairs_seed_{seed}_weight_{weight}.csv",
                index=False
            )

    # -----------------------------------------------------
    # Save final results
    # -----------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No GNN predictions were loaded. "
            "Check results_amazon_directed."
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        output_dir / "degree_matched_results.csv",
        index=False
    )

    # -----------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------

    summary = (
        results_df
        .groupby("edge_weight")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),

            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),

            matched_accuracy_mean=(
                "gnn_matched_accuracy",
                "mean"
            ),

            matched_accuracy_std=(
                "gnn_matched_accuracy",
                "std"
            ),

            mean_score_difference=(
                "gnn_mean_score_difference",
                "mean"
            ),

            median_score_difference=(
                "gnn_median_score_difference",
                "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        output_dir / "degree_matched_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL DEGREE-MATCHED SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nIn-degree matched accuracy:")
    print(
        degree_result["in_degree_matched_accuracy"]
    )

    print("\nResults saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()
