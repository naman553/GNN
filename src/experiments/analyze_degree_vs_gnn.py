
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

SEEDS = [42, 123, 456, 789, 1024]

EDGE_WEIGHTS = [20.0, 50.0]

N_SPLITS = 5

REGRESSION_ALPHA = 10.0

TOP_K = 28


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def evaluate_scores(name, labels, scores):

    """
    Higher scores must indicate greater anomaly likelihood.
    """

    scores = np.asarray(scores).reshape(-1)

    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite values in {name}")

    return {
        "method": name,
        "roc_auc": roc_auc_score(labels, scores),
        "average_precision": average_precision_score(
            labels,
            scores
        ),
    }


def get_degree_features(adj):

    """
    adj[i, j] = 1 means edge i -> j.

    Returns:
        in_degree
        out_degree
        degree_features
    """

    out_degree = adj.sum(axis=1)
    in_degree = adj.sum(axis=0)

    # Log transformation reduces degree scale differences.
    log_in = np.log1p(in_degree)
    log_out = np.log1p(out_degree)

    degree_features = np.column_stack([
        log_in,
        log_out
    ])

    return in_degree, out_degree, degree_features


def robust_degree_score(degree):

    """
    Two-sided degree deviation score.

    Higher score means degree is more unusual,
    whether unusually high or unusually low.
    """

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


# ---------------------------------------------------------
# Load GNN predictions
# ---------------------------------------------------------

def load_gnn_scores(results_dir, seed, weight):

    folder = (
        results_dir
        / f"directed_seed_{seed}_weight_{weight}"
    )

    prediction_file = folder / "predictions.csv"

    if not prediction_file.exists():

        print("Missing prediction file:", prediction_file)

        return None

    df = pd.read_csv(prediction_file)

    if "structural_score" not in df.columns:

        raise ValueError(
            f"'structural_score' column not found in "
            f"{prediction_file}\n"
            f"Available columns: {list(df.columns)}"
        )

    scores = df["structural_score"].to_numpy(dtype=float)

    if not np.all(np.isfinite(scores)):

        raise ValueError(
            f"Non-finite GNN scores in {prediction_file}"
        )

    return scores


# ---------------------------------------------------------
# Degree-controlled regression
# ---------------------------------------------------------

def calculate_residual_scores(
    degree_features,
    gnn_scores,
    seed
):

    """
    Predict GNN structural scores using degree information.

    Uses polynomial degree features and Ridge regression.

    Predictions are generated using 5-fold cross-validation.
    Each node's predicted score comes from a model that
    was not trained on that node.

    Residual = actual GNN score - predicted degree score.
    """

    model = make_pipeline(

        PolynomialFeatures(
            degree=2,
            include_bias=False
        ),

        StandardScaler(),

        Ridge(
            alpha=REGRESSION_ALPHA
        )
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed
    )

    predicted_scores = cross_val_predict(
        model,
        degree_features,
        gnn_scores,
        cv=kfold,
        method="predict"
    )

    residual_scores = gnn_scores - predicted_scores

    return predicted_scores, residual_scores


# ---------------------------------------------------------
# Top-K analysis
# ---------------------------------------------------------

def analyze_top_k(
    labels,
    degree_scores,
    gnn_scores,
    residual_scores
):

    k = min(TOP_K, len(labels))

    actual_anomalies = set(
        np.where(labels == 1)[0]
    )

    degree_top = set(
        np.argsort(degree_scores)[-k:]
    )

    gnn_top = set(
        np.argsort(gnn_scores)[-k:]
    )

    residual_top = set(
        np.argsort(residual_scores)[-k:]
    )

    degree_detected = degree_top.intersection(
        actual_anomalies
    )

    gnn_detected = gnn_top.intersection(
        actual_anomalies
    )

    residual_detected = residual_top.intersection(
        actual_anomalies
    )

    return {

        "degree_top_k_overlap": len(
            degree_top.intersection(gnn_top)
        ),

        "residual_top_k_overlap": len(
            residual_top.intersection(degree_top)
        ),

        "degree_detected": len(degree_detected),

        "gnn_detected": len(gnn_detected),

        "residual_detected": len(residual_detected),

        "gnn_only_anomalies": len(
            gnn_detected - degree_detected
        ),

        "residual_only_anomalies": len(
            residual_detected - degree_detected
        ),

        "residual_vs_gnn_overlap": len(
            residual_top.intersection(gnn_top)
        )
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("DEGREE-CONTROLLED DIRECTEDDOMINANT EVALUATION")
    print("=" * 65)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\nLoading Amazon Fail dataset...")

    from src.data.amazon_fail import load_amazon_fail

    data = load_amazon_fail()

    adj = data["adj"].numpy()
    labels = data["labels"].numpy()

    print("\nDataset information:")
    print("Nodes:", len(labels))
    print("Edges:", int(adj.sum()))
    print("Anomalies:", int(labels.sum()))
    print("Anomaly prevalence:", labels.mean())

    # -----------------------------------------------------
    # Calculate degree information
    # -----------------------------------------------------

    in_degree, out_degree, degree_features = (
        get_degree_features(adj)
    )

    in_degree_score = robust_degree_score(in_degree)

    out_degree_score = robust_degree_score(out_degree)

    combined_degree_score = (
        in_degree_score + out_degree_score
    ) / 2

    degree_baselines = {

        "In-degree deviation": in_degree_score,

        "Out-degree deviation": out_degree_score,

        "Combined degree deviation": combined_degree_score
    }

    print("\nDegree baseline metrics:")

    for name, scores in degree_baselines.items():

        result = evaluate_scores(
            name,
            labels,
            scores
        )

        print(
            f"{name}: "
            f"ROC-AUC={result['roc_auc']:.5f}, "
            f"AP={result['average_precision']:.5f}"
        )

    # -----------------------------------------------------
    # Results directories
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    results_dir = project_root / "results_amazon_directed"

    output_dir = project_root / "results_degree_controlled"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    # -----------------------------------------------------
    # Run all GNN configurations
    # -----------------------------------------------------

    for weight in EDGE_WEIGHTS:

        print("\n" + "=" * 65)
        print(f"EDGE WEIGHT: {weight}")
        print("=" * 65)

        for seed in SEEDS:

            print(
                f"\nProcessing seed={seed}, "
                f"weight={weight}"
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
                    f"Score count mismatch for "
                    f"seed={seed}, weight={weight}"
                )

            # -------------------------------------------------
            # Predict GNN scores from degree information
            # -------------------------------------------------

            predicted_scores, residual_scores = (
                calculate_residual_scores(
                    degree_features,
                    gnn_scores,
                    seed
                )
            )

            # -------------------------------------------------
            # Evaluate all three score types
            # -------------------------------------------------

            gnn_result = evaluate_scores(
                "Original GNN structural score",
                labels,
                gnn_scores
            )

            predicted_result = evaluate_scores(
                "Degree-predicted GNN score",
                labels,
                predicted_scores
            )

            residual_result = evaluate_scores(
                "Degree-controlled residual score",
                labels,
                residual_scores
            )

            degree_result = evaluate_scores(
                "In-degree deviation",
                labels,
                in_degree_score
            )

            # -------------------------------------------------
            # Correlations
            # -------------------------------------------------

                    
            correlation_degree = spearmanr(
                gnn_scores,
                predicted_scores
            )[0]

            correlation_residual = spearmanr(
                residual_scores,
                in_degree_score
            )[0]

            correlation_residual_gnn = spearmanr(
                residual_scores,
                gnn_scores
            )[0]


            # -------------------------------------------------
            # Top-K analysis
            # -------------------------------------------------

            top_k_result = analyze_top_k(
                labels,
                in_degree_score,
                gnn_scores,
                residual_scores
            )

            # -------------------------------------------------
            # Print results
            # -------------------------------------------------

            print("\nROC-AUC and AP:")

            print(
                f"Original GNN: "
                f"AUC={gnn_result['roc_auc']:.5f}, "
                f"AP={gnn_result['average_precision']:.5f}"
            )

            print(
                f"Degree prediction: "
                f"AUC={predicted_result['roc_auc']:.5f}, "
                f"AP={predicted_result['average_precision']:.5f}"
            )

            print(
                f"Residual GNN: "
                f"AUC={residual_result['roc_auc']:.5f}, "
                f"AP={residual_result['average_precision']:.5f}"
            )

            print("\nCorrelations:")

            print(
                "GNN vs degree-predicted:",
                f"{correlation_degree:.5f}"
            )

            print(
                "Residual vs in-degree:",
                f"{correlation_residual:.5f}"
            )

            print(
                "Residual vs original GNN:",
                f"{correlation_residual_gnn:.5f}"
            )

            print("\nTop-K analysis:")

            for key, value in top_k_result.items():

                print(f"{key}: {value}")

            # -------------------------------------------------
            # Save per-node scores
            # -------------------------------------------------

            node_scores = pd.DataFrame({

                "node_index": np.arange(len(labels)),

                "label": labels,

                "in_degree": in_degree,

                "out_degree": out_degree,

                "in_degree_score": in_degree_score,

                "gnn_structural_score": gnn_scores,

                "degree_predicted_score": predicted_scores,

                "degree_controlled_residual": residual_scores
            })

            node_scores.to_csv(
                output_dir
                / f"scores_seed_{seed}_weight_{weight}.csv",
                index=False
            )

            # -------------------------------------------------
            # Store experiment results
            # -------------------------------------------------

            row = {

                "seed": seed,

                "edge_weight": weight,

                "gnn_auc": gnn_result["roc_auc"],

                "gnn_ap": gnn_result["average_precision"],

                "degree_predicted_auc": predicted_result["roc_auc"],

                "degree_predicted_ap": predicted_result["average_precision"],

                "residual_auc": residual_result["roc_auc"],

                "residual_ap": residual_result["average_precision"],

                "in_degree_auc": degree_result["roc_auc"],

                "in_degree_ap": degree_result["average_precision"],

                "gnn_degree_correlation": correlation_degree,

                "residual_in_degree_correlation": correlation_residual,

                "residual_gnn_correlation": correlation_residual_gnn,

                **top_k_result
            }

            all_results.append(row)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    if not all_results:

        raise RuntimeError(
            "No GNN prediction files were loaded. "
            "Check the results_amazon_directed directory."
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        output_dir / "degree_controlled_results.csv",
        index=False
    )

    # -----------------------------------------------------
    # Aggregate across seeds
    # -----------------------------------------------------

    metric_columns = [

        "gnn_auc",

        "gnn_ap",

        "degree_predicted_auc",

        "degree_predicted_ap",

        "residual_auc",

        "residual_ap",

        "gnn_degree_correlation",

        "residual_in_degree_correlation",

        "residual_gnn_correlation",

        "degree_top_k_overlap",

        "residual_top_k_overlap",

        "gnn_only_anomalies",

        "residual_only_anomalies"
    ]

    summary = (
        results_df
        .groupby("edge_weight")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.to_csv(
        output_dir / "degree_controlled_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL DEGREE-CONTROLLED SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nResults saved in:")
    print(output_dir)

    print("\nGenerated files:")
    print("- degree_controlled_results.csv")
    print("- degree_controlled_summary.csv")
    print("- Per-seed node score CSV files")


if __name__ == "__main__":
    main()
