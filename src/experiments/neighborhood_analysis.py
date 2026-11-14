
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


# =========================================================
# CONFIGURATION
# =========================================================

SEEDS = [42, 123, 456, 789, 1024]

EDGE_WEIGHTS = [20.0, 50.0]

N_ESTIMATORS = 300


# =========================================================
# NEIGHBOURHOOD UTILITIES
# =========================================================

def get_neighbors(adj):

    """
    adj[i, j] = 1 means i -> j.

    incoming[v]: nodes pointing to v.
    outgoing[v]: nodes pointed to by v.
    """

    num_nodes = adj.shape[0]

    incoming = []
    outgoing = []

    for v in range(num_nodes):

        incoming.append(
            np.flatnonzero(adj[:, v]).tolist()
        )

        outgoing.append(
            np.flatnonzero(adj[v, :]).tolist()
        )

    return incoming, outgoing


def get_two_hop_neighbors(
    node,
    incoming,
    outgoing
):

    """
    Four directed two-hop path patterns:

    II: w -> u -> node
    IO: u -> node and u -> w
    OI: node -> u and w -> u
    OO: node -> u -> w

    Direct neighbours and the node itself are excluded
    from the resulting two-hop sets.
    """

    in_in = set()
    in_out = set()

    out_in = set()
    out_out = set()

    # Paths starting from incoming neighbours.
    for u in incoming[node]:

        in_in.update(incoming[u])
        in_out.update(outgoing[u])

    # Paths starting from outgoing neighbours.
    for u in outgoing[node]:

        out_in.update(incoming[u])
        out_out.update(outgoing[u])

    direct_neighbors = (
        set(incoming[node]) |
        set(outgoing[node])
    )

    excluded = direct_neighbors | {node}

    paths = {
        "II": in_in,
        "IO": in_out,
        "OI": out_in,
        "OO": out_out
    }

    for key in paths:

        paths[key] = paths[key] - excluded

    return paths


# =========================================================
# FEATURE AGGREGATION
# =========================================================

def aggregate_neighbor_degrees(
    nodes,
    in_degree,
    out_degree
):

    """
    Return mean and standard deviation of log in-degree
    and log out-degree for a collection of neighbours.

    Does not include the number of neighbours.
    This prevents directly reintroducing the node's
    own degree through the neighbour count.
    """

    nodes = np.asarray(sorted(nodes), dtype=int)

    if len(nodes) == 0:

        return [0.0, 0.0, 0.0, 0.0]

    log_in = np.log1p(in_degree[nodes])

    log_out = np.log1p(out_degree[nodes])

    return [
        float(np.mean(log_in)),
        float(np.std(log_in)),
        float(np.mean(log_out)),
        float(np.std(log_out))
    ]


# =========================================================
# BUILD NEIGHBOURHOOD FEATURES
# =========================================================

def build_neighborhood_features(adj):

    num_nodes = adj.shape[0]

    incoming, outgoing = get_neighbors(adj)

    # Node degrees.
    out_degree = adj.sum(axis=1)

    in_degree = adj.sum(axis=0)

    feature_rows = []

    feature_names = []

    # -----------------------------------------------------
    # Define feature names
    # -----------------------------------------------------

    one_hop_groups = ["incoming", "outgoing"]

    for group in one_hop_groups:

        feature_names.extend([
            f"{group}_neighbor_mean_log_in_degree",
            f"{group}_neighbor_std_log_in_degree",
            f"{group}_neighbor_mean_log_out_degree",
            f"{group}_neighbor_std_log_out_degree"
        ])

    two_hop_groups = ["II", "IO", "OI", "OO"]

    for group in two_hop_groups:

        feature_names.extend([
            f"{group}_count",
            f"{group}_mean_log_in_degree",
            f"{group}_std_log_in_degree",
            f"{group}_mean_log_out_degree",
            f"{group}_std_log_out_degree"
        ])

    # -----------------------------------------------------
    # Extract features for every node
    # -----------------------------------------------------

    print("\nExtracting neighbourhood features...")

    for node in range(num_nodes):

        features = []

        # -------------------------------------------------
        # 1-hop incoming neighbours
        # -------------------------------------------------

        features.extend(
            aggregate_neighbor_degrees(
                incoming[node],
                in_degree,
                out_degree
            )
        )

        # -------------------------------------------------
        # 1-hop outgoing neighbours
        # -------------------------------------------------

        features.extend(
            aggregate_neighbor_degrees(
                outgoing[node],
                in_degree,
                out_degree
            )
        )

        # -------------------------------------------------
        # 2-hop directed neighbourhoods
        # -------------------------------------------------

        paths = get_two_hop_neighbors(
            node,
            incoming,
            outgoing
        )

        for group in two_hop_groups:

            nodes = paths[group]

            # Number of unique two-hop nodes.
            features.append(
                float(np.log1p(len(nodes)))
            )

            # Degree statistics of those nodes.
            features.extend(
                aggregate_neighbor_degrees(
                    nodes,
                    in_degree,
                    out_degree
                )
            )

        feature_rows.append(features)

        if (node + 1) % 200 == 0:

            print(
                f"Processed {node + 1}/{num_nodes} nodes"
            )

    neighborhood_x = np.asarray(
        feature_rows,
        dtype=np.float64
    )

    # Own degree information.
    degree_x = np.column_stack([
        np.log1p(in_degree),
        np.log1p(out_degree)
    ])

    degree_feature_names = [
        "own_log_in_degree",
        "own_log_out_degree"
    ]

    # Combined feature matrix.
    combined_x = np.column_stack([
        neighborhood_x,
        degree_x
    ])

    combined_feature_names = (
        feature_names + degree_feature_names
    )

    return {
        "neighborhood_x": neighborhood_x,
        "combined_x": combined_x,
        "degree_x": degree_x,
        "feature_names": feature_names,
        "combined_feature_names": combined_feature_names,
        "in_degree": in_degree,
        "out_degree": out_degree
    }


# =========================================================
# ISOLATION FOREST
# =========================================================

def run_isolation_forest(
    x,
    labels,
    seed,
    method_name
):

    """
    Train an unsupervised Isolation Forest.

    Labels are never passed to the model.
    """

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=seed,
        n_jobs=-1
    )

    model.fit(x_scaled)

    # Lower decision_function means more anomalous.
    scores = -model.decision_function(x_scaled)

    auc = roc_auc_score(
        labels,
        scores
    )

    ap = average_precision_score(
        labels,
        scores
    )

    return {
        "method": method_name,
        "seed": seed,
        "roc_auc": auc,
        "average_precision": ap,
        "scores": scores
    }


# =========================================================
# LOAD DIRECTEDDOMINANT SCORES
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

        print("Missing GNN file:", prediction_file)

        return None

    df = pd.read_csv(prediction_file)

    if "structural_score" not in df.columns:

        raise ValueError(
            f"structural_score missing in {prediction_file}. "
            f"Columns found: {list(df.columns)}"
        )

    scores = df["structural_score"].to_numpy(
        dtype=float
    )

    if not np.all(np.isfinite(scores)):

        raise ValueError(
            f"Non-finite GNN scores in {prediction_file}"
        )

    return scores


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("NEIGHBOURHOOD STRUCTURAL ANOMALY ANALYSIS")
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

    # -----------------------------------------------------
    # Extract neighbourhood features
    # -----------------------------------------------------

    feature_data = build_neighborhood_features(adj)

    neighborhood_x = feature_data["neighborhood_x"]

    combined_x = feature_data["combined_x"]

    print("\nFeature information:")

    print(
        "Neighbourhood-only features:",
        neighborhood_x.shape
    )

    print(
        "Degree + neighbourhood features:",
        combined_x.shape
    )

    # -----------------------------------------------------
    # Save feature matrices
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    output_dir = project_root / "results_neighborhood"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df = pd.DataFrame(
        neighborhood_x,
        columns=feature_data["feature_names"]
    )

    feature_df.insert(
        0,
        "node_index",
        np.arange(len(labels))
    )

    feature_df["label"] = labels

    feature_df.to_csv(
        output_dir / "neighborhood_features.csv",
        index=False
    )

    combined_df = pd.DataFrame(
        combined_x,
        columns=feature_data["combined_feature_names"]
    )

    combined_df.insert(
        0,
        "node_index",
        np.arange(len(labels))
    )

    combined_df["label"] = labels

    combined_df.to_csv(
        output_dir / "combined_features.csv",
        index=False
    )

    # -----------------------------------------------------
    # Run feature-based baselines
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FEATURE-BASED ISOLATION FOREST")
    print("=" * 65)

    baseline_results = []

    score_files = {}

    feature_sets = {
        "Neighborhood-only IF": neighborhood_x,
        "Degree+Neighborhood IF": combined_x
    }

    for method_name, x in feature_sets.items():

        print("\nMethod:", method_name)

        for seed in SEEDS:

            result = run_isolation_forest(
                x,
                labels,
                seed,
                method_name
            )

            print(
                f"Seed {seed}: "
                f"ROC-AUC={result['roc_auc']:.5f}, "
                f"AP={result['average_precision']:.5f}"
            )

            baseline_results.append({
                "method": result["method"],
                "seed": seed,
                "edge_weight": None,
                "roc_auc": result["roc_auc"],
                "average_precision": result["average_precision"]
            })

            score_files[
                (method_name, seed)
            ] = result["scores"]

    # -----------------------------------------------------
    # Evaluate GNN
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("DIRECTEDDOMINANT COMPARISON")
    print("=" * 65)

    gnn_results_dir = project_root / "results_amazon_directed"

    for weight in EDGE_WEIGHTS:

        print("\nEdge weight:", weight)

        for seed in SEEDS:

            scores = load_gnn_scores(
                gnn_results_dir,
                seed,
                weight
            )

            if scores is None:
                continue

            if len(scores) != len(labels):

                raise ValueError(
                    f"GNN score count mismatch: "
                    f"seed={seed}, weight={weight}"
                )

            auc = roc_auc_score(
                labels,
                scores
            )

            ap = average_precision_score(
                labels,
                scores
            )

            print(
                f"Seed {seed}: "
                f"ROC-AUC={auc:.5f}, AP={ap:.5f}"
            )

            baseline_results.append({
                "method": "DirectedDOMINANT structural",
                "seed": seed,
                "edge_weight": weight,
                "roc_auc": auc,
                "average_precision": ap
            })

    # -----------------------------------------------------
    # Save individual results
    # -----------------------------------------------------

    results_df = pd.DataFrame(baseline_results)

    results_df.to_csv(
        output_dir / "neighborhood_results.csv",
        index=False
    )

    # -----------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------

    summary = (
        results_df
        .groupby(
            ["method", "edge_weight"],
            dropna=False
        )
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std")
        )
        .reset_index()
    )

    summary.to_csv(
        output_dir / "neighborhood_summary.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save score matrix
    # -----------------------------------------------------

    score_df = pd.DataFrame({
        "node_index": np.arange(len(labels)),
        "label": labels
    })

    for method_name in feature_sets:

        for seed in SEEDS:

            score_df[
                f"{method_name}_seed_{seed}"
            ] = score_files[(method_name, seed)]

    score_df.to_csv(
        output_dir / "neighborhood_scores.csv",
        index=False
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL NEIGHBOURHOOD EXPERIMENT SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\nResults saved to:")
    print(output_dir)

    print("\nGenerated files:")
    print("- neighborhood_features.csv")
    print("- combined_features.csv")
    print("- neighborhood_results.csv")
    print("- neighborhood_summary.csv")
    print("- neighborhood_scores.csv")


if __name__ == "__main__":
    main()
