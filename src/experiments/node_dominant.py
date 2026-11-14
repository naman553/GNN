import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.data.synthetic import make_attributed_graph
from src.models.gcn import normalize_adjacency
from src.models.dominant import DOMINANT, dominant_scores

from src.utils.common import (
    seed_everything,
    get_device,
    ensure_dirs,
    binary_metrics,
    save_loss_plot,
    save_score_plot,
)


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

EPOCHS = 80
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
ALPHA = 0.5

# Positive-edge weights and random seeds for multi-seed validation.
EDGE_WEIGHTS = [10, 20, 30, 40, 50]
SEEDS = [42, 123, 456, 789, 1024]


# ---------------------------------------------------------
# Structural loss
# ---------------------------------------------------------

def calculate_structural_loss(
    mode,
    logits,
    adj,
    off_diag_mask,
    edge_weight,
    device,
):
    """Calculate structural reconstruction loss."""

    predictions = torch.sigmoid(logits)

    if mode == "mse":
        return F.mse_loss(
            predictions[off_diag_mask],
            adj[off_diag_mask],
        )

    elif mode == "weighted_mse":
        squared_error = (predictions - adj) ** 2

        weights = torch.where(
            adj > 0,
            torch.tensor(
                edge_weight,
                device=device,
                dtype=adj.dtype,
            ),
            torch.tensor(
                1.0,
                device=device,
                dtype=adj.dtype,
            ),
        )

        weighted_error = squared_error * weights
        masked_error = weighted_error[off_diag_mask]
        masked_weights = weights[off_diag_mask]

        return (
            masked_error.sum()
            / masked_weights.sum().clamp_min(1e-8)
        )

    elif mode == "weighted_bce":
        pos_weight = torch.tensor(
            edge_weight,
            dtype=adj.dtype,
            device=device,
        )

        return F.binary_cross_entropy_with_logits(
            logits[off_diag_mask],
            adj[off_diag_mask],
            pos_weight=pos_weight,
        )

    else:
        raise ValueError(f"Unknown structural loss mode: {mode}")


# ---------------------------------------------------------
# Run one experiment for one seed and one edge weight
# ---------------------------------------------------------

def run_experiment(
    mode,
    data,
    epochs,
    seed,
    device,
    edge_weight,
):
    print("\n" + "=" * 60)
    print(
        f"Starting experiment: {mode} | "
        f"seed: {seed} | edge weight: {edge_weight}"
    )
    print("=" * 60)

    # Reset seed so each edge-weight configuration for the same
    # seed starts from the same model initialization.
    seed_everything(seed)

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    norm = normalize_adjacency(adj)

    n = adj.size(0)
    if n <= 1:
        raise ValueError("Graph must contain at least two nodes.")

    # Exclude self-loops from the reconstruction objective.
    off_diag_mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=device,
    )

    model = DOMINANT(in_features=x.size(1)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    losses = []

    # ---------------- Training ----------------
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_hat, adj_hat, z = model(x, norm)

        attribute_loss = F.mse_loss(x_hat, x)

        logits = model.structure_logits(z)

        structural_loss = calculate_structural_loss(
            mode=mode,
            logits=logits,
            adj=adj,
            off_diag_mask=off_diag_mask,
            edge_weight=edge_weight,
            device=device,
        )

        loss = (
            (1 - ALPHA) * attribute_loss
            + ALPHA * structural_loss
        )

        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

    # ---------------- Evaluation ----------------
    model.eval()

    with torch.no_grad():
        x_hat, adj_hat, _ = model(x, norm)

        scores, attr_error, struct_error = dominant_scores(
            x,
            adj,
            x_hat,
            adj_hat,
            ALPHA,
        )

        combined_scores = scores.cpu().numpy()
        attr_scores = attr_error.cpu().numpy()
        struct_scores = struct_error.cpu().numpy()

        edge_mask = adj.bool() & off_diag_mask
        non_edge_mask = (~adj.bool()) & off_diag_mask

        edge_predictions = adj_hat[edge_mask]
        non_edge_predictions = adj_hat[non_edge_mask]

        diagnostics = {
            "adjacency_density": adj.mean().item(),
            "reconstructed_mean": adj_hat.mean().item(),
            "reconstructed_std": adj_hat.std().item(),
            "edge_prediction_mean": edge_predictions.mean().item(),
            "non_edge_prediction_mean": non_edge_predictions.mean().item(),
            "edge_prediction_std": edge_predictions.std().item(),
            "non_edge_prediction_std": non_edge_predictions.std().item(),
        }

    # ---------------- Labels and metrics ----------------
    labels = data["labels"]
    attribute_labels = data["attribute_labels"]
    structural_labels = data["structural_labels"]

    result = {
        "loss_mode": mode,
        "edge_weight": float(edge_weight),
        "seed": int(seed),
        "final_loss": losses[-1],
        "attribute_loss": float(attribute_loss.detach().cpu()),
        "structural_loss": float(structural_loss.detach().cpu()),
        **binary_metrics(labels, combined_scores),
    }

    result["attribute_detection_roc_auc"] = binary_metrics(
        attribute_labels,
        attr_scores,
    )["roc_auc"]

    result["structural_detection_roc_auc"] = binary_metrics(
        structural_labels,
        struct_scores,
    )["roc_auc"]

    result["attribute_score_all_anomalies_roc_auc"] = binary_metrics(
        labels,
        attr_scores,
    )["roc_auc"]

    result["structural_score_all_anomalies_roc_auc"] = binary_metrics(
        labels,
        struct_scores,
    )["roc_auc"]

    print("\n--- Structural Diagnostics ---")
    for key, value in diagnostics.items():
        print(f"{key}: {value:.6f}")

    print("\n--- Experiment Results ---")
    print(result)

    # ---------------- Save per-run outputs ----------------
    # Include BOTH seed and edge weight to prevent overwriting.
    output_name = (
        f"dominant_{mode}_seed_{seed}_edge_weight_{edge_weight:g}"
    )

    prediction_path = (
        f"results_v2/predictions/{output_name}_scores.csv"
    )
    checkpoint_path = (
        f"results_v2/checkpoints/{output_name}.pt"
    )

    pd.DataFrame({
        "node_id": np.arange(len(labels)),
        "label": labels,
        "attribute_label": attribute_labels,
        "structural_label": structural_labels,
        "combined_score": combined_scores,
        "attribute_error": attr_scores,
        "structure_error": struct_scores,
    }).to_csv(prediction_path, index=False)

    torch.save(model.state_dict(), checkpoint_path)

    save_loss_plot(
        losses,
        f"{output_name}_loss.png",
        f"DOMINANT weighted MSE | seed {seed} | edge weight {edge_weight:g}",
    )

    save_score_plot(
        labels,
        combined_scores,
        f"{output_name}_scores.png",
        f"Node anomaly scores | seed {seed} | edge weight {edge_weight:g}",
    )

    return result


# ---------------------------------------------------------
# Multi-seed experiment runner
# ---------------------------------------------------------

def run(
    epochs=EPOCHS,
    num_nodes=260,
    seeds=None,
    edge_weights=None,
):
    ensure_dirs()

    os.makedirs("results_v2/predictions", exist_ok=True)
    os.makedirs("results_v2/checkpoints", exist_ok=True)

    if seeds is None:
        seeds = SEEDS

    if edge_weights is None:
        edge_weights = EDGE_WEIGHTS

    if not seeds:
        raise ValueError("At least one seed must be provided.")

    if not edge_weights:
        raise ValueError("At least one edge weight must be provided.")

    device = get_device()

    print("Device:", device)
    print("Nodes:", num_nodes)
    print("Epochs:", epochs)
    print("Seeds:", seeds)
    print("Edge weights:", edge_weights)
    print("Loss: weighted_mse")
    print("Total experiments:", len(seeds) * len(edge_weights))

    all_results = []

    # Outer loop: iterate over random seeds.
    for seed in seeds:
        print("\n" + "#" * 70)
        print(f"GENERATING GRAPH AND RUNNING EXPERIMENTS FOR SEED: {seed}")
        print("#" * 70)

        # Generate one graph for this seed, then reuse it for every
        # edge weight so the comparison within this seed is fair.
        seed_everything(seed)
        data = make_attributed_graph(
            num_nodes=num_nodes,
            seed=seed,
        )

        # Inner loop: compare edge weights on the same graph.
        for edge_weight in edge_weights:
            result = run_experiment(
                mode="weighted_mse",
                data=data,
                epochs=epochs,
                seed=seed,
                device=device,
                edge_weight=float(edge_weight),
            )
            all_results.append(result)

    # ---------------- Save all individual results ----------------
    results_df = pd.DataFrame(all_results)

    all_results_path = "results_v2/multi_seed_all_results.csv"
    results_df.to_csv(all_results_path, index=False)

    # ---------------- Aggregate across seeds ----------------
    summary_df = (
        results_df
        .groupby("edge_weight")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_std=("average_precision", "std"),
            structural_auc_mean=(
                "structural_detection_roc_auc", "mean"
            ),
            structural_auc_std=(
                "structural_detection_roc_auc", "std"
            ),
            attribute_auc_mean=(
                "attribute_detection_roc_auc", "mean"
            ),
            attribute_auc_std=(
                "attribute_detection_roc_auc", "std"
            ),
        )
        .reset_index()
    )

    summary_path = "results_v2/multi_seed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # ---------------- Plot mean ROC-AUC with standard deviation ----------------
    plt.figure(figsize=(9, 5))

    plt.errorbar(
        summary_df["edge_weight"],
        summary_df["roc_auc_mean"],
        yerr=summary_df["roc_auc_std"],
        marker="o",
        capsize=4,
        label="Combined ROC-AUC",
    )

    plt.errorbar(
        summary_df["edge_weight"],
        summary_df["structural_auc_mean"],
        yerr=summary_df["structural_auc_std"],
        marker="s",
        capsize=4,
        label="Structural ROC-AUC",
    )

    plt.xlabel("Positive-edge weight")
    plt.ylabel("ROC-AUC")
    plt.title("Multi-seed weighted MSE comparison")
    plt.xticks(summary_df["edge_weight"].tolist())
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = "results_v2/multi_seed_weight_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # ---------------- Print summary ----------------
    print("\n" + "=" * 90)
    print("MULTI-SEED EDGE-WEIGHT SUMMARY (MEAN ± STANDARD DEVIATION)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    print("\nSaved individual results:", all_results_path)
    print("Saved aggregate summary:", summary_path)
    print("Saved comparison plot:", plot_path)

    return summary_df


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-seed DOMINANT weighted-MSE edge-weight sweep"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=260,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Random seeds to evaluate",
    )

    parser.add_argument(
        "--edge-weights",
        type=float,
        nargs="+",
        default=EDGE_WEIGHTS,
        help="Positive-edge weights to test",
    )

    args = parser.parse_args()

    run(
        epochs=args.epochs,
        num_nodes=args.nodes,
        seeds=args.seeds,
        edge_weights=args.edge_weights,
    )
