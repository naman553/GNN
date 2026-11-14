
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.amazon_fail import load_amazon_fail
from src.models.directed_gcn import normalize_directed_adjacency
from src.models.directed_dominant import (
    DirectedDOMINANT,
    directed_dominant_scores,
)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_run(
    data,
    seed,
    edge_weight,
    epochs,
    learning_rate,
    weight_decay,
    alpha,
    device,
    results_dir,
):
    """
    Train one DirectedDOMINANT experiment.

    Anomaly labels are not used during training.
    """

    set_seed(seed)

    x = data["x"].to(device)
    adj = data["adj"].to(device)
    labels = data["labels"].cpu().numpy()

    incoming_adj, outgoing_adj = normalize_directed_adjacency(adj)

    incoming_adj = incoming_adj.to(device)
    outgoing_adj = outgoing_adj.to(device)

    num_nodes = x.size(0)

    # Exclude self-connections from structural reconstruction.
    off_diagonal_mask = ~torch.eye(
        num_nodes,
        dtype=torch.bool,
        device=device,
    )

    target_adj = adj[off_diagonal_mask]

    # Positive edge weight for weighted BCE.
    pos_weight = torch.tensor(
        edge_weight,
        dtype=torch.float32,
        device=device,
    )

    model = DirectedDOMINANT(
        in_features=x.size(1),
        hidden=64,
        embedding=32,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    # -------------------------------
    # Training
    # -------------------------------

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        x_hat, adj_hat, z = model(
            x,
            incoming_adj,
            outgoing_adj,
        )

        # Attribute reconstruction loss.
        attribute_loss = F.mse_loss(x_hat, x)

        # Obtain raw logits for numerically stable BCE.
        structure_logits = model.structure_logits(z)

        structure_logits = structure_logits[off_diagonal_mask]

        structure_loss = F.binary_cross_entropy_with_logits(
            structure_logits,
            target_adj,
            pos_weight=pos_weight,
        )

        total_loss = attribute_loss + structure_loss

        total_loss.backward()

        optimizer.step()

        history.append({
            "epoch": epoch + 1,
            "total_loss": total_loss.item(),
            "attribute_loss": attribute_loss.item(),
            "structure_loss": structure_loss.item(),
        })

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Seed {seed} | Weight {edge_weight} | "
                f"Epoch {epoch + 1}/{epochs} | "
                f"Total: {total_loss.item():.6f} | "
                f"Attr: {attribute_loss.item():.6f} | "
                f"Struct: {structure_loss.item():.6f}"
            )

    # -------------------------------
    # Evaluation
    # -------------------------------

    model.eval()

    with torch.no_grad():

        x_hat, adj_hat, z = model(
            x,
            incoming_adj,
            outgoing_adj,
        )

        scores, attr_scores, struct_scores = (
            directed_dominant_scores(
                x,
                adj,
                x_hat,
                adj_hat,
                alpha=alpha,
            )
        )

    scores = scores.cpu().numpy()
    attr_scores = attr_scores.cpu().numpy()
    struct_scores = struct_scores.cpu().numpy()

    combined_auc = roc_auc_score(labels, scores)
    combined_ap = average_precision_score(labels, scores)

    attribute_auc = roc_auc_score(labels, attr_scores)
    attribute_ap = average_precision_score(labels, attr_scores)

    structural_auc = roc_auc_score(labels, struct_scores)
    structural_ap = average_precision_score(labels, struct_scores)

    # -------------------------------
    # Save outputs
    # -------------------------------

    run_name = f"directed_seed_{seed}_weight_{edge_weight}"

    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    predictions = pd.DataFrame({
        "node_id": data["node_ids"],
        "product_id": data["product_ids"],
        "label": labels,
        "combined_score": scores,
        "attribute_score": attr_scores,
        "structural_score": struct_scores,
    })

    predictions.to_csv(
        run_dir / "predictions.csv",
        index=False,
    )

    torch.save(
        model.state_dict(),
        run_dir / "model.pt",
    )

    result = {
        "seed": seed,
        "edge_weight": edge_weight,
        "epochs": epochs,
        "final_loss": history[-1]["total_loss"],
        "final_attribute_loss": history[-1]["attribute_loss"],
        "final_structure_loss": history[-1]["structure_loss"],
        "combined_roc_auc": combined_auc,
        "combined_ap": combined_ap,
        "attribute_roc_auc": attribute_auc,
        "attribute_ap": attribute_ap,
        "structural_roc_auc": structural_auc,
        "structural_ap": structural_ap,
    }

    print("\nEvaluation:")
    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main():

    parser = argparse.ArgumentParser(
        description="Directed DOMINANT on Amazon Fail"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 1024],
    )

    parser.add_argument(
        "--edge-weights",
        type=float,
        nargs="+",
        default=[10, 20, 30, 50],
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    # -------------------------------
    # Device and dataset
    # -------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    data = load_amazon_fail()

    print("\nDataset:")
    print("Nodes:", data["x"].shape[0])
    print("Features:", data["x"].shape[1])
    print("Directed edges:", data["edge_index"].shape[1])
    print("Anomalies:", (data["labels"] == 1).sum().item())

    # -------------------------------
    # Results directory
    # -------------------------------

    project_root = Path(__file__).resolve().parents[2]

    results_dir = project_root / "results_amazon_directed"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # -------------------------------
    # Experiment sweep
    # -------------------------------

    for seed in args.seeds:

        for edge_weight in args.edge_weights:

            print("\n" + "=" * 65)
            print(
                f"Starting seed={seed}, "
                f"edge_weight={edge_weight}"
            )
            print("=" * 65)

            result = train_one_run(
                data=data,
                seed=seed,
                edge_weight=edge_weight,
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                alpha=args.alpha,
                device=device,
                results_dir=results_dir,
            )

            all_results.append(result)

    # -------------------------------
    # Save all individual results
    # -------------------------------

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        results_dir / "all_results.csv",
        index=False,
    )

    # -------------------------------
    # Aggregate across seeds
    # -------------------------------

    metrics = [
        "combined_roc_auc",
        "combined_ap",
        "attribute_roc_auc",
        "attribute_ap",
        "structural_roc_auc",
        "structural_ap",
    ]

    summary = results_df.groupby("edge_weight")[metrics].agg(
        ["mean", "std"]
    )

    summary.to_csv(
        results_dir / "summary.csv",
    )

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print(results_df.to_string(index=False))

    print("\nAggregate results:")
    print(summary.to_string())

    print("\nResults saved to:", results_dir)


if __name__ == "__main__":
    main()
