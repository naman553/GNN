
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

from src.data.amazon_fail import load_amazon_fail

from src.models.directed_gcn import (
    normalize_directed_adjacency,
    DirectedGCNEncoder,
)

from src.models.directed_dominant import (
    DirectedDOMINANT,
    directed_dominant_scores,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ABLATION GRAPH CONVOLUTION
# ============================================================

class AblationGraphConvolution(nn.Module):
    """
    Configurable directed graph convolution.

    Variants:

        incoming_only:
            Self + incoming aggregation

        outgoing_only:
            Self + outgoing aggregation

        no_self:
            Incoming + outgoing aggregation
            without self-node transformation
    """

    def __init__(
        self,
        in_features,
        out_features,
        variant,
    ):
        super().__init__()

        self.variant = variant

        self.self_linear = None
        self.incoming_linear = None
        self.outgoing_linear = None

        if variant != "no_self":
            self.self_linear = nn.Linear(
                in_features,
                out_features,
            )

        if variant != "outgoing_only":
            self.incoming_linear = nn.Linear(
                in_features,
                out_features,
                bias=False,
            )

        if variant != "incoming_only":
            self.outgoing_linear = nn.Linear(
                in_features,
                out_features,
                bias=False,
            )

    def forward(
        self,
        x,
        incoming_adj,
        outgoing_adj,
    ):
        output = None

        # Self-node information
        if self.self_linear is not None:
            output = self.self_linear(x)

        # Incoming-neighbour information
        if self.incoming_linear is not None:
            incoming_features = incoming_adj @ x

            incoming_part = self.incoming_linear(
                incoming_features
            )

            output = (
                incoming_part
                if output is None
                else output + incoming_part
            )

        # Outgoing-neighbour information
        if self.outgoing_linear is not None:
            outgoing_features = outgoing_adj @ x

            outgoing_part = self.outgoing_linear(
                outgoing_features
            )

            output = (
                outgoing_part
                if output is None
                else output + outgoing_part
            )

        if output is None:
            raise RuntimeError(
                "Graph convolution has no active transformation."
            )

        return output


# ============================================================
# ABLATION ENCODER
# ============================================================

class AblationDirectedEncoder(nn.Module):

    def __init__(
        self,
        in_features,
        hidden=64,
        embedding=32,
        dropout=0.2,
        variant="full",
    ):
        super().__init__()

        self.gc1 = AblationGraphConvolution(
            in_features,
            hidden,
            variant,
        )

        self.gc2 = AblationGraphConvolution(
            hidden,
            embedding,
            variant,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        incoming_adj,
        outgoing_adj,
    ):
        h = self.gc1(
            x,
            incoming_adj,
            outgoing_adj,
        )

        h = F.relu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        z = self.gc2(
            h,
            incoming_adj,
            outgoing_adj,
        )

        return z


# ============================================================
# SYMMETRIC STRUCTURE DECODER
# ============================================================

class SymmetricStructureDecoder(nn.Module):
    """
    Symmetric bilinear decoder.

    Uses a trainable matrix W but constrains it to be symmetric:

        W_sym = (W + W.T) / 2

        logits = Z @ W_sym @ Z.T

    Therefore:

        logits[i, j] = logits[j, i]
    """

    def __init__(self, embedding):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(embedding, embedding)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):

        symmetric_weight = (
            self.weight + self.weight.T
        ) / 2

        return (
            z
            @ symmetric_weight
            @ z.T
        )


# ============================================================
# ABLATION MODEL
# ============================================================

class AblationDirectedDOMINANT(nn.Module):
    """
    DirectedDOMINANT with configurable encoder/decoder.

    Variants:

        incoming_only
        outgoing_only
        no_self
        symmetric_decoder
        attribute_only
        structure_only
    """

    def __init__(
        self,
        in_features,
        variant,
        hidden=64,
        embedding=32,
        dropout=0.2,
    ):
        super().__init__()

        self.variant = variant

        # Symmetric decoder experiment uses the original
        # full directed encoder.
        if variant == "symmetric_decoder":

            self.encoder = DirectedGCNEncoder(
                in_features=in_features,
                hidden=hidden,
                embedding=embedding,
                dropout=dropout,
            )

            self.structure_decoder = (
                SymmetricStructureDecoder(embedding)
            )

            self.register_parameter(
                "structure_weight",
                None,
            )

        else:

            self.encoder = AblationDirectedEncoder(
                in_features=in_features,
                hidden=hidden,
                embedding=embedding,
                dropout=dropout,
                variant=variant,
            )

            # Original asymmetric bilinear decoder.
            self.structure_weight = nn.Parameter(
                torch.empty(embedding, embedding)
            )

            nn.init.xavier_uniform_(
                self.structure_weight
            )

            self.structure_decoder = None

        # Attribute reconstruction decoder.
        self.attribute_decoder = nn.Sequential(
            nn.Linear(embedding, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_features),
        )

    def structure_logits(self, z):

        # Symmetric bilinear decoder
        if self.variant == "symmetric_decoder":

            # Assign to a local variable so that
            # Pylance can correctly narrow the type.
            decoder = self.structure_decoder

            if decoder is None:
                raise RuntimeError(
                    "Symmetric structure decoder "
                    "was not initialized."
                )

            return decoder(z)

        # Original asymmetric bilinear decoder
        if self.structure_weight is None:
            raise RuntimeError(
                "Asymmetric structure weight "
                "was not initialized."
            )

        transformed = z @ self.structure_weight

        return transformed @ z.T

    def forward(
        self,
        x,
        incoming_adj,
        outgoing_adj,
    ):

        z = self.encoder(
            x,
            incoming_adj,
            outgoing_adj,
        )

        structure_logits = self.structure_logits(z)

        reconstructed_adj = torch.sigmoid(
            structure_logits
        )

        reconstructed_x = self.attribute_decoder(z)

        return reconstructed_x, reconstructed_adj, z


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    variant,
    in_features,
    device,
):

    # Original full model: no architectural changes.
    if variant == "full":

        model = DirectedDOMINANT(
            in_features=in_features,
            hidden=64,
            embedding=32,
            dropout=0.2,
        )

    else:

        model = AblationDirectedDOMINANT(
            in_features=in_features,
            variant=variant,
            hidden=64,
            embedding=32,
            dropout=0.2,
        )

    return model.to(device)


# ============================================================
# ONE EXPERIMENT
# ============================================================

def train_one_run(
    data,
    variant,
    seed,
    edge_weight,
    epochs,
    learning_rate,
    weight_decay,
    alpha,
    device,
    results_dir,
):

    print("\n" + "=" * 70)
    print(
        f"Variant: {variant} | "
        f"Seed: {seed} | "
        f"Edge weight: {edge_weight}"
    )
    print("=" * 70)

    set_seed(seed)

    x = data["x"].to(device)
    adj = data["adj"].to(device)

    # Labels are used only for evaluation.
    labels = data["labels"].cpu().numpy()

    # Normalize incoming and outgoing adjacency.
    incoming_adj, outgoing_adj = (
        normalize_directed_adjacency(adj)
    )

    incoming_adj = incoming_adj.to(device)
    outgoing_adj = outgoing_adj.to(device)

    num_nodes = x.size(0)

    # Exclude self-connections from structural loss.
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

    model = create_model(
        variant=variant,
        in_features=x.size(1),
        device=device,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(epochs):

        model.train()
        optimizer.zero_grad()

        x_hat, adj_hat, z = model(
            x,
            incoming_adj,
            outgoing_adj,
        )

        # Attribute reconstruction loss.
        attribute_loss = F.mse_loss(
            x_hat,
            x,
        )

        # Structural reconstruction loss.
        structure_logits = model.structure_logits(z)

        structure_logits = structure_logits[
            off_diagonal_mask
        ]

        structure_loss = F.binary_cross_entropy_with_logits(
            structure_logits,
            target_adj,
            pos_weight=pos_weight,
        )

        # Select training objective.
        if variant == "attribute_only":

            total_loss = attribute_loss

        elif variant == "structure_only":

            total_loss = structure_loss

        else:

            total_loss = (
                attribute_loss + structure_loss
            )

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
                f"Epoch {epoch + 1}/{epochs} | "
                f"Total: {total_loss.item():.6f} | "
                f"Attr: {attribute_loss.item():.6f} | "
                f"Struct: {structure_loss.item():.6f}"
            )

    # ========================================================
    # EVALUATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        x_hat, adj_hat, z = model(
            x,
            incoming_adj,
            outgoing_adj,
        )

        combined_scores, attr_scores, struct_scores = (
            directed_dominant_scores(
                x,
                adj,
                x_hat,
                adj_hat,
                alpha=alpha,
            )
        )

    combined_scores = combined_scores.cpu().numpy()
    attr_scores = attr_scores.cpu().numpy()
    struct_scores = struct_scores.cpu().numpy()

    # Combined metrics.
    combined_auc = roc_auc_score(
        labels,
        combined_scores,
    )

    combined_ap = average_precision_score(
        labels,
        combined_scores,
    )

    # Attribute metrics.
    attribute_auc = roc_auc_score(
        labels,
        attr_scores,
    )

    attribute_ap = average_precision_score(
        labels,
        attr_scores,
    )

    # Structural metrics.
    structural_auc = roc_auc_score(
        labels,
        struct_scores,
    )

    structural_ap = average_precision_score(
        labels,
        struct_scores,
    )

    # Primary score for each variant.
    if variant == "attribute_only":

        primary_scores = attr_scores

    elif variant == "structure_only":

        primary_scores = struct_scores

    else:

        primary_scores = combined_scores

    primary_auc = roc_auc_score(
        labels,
        primary_scores,
    )

    primary_ap = average_precision_score(
        labels,
        primary_scores,
    )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    run_name = (
        f"{variant}_seed_{seed}_weight_{edge_weight}"
    )

    run_dir = results_dir / run_name

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Training history.
    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv",
        index=False,
    )

    # Node predictions.
    predictions = pd.DataFrame({
        "node_id": data["node_ids"],
        "product_id": data["product_ids"],
        "label": labels,
        "combined_score": combined_scores,
        "attribute_score": attr_scores,
        "structural_score": struct_scores,
        "primary_score": primary_scores,
    })

    predictions.to_csv(
        run_dir / "predictions.csv",
        index=False,
    )

    # Model checkpoint.
    torch.save(
        model.state_dict(),
        run_dir / "model.pt",
    )

    result = {
        "variant": variant,
        "seed": seed,
        "edge_weight": edge_weight,
        "epochs": epochs,

        "final_loss": history[-1]["total_loss"],
        "final_attribute_loss": history[-1]["attribute_loss"],
        "final_structure_loss": history[-1]["structure_loss"],

        "primary_roc_auc": primary_auc,
        "primary_ap": primary_ap,

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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="DirectedDOMINANT ablation study"
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
        default=[20],
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

    variants = [
        "full",
        "incoming_only",
        "outgoing_only",
        "no_self",
        "symmetric_decoder",
        "attribute_only",
        "structure_only",
    ]

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # DATASET
    # ========================================================

    data = load_amazon_fail()

    print("\nDataset:")
    print("Nodes:", data["x"].shape[0])
    print("Features:", data["x"].shape[1])
    print("Directed edges:", data["edge_index"].shape[1])
    print(
        "Anomalies:",
        (data["labels"] == 1).sum().item()
    )

    # ========================================================
    # RESULTS DIRECTORY
    # ========================================================

    project_root = Path(__file__).resolve().parents[2]

    results_dir = (
        project_root / "results_amazon_ablation"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    # ========================================================
    # EXPERIMENT LOOP
    # ========================================================

    for seed in args.seeds:

        for edge_weight in args.edge_weights:

            for variant in variants:

                # Attribute-only training does not use
                # edge_weight. Run it once per seed.
                if (
                    variant == "attribute_only"
                    and edge_weight != args.edge_weights[0]
                ):
                    continue

                result = train_one_run(
                    data=data,
                    variant=variant,
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

    # ========================================================
    # SAVE INDIVIDUAL RESULTS
    # ========================================================

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        results_dir / "all_results.csv",
        index=False,
    )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    metrics = [
        "primary_roc_auc",
        "primary_ap",
        "combined_roc_auc",
        "combined_ap",
        "attribute_roc_auc",
        "attribute_ap",
        "structural_roc_auc",
        "structural_ap",
    ]

    summary = results_df.groupby(
        ["variant", "edge_weight"]
    )[metrics].agg(
        ["mean", "std"]
    )

    summary.to_csv(
        results_dir / "summary.csv",
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)

    print(
        results_df.to_string(index=False)
    )

    print("\nAggregate results:")

    print(
        summary.to_string()
    )

    print("\nResults saved to:", results_dir)


if __name__ == "__main__":
    main()
