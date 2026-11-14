
from pathlib import Path
import csv

import networkx as nx
import numpy as np
import torch


def load_amazon_fail(data_dir=None):
    """
    Load the Amazon Fail attributed directed graph.

    Returns:
        dict containing:
            x:             Node feature matrix [N, F]
            adj:           Directed adjacency matrix [N, N]
            edge_index:    Directed edges [2, E]
            labels:        Ground-truth labels [N]
            node_ids:      Original numeric graph node IDs
            product_ids:   Original Amazon product IDs (ISBNs)
            feature_names: Names of retained node features
            graph:         Original NetworkX DiGraph

    Label convention:
        0 = normal
        1 = anomalous
    """

    # --------------------------------------------------
    # 1. Locate dataset files
    # --------------------------------------------------

    if data_dir is None:
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "datasets" / "AmazonFail"
    else:
        data_dir = Path(data_dir)

    graph_path = data_dir / "AmazonFailNumeric.graphml"
    labels_path = data_dir / "AmazonFailNumeric.true"
    mapping_path = data_dir / "AmazonFailNumericIDS.csv"

    for path in (graph_path, labels_path, mapping_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    # --------------------------------------------------
    # 2. Load original directed graph
    # --------------------------------------------------

    graph = nx.read_graphml(graph_path)

    if not graph.is_directed():
        raise ValueError("Expected a directed GraphML graph.")

    # GraphML stores node IDs as strings.
    # Sort numerically to establish a consistent node order.
    node_ids = sorted(graph.nodes(), key=lambda node: int(node))

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate node IDs found.")

    node_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    # --------------------------------------------------
    # 3. Load ground-truth labels
    # --------------------------------------------------

    label_map = {}

    with open(labels_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split(";")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid label format at line {line_number}: {line}"
                )

            node_id, label = parts
            label = int(label)

            if label not in (0, 1):
                raise ValueError(
                    f"Unexpected label {label} for node {node_id}"
                )

            if node_id in label_map:
                raise ValueError(f"Duplicate label for node {node_id}")

            label_map[node_id] = label

    # --------------------------------------------------
    # 4. Load ISBN-to-node-ID mapping
    # --------------------------------------------------

    product_id_map = {}

    with open(
        mapping_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.reader(file, delimiter=";")

        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue

            if len(row) != 2:
                raise ValueError(
                    f"Invalid mapping format at line {line_number}: {row}"
                )

            product_id, node_id = row[0].strip(), row[1].strip()

            if node_id in product_id_map:
                raise ValueError(
                    f"Duplicate mapping for node {node_id}"
                )

            product_id_map[node_id] = product_id

    # --------------------------------------------------
    # 5. Validate node IDs across all three files
    # --------------------------------------------------

    graph_ids = set(node_ids)

    if graph_ids != set(label_map):
        raise ValueError("Graph node IDs and label IDs do not match.")

    if graph_ids != set(product_id_map):
        raise ValueError("Graph node IDs and mapping IDs do not match.")

    # --------------------------------------------------
    # 6. Extract node features
    # --------------------------------------------------

    # rnmlcCluster is constant across all nodes, so it
    # cannot help distinguish nodes. Exclude it.
    excluded_features = {"rnmlcCluster"}

    all_feature_names = list(graph.nodes[node_ids[0]].keys())

    feature_names = [
        name for name in all_feature_names
        if name not in excluded_features
    ]

    if not feature_names:
        raise ValueError("No usable node features found.")

    feature_rows = []

    for node_id in node_ids:
        attributes = graph.nodes[node_id]
        row = []

        for feature_name in feature_names:
            if feature_name not in attributes:
                raise ValueError(
                    f"Missing feature {feature_name} for node {node_id}"
                )

            try:
                value = float(attributes[feature_name])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Non-numeric feature {feature_name} "
                    f"for node {node_id}"
                )

            if not np.isfinite(value):
                raise ValueError(
                    f"Non-finite value in {feature_name} "
                    f"for node {node_id}"
                )

            row.append(value)

        feature_rows.append(row)

    x = torch.tensor(
        np.asarray(feature_rows, dtype=np.float32),
        dtype=torch.float32
    )

    # --------------------------------------------------
    # 7. Construct directed adjacency matrix
    # --------------------------------------------------

    num_nodes = len(node_ids)

    adj = torch.zeros(
        (num_nodes, num_nodes),
        dtype=torch.float32
    )

    edge_pairs = []

    for source, target in graph.edges():
        source_index = node_to_index[source]
        target_index = node_to_index[target]

        adj[source_index, target_index] = 1.0

        edge_pairs.append((source_index, target_index))

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long
        ).T.contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # --------------------------------------------------
    # 8. Align labels and product IDs with node order
    # --------------------------------------------------

    labels = torch.tensor(
        [label_map[node_id] for node_id in node_ids],
        dtype=torch.long
    )

    product_ids = [
        product_id_map[node_id]
        for node_id in node_ids
    ]

    # --------------------------------------------------
    # 9. Final consistency checks
    # --------------------------------------------------

    if x.shape[0] != num_nodes:
        raise RuntimeError("Feature and node counts differ.")

    if adj.shape != (num_nodes, num_nodes):
        raise RuntimeError("Invalid adjacency matrix shape.")

    if labels.shape[0] != num_nodes:
        raise RuntimeError("Label and node counts differ.")

    if edge_index.shape[0] != 2:
        raise RuntimeError("Invalid edge_index shape.")

    if not torch.equal(
        adj.sum().long(),
        torch.tensor(len(edge_pairs))
    ):
        raise RuntimeError("Adjacency and edge counts differ.")

    return {
        "x": x,
        "adj": adj,
        "edge_index": edge_index,
        "labels": labels,
        "node_ids": node_ids,
        "product_ids": product_ids,
        "feature_names": feature_names,
        "graph": graph,
    }
