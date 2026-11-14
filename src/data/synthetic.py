
import numpy as np
import torch
import networkx as nx


def make_attributed_graph(
    num_nodes=260,
    feature_dim=16,
    anomaly_fraction=0.08,
    seed=42
):
    """
    Generate a synthetic attributed graph.

    Includes:
    - Normal nodes
    - Attribute anomalies
    - Structural anomalies

    Returns:
        x: Node feature tensor
        adj: Adjacency matrix tensor
        edge_index: Bidirectional edge index
        labels: Overall node anomaly labels
        attribute_labels: Attribute anomaly labels
        structural_labels: Structural anomaly labels
        graph: NetworkX graph
        communities: Community assignments
    """

    if num_nodes < 4:
        raise ValueError("num_nodes must be at least 4.")

    if feature_dim < 1:
        raise ValueError("feature_dim must be positive.")

    if not 0 <= anomaly_fraction <= 1:
        raise ValueError("anomaly_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)

    n = num_nodes
    groups = 4

    # Assign nodes to communities
    community = np.arange(n) % groups

    # Generate community-specific feature distributions
    centers = rng.normal(
        0,
        1,
        (groups, feature_dim)
    )

    x = (
        centers[community]
        + rng.normal(0, 0.35, (n, feature_dim))
    ).astype(np.float32)

    # Generate an undirected graph
    g = nx.Graph()
    g.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):

            prob = (
                0.12
                if community[i] == community[j]
                else 0.008
            )

            if rng.random() < prob:
                g.add_edge(i, j)

    # Select anomalous nodes
    count = min(
        n,
        max(4, int(n * anomaly_fraction))
    )

    anomaly_nodes = rng.choice(
        n,
        count,
        replace=False
    )

    # Split anomalies into two groups
    n_attr = count // 2

    attribute_nodes = anomaly_nodes[:n_attr]
    structural_nodes = anomaly_nodes[n_attr:]

    # Attribute anomaly injection
    x[attribute_nodes] = rng.normal(
        4.0,
        1.0,
        (len(attribute_nodes), feature_dim)
    ).astype(np.float32)

    # Structural anomaly injection
    for u in structural_nodes:

        candidates = np.where(
            community != community[u]
        )[0]

        selected = rng.choice(
            candidates,
            size=min(5, len(candidates)),
            replace=False
        )

        for v in selected:
            g.add_edge(int(u), int(v))

    # Generate adjacency matrix
    adj = nx.to_numpy_array(g).astype(np.float32)

    np.fill_diagonal(adj, 0)

    # Generate bidirectional edge_index
    edges = np.array(
        list(g.edges),
        dtype=np.int64
    ).reshape(-1, 2)

    if len(edges) > 0:
        edge_index = np.concatenate(
            [edges.T, edges.T[::-1]],
            axis=1
        )
    else:
        edge_index = np.empty(
            (2, 0),
            dtype=np.int64
        )

    # Separate anomaly labels
    attribute_labels = np.zeros(n, dtype=np.int64)
    structural_labels = np.zeros(n, dtype=np.int64)

    attribute_labels[attribute_nodes] = 1
    structural_labels[structural_nodes] = 1

    # Overall anomaly labels
    labels = np.maximum(
        attribute_labels,
        structural_labels
    )

    return {
        "x": torch.tensor(
            x,
            dtype=torch.float32
        ),

        "adj": torch.tensor(
            adj,
            dtype=torch.float32
        ),

        "edge_index": torch.tensor(
            edge_index,
            dtype=torch.long
        ),

        "labels": labels,

        "attribute_labels": attribute_labels,

        "structural_labels": structural_labels,

        "graph": g,

        "communities": community
    }


def make_graph_collection(
    n_graphs=180,
    nodes_per_graph=24,
    feature_dim=8,
    seed=7
):
    """
    Generate a collection of normal and anomalous graphs.

    Normal graphs:
        Erdos-Renyi graphs with higher edge probability.

    Anomalous graphs:
        Attribute shift and structural hub injection.

    Returns:
        graphs: List of (node_features, adjacency_matrix)
        labels: Graph-level anomaly labels
    """

    if n_graphs < 2:
        raise ValueError("n_graphs must be at least 2.")

    if nodes_per_graph < 2:
        raise ValueError("nodes_per_graph must be at least 2.")

    if feature_dim < 1:
        raise ValueError("feature_dim must be positive.")

    rng = np.random.default_rng(seed)

    graphs = []

    # Generate and randomize graph-level labels
    n_anomalous = max(1, int(0.25 * n_graphs))

    anomaly_flags = np.zeros(
        n_graphs,
        dtype=np.int64
    )

    anomaly_flags[:n_anomalous] = 1

    rng.shuffle(anomaly_flags)

    for idx in range(n_graphs):

        anomalous = int(anomaly_flags[idx])

        # Generate graph structure
        prob = 0.07 if anomalous else 0.24

        graph_seed = int(
            rng.integers(1_000_000)
        )

        g = nx.erdos_renyi_graph(
            nodes_per_graph,
            prob,
            seed=graph_seed
        )

        # Generate node features
        x = rng.normal(
            0,
            1,
            (nodes_per_graph, feature_dim)
        ).astype(np.float32)

        if anomalous:

            # Attribute anomaly injection
            x += 1.7

            # Structural hub anomaly injection
            hub = int(
                rng.integers(nodes_per_graph)
            )

            for v in range(nodes_per_graph):

                if v != hub and rng.random() < 0.75:
                    g.add_edge(hub, v)

        # Generate adjacency matrix
        adj = nx.to_numpy_array(g).astype(np.float32)

        graphs.append((x, adj))

    labels = anomaly_flags.astype(np.int64)

    return graphs, labels


def make_temporal_data(
    t_steps=10,
    n=120,
    f=10,
    seed=46
):
    """
    Generate a sequence of graph snapshots.

    Normal snapshots:
        Community-based graph structure and node features.

    Final snapshot:
        Injected attribute and structural anomalies.

    Returns:
        xs: Node features across time
        ads: Adjacency matrices across time
        labels: Final snapshot node anomaly labels
    """

    if t_steps < 1:
        raise ValueError("t_steps must be at least 1.")

    if n < 2:
        raise ValueError("n must be at least 2.")

    if f < 1:
        raise ValueError("f must be positive.")

    rng = np.random.default_rng(seed)

    # Community assignments
    groups = np.arange(n) % 4

    # Community-specific feature distributions
    centers = rng.normal(
        0,
        0.7,
        (4, f)
    )

    base = (
        centers[groups]
        + rng.normal(0, 0.1, (n, f))
    )

    xs = []
    ads = []

    # Generate normal temporal snapshots
    for t in range(t_steps):

        x = base + rng.normal(
            0,
            0.04,
            (n, f)
        )

        a = np.zeros(
            (n, n),
            dtype=np.float32
        )

        for i in range(n):
            for j in range(i + 1, n):

                prob = (
                    0.10
                    if groups[i] == groups[j]
                    else 0.015
                )

                if rng.random() < prob:
                    a[i, j] = 1
                    a[j, i] = 1

        xs.append(x.astype(np.float32))
        ads.append(a)

    # Inject anomalies into the final snapshot
    labels = np.zeros(n, dtype=np.int64)

    n_anomalous = min(
        n,
        max(4, n // 12)
    )

    chosen = rng.choice(
        n,
        n_anomalous,
        replace=False
    )

    labels[chosen] = 1

    # Attribute anomalies
    xs[-1][chosen] += 2.8

    # Structural anomalies
    for u in chosen:

        candidates = np.delete(
            np.arange(n),
            u
        )

        selected = rng.choice(
            candidates,
            size=min(9, len(candidates)),
            replace=False
        )

        for v in selected:
            ads[-1][u, v] = 1
            ads[-1][v, u] = 1

    return (
        torch.tensor(
            np.stack(xs),
            dtype=torch.float32
        ),

        torch.tensor(
            np.stack(ads),
            dtype=torch.float32
        ),

        labels
    )
