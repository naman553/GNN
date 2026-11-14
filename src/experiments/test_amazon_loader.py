import networkx as nx
from src.data.amazon_fail import load_amazon_fail

data = load_amazon_fail()

print("\n========== AMAZON FAIL DATASET ==========")

print("Nodes:", data["x"].shape[0])
print("Features:", data["x"].shape[1])
print("Edges:", data["edge_index"].shape[1])

print("\nTensor shapes:")
print("x:", data["x"].shape)
print("adj:", data["adj"].shape)
print("edge_index:", data["edge_index"].shape)
print("labels:", data["labels"].shape)

print("\nLabel distribution:")
print("Normal:", (data["labels"] == 0).sum().item())
print("Anomalous:", (data["labels"] == 1).sum().item())

print("\nFeature names:")
print(data["feature_names"])

print("\nGraph properties:")
print("Directed:", data["graph"].is_directed())
print("Self-loops:", sum(nx.number_of_selfloops(data["graph"]) for _ in [0]))
print("Adjacency symmetric:", data["adj"].equal(data["adj"].T))

print("\nFirst five node mappings:")
for i in range(5):
    print(
        data["node_ids"][i],
        "->",
        data["product_ids"][i],
        "| label:",
        data["labels"][i].item()
    )

print("\nAll checks completed.")
