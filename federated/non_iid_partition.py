import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def _safe_mask(data, name, subset):
    if hasattr(data, name):
        mask = getattr(data, name)
        if mask is not None:
            return mask[subset]
    return torch.zeros(len(subset), dtype=torch.bool)


def non_iid_partition(data, num_clients, alpha=0.5, seed=42, verbose=False):
    rng = np.random.default_rng(seed)
    labels = data.y.cpu().numpy()
    classes = np.unique(labels)
    client_nodes = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        counts = rng.multinomial(len(cls_idx), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            selected = cls_idx[start:start + count]
            client_nodes[client_id].extend(selected.tolist())
            start += count

    for empty_client in [i for i, nodes in enumerate(client_nodes) if len(nodes) == 0]:
        largest_client = max(range(num_clients), key=lambda idx: len(client_nodes[idx]))
        if len(client_nodes[largest_client]) > 1:
            client_nodes[empty_client].append(client_nodes[largest_client].pop())

    local_graphs = []
    for client_id, node_list in enumerate(client_nodes):
        subset = torch.tensor(sorted(set(node_list)), dtype=torch.long)
        edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        local_data = Data(
            x=data.x[subset],
            edge_index=edge_index,
            y=data.y[subset],
            train_mask=_safe_mask(data, "train_mask", subset),
            val_mask=_safe_mask(data, "val_mask", subset),
            test_mask=_safe_mask(data, "test_mask", subset)
        )
        local_data.global_node_indices = subset

        if verbose:
            print(f"Client {client_id} | Nodes: {local_data.num_nodes} | Edges: {local_data.num_edges}")

        local_graphs.append(local_data)

    return local_graphs
