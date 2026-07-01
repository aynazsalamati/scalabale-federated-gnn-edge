import numpy as np
import torch
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils import subgraph
from torch_geometric.data import Data


def set_fallback_masks(data, seed=42, train_ratio=0.60, val_ratio=0.20):
    rng = np.random.default_rng(seed)
    idx = np.arange(data.num_nodes)
    rng.shuffle(idx)

    train_end = int(train_ratio * data.num_nodes)
    val_end = int((train_ratio + val_ratio) * data.num_nodes)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(idx[:train_end])] = True
    val_mask[torch.tensor(idx[train_end:val_end])] = True
    test_mask[torch.tensor(idx[val_end:])] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def ensure_masks(data, seed=42):
    for name in ["train_mask", "val_mask", "test_mask"]:
        if not hasattr(data, name) or getattr(data, name) is None or int(getattr(data, name).sum()) == 0:
            return set_fallback_masks(data, seed=seed)
    return data


def sample_induced_subgraph(data, num_sample_nodes=8000, seed=42, preserve_masks=True):
    if num_sample_nodes is None or num_sample_nodes >= data.num_nodes:
        return ensure_masks(data, seed=seed)

    rng = np.random.default_rng(seed)
    y = data.y.cpu().numpy()
    classes = np.unique(y)
    selected = []

    per_class = max(1, num_sample_nodes // len(classes))
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > 0:
            take = min(per_class, len(cls_idx))
            selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())

    selected = list(set(selected))
    if len(selected) < num_sample_nodes:
        remaining = np.setdiff1d(np.arange(data.num_nodes), np.array(selected))
        extra = min(num_sample_nodes - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())

    selected = np.array(selected[:num_sample_nodes])
    selected.sort()
    subset = torch.tensor(selected, dtype=torch.long)

    edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

    sampled = Data(x=data.x[subset], edge_index=edge_index, y=data.y[subset])
    if preserve_masks:
        for name in ["train_mask", "val_mask", "test_mask"]:
            if hasattr(data, name) and getattr(data, name) is not None:
                setattr(sampled, name, getattr(data, name)[subset])

    sampled = ensure_masks(sampled, seed=seed)
    sampled.original_num_nodes = data.num_nodes
    sampled.sampled_num_nodes = sampled.num_nodes
    return sampled


def load_graph_dataset(dataset_name, root="data", reddit_sample_nodes=8000, seed=42):
    name = dataset_name.lower()

    if name == "cora":
        dataset = Planetoid(root=f"{root}/Cora", name="Cora")
        data = ensure_masks(dataset[0], seed=seed)
        return dataset, data

    if name == "reddit":
        dataset = Reddit(root=f"{root}/Reddit")
        data = sample_induced_subgraph(dataset[0], num_sample_nodes=reddit_sample_nodes, seed=seed)
        return dataset, data

    raise ValueError(f"Unknown dataset: {dataset_name}")
