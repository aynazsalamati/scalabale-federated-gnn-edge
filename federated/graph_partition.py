import torch
from torch_geometric.data import Data


def partition_graph(data, num_clients):

    """
    Split graph nodes among clients
    """

    num_nodes = data.num_nodes

    node_indices = torch.randperm(num_nodes)

    split_size = num_nodes // num_clients

    client_subgraphs = []

    for i in range(num_clients):

        start = i * split_size

        if i == num_clients - 1:
            end = num_nodes
        else:
            end = (i + 1) * split_size

        client_nodes = node_indices[start:end]

        # Create node mask
        node_mask = torch.zeros(
            num_nodes,
            dtype=torch.bool
        )

        node_mask[client_nodes] = True

        # Filter edges
        edge_mask = (
            node_mask[data.edge_index[0]] &
            node_mask[data.edge_index[1]]
        )

        sub_edge_index = data.edge_index[
            :,
            edge_mask
        ]

        # Create subgraph
        subgraph = Data(
            x=data.x,
            edge_index=sub_edge_index,
            y=data.y,
            train_mask=data.train_mask,
            test_mask=data.test_mask
        )

        client_subgraphs.append(subgraph)

    return client_subgraphs