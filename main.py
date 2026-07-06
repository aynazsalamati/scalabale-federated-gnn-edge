import os
import copy
import csv
import pickle
import random
import numpy as np
import torch

from models.gnn_model import LocalGNN

from federated.client import FederatedClient
from federated.server import FederatedServer
from federated.scheduler import AdaptiveScheduler
from federated.topology import TopologyManager
from federated.non_iid_partition import non_iid_partition
from federated.metrics import MetricsTracker
from federated.datasets import load_graph_dataset

from federated.visualization import (
    plot_multi_accuracy,
    plot_multi_loss,
    plot_multi_participation,
    plot_multi_communication,
    plot_metric_grid
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_global_model(model, data, device="cpu"):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

        if (
            hasattr(data, "test_mask")
            and data.test_mask is not None
            and int(data.test_mask.sum()) > 0
        ):
            test_mask = data.test_mask
        else:
            test_mask = torch.ones(
                data.num_nodes,
                dtype=torch.bool,
                device=device
            )

        correct = (pred[test_mask] == data.y[test_mask]).sum()
        acc = int(correct) / max(int(test_mask.sum()), 1)

    return acc


def create_clients(
    client_subgraphs,
    dataset,
    hidden_dim=64,
    lr=0.01,
    device="cpu"
):
    clients = []

    for client_id, subgraph in enumerate(client_subgraphs):
        client = FederatedClient(
            client_id=client_id,
            data=subgraph,
            input_dim=dataset.num_features,
            hidden_dim=hidden_dim,
            output_dim=dataset.num_classes,
            lr=lr,
            device=device
        )
        clients.append(client)

    return clients


def get_experiments():

    experiments = {
        "FedAvg-GNN": {
            "adaptive": False,
            "topology_aware": False,
            "threshold": 0.0,
            "availability": 1.0,
            "min_send_probability": 1.0
        },

        "FedAvg-GNN + Adaptive Scheduling": {
            "adaptive": True,
            "topology_aware": False,
            "threshold": 0.0025,
            "availability": 0.85,
            "min_send_probability": 0.45
        },

        "SF-GNN": {
            "adaptive": True,
            "topology_aware": True,
            "threshold": 0.0020,
            "availability": 0.90,
            "min_send_probability": 0.55
        }
    }

    return experiments


def run_single_method(
    method_name,
    config,
    dataset,
    data,
    num_clients=20,
    rounds=50,
    hidden_dim=64,
    local_epochs=3,
    learning_rate=0.01,
    seed=42,
    alpha=0.5,
    device="cpu",
    verbose=False
):
    set_seed(seed)

    if verbose:
        print("\n==============================")
        print(f"Method: {method_name}")
        print(f"Seed: {seed}")
        print(f"Clients: {num_clients}")
        print("==============================")

    client_subgraphs = non_iid_partition(
        data=data.cpu(),
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
        verbose=False
    )

    global_model = LocalGNN(
        input_dim=dataset.num_features,
        hidden_dim=hidden_dim,
        output_dim=dataset.num_classes
    ).to(device)

    server = FederatedServer(global_model, device=device)

    scheduler = AdaptiveScheduler(
        threshold=config["threshold"],
        availability=config["availability"],
        min_send_probability=config["min_send_probability"],
        verbose=False
    )

    topology = TopologyManager(num_clients=num_clients)
    metrics = MetricsTracker()

    clients = create_clients(
        client_subgraphs=client_subgraphs,
        dataset=dataset,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        device=device
    )

    initial_weights = copy.deepcopy(global_model.state_dict())

    for client in clients:
        client.set_weights(initial_weights)

    for round_num in range(rounds):
        client_updates = []
        round_losses = []

        for client in clients:
            old_weights = copy.deepcopy(client.get_weights())

            loss = client.train(epochs=local_epochs)

            # If a client has no local train_mask samples, skip supervised update
            # to avoid train/test leakage.
            if loss is None:
                continue

            round_losses.append(loss)
            metrics.log_client_loss(client.client_id, loss)

            new_weights = copy.deepcopy(client.get_weights())

            if config["adaptive"]:
                should_send = scheduler.should_send_update(
                    old_weights,
                    new_weights
                )
            else:
                should_send = True

            if should_send:
                if config["topology_aware"]:
                    importance = topology.get_client_weight(client.client_id)
                else:
                    importance = 1.0

                client_updates.append((new_weights, importance))

        if len(client_updates) > 0:
            global_weights = server.aggregate(client_updates)

            for client in clients:
                client.set_weights(global_weights)

        acc = evaluate_global_model(server.global_model, data, device=device)
        participation_rate = len(client_updates) / num_clients
        communication_cost = len(client_updates)

        if len(round_losses) > 0:
            avg_loss = sum(round_losses) / len(round_losses)
        else:
            avg_loss = metrics.round_losses[-1] if metrics.round_losses else 0.0

        metrics.log_round(
            loss=avg_loss,
            accuracy=acc,
            participation_rate=participation_rate,
            communication_cost=communication_cost
        )

        if verbose:
            print(
                f"Round {round_num + 1:03d} | "
                f"Acc: {acc:.4f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Part: {participation_rate:.2f} | "
                f"Comm: {communication_cost}"
            )

    communication = metrics.get_communication_costs()

    history = {
        "accuracy": metrics.get_accuracies(),
        "loss": metrics.get_losses(),
        "participation": metrics.get_participation_rates(),
        "communication": communication,
        "cumulative_communication": list(np.cumsum(communication)),
        "client_losses": metrics.get_client_losses()
    }

    return history


def average_histories(histories):
    metric_keys = [
        "accuracy",
        "loss",
        "participation",
        "communication",
        "cumulative_communication"
    ]

    averaged = {}

    for key in metric_keys:
        min_len = min(len(history[key]) for history in histories)
        values = np.array([history[key][:min_len] for history in histories])

        averaged[key] = values.mean(axis=0).tolist()
        averaged[f"{key}_std"] = values.std(axis=0).tolist()

    return averaged


def run_method_over_seeds(
    method_name,
    config,
    dataset,
    data,
    num_clients,
    rounds,
    seeds,
    alpha=0.5,
    device="cpu",
    verbose=False
):
    histories = []

    for seed in seeds:
        history = run_single_method(
            method_name=method_name,
            config=config,
            dataset=dataset,
            data=data,
            num_clients=num_clients,
            rounds=rounds,
            seed=seed,
            alpha=alpha,
            device=device,
            verbose=verbose
        )
        histories.append(history)

    return average_histories(histories)


def run_all_methods(
    dataset,
    data,
    num_clients=20,
    rounds=50,
    seeds=None,
    alpha=0.5,
    device="cpu",
    verbose=False
):
    if seeds is None:
        seeds = [42, 43, 44]

    experiments = get_experiments()
    all_results = {}

    for method_name, config in experiments.items():
        print(
            f"Running {method_name} | "
            f"Clients: {num_clients}"
        )

        result = run_method_over_seeds(
            method_name=method_name,
            config=config,
            dataset=dataset,
            data=data,
            num_clients=num_clients,
            rounds=rounds,
            seeds=seeds,
            alpha=alpha,
            device=device,
            verbose=verbose
        )

        all_results[method_name] = result

    return all_results


def print_final_summary(dataset_name, results):
    print(f"\n===== FINAL SUMMARY: {dataset_name} =====")

    for method_name, history in results.items():
        final_acc = history["accuracy"][-1]
        final_loss = history["loss"][-1]
        total_comm = history["cumulative_communication"][-1]
        avg_part = np.mean(history["participation"])

        print(
            f"{method_name:40s} | "
            f"Acc: {final_acc:.4f} | "
            f"Loss: {final_loss:.4f} | "
            f"Total Comm: {total_comm:.1f} | "
            f"Avg Part: {avg_part:.2f}"
        )


def save_final_summary_csv(dataset_name, results, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_name}_summary.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset",
            "Method",
            "Final Accuracy",
            "Final Loss",
            "Average Participation",
            "Total Updates Sent"
        ])

        for method_name, history in results.items():
            writer.writerow([
                dataset_name,
                method_name,
                history["accuracy"][-1],
                history["loss"][-1],
                float(np.mean(history["participation"])),
                history["cumulative_communication"][-1]
            ])

    return csv_path


def make_cache_name(
    dataset_name,
    num_clients,
    rounds,
    seeds,
    alpha,
    reddit_sample_nodes,
    ogbn_arxiv_sample_nodes,
    planetoid_split
):
    seeds_text = "_".join(str(seed) for seed in seeds)
    reddit_text = "full" if reddit_sample_nodes is None else str(reddit_sample_nodes)
    arxiv_text = "full" if ogbn_arxiv_sample_nodes is None else str(ogbn_arxiv_sample_nodes)

    return (
        f"{dataset_name.lower()}_"
        f"clients_{num_clients}_"
        f"rounds_{rounds}_"
        f"seeds_{seeds_text}_"
        f"alpha_{alpha}_"
        f"planetoid_{planetoid_split}_"
        f"reddit_{reddit_text}_"
        f"arxiv_{arxiv_text}.pkl"
    )


def load_or_run(
    dataset_name,
    dataset,
    data,
    num_clients,
    rounds,
    seeds,
    alpha,
    reddit_sample_nodes,
    ogbn_arxiv_sample_nodes,
    planetoid_split,
    device,
    cache_dir="outputs/cache"
):
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(
        cache_dir,
        make_cache_name(
            dataset_name=dataset_name,
            num_clients=num_clients,
            rounds=rounds,
            seeds=seeds,
            alpha=alpha,
            reddit_sample_nodes=reddit_sample_nodes,
            ogbn_arxiv_sample_nodes=ogbn_arxiv_sample_nodes,
            planetoid_split=planetoid_split
        )
    )

    if os.path.exists(cache_path):
        print(f"Loading cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    results = run_all_methods(
        dataset=dataset,
        data=data,
        num_clients=num_clients,
        rounds=rounds,
        seeds=seeds,
        alpha=alpha,
        device=device,
        verbose=False
    )

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    return results


def get_accuracy_ylim(dataset_name):
    name = dataset_name.lower()

    if "reddit" in name:
        return (0.30, 0.97)
    if "arxiv" in name:
        return (0.20, 0.85)
    if "pubmed" in name:
        return (0.20, 0.90)
    return (0.10, 0.90)


if __name__ == "__main__":
    set_seed(42)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Best practical Q1-oriented combination:
    # Cora + CiteSeer + PubMed + ogbn-arxiv + sampled Reddit
    DATASETS = [
        "Cora",
        "CiteSeer",
        "PubMed",
        "ogbn-arxiv",
        "Reddit"
    ]

    # Reddit is kept as a sampled induced subgraph for computational feasibility.
    REDDIT_SAMPLE_NODES = 8000

    # Use None for full ogbn-arxiv. If your system is weak, use 20000 or 50000,
    # but report it as sampled ogbn-arxiv in the paper.
    OGBN_ARXIV_SAMPLE_NODES = None

    # Planetoid split options: "public", "full", "geom-gcn", "random".
    # "full" is more stable for federated simulations with many clients.
    PLANETOID_SPLIT = "full"

    ROUNDS = 50
    SEEDS = [42, 43, 44]
    ALPHA = 0.5
    SINGLE_NUM_CLIENTS = 20

    RUN_GRID = True

    GRID_CLIENT_SCENARIOS = [
        5,
        10,
        15,
        20,
        30,
        50
    ]

    all_csv_paths = []

    for dataset_name in DATASETS:
        print("\n==============================")
        print(f"Dataset: {dataset_name}")
        print("==============================")

        dataset, data = load_graph_dataset(
            dataset_name=dataset_name,
            reddit_sample_nodes=REDDIT_SAMPLE_NODES,
            ogbn_arxiv_sample_nodes=OGBN_ARXIV_SAMPLE_NODES,
            seed=42,
            planetoid_split=PLANETOID_SPLIT
        )

        print(
            f"Dataset Loaded: {dataset.name} | "
            f"Nodes: {data.num_nodes} | "
            f"Edges: {data.num_edges} | "
            f"Features: {dataset.num_features} | "
            f"Classes: {dataset.num_classes} | "
            f"Train: {int(data.train_mask.sum())} | "
            f"Val: {int(data.val_mask.sum())} | "
            f"Test: {int(data.test_mask.sum())}"
        )

        safe_dataset_name = dataset_name.replace("/", "_").replace(" ", "_")
        output_dir = f"outputs/{safe_dataset_name}"

        results = load_or_run(
            dataset_name=dataset_name,
            dataset=dataset,
            data=data,
            num_clients=SINGLE_NUM_CLIENTS,
            rounds=ROUNDS,
            seeds=SEEDS,
            alpha=ALPHA,
            reddit_sample_nodes=REDDIT_SAMPLE_NODES,
            ogbn_arxiv_sample_nodes=OGBN_ARXIV_SAMPLE_NODES,
            planetoid_split=PLANETOID_SPLIT,
            device=DEVICE
        )

        print_final_summary(dataset_name, results)

        csv_path = save_final_summary_csv(
            dataset_name=safe_dataset_name,
            results=results,
            output_dir=output_dir
        )
        all_csv_paths.append(csv_path)
        print(f"Summary CSV saved to: {csv_path}")

        plot_multi_accuracy(results, output_dir=output_dir)
        plot_multi_loss(results, output_dir=output_dir)
        plot_multi_participation(results, output_dir=output_dir)
        plot_multi_communication(results, output_dir=output_dir)

        if RUN_GRID:
            panel_results = {}

            for n_clients in GRID_CLIENT_SCENARIOS:
                print(
                    f"\n######## Dataset: {dataset_name} | "
                    f"Scenario: {n_clients} Clients ########"
                )

                scenario_results = load_or_run(
                    dataset_name=f"{dataset_name}_grid",
                    dataset=dataset,
                    data=data,
                    num_clients=n_clients,
                    rounds=ROUNDS,
                    seeds=[42],
                    alpha=ALPHA,
                    reddit_sample_nodes=REDDIT_SAMPLE_NODES,
                    ogbn_arxiv_sample_nodes=OGBN_ARXIV_SAMPLE_NODES,
                    planetoid_split=PLANETOID_SPLIT,
                    device=DEVICE
                )

                panel_results[f"{dataset_name} - {n_clients} Clients"] = scenario_results

            plot_metric_grid(
                panel_results=panel_results,
                metric_key="accuracy",
                title="Global Accuracy Across Client Scenarios",
                ylabel="Accuracy",
                save_path=f"{output_dir}/accuracy_grid.png",
                smooth=True,
                ylim=get_accuracy_ylim(dataset_name)
            )

            plot_metric_grid(
                panel_results=panel_results,
                metric_key="loss",
                title="Training Loss Across Client Scenarios",
                ylabel="Loss",
                save_path=f"{output_dir}/loss_grid.png",
                smooth=True,
                ylim=(0.0, 1.20)
            )

            plot_metric_grid(
                panel_results=panel_results,
                metric_key="cumulative_communication",
                title="Cumulative Communication Cost Across Client Scenarios",
                ylabel="Cumulative Updates Sent",
                save_path=f"{output_dir}/communication_grid.png",
                smooth=False
            )

    print("\nAll dataset summaries:")
    for path in all_csv_paths:
        print(path)
