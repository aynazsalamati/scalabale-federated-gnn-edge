import os
import numpy as np
import matplotlib.pyplot as plt


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 4,
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
    })


def make_output_dir(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)


def save_figure(save_path):
    if save_path is None:
        return

    output_dir = os.path.dirname(save_path)
    if output_dir:
        make_output_dir(output_dir)

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    if save_path.endswith(".png"):
        plt.savefig(
            save_path.replace(".png", ".pdf"),
            bbox_inches="tight"
        )


def smooth_curve(values, window=3):
    values = np.array(values, dtype=float)

    if window is None or window <= 1 or len(values) < window:
        return values

    smoothed = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(values[start:i + 1])))

    return np.array(smoothed)


def get_series(history, metric_key, smooth=True, window=3):
    values = history[metric_key]

    if smooth and metric_key in ["accuracy", "loss", "participation"]:
        values = smooth_curve(values, window=window)

    return np.array(values, dtype=float)


def get_method_styles():
    return {
        "FedAvg-GNN": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.0,
            "zorder": 5
        },
        "FedAvg-GNN + Topology": {
            "color": "#ff7f0e",
            "marker": "s",
            "linestyle": "--",
            "linewidth": 2.0,
            "zorder": 4
        },
        "FedAvg-GNN + Adaptive Scheduling": {
            "color": "#2ca02c",
            "marker": "^",
            "linestyle": "-.",
            "linewidth": 2.0,
            "zorder": 3
        },
        "SF-GNN": {
            "color": "#d62728",
            "marker": "D",
            "linestyle": "-",
            "linewidth": 2.2,
            "zorder": 6
        }
    }


def filter_results_for_plot(results, metric_key):
    """
    FedAvg-GNN + Topology has the same communication/participation pattern
    as FedAvg-GNN because both use full participation.
    It is removed only from communication-related figures to avoid overlap.
    """

    if metric_key in [
        "participation",
        "communication",
        "cumulative_communication"
    ]:
        return {
            method: history
            for method, history in results.items()
            if method != "FedAvg-GNN + Topology"
        }

    return results


def plot_multi_metric(
    results,
    metric_key,
    title,
    ylabel,
    xlabel="Communication Round",
    save_path=None,
    smooth=True,
    window=3,
    show_std=True,
    ylim=None
):
    set_publication_style()

    results = filter_results_for_plot(
        results,
        metric_key
    )

    styles = get_method_styles()

    plt.figure(
        figsize=(8.6, 5.4)
    )

    for method_name, history in results.items():
        values = get_series(
            history=history,
            metric_key=metric_key,
            smooth=smooth,
            window=window
        )

        rounds = np.arange(
            1,
            len(values) + 1
        )

        st = styles[method_name]

        plt.plot(
            rounds,
            values,
            label=method_name,
            color=st["color"],
            marker=st["marker"],
            linestyle=st["linestyle"],
            linewidth=st["linewidth"],
            markersize=4,
            markevery=max(1, len(rounds) // 20),
            zorder=st["zorder"]
        )

        std_key = f"{metric_key}_std"

        if show_std and std_key in history:
            std_values = np.array(
                history[std_key],
                dtype=float
            )

            if smooth and metric_key in ["accuracy", "loss", "participation"]:
                std_values = smooth_curve(
                    std_values,
                    window=window
                )

            plt.fill_between(
                rounds,
                values - std_values,
                values + std_values,
                color=st["color"],
                alpha=0.08,
                linewidth=0
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(1, None)

    if ylim is not None:
        plt.ylim(ylim)

    plt.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        edgecolor="black"
    )

    plt.tight_layout()
    save_figure(save_path)
    plt.show()


def plot_multi_accuracy(
    results,
    output_dir="outputs"
):
    if "Reddit" in output_dir or "reddit" in output_dir:
        ylim = (0.30, 0.97)
    else:
        ylim = (0.15, 0.85)

    plot_multi_metric(
        results=results,
        metric_key="accuracy",
        title="Global Accuracy Comparison",
        ylabel="Accuracy",
        save_path=f"{output_dir}/accuracy_comparison.png",
        smooth=True,
        window=3,
        show_std=True,
        ylim=ylim
    )


def plot_multi_loss(
    results,
    output_dir="outputs"
):
    plot_multi_metric(
        results=results,
        metric_key="loss",
        title="Training Loss Comparison",
        ylabel="Loss",
        save_path=f"{output_dir}/loss_comparison.png",
        smooth=True,
        window=3,
        show_std=True,
        ylim=(0.0, 1.20)
    )


def plot_multi_participation(
    results,
    output_dir="outputs"
):
    plot_multi_metric(
        results=results,
        metric_key="participation",
        title="Client Participation Rate Comparison",
        ylabel="Participation Rate",
        save_path=f"{output_dir}/participation_comparison.png",
        smooth=True,
        window=3,
        show_std=True,
        ylim=(0.70, 1.05)
    )


def plot_multi_communication(
    results,
    output_dir="outputs"
):
    plot_multi_metric(
        results=results,
        metric_key="cumulative_communication",
        title="Cumulative Communication Cost Comparison",
        ylabel="Cumulative Updates Sent",
        save_path=f"{output_dir}/cumulative_communication_comparison.png",
        smooth=False,
        show_std=True
    )


def plot_metric_grid(
    panel_results,
    metric_key,
    title,
    ylabel,
    xlabel="Communication Round",
    ncols=2,
    save_path=None,
    smooth=True,
    window=3,
    ylim=None
):
    set_publication_style()

    panel_names = list(
        panel_results.keys()
    )

    n_panels = len(panel_names)
    nrows = int(
        np.ceil(n_panels / ncols)
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 4.8 * nrows)
    )

    axes = np.array(
        axes
    ).flatten()

    styles = get_method_styles()

    panel_letters = [
        "a", "b", "c", "d", "e", "f",
        "g", "h", "i", "j"
    ]

    handles = []
    labels = []

    for panel_idx, panel_name in enumerate(panel_names):
        ax = axes[panel_idx]

        results = filter_results_for_plot(
            panel_results[panel_name],
            metric_key
        )

        for method_name, history in results.items():
            values = get_series(
                history=history,
                metric_key=metric_key,
                smooth=smooth,
                window=window
            )

            rounds = np.arange(
                1,
                len(values) + 1
            )

            st = styles[method_name]

            line, = ax.plot(
                rounds,
                values,
                label=method_name,
                color=st["color"],
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=1.6,
                markersize=3,
                markevery=max(1, len(rounds) // 15),
                zorder=st["zorder"]
            )

            if method_name not in labels:
                handles.append(line)
                labels.append(method_name)

        clean_title = panel_name.replace(
            "Cora - ",
            ""
        ).replace(
            "Reddit - ",
            ""
        )

        ax.set_title(
            f"{panel_letters[panel_idx]}) {clean_title}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlim(1, None)
        ax.grid(
            True,
            linestyle="--",
            alpha=0.35
        )

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        title,
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(
        rect=[0, 0.06, 1, 0.95]
    )

    save_figure(save_path)
    plt.show()
