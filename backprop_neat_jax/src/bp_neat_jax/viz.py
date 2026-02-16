from __future__ import annotations

from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

from .datasets import TaskData
from .genome import Genome


def plot_training_curves(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return

    g = np.array([r["generation"] for r in history], dtype=float)
    best_fit = np.array([r["best_fitness"] for r in history], dtype=float)
    mean_fit = np.array([r["mean_fitness"] for r in history], dtype=float)

    best_acc = np.array([r["best_test_acc"] for r in history], dtype=float)
    mean_acc = np.array([r["mean_test_acc"] for r in history], dtype=float)
    best_loss = np.array([r["best_test_loss"] for r in history], dtype=float)

    mean_hidden = np.array([r["mean_hidden_nodes"] for r in history], dtype=float)
    mean_conn = np.array([r["mean_enabled_connections"] for r in history], dtype=float)
    species = np.array([r["species_count"] for r in history], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(g, best_fit, label="best fitness", linewidth=2)
    axes[0].plot(g, mean_fit, label="mean fitness", linewidth=1.6)
    axes[0].set_ylabel("fitness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(g, best_acc, label="best test acc", linewidth=2)
    axes[1].plot(g, mean_acc, label="mean test acc", linewidth=1.6)
    axes[1].plot(g, best_loss, label="best test loss", linewidth=1.2)
    axes[1].set_ylabel("accuracy/loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(g, mean_hidden, label="mean hidden", linewidth=2)
    axes[2].plot(g, mean_conn, label="mean enabled conn", linewidth=2)
    axes[2].set_ylabel("complexity")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(g, species, label="species", linewidth=2)
    axes[3].set_ylabel("species")
    axes[3].set_xlabel("generation")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_species_sizes(species_sizes: list[dict[int, int]], path: Path) -> None:
    if not species_sizes:
        return

    all_species = sorted({sid for row in species_sizes for sid in row.keys()})
    g = np.arange(len(species_sizes))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cumulative = np.zeros(len(species_sizes), dtype=float)
    for sid in all_species:
        vals = np.array([row.get(sid, 0) for row in species_sizes], dtype=float)
        ax.fill_between(g, cumulative, cumulative + vals, alpha=0.45, label=f"species {sid}")
        cumulative += vals

    ax.set_xlabel("generation")
    ax.set_ylabel("members")
    ax.set_title("Species Composition")
    ax.grid(True, alpha=0.25)
    if len(all_species) <= 12:
        ax.legend(loc="upper right", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_network(genome: Genome, path: Path, title: str = "Best Network") -> None:
    layers = sorted({n.layer for n in genome.nodes.values()})
    layer_to_x = {layer: i for i, layer in enumerate(layers)}

    nodes_by_layer: dict[float, list[int]] = {}
    for node in genome.nodes.values():
        nodes_by_layer.setdefault(node.layer, []).append(node.node_id)
    for layer in nodes_by_layer:
        nodes_by_layer[layer] = sorted(nodes_by_layer[layer])

    pos: dict[int, tuple[float, float]] = {}
    for layer in layers:
        ids = nodes_by_layer[layer]
        ys = np.linspace(0.1, 0.9, max(2, len(ids)))
        if len(ids) == 1:
            ys = np.array([0.5])
        for i, nid in enumerate(ids):
            pos[nid] = (layer_to_x[layer], ys[i])

    fig, ax = plt.subplots(figsize=(11, 6))

    for conn in genome.connections.values():
        if conn.src not in pos or conn.dst not in pos:
            continue
        x1, y1 = pos[conn.src]
        x2, y2 = pos[conn.dst]
        color = "#1f77b4" if conn.weight >= 0 else "#d62728"
        alpha = 0.7 if conn.enabled else 0.16
        lw = 0.8 + min(2.4, abs(conn.weight))
        style = "-" if conn.enabled else "--"
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, linestyle=style)

    color_map = {"input": "#2ca02c", "hidden": "#9467bd", "output": "#ff7f0e"}
    for nid, node in sorted(genome.nodes.items()):
        x, y = pos[nid]
        ax.scatter([x], [y], s=170, color=color_map.get(node.kind, "#7f7f7f"), edgecolors="black", zorder=3)
        ax.text(x, y + 0.03, f"{nid}:{node.activation}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("layer")
    ax.set_ylabel("node position")
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_activation_usage(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    last = history[-1]
    pairs = [(k[4:], v) for k, v in last.items() if k.startswith("act_")]
    if not pairs:
        return

    names = [k for k, _ in pairs]
    vals = [v for _, v in pairs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals)
    ax.set_ylabel("hidden-node count")
    ax.set_title("Activation Usage (Final Population)")
    ax.grid(True, axis="y", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_decision_boundary(genome: Genome, data: TaskData, path: Path, grid_n: int = 220) -> None:
    phenotype = genome.build_phenotype()
    params = phenotype.initial_params()

    train_x = np.asarray(data.train_x)
    test_x = np.asarray(data.test_x)
    all_x = np.concatenate([train_x, test_x], axis=0)

    x_min, x_max = float(np.min(all_x[:, 0])) - 1.0, float(np.max(all_x[:, 0])) + 1.0
    y_min, y_max = float(np.min(all_x[:, 1])) - 1.0, float(np.max(all_x[:, 1])) + 1.0

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n), np.linspace(y_min, y_max, grid_n))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    logits = phenotype.forward(params, jax.numpy.asarray(grid)).reshape(-1)
    prob = np.asarray(jax.nn.sigmoid(logits)).reshape(xx.shape)

    train_y = np.asarray(data.train_y)
    test_y = np.asarray(data.test_y)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    contour = ax.contourf(xx, yy, prob, levels=32, cmap="RdBu", alpha=0.65)
    plt.colorbar(contour, ax=ax, label="P(class=1)")

    ax.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap="RdBu", edgecolors="k", s=26, alpha=0.9, label="train")
    ax.scatter(test_x[:, 0], test_x[:, 1], c=test_y, cmap="RdBu", marker="x", s=30, alpha=0.8, label="test")

    ax.set_title(f"Decision Boundary ({data.name})")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.legend(loc="upper right")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
