from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .genome import Genome


def plot_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return

    g = np.array([h["generation"] for h in history], dtype=float)
    best = np.array([h["best_fitness"] for h in history], dtype=float)
    mean = np.array([h["mean_fitness"] for h in history], dtype=float)
    species = np.array([h["species_count"] for h in history], dtype=float)
    mean_hidden = np.array([h["mean_hidden_nodes"] for h in history], dtype=float)
    mean_conn = np.array([h["mean_enabled_connections"] for h in history], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(g, best, label="best fitness", linewidth=2)
    axes[0].plot(g, mean, label="mean fitness", linewidth=1.6)
    axes[0].set_ylabel("fitness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(g, mean_hidden, label="mean hidden nodes", linewidth=2)
    axes[1].plot(g, mean_conn, label="mean enabled connections", linewidth=2)
    axes[1].set_ylabel("complexity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(g, species, label="species count", linewidth=2)
    axes[2].set_ylabel("species")
    axes[2].set_xlabel("generation")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_species_sizes(species_sizes: list[dict[int, int]], path: Path) -> None:
    if not species_sizes:
        return

    all_species = sorted({sid for row in species_sizes for sid in row})
    gen = np.arange(len(species_sizes))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cumulative = np.zeros(len(species_sizes), dtype=float)
    for sid in all_species:
        values = np.array([row.get(sid, 0) for row in species_sizes], dtype=float)
        ax.fill_between(gen, cumulative, cumulative + values, alpha=0.45, label=f"species {sid}")
        cumulative += values

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


def plot_genome(genome: Genome, path: Path, title: str = "Genome Topology") -> None:
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

    # Draw connections.
    for conn in genome.connections.values():
        if conn.src not in pos or conn.dst not in pos:
            continue
        x1, y1 = pos[conn.src]
        x2, y2 = pos[conn.dst]
        color = "#1f77b4" if conn.weight >= 0 else "#d62728"
        alpha = 0.65 if conn.enabled else 0.18
        lw = 0.7 + min(2.5, abs(conn.weight))
        ls = "-" if conn.enabled else "--"
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, linestyle=ls)

    # Draw nodes.
    kind_color = {"input": "#2ca02c", "hidden": "#9467bd", "output": "#ff7f0e"}
    for nid, node in sorted(genome.nodes.items()):
        x, y = pos[nid]
        ax.scatter([x], [y], s=160, color=kind_color.get(node.kind, "#7f7f7f"), edgecolors="black", zorder=3)
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


def plot_activation_usage(activation_counts: dict[str, int], path: Path) -> None:
    if not activation_counts:
        return
    names = list(activation_counts.keys())
    values = [activation_counts[k] for k in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values)
    ax.set_ylabel("hidden node count")
    ax.set_title("Activation Function Usage (Population)")
    ax.grid(True, axis="y", alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
