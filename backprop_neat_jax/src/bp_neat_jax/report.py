from __future__ import annotations

from pathlib import Path

from .genome import Genome


def _safe(history: list[dict[str, float]], key: str, default: float = 0.0) -> tuple[float, float]:
    if not history:
        return default, default
    return float(history[0].get(key, default)), float(history[-1].get(key, default))


def _activation_desc(genome: Genome) -> str:
    counts: dict[str, int] = {}
    for node in genome.nodes.values():
        if node.kind != "hidden":
            continue
        counts[node.activation] = counts.get(node.activation, 0) + 1
    if not counts:
        return "no hidden activations (minimal network)."
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))


def write_report(
    path: Path,
    task: str,
    history: list[dict[str, float]],
    best: Genome,
    artifacts: dict[str, Path],
) -> None:
    fit0, fit1 = _safe(history, "best_fitness")
    acc0, acc1 = _safe(history, "best_test_acc")
    hid0, hid1 = _safe(history, "champ_hidden_nodes")
    con0, con1 = _safe(history, "champ_enabled_connections")

    lines = [
        "# Backprop-NEAT JAX Report",
        "",
        f"## Task: `{task}`",
        "",
        "## Summary",
        "",
        f"- Best fitness: {fit0:.4f} -> {fit1:.4f} (delta {fit1-fit0:+.4f})",
        f"- Best test accuracy: {acc0:.4f} -> {acc1:.4f} (delta {acc1-acc0:+.4f})",
        f"- Champion hidden nodes: {hid0:.1f} -> {hid1:.1f}",
        f"- Champion enabled connections: {con0:.1f} -> {con1:.1f}",
        "",
        "## Complexification Commentary",
        "",
        "- Architectures are evolved with feed-forward-only constraints and no skip bypass once intermediate layers emerge.",
        "- Weight parameters are optimized by backprop (JAX Adam) at fitness time, then inherited Lamarckian-style into the next generation.",
        "- Complexity is penalized in fitness to encourage compact yet expressive topologies.",
        f"- Final champion hidden activation usage: {_activation_desc(best)}",
        "",
        "## Visual Assets",
        "",
    ]

    for name, p in sorted(artifacts.items()):
        lines.append(f"- {name}: `{p}`")

    lines.extend(
        [
            "",
            "## Notes Worth Writing About",
            "",
            "- Which activation families became dominant and in which layers.",
            "- Whether complexity growth correlated with jumps in test accuracy.",
            "- Cases where backprop improved shallow topologies versus forcing larger ones.",
            "- Any elegant low-complexity champions that rivaled larger networks.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
