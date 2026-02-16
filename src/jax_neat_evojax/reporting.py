from __future__ import annotations

from pathlib import Path

from .genome import Genome


def _safe_delta(start: float, end: float) -> str:
    return f"{start:.2f} -> {end:.2f} (delta {end - start:+.2f})"


def build_complexification_commentary(history: list[dict[str, float]], champion: Genome) -> str:
    if not history:
        return "No generation history was recorded."

    start = history[0]
    end = history[-1]

    lines = []
    lines.append("### Complexification Notes")
    lines.append("")
    lines.append(
        f"- Mean hidden nodes: {_safe_delta(start['mean_hidden_nodes'], end['mean_hidden_nodes'])}."
    )
    lines.append(
        f"- Mean enabled connections: {_safe_delta(start['mean_enabled_connections'], end['mean_enabled_connections'])}."
    )
    lines.append(
        f"- Best fitness: {_safe_delta(start['best_fitness'], end['best_fitness'])}."
    )

    hidden_nodes = [n for n in champion.nodes.values() if n.kind == "hidden"]
    activation_counts: dict[str, int] = {}
    for node in hidden_nodes:
        activation_counts[node.activation] = activation_counts.get(node.activation, 0) + 1

    if activation_counts:
        act_desc = ", ".join(f"{k}: {v}" for k, v in sorted(activation_counts.items()))
        lines.append(f"- Champion hidden activations: {act_desc}.")
    else:
        lines.append("- Champion has no hidden nodes (no topology complexification observed).")

    lines.append(
        "- No residual/skip links remain enabled once intermediate layers exist; "
        "complexification therefore proceeds via inserted hidden nodes and adjacent-layer paths."
    )

    return "\n".join(lines)


def write_markdown_report(
    path: Path,
    mode: str,
    history: list[dict[str, float]],
    champion: Genome,
    artifacts: dict[str, Path],
) -> None:
    commentary = build_complexification_commentary(history, champion)

    lines = [
        "# NEAT in JAX + EvoJAX SlimeVolley Report",
        "",
        f"## Mode: `{mode}`",
        "",
        "## Artifacts",
        "",
    ]

    for name, p in sorted(artifacts.items()):
        lines.append(f"- {name}: `{p}`")

    lines.extend(
        [
            "",
            "## Visualization Summary",
            "",
            "- Include the following in your Google Doc:",
            f"  - GIF: `{artifacts.get('gif_vs_builtin', Path('N/A'))}`",
            f"  - GIF (self-play): `{artifacts.get('gif_vs_runnerup', Path('N/A'))}`",
            "  - Network plots from the `plots/` directory.",
            "",
            commentary,
            "",
            "## Suggested Write-up Points",
            "",
            "- How species diversity changed and whether threshold adaptation stabilized species count.",
            "- Whether hidden-node growth was smooth or occurred in bursts.",
            "- Which activation families dominated in the champion and why that might help this task.",
            "- Whether evolved topologies used narrow or wide hidden bottlenecks.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
