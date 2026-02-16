from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from .config import EvolutionConfig
from .evolution import NEATTrainer
from .reporting import write_markdown_report
from .visualization import (
    plot_activation_usage,
    plot_genome,
    plot_history,
    plot_species_sizes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feed-forward NEAT on EvoJAX SlimeVolley")
    p.add_argument("--mode", choices=["direct_vs_builtin", "selfplay_then_builtin"], required=True)
    p.add_argument("--pop-size", type=int, default=80)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--episodes-per-genome", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-root", type=str, default="artifacts")
    p.add_argument("--evojax-path", type=str, default="evojax")
    return p.parse_args()


def _final_activation_counts(population) -> dict[str, int]:
    counts: dict[str, int] = {}
    for genome in population:
        for node in genome.nodes.values():
            if node.kind != "hidden":
                continue
            counts[node.activation] = counts.get(node.activation, 0) + 1
    return counts


def main() -> None:
    args = parse_args()

    evojax_path = Path(args.evojax_path).resolve()
    if str(evojax_path) not in sys.path:
        sys.path.insert(0, str(evojax_path))

    cfg = EvolutionConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        episodes_per_genome=args.episodes_per_genome,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).resolve() / f"{args.mode}_{ts}"

    trainer = NEATTrainer(cfg=cfg, mode=args.mode, out_dir=out_dir)
    champion, runnerup, artifacts = trainer.run()

    plots_dir = out_dir / "plots"
    plot_history(trainer.history, plots_dir / "fitness_complexity.png")
    plot_species_sizes(trainer.species_sizes, plots_dir / "species_sizes.png")
    plot_genome(champion, plots_dir / "champion_network.png", title="Champion Topology")

    if trainer.champion_snapshots:
        key_gens = sorted(
            {
                trainer.champion_snapshots[0][0],
                trainer.champion_snapshots[len(trainer.champion_snapshots) // 2][0],
                trainer.champion_snapshots[-1][0],
            }
        )
        snap_by_gen = {g: genome for g, genome in trainer.champion_snapshots}
        for g in key_gens:
            plot_genome(
                snap_by_gen[g],
                plots_dir / f"champion_network_gen_{g}.png",
                title=f"Champion Topology (Generation {g})",
            )

    ranked = sorted(
        trainer.population,
        key=lambda g: g.fitness if g.fitness is not None else -1e9,
        reverse=True,
    )
    for rank, genome in enumerate(ranked[:3], start=1):
        plot_genome(genome, plots_dir / f"network_rank_{rank}.png", title=f"Final Population Rank {rank}")

    act_counts = _final_activation_counts(trainer.population)
    plot_activation_usage(act_counts, plots_dir / "final_activation_usage.png")

    report_path = out_dir / "report.md"
    write_markdown_report(
        path=report_path,
        mode=args.mode,
        history=trainer.history,
        champion=champion,
        artifacts={**artifacts, "plots_dir": plots_dir},
    )

    print(f"Run complete: {out_dir}")
    for name, p in sorted({**artifacts, "report": report_path}.items()):
        print(f"{name}: {p}")


if __name__ == "__main__":
    main()
