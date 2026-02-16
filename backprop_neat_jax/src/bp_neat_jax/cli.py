from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from .config import BPNEATConfig
from .datasets import make_dataset
from .report import write_report
from .trainer import BackpropNEATTrainer
from .viz import (
    plot_activation_usage,
    plot_decision_boundary,
    plot_network,
    plot_species_sizes,
    plot_training_curves,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backprop-NEAT in JAX for 2D classification")
    p.add_argument("--task", choices=["circle", "xor", "spiral", "all"], default="all")
    p.add_argument("--pop-size", type=int, default=60)
    p.add_argument("--generations", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--train-size", type=int, default=200)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.5)

    p.add_argument("--bp-steps", type=int, default=220)
    p.add_argument("--bp-batch-size", type=int, default=32)
    p.add_argument("--bp-lr", type=float, default=0.01)

    p.add_argument("--conn-penalty", type=float, default=0.02)
    p.add_argument("--hidden-penalty", type=float, default=0.015)
    p.add_argument("--loss-weight", type=float, default=0.25)

    p.add_argument("--out-root", type=str, default="artifacts")
    return p.parse_args()


def _run_single_task(task: str, args: argparse.Namespace) -> Path:
    cfg = BPNEATConfig(pop_size=args.pop_size, generations=args.generations, seed=args.seed)
    cfg.dataset.train_size = args.train_size
    cfg.dataset.test_size = args.test_size
    cfg.dataset.noise = args.noise

    cfg.backprop.steps = args.bp_steps
    cfg.backprop.batch_size = args.bp_batch_size
    cfg.backprop.learning_rate = args.bp_lr

    cfg.fitness.complexity_conn_penalty = args.conn_penalty
    cfg.fitness.complexity_hidden_penalty = args.hidden_penalty
    cfg.fitness.loss_weight = args.loss_weight

    data = make_dataset(
        task=task,
        train_size=cfg.dataset.train_size,
        test_size=cfg.dataset.test_size,
        noise=cfg.dataset.noise,
        seed=cfg.seed,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root).resolve() / f"{task}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = BackpropNEATTrainer(cfg=cfg, task_data=data, out_dir=out_dir)
    best = trainer.run()

    history_csv = out_dir / "history.csv"
    species_csv = out_dir / "species_sizes.csv"
    best_json = out_dir / "best_genome.json"
    trainer.save_history_csv(history_csv)
    trainer.save_species_csv(species_csv)
    trainer.save_genome_json(best, best_json)

    plots = out_dir / "plots"
    plot_training_curves(trainer.history, plots / "training_curves.png")
    plot_species_sizes(trainer.species_sizes, plots / "species_sizes.png")
    plot_activation_usage(trainer.history, plots / "activation_usage.png")
    plot_network(best, plots / "network_best.png", title=f"Best Network ({task})")
    plot_decision_boundary(best, data, plots / "decision_boundary_best.png")

    if trainer.champion_snapshots:
        key_gens = sorted(
            {
                trainer.champion_snapshots[0][0],
                trainer.champion_snapshots[len(trainer.champion_snapshots) // 2][0],
                trainer.champion_snapshots[-1][0],
            }
        )
        snap_map = {g: genome for g, genome in trainer.champion_snapshots}
        for g in key_gens:
            plot_network(
                snap_map[g],
                plots / f"network_gen_{g}.png",
                title=f"Champion Topology (Generation {g}, task={task})",
            )

    report_path = out_dir / "report.md"
    artifacts = {
        "history_csv": history_csv,
        "species_csv": species_csv,
        "best_genome_json": best_json,
        "training_curves": plots / "training_curves.png",
        "species_plot": plots / "species_sizes.png",
        "activation_usage": plots / "activation_usage.png",
        "decision_boundary": plots / "decision_boundary_best.png",
        "best_network": plots / "network_best.png",
    }
    write_report(report_path, task=task, history=trainer.history, best=best, artifacts=artifacts)

    print(f"Task {task} complete: {out_dir}")
    print(f"Best fitness={best.fitness:.4f}, train_acc={best.train_acc:.4f}, test_acc={best.test_acc:.4f}")
    return out_dir


def main() -> None:
    args = parse_args()
    tasks = [args.task] if args.task != "all" else ["circle", "xor", "spiral"]

    outputs = []
    for i, task in enumerate(tasks):
        task_args = argparse.Namespace(**vars(args))
        task_args.seed = args.seed + i * 101
        outputs.append(_run_single_task(task, task_args))

    print("\nRun outputs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
