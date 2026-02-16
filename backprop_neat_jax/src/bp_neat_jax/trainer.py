from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import BPNEATConfig
from .datasets import TaskData
from .genome import Genome, create_initial_genome, crossover
from .innovation import InnovationTracker
from .species import SpeciesManager


class BackpropNEATTrainer:
    def __init__(self, cfg: BPNEATConfig, task_data: TaskData, out_dir: Path):
        self.cfg = cfg
        self.data = task_data
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(cfg.seed)
        self.tracker = InnovationTracker(
            next_innovation=0,
            next_node_id=cfg.input_size + cfg.output_size,
        )
        self.species_mgr = SpeciesManager(cfg)

        self.population: list[Genome] = [
            create_initial_genome(i, cfg, self.tracker, self.rng)
            for i in range(cfg.pop_size)
        ]
        self.next_genome_id = cfg.pop_size

        self.history: list[dict[str, float]] = []
        self.species_sizes: list[dict[int, int]] = []
        self.champion_snapshots: list[tuple[int, Genome]] = []

    def run(self) -> Genome:
        for gen in range(self.cfg.generations):
            self._evaluate_population(gen)
            g2s = self.species_mgr.speciate(self.population, self.rng)
            self._record_generation(gen, g2s)

            if gen < self.cfg.generations - 1:
                self.population = self._reproduce(g2s)

        return self.best_genome().clone()

    def best_genome(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness if g.fitness is not None else -1e9)

    def _evaluate_population(self, gen: int) -> None:
        for i, genome in enumerate(self.population):
            seed = (
                self.cfg.seed
                + 1_100_003 * (gen + 1)
                + 5_009 * (i + 1)
            ) & 0xFFFFFFFF
            params, train_loss, train_acc = self._train_genome_with_backprop(genome, seed)
            phenotype = genome.build_phenotype()

            test_loss = float(phenotype.loss(params, self.data.test_x, self.data.test_y))
            test_acc = float(phenotype.accuracy(params, self.data.test_x, self.data.test_y))

            hidden, enabled_conn = genome.complexity()
            penalty = (
                self.cfg.fitness.complexity_conn_penalty * np.sqrt(max(enabled_conn, 1))
                + self.cfg.fitness.complexity_hidden_penalty * hidden
            )

            fitness = test_acc - self.cfg.fitness.loss_weight * test_loss - penalty
            genome.fitness = float(fitness)
            genome.train_acc = float(train_acc)
            genome.test_acc = float(test_acc)
            genome.test_loss = float(test_loss)

            if self.cfg.backprop.lamarckian_inheritance:
                phenotype.update_genome(genome, params)

    def _train_genome_with_backprop(
        self,
        genome: Genome,
        seed: int,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], float, float]:
        phenotype = genome.build_phenotype()
        params = phenotype.initial_params()

        bp = self.cfg.backprop
        train_x = self.data.train_x
        train_y = self.data.train_y
        n = int(train_x.shape[0])

        loss_grad = jax.jit(jax.value_and_grad(phenotype.loss))
        eval_loss = jax.jit(phenotype.loss)
        eval_acc = jax.jit(phenotype.accuracy)

        w, b = params
        mw = jnp.zeros_like(w)
        mb = jnp.zeros_like(b)
        vw = jnp.zeros_like(w)
        vb = jnp.zeros_like(b)

        best_params = (w, b)
        best_train_loss = float(eval_loss((w, b), train_x, train_y))
        bad_windows = 0

        rng = np.random.default_rng(seed)

        for t in range(1, bp.steps + 1):
            batch_idx = rng.integers(0, n, size=(bp.batch_size,), endpoint=False)
            xb = train_x[batch_idx]
            yb = train_y[batch_idx]

            (loss_val, grads) = loss_grad((w, b), xb, yb)
            gw, gb = grads

            # Weight decay on trainable connection weights only.
            gw = gw + bp.weight_decay * w

            # Global gradient clipping for stability.
            grad_norm = jnp.sqrt(jnp.sum(gw * gw) + jnp.sum(gb * gb) + 1e-12)
            clip_scale = jnp.minimum(1.0, bp.grad_clip / (grad_norm + 1e-8))
            gw = gw * clip_scale
            gb = gb * clip_scale

            mw = bp.beta1 * mw + (1.0 - bp.beta1) * gw
            mb = bp.beta1 * mb + (1.0 - bp.beta1) * gb
            vw = bp.beta2 * vw + (1.0 - bp.beta2) * (gw * gw)
            vb = bp.beta2 * vb + (1.0 - bp.beta2) * (gb * gb)

            mw_hat = mw / (1.0 - bp.beta1**t)
            mb_hat = mb / (1.0 - bp.beta1**t)
            vw_hat = vw / (1.0 - bp.beta2**t)
            vb_hat = vb / (1.0 - bp.beta2**t)

            w = w - bp.learning_rate * mw_hat / (jnp.sqrt(vw_hat) + bp.epsilon)
            b = b - bp.learning_rate * mb_hat / (jnp.sqrt(vb_hat) + bp.epsilon)

            if bp.eval_interval > 0 and (t % bp.eval_interval == 0):
                full_train_loss = float(eval_loss((w, b), train_x, train_y))
                if full_train_loss + 1e-6 < best_train_loss:
                    best_train_loss = full_train_loss
                    best_params = (w, b)
                    bad_windows = 0
                else:
                    bad_windows += 1
                    if bad_windows >= bp.patience_windows:
                        break

        final_params = best_params
        final_train_loss = float(eval_loss(final_params, train_x, train_y))
        final_train_acc = float(eval_acc(final_params, train_x, train_y))
        return final_params, final_train_loss, final_train_acc

    def _record_generation(self, gen: int, genome_to_species: dict[int, int]) -> None:
        del genome_to_species
        best = self.best_genome()

        fit = np.array([g.fitness if g.fitness is not None else -1e9 for g in self.population], dtype=float)
        tr_acc = np.array([g.train_acc if g.train_acc is not None else 0.0 for g in self.population], dtype=float)
        te_acc = np.array([g.test_acc if g.test_acc is not None else 0.0 for g in self.population], dtype=float)
        te_loss = np.array([g.test_loss if g.test_loss is not None else 1.0 for g in self.population], dtype=float)

        hidden_vals = np.array([g.complexity()[0] for g in self.population], dtype=float)
        conn_vals = np.array([g.complexity()[1] for g in self.population], dtype=float)

        record = {
            "generation": float(gen),
            "best_fitness": float(np.max(fit)),
            "mean_fitness": float(np.mean(fit)),
            "best_train_acc": float(best.train_acc or 0.0),
            "best_test_acc": float(best.test_acc or 0.0),
            "best_test_loss": float(best.test_loss or 0.0),
            "mean_train_acc": float(np.mean(tr_acc)),
            "mean_test_acc": float(np.mean(te_acc)),
            "mean_test_loss": float(np.mean(te_loss)),
            "species_count": float(len(self.species_mgr.species)),
            "mean_hidden_nodes": float(np.mean(hidden_vals)),
            "mean_enabled_connections": float(np.mean(conn_vals)),
            "champ_hidden_nodes": float(best.complexity()[0]),
            "champ_enabled_connections": float(best.complexity()[1]),
            "compat_threshold": float(self.species_mgr.threshold),
        }

        act_counts: dict[str, int] = {}
        for genome in self.population:
            for node in genome.nodes.values():
                if node.kind != "hidden":
                    continue
                act_counts[node.activation] = act_counts.get(node.activation, 0) + 1
        for name in self.cfg.activation_options:
            record[f"act_{name}"] = float(act_counts.get(name, 0))

        self.history.append(record)
        self.champion_snapshots.append((gen, best.clone()))
        self.species_sizes.append({sid: len(sp.members) for sid, sp in self.species_mgr.species.items()})

    def _reproduce(self, genome_to_species: dict[int, int]) -> list[Genome]:
        del genome_to_species
        pop_by_id = {g.genome_id: g for g in self.population}

        adjusted: dict[int, float] = {}
        species_members: dict[int, list[Genome]] = {}
        for sid, sp in self.species_mgr.species.items():
            members = [pop_by_id[gid] for gid in sp.members if gid in pop_by_id]
            if not members:
                continue
            members.sort(key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
            species_members[sid] = members
            denom = float(len(members))
            for g in members:
                adjusted[g.genome_id] = (g.fitness if g.fitness is not None else -1e9) / denom

        if not species_members:
            return [
                create_initial_genome(i, self.cfg, self.tracker, self.rng)
                for i in range(self.cfg.pop_size)
            ]

        min_adj = min(adjusted.values())
        shift = -min_adj + 1e-6 if min_adj <= 0 else 0.0

        species_score: dict[int, float] = {}
        for sid, members in species_members.items():
            vals = [adjusted[g.genome_id] + shift for g in members]
            species_score[sid] = float(np.mean(vals))

        total_score = sum(species_score.values())
        if total_score <= 0:
            probs = {sid: 1.0 / len(species_score) for sid in species_score}
        else:
            probs = {sid: score / total_score for sid, score in species_score.items()}

        raw = {sid: probs[sid] * self.cfg.pop_size for sid in probs}
        spawn = {sid: int(np.floor(raw[sid])) for sid in raw}

        elitism = self.cfg.reproduction.elitism
        for sid, members in species_members.items():
            if members:
                spawn[sid] = max(spawn[sid], min(elitism, len(members)))

        n_now = sum(spawn.values())
        frac_order = sorted(raw.keys(), key=lambda sid: raw[sid] - np.floor(raw[sid]), reverse=True)

        i = 0
        while n_now < self.cfg.pop_size:
            sid = frac_order[i % len(frac_order)]
            spawn[sid] += 1
            n_now += 1
            i += 1

        i = 0
        while n_now > self.cfg.pop_size:
            sid = frac_order[i % len(frac_order)]
            min_keep = min(elitism, len(species_members[sid]))
            if spawn[sid] > min_keep:
                spawn[sid] -= 1
                n_now -= 1
            i += 1
            if i > 10000:
                break

        new_population: list[Genome] = []

        for sid, members in species_members.items():
            n_offspring = spawn.get(sid, 0)
            if n_offspring <= 0:
                continue

            members = sorted(members, key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
            elite_n = min(elitism, len(members), n_offspring)

            for e in range(elite_n):
                elite = members[e].clone(new_id=self.next_genome_id)
                self.next_genome_id += 1
                elite.fitness = None
                elite.train_acc = None
                elite.test_acc = None
                elite.test_loss = None
                new_population.append(elite)

            remaining = n_offspring - elite_n
            parent_cut = max(2, int(np.ceil(len(members) * self.cfg.reproduction.survival_fraction)))
            parent_pool = members[:parent_cut]

            for _ in range(remaining):
                if len(parent_pool) >= 2 and self.rng.random() < self.cfg.reproduction.crossover_rate:
                    p1 = parent_pool[self.rng.integers(len(parent_pool))]
                    p2 = parent_pool[self.rng.integers(len(parent_pool))]
                    child = crossover(self.rng, p1, p2, self.next_genome_id)
                else:
                    p = parent_pool[self.rng.integers(len(parent_pool))]
                    child = p.clone(new_id=self.next_genome_id)
                self.next_genome_id += 1
                child.fitness = None
                child.train_acc = None
                child.test_acc = None
                child.test_loss = None
                child.mutate(self.rng, self.cfg, self.tracker)
                new_population.append(child)

        if len(new_population) < self.cfg.pop_size:
            seed_parent = self.best_genome().clone()
            while len(new_population) < self.cfg.pop_size:
                child = seed_parent.clone(new_id=self.next_genome_id)
                self.next_genome_id += 1
                child.fitness = None
                child.train_acc = None
                child.test_acc = None
                child.test_loss = None
                child.mutate(self.rng, self.cfg, self.tracker)
                new_population.append(child)
        elif len(new_population) > self.cfg.pop_size:
            new_population = new_population[: self.cfg.pop_size]

        return new_population

    def save_history_csv(self, path: Path) -> None:
        if not self.history:
            return
        fieldnames = list(self.history[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def save_species_csv(self, path: Path) -> None:
        all_species = sorted({sid for row in self.species_sizes for sid in row.keys()})
        fieldnames = ["generation"] + [f"species_{sid}" for sid in all_species]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for gen, row in enumerate(self.species_sizes):
                out = {"generation": gen}
                for sid in all_species:
                    out[f"species_{sid}"] = row.get(sid, 0)
                writer.writerow(out)

    def save_genome_json(self, genome: Genome, path: Path) -> None:
        data = {
            "genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "train_acc": genome.train_acc,
            "test_acc": genome.test_acc,
            "test_loss": genome.test_loss,
            "nodes": [asdict(n) for n in sorted(genome.nodes.values(), key=lambda x: x.node_id)],
            "connections": [
                asdict(c)
                for c in sorted(genome.connections.values(), key=lambda x: (x.innovation, x.src, x.dst))
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
