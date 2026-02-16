from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import EvolutionConfig
from .envs.slimevolley_env import SlimeVolleyEvaluator, save_gif
from .genome import Genome, create_initial_genome, crossover
from .innovation import InnovationTracker
from .species import SpeciesManager


class NEATTrainer:
    def __init__(self, cfg: EvolutionConfig, mode: str, out_dir: Path):
        if mode not in {"direct_vs_builtin", "selfplay_then_builtin"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.cfg = cfg
        self.mode = mode
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(cfg.seed)
        self.tracker = InnovationTracker(
            next_innovation=0,
            next_node_id=cfg.input_size + cfg.output_size,
        )
        self.species_mgr = SpeciesManager(cfg)
        self.evaluator = SlimeVolleyEvaluator(max_steps=cfg.max_steps)

        self.population: list[Genome] = [
            create_initial_genome(i, cfg, self.tracker, self.rng)
            for i in range(cfg.pop_size)
        ]
        self.next_genome_id = cfg.pop_size

        self.history: list[dict[str, float]] = []
        self.species_sizes: list[dict[int, int]] = []
        self.champion_snapshots: list[tuple[int, Genome]] = []
        self.live_history_path = self.out_dir / "history_live.csv"
        self.live_species_path = self.out_dir / "species_sizes_live.csv"
        self.live_progress_path = self.out_dir / "progress.json"
        self.live_champion_path = self.out_dir / "champion_genome_live.json"

    def run(self) -> tuple[Genome, Genome | None, dict[str, Path]]:
        self._write_live_progress(
            generation_completed=-1,
            status="starting",
        )
        for gen in range(self.cfg.generations):
            self._write_live_progress(
                generation_completed=gen - 1,
                status=f"evaluating_generation_{gen}",
            )
            if self.mode == "direct_vs_builtin":
                self._evaluate_direct_vs_builtin(gen)
                builtin_eval = float(
                    max(g.fitness if g.fitness is not None else -1e9 for g in self.population)
                )
            else:
                self._evaluate_selfplay(gen)
                builtin_eval = self._evaluate_single_vs_builtin(
                    self._best_genome(),
                    episodes=max(1, self.cfg.episodes_per_genome),
                    gen=gen,
                )

            g2s = self.species_mgr.speciate(self.population, self.rng)
            self._record_generation(gen, g2s, builtin_eval)
            self._write_live_generation_files(gen)
            self._write_live_progress(
                generation_completed=gen,
                status=f"completed_generation_{gen}",
            )

            if gen < self.cfg.generations - 1:
                self.population = self._reproduce(g2s)

        champion, runnerup = self._final_ranking()
        artifacts = self._save_artifacts(champion, runnerup)
        self._write_live_progress(
            generation_completed=self.cfg.generations - 1,
            status="finished",
        )
        return champion, runnerup, artifacts

    def _write_live_generation_files(self, gen: int) -> None:
        # These files are rewritten every generation so progress is visible while training is running.
        self._write_history_csv(self.live_history_path)
        self._write_species_csv(self.live_species_path)
        self._write_genome_json(self._best_genome(), self.live_champion_path)
        self._write_live_progress(
            generation_completed=gen,
            status=f"completed_generation_{gen}",
        )

    def _write_live_progress(self, generation_completed: int, status: str) -> None:
        progress = {
            "generation_completed": generation_completed,
            "generations_total": self.cfg.generations,
            "mode": self.mode,
            "status": status,
            "out_dir": str(self.out_dir),
            "best_fitness": self.history[-1]["best_fitness"] if self.history else None,
            "mean_fitness": self.history[-1]["mean_fitness"] if self.history else None,
        }
        with self.live_progress_path.open("w", encoding="utf-8") as f:
            json.dump(progress, f, indent=2)
            f.flush()

    def _policy_fn(self, genome: Genome):
        phenotype = genome.to_phenotype()
        return phenotype.action_binary

    def _evaluate_single_vs_builtin(self, genome: Genome, episodes: int, gen: int) -> float:
        policy = self._policy_fn(genome)
        rewards = []
        for ep in range(episodes):
            seed = (self.cfg.seed + 1_000_003 * (gen + 1) + 10_007 * (ep + 1)) & 0xFFFFFFFF
            roll_rng = np.random.default_rng(seed)
            result = self.evaluator.play(
                rng=roll_rng,
                right_policy=policy,
                left_policy=None,
                capture_frames=False,
                early_terminate=True,
            )
            rewards.append(result.right_reward)
        return float(np.mean(rewards))

    def _evaluate_direct_vs_builtin(self, gen: int) -> None:
        for i, genome in enumerate(self.population):
            rewards = []
            policy = self._policy_fn(genome)
            for ep in range(self.cfg.episodes_per_genome):
                seed = (
                    self.cfg.seed
                    + 99991 * (gen + 1)
                    + 613 * (i + 1)
                    + 31 * (ep + 1)
                ) & 0xFFFFFFFF
                roll_rng = np.random.default_rng(seed)
                result = self.evaluator.play(
                    rng=roll_rng,
                    right_policy=policy,
                    left_policy=None,
                    capture_frames=False,
                    early_terminate=True,
                )
                rewards.append(result.right_reward)
            genome.fitness = float(np.mean(rewards))

    def _evaluate_selfplay(self, gen: int) -> None:
        scores = {g.genome_id: 0.0 for g in self.population}
        games = {g.genome_id: 0 for g in self.population}

        indices = np.arange(len(self.population))
        self.rng.shuffle(indices)
        if len(indices) % 2 == 1:
            indices = np.append(indices, indices[-1])

        for p in range(0, len(indices), 2):
            i = int(indices[p])
            j = int(indices[p + 1])
            g_i = self.population[i]
            g_j = self.population[j]

            policy_i = self._policy_fn(g_i)
            policy_j = self._policy_fn(g_j)

            for ep in range(self.cfg.episodes_per_genome):
                seed_a = (
                    self.cfg.seed
                    + 200003 * (gen + 1)
                    + 997 * (p + 1)
                    + 17 * (ep + 1)
                ) & 0xFFFFFFFF
                seed_b = (seed_a + 1337) & 0xFFFFFFFF

                res_a = self.evaluator.play(
                    rng=np.random.default_rng(seed_a),
                    right_policy=policy_i,
                    left_policy=policy_j,
                    capture_frames=False,
                    early_terminate=True,
                )
                scores[g_i.genome_id] += res_a.right_reward
                scores[g_j.genome_id] += res_a.left_reward
                games[g_i.genome_id] += 1
                games[g_j.genome_id] += 1

                res_b = self.evaluator.play(
                    rng=np.random.default_rng(seed_b),
                    right_policy=policy_j,
                    left_policy=policy_i,
                    capture_frames=False,
                    early_terminate=True,
                )
                scores[g_j.genome_id] += res_b.right_reward
                scores[g_i.genome_id] += res_b.left_reward
                games[g_i.genome_id] += 1
                games[g_j.genome_id] += 1

        for genome in self.population:
            n = max(1, games[genome.genome_id])
            genome.fitness = scores[genome.genome_id] / n

    def _reproduce(self, genome_to_species: dict[int, int]) -> list[Genome]:
        pop_by_id = {g.genome_id: g for g in self.population}

        # Explicit fitness sharing.
        adjusted: dict[int, float] = {}
        species_members: dict[int, list[Genome]] = {}
        for sid, sp in self.species_mgr.species.items():
            members = [pop_by_id[gid] for gid in sp.members if gid in pop_by_id]
            if not members:
                continue
            members.sort(key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
            species_members[sid] = members
            s = float(len(members))
            for g in members:
                fit = g.fitness if g.fitness is not None else -1e9
                adjusted[g.genome_id] = fit / s

        if not species_members:
            # Safety fallback.
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

        # Respect elitism if species has members.
        elitism = self.cfg.reproduction.elitism
        for sid, members in species_members.items():
            if members:
                spawn[sid] = max(spawn[sid], min(elitism, len(members)))

        # Adjust to exact population size.
        n_now = sum(spawn.values())
        fracs = sorted(raw.keys(), key=lambda sid: raw[sid] - np.floor(raw[sid]), reverse=True)
        i = 0
        while n_now < self.cfg.pop_size:
            sid = fracs[i % len(fracs)]
            spawn[sid] += 1
            n_now += 1
            i += 1

        i = 0
        while n_now > self.cfg.pop_size:
            sid = fracs[i % len(fracs)]
            min_keep = min(elitism, len(species_members[sid]))
            if spawn[sid] > min_keep:
                spawn[sid] -= 1
                n_now -= 1
            i += 1
            if i > 10_000:
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
                    parent = parent_pool[self.rng.integers(len(parent_pool))]
                    child = parent.clone(new_id=self.next_genome_id)
                self.next_genome_id += 1
                child.fitness = None
                child.mutate(self.rng, self.cfg, self.tracker)
                new_population.append(child)

        # Final safety correction.
        if len(new_population) < self.cfg.pop_size:
            best = self._best_genome().clone()
            while len(new_population) < self.cfg.pop_size:
                child = best.clone(new_id=self.next_genome_id)
                self.next_genome_id += 1
                child.fitness = None
                child.mutate(self.rng, self.cfg, self.tracker)
                new_population.append(child)
        elif len(new_population) > self.cfg.pop_size:
            new_population = new_population[: self.cfg.pop_size]

        return new_population

    def _best_genome(self) -> Genome:
        return max(self.population, key=lambda g: g.fitness if g.fitness is not None else -1e9)

    def _final_ranking(self) -> tuple[Genome, Genome | None]:
        ranked = sorted(self.population, key=lambda g: g.fitness if g.fitness is not None else -1e9, reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        return best, second

    def _activation_usage(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for genome in self.population:
            for node in genome.nodes.values():
                if node.kind != "hidden":
                    continue
                counts[node.activation] = counts.get(node.activation, 0) + 1
        return counts

    def _record_generation(self, gen: int, genome_to_species: dict[int, int], builtin_eval: float) -> None:
        fitness = np.array([g.fitness if g.fitness is not None else -1e9 for g in self.population], dtype=float)
        best = self._best_genome()
        mean_hidden = float(np.mean([g.complexity()[0] for g in self.population]))
        mean_conn = float(np.mean([g.complexity()[1] for g in self.population]))
        best_hidden, best_conn = best.complexity()

        record = {
            "generation": float(gen),
            "best_fitness": float(np.max(fitness)),
            "mean_fitness": float(np.mean(fitness)),
            "species_count": float(len(self.species_mgr.species)),
            "mean_hidden_nodes": mean_hidden,
            "mean_enabled_connections": mean_conn,
            "champ_hidden_nodes": float(best_hidden),
            "champ_enabled_connections": float(best_conn),
            "compat_threshold": float(self.species_mgr.threshold),
            "champ_vs_builtin": float(builtin_eval),
        }

        usage = self._activation_usage()
        for k in self.cfg.activation_options:
            record[f"act_{k}"] = float(usage.get(k, 0))

        self.history.append(record)
        self.champion_snapshots.append((gen, best.clone()))

        size_map = {sid: len(sp.members) for sid, sp in self.species_mgr.species.items()}
        self.species_sizes.append(size_map)
        print(
            f"[gen {gen + 1:03d}/{self.cfg.generations:03d}] "
            f"best={record['best_fitness']:.3f} "
            f"mean={record['mean_fitness']:.3f} "
            f"species={int(record['species_count'])} "
            f"champ_vs_builtin={record['champ_vs_builtin']:.3f}"
        )

    def _save_artifacts(self, champion: Genome, runnerup: Genome | None) -> dict[str, Path]:
        artifacts: dict[str, Path] = {}

        self._write_history_csv(self.out_dir / "history.csv")
        self._write_species_csv(self.out_dir / "species_sizes.csv")
        self._write_genome_json(champion, self.out_dir / "champion_genome.json")
        artifacts["history_csv"] = self.out_dir / "history.csv"
        artifacts["species_csv"] = self.out_dir / "species_sizes.csv"
        artifacts["champion_json"] = self.out_dir / "champion_genome.json"
        artifacts["history_live_csv"] = self.live_history_path
        artifacts["species_live_csv"] = self.live_species_path
        artifacts["progress_json"] = self.live_progress_path
        artifacts["champion_live_json"] = self.live_champion_path

        # GIFs.
        champion_policy = self._policy_fn(champion)
        gif_rng = np.random.default_rng(self.cfg.seed + 424242)
        vs_builtin = self.evaluator.play(
            rng=gif_rng,
            right_policy=champion_policy,
            left_policy=None,
            capture_frames=True,
            early_terminate=True,
        )
        gif_dir = self.out_dir / "gifs"
        gif_dir.mkdir(parents=True, exist_ok=True)
        gif_builtin = gif_dir / "champion_vs_builtin.gif"
        save_gif(vs_builtin.frames, gif_builtin)
        artifacts["gif_vs_builtin"] = gif_builtin

        if runnerup is not None:
            runner_policy = self._policy_fn(runnerup)
            sp = self.evaluator.play(
                rng=np.random.default_rng(self.cfg.seed + 515151),
                right_policy=champion_policy,
                left_policy=runner_policy,
                capture_frames=True,
                early_terminate=True,
            )
            gif_selfplay = gif_dir / "champion_vs_runnerup.gif"
            save_gif(sp.frames, gif_selfplay)
            artifacts["gif_vs_runnerup"] = gif_selfplay

        return artifacts

    def _write_history_csv(self, path: Path) -> None:
        if not self.history:
            return
        fieldnames = list(self.history[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

    def _write_species_csv(self, path: Path) -> None:
        all_species = sorted({sid for m in self.species_sizes for sid in m.keys()})
        fields = ["generation"] + [f"species_{sid}" for sid in all_species]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for gen, sizes in enumerate(self.species_sizes):
                row = {"generation": gen}
                for sid in all_species:
                    row[f"species_{sid}"] = sizes.get(sid, 0)
                writer.writerow(row)

    def _write_genome_json(self, genome: Genome, path: Path) -> None:
        data = {
            "genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "nodes": [asdict(n) for n in sorted(genome.nodes.values(), key=lambda x: x.node_id)],
            "connections": [
                asdict(c)
                for c in sorted(genome.connections.values(), key=lambda x: (x.innovation, x.src, x.dst))
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
