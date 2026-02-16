from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EvolutionConfig
from .genome import Genome, compatibility_distance


@dataclass
class Species:
    species_id: int
    representative: Genome
    members: list[int] = field(default_factory=list)


class SpeciesManager:
    def __init__(self, cfg: EvolutionConfig):
        self.cfg = cfg
        self.next_species_id = 0
        self.threshold = cfg.species.compatibility_threshold
        self.species: dict[int, Species] = {}

    def speciate(self, population: list[Genome], rng: np.random.Generator) -> dict[int, int]:
        for sp in self.species.values():
            sp.members = []

        genome_to_species: dict[int, int] = {}

        for genome in population:
            best_sid = None
            best_dist = float("inf")
            for sid, sp in self.species.items():
                dist = compatibility_distance(genome, sp.representative, self.cfg)
                if dist < best_dist:
                    best_dist = dist
                    best_sid = sid

            if best_sid is None or best_dist > self.threshold:
                sid = self.next_species_id
                self.next_species_id += 1
                self.species[sid] = Species(
                    species_id=sid,
                    representative=genome.clone(),
                    members=[genome.genome_id],
                )
                genome_to_species[genome.genome_id] = sid
            else:
                self.species[best_sid].members.append(genome.genome_id)
                genome_to_species[genome.genome_id] = best_sid

        # Drop empty species and refresh representatives.
        alive: dict[int, Species] = {}
        pop_by_id = {g.genome_id: g for g in population}
        for sid, sp in self.species.items():
            if not sp.members:
                continue
            rep_gid = sp.members[rng.integers(len(sp.members))]
            sp.representative = pop_by_id[rep_gid].clone()
            alive[sid] = sp
        self.species = alive

        self._adjust_threshold()
        return genome_to_species

    def _adjust_threshold(self) -> None:
        target = self.cfg.species.target_species
        if target <= 0:
            return

        n_species = len(self.species)
        step = self.cfg.species.threshold_adjust_step
        if n_species > target:
            self.threshold += step
        elif n_species < target:
            self.threshold -= step

        self.threshold = max(self.threshold, self.cfg.species.min_threshold)
