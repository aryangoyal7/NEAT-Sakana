from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MutationConfig:
    weight_mutate_rate: float = 0.8
    weight_mutate_power: float = 0.5
    weight_replace_rate: float = 0.1
    bias_mutate_rate: float = 0.3
    bias_mutate_power: float = 0.2
    activation_mutate_rate: float = 0.03
    add_node_rate: float = 0.05
    add_conn_rate: float = 0.10
    toggle_conn_rate: float = 0.02
    weight_clip: float = 5.0


@dataclass
class ReproductionConfig:
    elitism: int = 1
    survival_fraction: float = 0.3
    crossover_rate: float = 0.75


@dataclass
class SpeciesConfig:
    compatibility_threshold: float = 2.8
    compatibility_disjoint_coeff: float = 1.0
    compatibility_weight_coeff: float = 0.5
    compatibility_bias_coeff: float = 0.2
    compatibility_activation_coeff: float = 0.4
    target_species: int = 8
    threshold_adjust_step: float = 0.05
    min_threshold: float = 0.5


@dataclass
class EvolutionConfig:
    pop_size: int = 80
    generations: int = 60
    input_size: int = 12
    output_size: int = 3
    seed: int = 0
    episodes_per_genome: int = 3
    max_steps: int = 3000
    activation_options: tuple[str, ...] = (
        "tanh",
        "relu",
        "sigmoid",
        "sin",
        "gauss",
        "identity",
    )
    mutation: MutationConfig = field(default_factory=MutationConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    species: SpeciesConfig = field(default_factory=SpeciesConfig)
