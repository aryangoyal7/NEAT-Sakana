from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MutationConfig:
    weight_mutate_rate: float = 0.85
    weight_mutate_power: float = 0.12
    weight_replace_rate: float = 0.05
    bias_mutate_rate: float = 0.30
    bias_mutate_power: float = 0.08
    activation_mutate_rate: float = 0.03
    add_node_rate: float = 0.18
    add_conn_rate: float = 0.45
    toggle_conn_rate: float = 0.02
    weight_clip: float = 6.0


@dataclass
class ReproductionConfig:
    elitism: int = 2
    survival_fraction: float = 0.35
    crossover_rate: float = 0.75


@dataclass
class SpeciesConfig:
    compatibility_threshold: float = 2.6
    compatibility_disjoint_coeff: float = 1.0
    compatibility_weight_coeff: float = 0.5
    compatibility_bias_coeff: float = 0.25
    compatibility_activation_coeff: float = 0.5
    target_species: int = 7
    threshold_adjust_step: float = 0.05
    min_threshold: float = 0.4


@dataclass
class BackpropConfig:
    steps: int = 220
    batch_size: int = 32
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    eval_interval: int = 20
    patience_windows: int = 4
    lamarckian_inheritance: bool = True


@dataclass
class FitnessConfig:
    complexity_conn_penalty: float = 0.02
    complexity_hidden_penalty: float = 0.015
    loss_weight: float = 0.25


@dataclass
class DatasetConfig:
    train_size: int = 200
    test_size: int = 200
    noise: float = 0.5


@dataclass
class BPNEATConfig:
    seed: int = 0
    input_size: int = 2
    output_size: int = 1
    pop_size: int = 60
    generations: int = 40
    activation_options: tuple[str, ...] = (
        "tanh",
        "relu",
        "sigmoid",
        "swish",
        "softplus",
    )
    mutation: MutationConfig = field(default_factory=MutationConfig)
    reproduction: ReproductionConfig = field(default_factory=ReproductionConfig)
    species: SpeciesConfig = field(default_factory=SpeciesConfig)
    backprop: BackpropConfig = field(default_factory=BackpropConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
