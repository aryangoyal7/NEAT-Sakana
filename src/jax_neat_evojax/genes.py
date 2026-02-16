from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeGene:
    node_id: int
    kind: str
    layer: float
    activation: str
    bias: float = 0.0


@dataclass
class ConnectionGene:
    innovation: int
    src: int
    dst: int
    weight: float
    enabled: bool = True
