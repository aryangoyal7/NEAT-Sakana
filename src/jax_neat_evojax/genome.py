from __future__ import annotations

import copy
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .activations import apply_activation
from .config import EvolutionConfig
from .genes import ConnectionGene, NodeGene
from .innovation import InnovationTracker


@dataclass
class Genome:
    genome_id: int
    nodes: dict[int, NodeGene]
    connections: dict[tuple[int, int], ConnectionGene]
    fitness: float | None = None

    def clone(self, new_id: int | None = None) -> "Genome":
        return Genome(
            genome_id=self.genome_id if new_id is None else new_id,
            nodes=copy.deepcopy(self.nodes),
            connections=copy.deepcopy(self.connections),
            fitness=self.fitness,
        )

    @property
    def input_ids(self) -> list[int]:
        return sorted(k for k, n in self.nodes.items() if n.kind == "input")

    @property
    def output_ids(self) -> list[int]:
        return sorted(k for k, n in self.nodes.items() if n.kind == "output")

    @property
    def hidden_ids(self) -> list[int]:
        return sorted(k for k, n in self.nodes.items() if n.kind == "hidden")

    def complexity(self) -> tuple[int, int]:
        enabled = sum(1 for c in self.connections.values() if c.enabled)
        return len(self.hidden_ids), enabled

    def to_phenotype(self) -> "FeedForwardPhenotype":
        return FeedForwardPhenotype(self)

    def mutate(
        self,
        rng: np.random.Generator,
        cfg: EvolutionConfig,
        tracker: InnovationTracker,
    ) -> None:
        self._mutate_weights_and_nodes(rng, cfg)

        if rng.random() < cfg.mutation.add_node_rate:
            self._mutate_add_node(rng, cfg, tracker)

        if rng.random() < cfg.mutation.add_conn_rate:
            self._mutate_add_connection(rng, cfg, tracker)

        if rng.random() < cfg.mutation.toggle_conn_rate and self.connections:
            pair = list(self.connections.keys())[rng.integers(len(self.connections))]
            self.connections[pair].enabled = not self.connections[pair].enabled

        self._disable_skip_connections()

    def _mutate_weights_and_nodes(
        self,
        rng: np.random.Generator,
        cfg: EvolutionConfig,
    ) -> None:
        mcfg = cfg.mutation

        for conn in self.connections.values():
            if rng.random() < mcfg.weight_mutate_rate:
                if rng.random() < mcfg.weight_replace_rate:
                    conn.weight = float(rng.uniform(-1.0, 1.0))
                else:
                    conn.weight += float(rng.normal(0.0, mcfg.weight_mutate_power))
                conn.weight = float(np.clip(conn.weight, -mcfg.weight_clip, mcfg.weight_clip))

        for node in self.nodes.values():
            if node.kind == "input":
                continue
            if rng.random() < mcfg.bias_mutate_rate:
                node.bias += float(rng.normal(0.0, mcfg.bias_mutate_power))
                node.bias = float(np.clip(node.bias, -mcfg.weight_clip, mcfg.weight_clip))
            if node.kind == "hidden" and rng.random() < mcfg.activation_mutate_rate:
                node.activation = rng.choice(cfg.activation_options).item()

    def _mutate_add_connection(
        self,
        rng: np.random.Generator,
        cfg: EvolutionConfig,
        tracker: InnovationTracker,
    ) -> None:
        candidates: list[tuple[int, int]] = []
        node_items = list(self.nodes.items())

        for src_id, src_node in node_items:
            if src_node.kind == "output":
                continue
            for dst_id, dst_node in node_items:
                if dst_node.kind == "input" or src_id == dst_id:
                    continue
                if src_node.layer >= dst_node.layer:
                    continue
                if (src_id, dst_id) in self.connections:
                    continue
                if self._has_intermediate_layer(src_node.layer, dst_node.layer):
                    continue
                candidates.append((src_id, dst_id))

        if not candidates:
            return

        src, dst = candidates[rng.integers(len(candidates))]
        innov = tracker.get_connection_innovation(src, dst)
        self.connections[(src, dst)] = ConnectionGene(
            innovation=innov,
            src=src,
            dst=dst,
            weight=float(rng.uniform(-1.0, 1.0)),
            enabled=True,
        )

    def _mutate_add_node(
        self,
        rng: np.random.Generator,
        cfg: EvolutionConfig,
        tracker: InnovationTracker,
    ) -> None:
        enabled_pairs = [k for k, c in self.connections.items() if c.enabled]
        if not enabled_pairs:
            return

        pair = enabled_pairs[rng.integers(len(enabled_pairs))]
        old_conn = self.connections[pair]
        old_conn.enabled = False

        src, dst = pair
        src_layer = self.nodes[src].layer
        dst_layer = self.nodes[dst].layer
        new_node_id, in_innov, out_innov = tracker.get_split(src, dst)

        if new_node_id not in self.nodes:
            self.nodes[new_node_id] = NodeGene(
                node_id=new_node_id,
                kind="hidden",
                layer=(src_layer + dst_layer) / 2.0,
                activation=rng.choice(cfg.activation_options).item(),
                bias=0.0,
            )

        self.connections[(src, new_node_id)] = ConnectionGene(
            innovation=in_innov,
            src=src,
            dst=new_node_id,
            weight=1.0,
            enabled=True,
        )
        self.connections[(new_node_id, dst)] = ConnectionGene(
            innovation=out_innov,
            src=new_node_id,
            dst=dst,
            weight=old_conn.weight,
            enabled=True,
        )

    def _has_intermediate_layer(self, src_layer: float, dst_layer: float) -> bool:
        eps = 1e-9
        for node in self.nodes.values():
            if src_layer + eps < node.layer < dst_layer - eps:
                return True
        return False

    def _disable_skip_connections(self) -> None:
        for conn in self.connections.values():
            if not conn.enabled:
                continue
            src_layer = self.nodes[conn.src].layer
            dst_layer = self.nodes[conn.dst].layer
            if self._has_intermediate_layer(src_layer, dst_layer):
                conn.enabled = False


class FeedForwardPhenotype:
    def __init__(self, genome: Genome):
        self._nodes = genome.nodes
        self._input_ids = genome.input_ids
        self._output_ids = genome.output_ids

        self._compute_order = sorted(
            [
                n.node_id
                for n in genome.nodes.values()
                if n.kind in ("hidden", "output")
            ],
            key=lambda nid: (genome.nodes[nid].layer, nid),
        )

        incoming: dict[int, list[ConnectionGene]] = {nid: [] for nid in self._compute_order}
        for conn in genome.connections.values():
            if conn.enabled and conn.dst in incoming:
                incoming[conn.dst].append(conn)
        for nid in incoming:
            incoming[nid] = sorted(incoming[nid], key=lambda c: c.src)
        self._incoming = incoming

    def forward(self, obs: jnp.ndarray) -> jnp.ndarray:
        obs = jnp.asarray(obs, dtype=jnp.float32).reshape(-1)
        acts: dict[int, jnp.ndarray] = {}

        for i, nid in enumerate(self._input_ids):
            acts[nid] = obs[i] if i < obs.shape[0] else jnp.float32(0.0)

        for nid in self._compute_order:
            node = self._nodes[nid]
            total = jnp.float32(node.bias)
            for conn in self._incoming[nid]:
                if conn.src not in acts:
                    continue
                total = total + acts[conn.src] * jnp.float32(conn.weight)
            acts[nid] = apply_activation(node.activation, total)

        if not self._output_ids:
            return jnp.zeros((0,), dtype=jnp.float32)
        return jnp.stack([acts[nid] for nid in self._output_ids])

    def action_binary(self, obs: np.ndarray) -> np.ndarray:
        out = self.forward(jnp.asarray(obs, dtype=jnp.float32))
        return np.asarray((out > 0).astype(jnp.float32))



def create_initial_genome(
    genome_id: int,
    cfg: EvolutionConfig,
    tracker: InnovationTracker,
    rng: np.random.Generator,
) -> Genome:
    nodes: dict[int, NodeGene] = {}
    connections: dict[tuple[int, int], ConnectionGene] = {}

    for nid in range(cfg.input_size):
        nodes[nid] = NodeGene(
            node_id=nid,
            kind="input",
            layer=0.0,
            activation="identity",
            bias=0.0,
        )

    out_start = cfg.input_size
    for oid in range(cfg.output_size):
        nid = out_start + oid
        nodes[nid] = NodeGene(
            node_id=nid,
            kind="output",
            layer=1.0,
            activation="tanh",
            bias=float(rng.normal(0.0, 0.1)),
        )

    for src in range(cfg.input_size):
        for dst in range(out_start, out_start + cfg.output_size):
            innov = tracker.get_connection_innovation(src, dst)
            connections[(src, dst)] = ConnectionGene(
                innovation=innov,
                src=src,
                dst=dst,
                weight=float(rng.uniform(-1.0, 1.0)),
                enabled=True,
            )

    return Genome(genome_id=genome_id, nodes=nodes, connections=connections)



def compatibility_distance(genome_a: Genome, genome_b: Genome, cfg: EvolutionConfig) -> float:
    scfg = cfg.species

    a_by_innov = {c.innovation: c for c in genome_a.connections.values()}
    b_by_innov = {c.innovation: c for c in genome_b.connections.values()}
    a_innovs = set(a_by_innov.keys())
    b_innovs = set(b_by_innov.keys())

    matching = a_innovs & b_innovs
    union = a_innovs | b_innovs

    if a_innovs and b_innovs:
        a_max = max(a_innovs)
        b_max = max(b_innovs)
    else:
        a_max = b_max = 0

    excess = 0
    disjoint = 0
    for innov in union - matching:
        if innov > a_max or innov > b_max:
            excess += 1
        else:
            disjoint += 1

    n = max(len(a_by_innov), len(b_by_innov), 1)
    if n < 20:
        n = 1

    if matching:
        w_diff = float(np.mean([abs(a_by_innov[i].weight - b_by_innov[i].weight) for i in matching]))
    else:
        w_diff = 0.0

    non_input_a = {nid: n for nid, n in genome_a.nodes.items() if n.kind != "input"}
    non_input_b = {nid: n for nid, n in genome_b.nodes.items() if n.kind != "input"}

    node_common = set(non_input_a.keys()) & set(non_input_b.keys())
    node_union = set(non_input_a.keys()) | set(non_input_b.keys())
    node_disjoint = len(node_union - node_common)

    if node_common:
        b_diff = float(np.mean([abs(non_input_a[i].bias - non_input_b[i].bias) for i in node_common]))
        a_mismatch = float(np.mean([non_input_a[i].activation != non_input_b[i].activation for i in node_common]))
    else:
        b_diff = 0.0
        a_mismatch = 0.0

    topo_term = scfg.compatibility_disjoint_coeff * (disjoint + excess) / n
    node_term = scfg.compatibility_disjoint_coeff * node_disjoint / max(len(node_union), 1)
    weight_term = scfg.compatibility_weight_coeff * w_diff
    bias_term = scfg.compatibility_bias_coeff * b_diff
    act_term = scfg.compatibility_activation_coeff * a_mismatch
    return topo_term + node_term + weight_term + bias_term + act_term



def crossover(
    rng: np.random.Generator,
    parent_a: Genome,
    parent_b: Genome,
    child_id: int,
) -> Genome:
    if parent_b.fitness is not None and parent_a.fitness is not None:
        if parent_b.fitness > parent_a.fitness:
            fitter, other = parent_b, parent_a
        elif parent_a.fitness > parent_b.fitness:
            fitter, other = parent_a, parent_b
        else:
            fitter, other = (parent_a, parent_b) if rng.random() < 0.5 else (parent_b, parent_a)
    else:
        fitter, other = parent_a, parent_b

    child_nodes = copy.deepcopy(fitter.nodes)
    for nid, node in other.nodes.items():
        if nid not in child_nodes:
            child_nodes[nid] = copy.deepcopy(node)

    child_conn: dict[tuple[int, int], ConnectionGene] = {}
    f_innov = {c.innovation: c for c in fitter.connections.values()}
    o_innov = {c.innovation: c for c in other.connections.values()}

    for innov in sorted(set(f_innov.keys()) | set(o_innov.keys())):
        gene = None
        if innov in f_innov and innov in o_innov:
            pick = f_innov[innov] if rng.random() < 0.5 else o_innov[innov]
            gene = copy.deepcopy(pick)
            if (not f_innov[innov].enabled or not o_innov[innov].enabled) and rng.random() < 0.75:
                gene.enabled = False
        elif innov in f_innov:
            gene = copy.deepcopy(f_innov[innov])
        else:
            # Disjoint/excess from less fit parent are dropped.
            continue

        if gene.src not in child_nodes or gene.dst not in child_nodes:
            continue
        if child_nodes[gene.src].layer >= child_nodes[gene.dst].layer:
            continue
        child_conn[(gene.src, gene.dst)] = gene

    child = Genome(genome_id=child_id, nodes=child_nodes, connections=child_conn)
    child._disable_skip_connections()
    return child
