from __future__ import annotations

import copy
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .activations import ACTIVATION_TO_ID, apply_activation_by_id
from .config import BPNEATConfig
from .genes import ConnectionGene, NodeGene
from .innovation import InnovationTracker


@dataclass
class Genome:
    genome_id: int
    nodes: dict[int, NodeGene]
    connections: dict[tuple[int, int], ConnectionGene]
    fitness: float | None = None
    train_acc: float | None = None
    test_acc: float | None = None
    test_loss: float | None = None

    def clone(self, new_id: int | None = None) -> "Genome":
        return Genome(
            genome_id=self.genome_id if new_id is None else new_id,
            nodes=copy.deepcopy(self.nodes),
            connections=copy.deepcopy(self.connections),
            fitness=self.fitness,
            train_acc=self.train_acc,
            test_acc=self.test_acc,
            test_loss=self.test_loss,
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
        enabled_conn = sum(1 for c in self.connections.values() if c.enabled)
        return len(self.hidden_ids), enabled_conn

    def mutate(self, rng: np.random.Generator, cfg: BPNEATConfig, tracker: InnovationTracker) -> None:
        self._mutate_params(rng, cfg)

        if rng.random() < cfg.mutation.add_node_rate:
            self._mutate_add_node(rng, cfg, tracker)

        if rng.random() < cfg.mutation.add_conn_rate:
            self._mutate_add_connection(rng, tracker)

        if rng.random() < cfg.mutation.toggle_conn_rate and self.connections:
            pair = list(self.connections.keys())[rng.integers(len(self.connections))]
            self.connections[pair].enabled = not self.connections[pair].enabled

        self._disable_skip_connections()

    def _mutate_params(self, rng: np.random.Generator, cfg: BPNEATConfig) -> None:
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

    def _mutate_add_connection(self, rng: np.random.Generator, tracker: InnovationTracker) -> None:
        candidates: list[tuple[int, int]] = []
        node_items = list(self.nodes.items())

        for src_id, src_node in node_items:
            if src_node.kind == "output":
                continue
            for dst_id, dst_node in node_items:
                if src_id == dst_id or dst_node.kind == "input":
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

    def _mutate_add_node(self, rng: np.random.Generator, cfg: BPNEATConfig, tracker: InnovationTracker) -> None:
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

    def build_phenotype(self) -> "DifferentiablePhenotype":
        return DifferentiablePhenotype(self)


class DifferentiablePhenotype:
    def __init__(self, genome: Genome):
        self.genome = genome

        self.node_ids = sorted(genome.nodes.keys(), key=lambda nid: (genome.nodes[nid].layer, nid))
        self.id_to_idx = {nid: i for i, nid in enumerate(self.node_ids)}

        self.input_node_ids = genome.input_ids
        self.output_node_ids = genome.output_ids

        self.input_idx = np.asarray([self.id_to_idx[nid] for nid in self.input_node_ids], dtype=np.int32)
        self.output_idx = np.asarray([self.id_to_idx[nid] for nid in self.output_node_ids], dtype=np.int32)

        self.compute_idx = np.asarray(
            [
                self.id_to_idx[nid]
                for nid in sorted(
                    [n.node_id for n in genome.nodes.values() if n.kind in ("hidden", "output")],
                    key=lambda nid: (genome.nodes[nid].layer, nid),
                )
            ],
            dtype=np.int32,
        )

        self.activation_ids = np.asarray(
            [ACTIVATION_TO_ID.get(genome.nodes[nid].activation, 0) for nid in self.node_ids],
            dtype=np.int32,
        )

        enabled_conns = sorted(
            [c for c in genome.connections.values() if c.enabled],
            key=lambda c: (c.innovation, c.src, c.dst),
        )

        self.conn_pairs = [(c.src, c.dst) for c in enabled_conns]
        self.edge_src_idx = np.asarray([self.id_to_idx[c.src] for c in enabled_conns], dtype=np.int32)
        self.edge_dst_idx = np.asarray([self.id_to_idx[c.dst] for c in enabled_conns], dtype=np.int32)
        self.initial_edge_weights = np.asarray([c.weight for c in enabled_conns], dtype=np.float32)

        self.bias_node_ids = [nid for nid in self.node_ids if genome.nodes[nid].kind != "input"]
        self.bias_node_idx = np.asarray([self.id_to_idx[nid] for nid in self.bias_node_ids], dtype=np.int32)
        self.initial_biases = np.asarray([genome.nodes[nid].bias for nid in self.bias_node_ids], dtype=np.float32)

        idx_to_bias = {self.id_to_idx[nid]: i for i, nid in enumerate(self.bias_node_ids)}
        self.node_to_bias_param = {idx: idx_to_bias[idx] for idx in idx_to_bias}

        incoming_src: dict[int, np.ndarray] = {}
        incoming_edge: dict[int, np.ndarray] = {}
        for node_idx in self.compute_idx.tolist():
            edge_ids = np.where(self.edge_dst_idx == node_idx)[0]
            incoming_edge[node_idx] = edge_ids.astype(np.int32)
            incoming_src[node_idx] = self.edge_src_idx[edge_ids].astype(np.int32)

        self.incoming_edge = incoming_edge
        self.incoming_src = incoming_src

    def initial_params(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return (
            jnp.asarray(self.initial_edge_weights, dtype=jnp.float32),
            jnp.asarray(self.initial_biases, dtype=jnp.float32),
        )

    def _forward_single(self, params: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        weights, biases = params
        acts = jnp.zeros((len(self.node_ids),), dtype=jnp.float32)
        acts = acts.at[self.input_idx].set(x)

        for node_idx in self.compute_idx.tolist():
            src_idx = self.incoming_src[node_idx]
            edge_idx = self.incoming_edge[node_idx]
            total = jnp.float32(0.0)
            if edge_idx.size > 0:
                total = total + jnp.sum(acts[src_idx] * weights[edge_idx])
            b_ix = self.node_to_bias_param.get(node_idx, None)
            if b_ix is not None:
                total = total + biases[b_ix]
            act_id = int(self.activation_ids[node_idx])
            acts = acts.at[node_idx].set(apply_activation_by_id(act_id, total))

        logits = acts[self.output_idx]
        return logits.reshape(-1)

    def forward(self, params: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 1:
            return self._forward_single(params, x)
        return jax.vmap(lambda xi: self._forward_single(params, xi))(x)

    def loss(self, params: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        logits = self.forward(params, x).reshape(-1)
        y = y.reshape(-1)
        # Stable logistic loss.
        losses = jnp.maximum(logits, 0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
        return jnp.mean(losses)

    def accuracy(self, params: tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        logits = self.forward(params, x).reshape(-1)
        preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
        return jnp.mean((preds == y.reshape(-1)).astype(jnp.float32))

    def update_genome(self, genome: Genome, params: tuple[jnp.ndarray, jnp.ndarray]) -> None:
        weights, biases = params
        weights_np = np.asarray(weights, dtype=np.float32)
        biases_np = np.asarray(biases, dtype=np.float32)

        for i, pair in enumerate(self.conn_pairs):
            if pair in genome.connections:
                genome.connections[pair].weight = float(weights_np[i])

        for i, node_id in enumerate(self.bias_node_ids):
            if node_id in genome.nodes:
                genome.nodes[node_id].bias = float(biases_np[i])


def create_initial_genome(
    genome_id: int,
    cfg: BPNEATConfig,
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
    for i in range(cfg.output_size):
        nid = out_start + i
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
                weight=float(rng.uniform(-0.3, 0.3)),
                enabled=True,
            )

    return Genome(genome_id=genome_id, nodes=nodes, connections=connections)


def compatibility_distance(a: Genome, b: Genome, cfg: BPNEATConfig) -> float:
    sc = cfg.species

    a_conn = {c.innovation: c for c in a.connections.values()}
    b_conn = {c.innovation: c for c in b.connections.values()}

    a_set = set(a_conn.keys())
    b_set = set(b_conn.keys())
    matching = a_set & b_set
    union = a_set | b_set

    if a_set and b_set:
        a_max = max(a_set)
        b_max = max(b_set)
    else:
        a_max = b_max = 0

    excess = 0
    disjoint = 0
    for innov in union - matching:
        if innov > a_max or innov > b_max:
            excess += 1
        else:
            disjoint += 1

    n_conn = max(len(a_conn), len(b_conn), 1)
    if n_conn < 20:
        n_conn = 1

    if matching:
        w_diff = float(np.mean([abs(a_conn[i].weight - b_conn[i].weight) for i in matching]))
    else:
        w_diff = 0.0

    non_input_a = {nid: n for nid, n in a.nodes.items() if n.kind != "input"}
    non_input_b = {nid: n for nid, n in b.nodes.items() if n.kind != "input"}

    node_common = set(non_input_a.keys()) & set(non_input_b.keys())
    node_union = set(non_input_a.keys()) | set(non_input_b.keys())
    node_disjoint = len(node_union - node_common)

    if node_common:
        b_diff = float(np.mean([abs(non_input_a[i].bias - non_input_b[i].bias) for i in node_common]))
        act_diff = float(np.mean([non_input_a[i].activation != non_input_b[i].activation for i in node_common]))
    else:
        b_diff = 0.0
        act_diff = 0.0

    topo_term = sc.compatibility_disjoint_coeff * (disjoint + excess) / n_conn
    node_term = sc.compatibility_disjoint_coeff * node_disjoint / max(len(node_union), 1)
    weight_term = sc.compatibility_weight_coeff * w_diff
    bias_term = sc.compatibility_bias_coeff * b_diff
    act_term = sc.compatibility_activation_coeff * act_diff

    return topo_term + node_term + weight_term + bias_term + act_term


def crossover(rng: np.random.Generator, parent_a: Genome, parent_b: Genome, child_id: int) -> Genome:
    if parent_a.fitness is not None and parent_b.fitness is not None:
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
        if innov in f_innov and innov in o_innov:
            pick = f_innov[innov] if rng.random() < 0.5 else o_innov[innov]
            gene = copy.deepcopy(pick)
            if (not f_innov[innov].enabled or not o_innov[innov].enabled) and rng.random() < 0.75:
                gene.enabled = False
        elif innov in f_innov:
            gene = copy.deepcopy(f_innov[innov])
        else:
            continue

        if gene.src not in child_nodes or gene.dst not in child_nodes:
            continue
        if child_nodes[gene.src].layer >= child_nodes[gene.dst].layer:
            continue
        child_conn[(gene.src, gene.dst)] = gene

    child = Genome(genome_id=child_id, nodes=child_nodes, connections=child_conn)
    child._disable_skip_connections()
    return child
