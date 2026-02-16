# Backprop-NEAT in JAX (Toy 2D Classification)

This folder contains a dedicated Backprop-NEAT implementation in JAX for:

- Circle classification
- XOR classification
- Spiral classification

The implementation evolves **feed-forward architectures** with NEAT while training each genome's weights using **backpropagation** (Adam) in JAX.

## Constraints applied

- Feed-forward only (acyclic).
- Differentiable activations only (default: `tanh`, `relu`, `sigmoid`, `swish`, `softplus`).
- Complexity penalty in fitness to encourage compact architectures.

## Engineering optimizations included

- JIT-compiled per-genome loss/gradient functions.
- Mini-batch Adam for fast inner-loop weight optimization.
- Early stopping checks during backprop cycles.
- Lamarckian inheritance of trained weights across generations.

## Install

```bash
cd /Users/aryangoyal/Documents/New\ project/backprop_neat_jax
pip install -e .
```

## Run

Single task:

```bash
python -m bp_neat_jax.cli --task spiral --generations 40 --pop-size 60
```

All tasks:

```bash
python -m bp_neat_jax.cli --task all --generations 40 --pop-size 60
```

## Outputs

Each run writes to:

```text
backprop_neat_jax/artifacts/<task>_<timestamp>/
```

Artifacts include:

- `history.csv`
- `plots/training_curves.png`
- `plots/decision_boundary_best.png`
- `plots/network_best.png`
- `plots/network_gen_*.png`
- `report.md`

`report.md` contains commentary on complexification, activation usage, and interesting structures.
