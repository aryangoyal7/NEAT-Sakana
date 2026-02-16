# Backprop-NEAT (JAX) Implementation Notes

## Scope

This subfolder implements a JAX-based Backprop-NEAT workflow inspired by:

- https://blog.otoro.net/2016/05/07/backprop-neat/
- https://github.com/hardmaru/backprop-neat-js/

Target tasks:

- Circle
- XOR
- Spiral

## Key design choices

- **Differentiable activations only**: `tanh`, `relu`, `sigmoid`, `swish`, `softplus`.
- **Feed-forward-only topology**: recurrent links are disallowed.
- **No skip bypass when intermediate layers exist**: encourages cleaner progressive complexification.
- **Backprop inside fitness**: each genome receives mini-batch Adam updates before scoring.
- **Complexity regularization**: fitness subtracts penalties for hidden-node and connection complexity.

## Performance-oriented additions

- JIT-compiled loss+grad functions per genome phenotype.
- Early stopping windows during genome-level backprop.
- Gradient clipping + weight decay for stable optimization.
- Lamarckian inheritance of trained parameters through genome gene updates.

## Main entry point

`/Users/aryangoyal/Documents/New project/backprop_neat_jax/src/bp_neat_jax/cli.py`

Example:

```bash
cd /Users/aryangoyal/Documents/New\ project/backprop_neat_jax
PYTHONPATH=src python -m bp_neat_jax.cli --task all --generations 40 --pop-size 60
```

## Output assets

For each task, outputs are written to:

`backprop_neat_jax/artifacts/<task>_<timestamp>/`

Including:

- `history.csv`
- `species_sizes.csv`
- `best_genome.json`
- `plots/training_curves.png`
- `plots/species_sizes.png`
- `plots/activation_usage.png`
- `plots/decision_boundary_best.png`
- `plots/network_best.png`
- `plots/network_gen_*.png`
- `report.md`
