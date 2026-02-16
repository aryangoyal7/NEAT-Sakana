# Implementation Notes: Feed-Forward NEAT in JAX + EvoJAX

## What was implemented

A full NEAT pipeline was added under `src/jax_neat_evojax/` with:

- Genome representation (node and connection genes with innovation IDs).
- Global innovation tracking for connection additions and node-split mutations.
- Feed-forward only topology constraints.
- Skip/residual-style bypass link suppression once intermediate layers exist.
- Mutation operators:
  - Weight/bias perturbation and weight replacement.
  - Hidden activation mutation.
  - Add node (split edge).
  - Add connection (acyclic, adjacent-layer only under the no-skip rule).
  - Enable/disable toggling.
- NEAT crossover by innovation alignment.
- Compatibility-distance-based speciation with adaptive threshold.
- Fitness sharing, elitism, and species-wise offspring allocation.

## EvoJAX integration

A SlimeVolley wrapper was implemented in:

- `src/jax_neat_evojax/envs/slimevolley_env.py`

It supports:

- NEAT policy on the right side vs built-in left policy.
- NEAT vs NEAT by setting both left and right actions.
- Frame capture for GIF output.

## Training modes implemented

- `direct_vs_builtin`
  - Every genome is evaluated directly against the built-in SlimeVolley agent.

- `selfplay_then_builtin`
  - Fitness is computed from NEAT-vs-NEAT pairwise matches (sides swapped).
  - Champion-vs-built-in scores are tracked for analysis.
  - Final champion is rendered vs built-in.

## Artifact generation

The CLI generates:

- `history.csv`
- `species_sizes.csv`
- champion JSON genome dump
- fitness/complexity/speciation plots
- champion topology plots across early/mid/final generations
- top-3 final network plots
- activation usage chart
- champion-vs-built-in GIF
- champion-vs-runnerup GIF (when available)
- markdown report with complexification commentary template

## How to run

```bash
PYTHONPATH=/Users/aryangoyal/Documents/New\ project/src:/Users/aryangoyal/Documents/New\ project/evojax \
python -m jax_neat_evojax.cli \
  --mode direct_vs_builtin \
  --generations 60 \
  --pop-size 80 \
  --episodes-per-genome 3
```

Or:

```bash
PYTHONPATH=/Users/aryangoyal/Documents/New\ project/src:/Users/aryangoyal/Documents/New\ project/evojax \
python -m jax_neat_evojax.cli \
  --mode selfplay_then_builtin \
  --generations 60 \
  --pop-size 80 \
  --episodes-per-genome 2
```

## Notes / limitations in this environment

- Code compiles (`py_compile`) successfully.
- End-to-end execution was blocked in this machine because `jax` is not currently installed in the active Python environment.
- Once `jax` is installed, the CLI should produce all requested plots/GIFs/report artifacts in `artifacts/<mode>_<timestamp>/`.
