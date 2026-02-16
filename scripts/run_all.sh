#!/usr/bin/env bash
set -euo pipefail

python -m jax_neat_evojax.cli \
  --mode direct_vs_builtin \
  --generations 60 \
  --pop-size 80 \
  --episodes-per-genome 3

python -m jax_neat_evojax.cli \
  --mode selfplay_then_builtin \
  --generations 60 \
  --pop-size 80 \
  --episodes-per-genome 2
