#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHONPATH=src python -m bp_neat_jax.cli \
  --task all \
  --generations 40 \
  --pop-size 60 \
  --bp-steps 220 \
  --bp-batch-size 32 \
  --bp-lr 0.01
