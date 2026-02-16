"""Backprop-NEAT in JAX for toy 2D classification tasks."""

from .config import BPNEATConfig
from .trainer import BackpropNEATTrainer

__all__ = ["BPNEATConfig", "BackpropNEATTrainer"]
