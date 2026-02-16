"""Feed-forward NEAT in JAX with EvoJAX integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import EvolutionConfig

if TYPE_CHECKING:
    from .evolution import NEATTrainer

__all__ = ["EvolutionConfig", "NEATTrainer"]


def __getattr__(name: str):
    if name == "NEATTrainer":
        from .evolution import NEATTrainer as _NEATTrainer

        return _NEATTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
