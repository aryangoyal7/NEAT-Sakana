from __future__ import annotations

import jax.numpy as jnp


def apply_activation(name: str, x: jnp.ndarray) -> jnp.ndarray:
    if name == "tanh":
        return jnp.tanh(x)
    if name == "relu":
        return jnp.maximum(0.0, x)
    if name == "sigmoid":
        return jax_sigmoid(x)
    if name == "sin":
        return jnp.sin(jnp.pi * x)
    if name == "gauss":
        return jnp.exp(-(x * x) / 2.0)
    return x


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
