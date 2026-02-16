from __future__ import annotations

import jax.numpy as jnp


ACTIVATION_TO_ID = {
    "identity": 0,
    "tanh": 1,
    "relu": 2,
    "sigmoid": 3,
    "swish": 4,
    "softplus": 5,
}

ID_TO_ACTIVATION = {v: k for k, v in ACTIVATION_TO_ID.items()}


def apply_activation_by_id(act_id: int, x: jnp.ndarray) -> jnp.ndarray:
    y = x
    y = jnp.where(act_id == 1, jnp.tanh(x), y)
    y = jnp.where(act_id == 2, jnp.maximum(0.0, x), y)
    y = jnp.where(act_id == 3, jax_sigmoid(x), y)
    y = jnp.where(act_id == 4, x * jax_sigmoid(x), y)
    y = jnp.where(act_id == 5, jnp.log1p(jnp.exp(x)), y)
    return y


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))
