from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class TaskData:
    name: str
    train_x: jnp.ndarray
    train_y: jnp.ndarray
    test_x: jnp.ndarray
    test_y: jnp.ndarray


def _shuffle(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    return x[idx], y[idx]


def _generate_xor(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-5.0, 5.0, size=(n, 2)) + rng.normal(0.0, noise, size=(n, 2))
    y = ((x[:, 0] > 0) & (x[:, 1] > 0)) | ((x[:, 0] < 0) & (x[:, 1] < 0))
    return x.astype(np.float32), y.astype(np.float32)


def _generate_spiral(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2

    def _one_spiral(delta_t: float, label: float) -> tuple[np.ndarray, np.ndarray]:
        i = np.arange(half, dtype=np.float32)
        r = i / max(half, 1) * 6.0
        t = 1.75 * i / max(half, 1) * 2.0 * np.pi + delta_t
        x = r * np.sin(t) + rng.uniform(-1.0, 1.0, size=(half,)) * noise
        y = r * np.cos(t) + rng.uniform(-1.0, 1.0, size=(half,)) * noise
        pts = np.stack([x, y], axis=1)
        labels = np.full((half,), label, dtype=np.float32)
        return pts, labels

    p1, l1 = _one_spiral(0.0, 1.0)
    p2, l2 = _one_spiral(np.pi, 0.0)
    x = np.concatenate([p1, p2], axis=0)
    y = np.concatenate([l1, l2], axis=0)

    if x.shape[0] < n:
        x_extra, y_extra = _generate_spiral(2, noise, rng)
        x = np.concatenate([x, x_extra[: n - x.shape[0]]], axis=0)
        y = np.concatenate([y, y_extra[: n - y.shape[0]]], axis=0)

    return x.astype(np.float32), y.astype(np.float32)


def _generate_circle(n: int, noise: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    radius = 5.0

    def label_fn(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
        return ((xv * xv + yv * yv) < (radius * 0.5) ** 2).astype(np.float32)

    r_in = rng.uniform(0.0, radius * 0.5, size=(half,))
    a_in = rng.uniform(0.0, 2 * np.pi, size=(half,))
    x_in = r_in * np.sin(a_in)
    y_in = r_in * np.cos(a_in)
    x_in += rng.uniform(-radius, radius, size=(half,)) * (noise / 3.0)
    y_in += rng.uniform(-radius, radius, size=(half,)) * (noise / 3.0)
    l_in = label_fn(x_in, y_in)

    r_out = rng.uniform(radius * 0.75, radius, size=(half,))
    a_out = rng.uniform(0.0, 2 * np.pi, size=(half,))
    x_out = r_out * np.sin(a_out)
    y_out = r_out * np.cos(a_out)
    x_out += rng.uniform(-radius, radius, size=(half,)) * (noise / 3.0)
    y_out += rng.uniform(-radius, radius, size=(half,)) * (noise / 3.0)
    l_out = label_fn(x_out, y_out)

    x = np.concatenate([np.stack([x_in, y_in], axis=1), np.stack([x_out, y_out], axis=1)], axis=0)
    y = np.concatenate([l_in, l_out], axis=0)

    if x.shape[0] < n:
        x2, y2 = _generate_circle(2, noise, rng)
        x = np.concatenate([x, x2[: n - x.shape[0]]], axis=0)
        y = np.concatenate([y, y2[: n - y.shape[0]]], axis=0)

    return x.astype(np.float32), y.astype(np.float32)


def make_dataset(task: str, train_size: int, test_size: int, noise: float, seed: int) -> TaskData:
    rng = np.random.default_rng(seed)

    if task == "circle":
        tx, ty = _generate_circle(train_size, noise, rng)
        vx, vy = _generate_circle(test_size, noise, rng)
    elif task == "xor":
        tx, ty = _generate_xor(train_size, noise, rng)
        vx, vy = _generate_xor(test_size, noise, rng)
    elif task == "spiral":
        tx, ty = _generate_spiral(train_size, noise, rng)
        vx, vy = _generate_spiral(test_size, noise, rng)
    else:
        raise ValueError(f"Unknown task: {task}")

    tx, ty = _shuffle(tx, ty, rng)
    vx, vy = _shuffle(vx, vy, rng)

    return TaskData(
        name=task,
        train_x=jnp.asarray(tx, dtype=jnp.float32),
        train_y=jnp.asarray(ty, dtype=jnp.float32),
        test_x=jnp.asarray(vx, dtype=jnp.float32),
        test_y=jnp.asarray(vy, dtype=jnp.float32),
    )
