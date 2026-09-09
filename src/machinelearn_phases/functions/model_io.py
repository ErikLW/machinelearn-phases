"""Save and restore network weights and biases without pickle."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np


def save_model_params(params, path):
    """Save a parameter tuple to a compressed .npz archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    weights, biases = params
    arrays = {"n_layers": np.asarray(len(weights))}
    arrays.update({f"W_{i}": np.asarray(W) for i, W in enumerate(weights)})
    arrays.update({f"b_{i}": np.asarray(b) for i, b in enumerate(biases)})
    np.savez_compressed(path, **arrays)


def load_model_params(path):
    """Restore the (weights, biases) tuple as JAX arrays."""
    with np.load(path, allow_pickle=False) as archive:
        n_layers = int(archive["n_layers"])
        weights = [jnp.asarray(archive[f"W_{i}"]) for i in range(n_layers)]
        biases = [jnp.asarray(archive[f"b_{i}"]) for i in range(n_layers)]
    return weights, biases
