from numbers import Integral

import jax.numpy as jnp
import ising_mc as ising


def get_samples(T, num_samples, *, L=10, burn_in_sweeps=1000,
                sweeps_between_samples=10):
    """Return flattened L-by-L configurations; one sweep is L**2 attempts."""

    *_, configs = ising.run_metropolis(
        L, 1/T, -1, 0.01, num_samples,
        burn_in_sweeps * L**2, sweeps_between_samples * L**2,
        return_configurations=True,
    )

    configs = jnp.asarray(configs)

    if configs.shape != (num_samples, L, L):
        raise ValueError("Simulator returned an unexpected configuration shape.")
    return configs.reshape(num_samples, L**2)


def get_training_set(T1, T2, *, L=10, num_samples=3000,
                     burn_in_sweeps=1000, sweeps_between_samples=10):
    """Generate equally sized classes: T1 -> [0, 1], T2 -> [1, 0]."""

    options = dict(L=L, burn_in_sweeps=burn_in_sweeps,
                   sweeps_between_samples=sweeps_between_samples)
    
    x1 = get_samples(T1, num_samples, **options)
    x2 = get_samples(T2, num_samples, **options)

    X = jnp.concatenate([x1, x2], axis=0)
    Y = jnp.concatenate([
        jnp.tile(jnp.array([0, 1]), (num_samples, 1)),
        jnp.tile(jnp.array([1, 0]), (num_samples, 1)),
    ], axis=0)
    
    return X, Y
