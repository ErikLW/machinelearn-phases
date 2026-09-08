import jax.numpy as jnp
import ising_mc as ising


def get_training_set(T1, T2):

    #This should be done with variable size in the future

    temp, energy, vare, m, varm, absm, configs_T1 = ising.run_metropolis(10, 1/T1, 1, 0, 3000, 100000, 1000, return_configurations=True)

    temp, energy, vare, m, varm, absm, configs_T2 = ising.run_metropolis(10, 1/T2, 1, 0, 3000, 100000, 1000, return_configurations=True)

    x_in1 = jnp.stack([x.flatten() for x in configs_T1[:3000]], axis=0)
    print(x_in1.shape)
    x_in2 = jnp.stack([x.flatten() for x in configs_T2[:3000]], axis=0)
    print(x_in2.shape)

    print('stacked from now')
    X = jnp.concatenate([x_in1, x_in2], axis=0)
    print(X.shape)

    y_in1 = jnp.stack([jnp.array([0, 1])] * 3000, axis = 0)
    y_in2 = jnp.stack([jnp.array([1, 0])] * 3000, axis = 0)
    Y = jnp.concatenate([y_in1,y_in2], axis = 0)
    print(Y.shape)

    return X, Y