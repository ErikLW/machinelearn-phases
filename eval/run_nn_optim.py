import jax
import jax.numpy as jnp
import machinelearn_phases as mlp
import pandas as pd
import numpy as np


data = np.load("../training_data/training_set.npz")

X = jnp.array(data["X"])
Y = jnp.array(data["Y"])

key_paras = jax.random.PRNGKey(1)
params = mlp.init_network_params((100,25,6,2), key_paras, ini_bias_zero=True)

results = []
key = jax.random.PRNGKey(0)

for i in range(5000):
    #cost, gradval = jax.value_and_grad(mlp.cost)(params, x=x_in, y_target=y_t)
    key, subkey = jax.random.split(key)

    cost, gradval = jax.value_and_grad(mlp.batch_cost)(params, X = X, Y = Y, batch_size = 32, rand_key = subkey)
    results.append(
        {"iter": i,
         "cost": cost}
    )
    if i % 100 == 0:
        print(subkey)
        print(i)
        print(cost)

    params = mlp.update(params, gradval, 0.01)

df = pd.DataFrame(results)
df.to_csv("results.csv", index = False)