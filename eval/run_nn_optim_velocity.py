from pathlib import Path

import jax
import jax.numpy as jnp
import machinelearn_phases as mlp
import pandas as pd
import numpy as np


data = np.load("../training_data/training_set.npz")

X = jnp.array(data["X"])
Y = jnp.array(data["Y"])

# Training options; defaults preserve the original loss and initialization.
loss = "cross_entropy" #squared_error"  # or "cross_entropy"
scale_by_layer_width = True

key_paras = jax.random.PRNGKey(1)
params = mlp.init_network_params(
    (100,25,6,2), key_paras, ini_bias_zero=False,
    scale_by_layer_width=scale_by_layer_width,
)

results = []
key = jax.random.PRNGKey(0)
velocity = None
for i in range(15000):
    #cost, gradval = jax.value_and_grad(mlp.cost)(params, x=x_in, y_target=y_t)
    key, subkey = jax.random.split(key)

    cost, gradval = jax.value_and_grad(mlp.batch_cost)(params, X = X, Y = Y, batch_size = 32, rand_key = subkey, loss=loss)
    results.append(
        {"iter": i,
         "cost": cost}
    )
    if i % 100 == 0:
        print(subkey)
        print(i)
        print(cost)

    params, velocity = mlp.update(params, gradval, 0.1, momentum=0.9, velocity=velocity)

df = pd.DataFrame(results)
df.to_csv("results_velo_fast.csv", index = False)
model_path = Path(__file__).resolve().parents[1] / "data" / "model_params_velo_fast.npz"
mlp.save_model_params(params, model_path)