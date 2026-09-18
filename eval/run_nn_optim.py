from pathlib import Path

import jax
import jax.numpy as jnp
import machinelearn_phases as mlp
import pandas as pd
import numpy as np


L = 10
root = Path(__file__).resolve().parents[1]

data = mlp.load_lattice_data(root / "training_data" / f"training_set_L{L}.npz", L=L)

X = jnp.array(data["X"])
Y = jnp.array(data["Y"])

data_vali = mlp.load_lattice_data(root / "training_data" / f"validation_set_easy_L{L}.npz", L=L)

X_vali = jnp.array(data_vali["X"])
Y_vali = jnp.array(data_vali["Y"])


# Training options; defaults preserve the original loss and initialization.
loss = "cross_entropy" #squared_error"  # or "cross_entropy"
scale_by_layer_width = True

key_paras = jax.random.PRNGKey(1)
params = mlp.init_network_params(
    (X.shape[1],25,6,2), key_paras, ini_bias_zero=False,
    scale_by_layer_width=scale_by_layer_width,
)

results = []
key = jax.random.PRNGKey(0)

for i in range(15000):
    #cost, gradval = jax.value_and_grad(mlp.cost)(params, x=x_in, y_target=y_t)
    key, subkey = jax.random.split(key)

    cost, gradval = jax.value_and_grad(mlp.batch_cost)(params, X = X, Y = Y, batch_size = 32, rand_key = subkey, loss=loss)

    cost_vali = mlp.batch_cost(params, X = X_vali, Y = Y_vali, batch_size = 32, rand_key = subkey, loss=loss)

    results.append(
        {"iter": i,
         "cost": cost,
         "cost_vali": cost_vali}
    )
    if i % 100 == 0:
        print(subkey)
        print(i)
        print(cost)
        print(cost_vali)

    params = mlp.update(params, gradval, 0.01)

df = pd.DataFrame(results)
df.to_csv(root / "eval" / f"results_L{L}.csv", index = False)
model_path = root / "data" / f"model_params_L{L}.npz"
mlp.save_model_params(params, model_path)
