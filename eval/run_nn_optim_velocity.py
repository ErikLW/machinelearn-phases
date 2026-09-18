from pathlib import Path

import jax
import jax.numpy as jnp
import machinelearn_phases as mlp
import pandas as pd
import numpy as np


L = 20
root = Path(__file__).resolve().parents[1]

data = mlp.load_lattice_data(root / "training_data" / f"training_set_L{L}.npz", L=L)

X = jnp.array(data["X"])
Y = jnp.array(data["Y"])

#flip all configurations in the training set to avoid learning only one direction
X = jnp.concatenate([X, -X], axis=0)
Y = jnp.concatenate([Y,  Y], axis=0)

data_vali1 = mlp.load_lattice_data(root / "training_data" / f"validation_set_easy_L{L}.npz", L=L)

X_vali1 = jnp.array(data_vali1["X"])
Y_vali1 = jnp.array(data_vali1["Y"])

data_vali2 = mlp.load_lattice_data(root / "training_data" / f"validation_set_hard_L{L}.npz", L=L)

X_vali2 = jnp.array(data_vali2["X"])
Y_vali2 = jnp.array(data_vali2["Y"])



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
velocity = None
for i in range(5000):
    #cost, gradval = jax.value_and_grad(mlp.cost)(params, x=x_in, y_target=y_t)
    key, subkey = jax.random.split(key)

    cost, gradval = jax.value_and_grad(mlp.batch_cost)(params, X = X, Y = Y, batch_size = 32, rand_key = subkey, loss=loss)

    cost_vali_easy = mlp.batch_cost(params, X = X_vali1, Y = Y_vali1, batch_size = 32, rand_key = subkey, loss=loss)
    cost_vali_hard = mlp.batch_cost(params, X = X_vali2, Y = Y_vali2, batch_size = 32, rand_key = subkey, loss=loss)


    results.append(
        {"iter": i,
         "cost": cost,
         "cost_vali_easy": cost_vali_easy,
         "cost_vali_hard": cost_vali_hard}
    )
    if i % 100 == 0:
        print(subkey)
        print(i)
        print(cost)
        print(cost_vali_easy)
        print(cost_vali_hard)

    params, velocity = mlp.update(params, gradval, 0.01, momentum=0.9, velocity=velocity)

df = pd.DataFrame(results)
df.to_csv(root / "eval" / f"results_velo_fast_L{L}.csv", index = False)
model_path = root / "optimized_nn_params" / f"model_params_velo_fast_L{L}.npz"
mlp.save_model_params(params, model_path)
