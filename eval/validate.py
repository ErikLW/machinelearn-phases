from pathlib import Path

import machinelearn_phases as mlp
import numpy as np
import jax.numpy as jnp

L = 20
root = Path(__file__).resolve().parents[1]
params = mlp.load_model_params(
    root / "optimized_nn_params" / f"model_params_velo_fast_L{L}.npz", L=L
)

data = mlp.load_lattice_data(root / "training_data" / f"validation_set_hard_L{L}.npz", L=L)

X = jnp.array(data["X"])
Y = jnp.array(data["Y"])


res_arr = []

for i in range(len(X)):
    x = X[i]
    probabilities = mlp.classifier(params, x)
    #print(probabilities)
    prediction = mlp.max_unit_vector(probabilities)
    if (prediction == Y[i]).all():
        res_arr.append(1)
    else:
        res_arr.append(0)

    #deviation = np.linalg.norm(probabilities - Y[i])
    #print(deviation)
    #res_arr.append(deviation)

#print(probabilities)
#print(res_arr)
correctness_percent = np.mean(res_arr)*100
print(f"the prediction is true in {correctness_percent}% of the time!")