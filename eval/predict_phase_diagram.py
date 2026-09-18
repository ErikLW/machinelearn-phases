from pathlib import Path

import machinelearn_phases as mlp
import numpy as np
import jax.numpy as jnp
import pandas as pd

N = 99 #number of samples to look at for majority voting.

L = 20

root = Path(__file__).resolve().parents[1]
params = mlp.load_model_params(
    root / "optimized_nn_params" / f"model_params_velo_fast_L{L}.npz", L=L
)


temps = np.linspace(1.5, 3, 30)

phase_arr = []
ordered_fraction_arr = []

for T in temps:
    data = mlp.load_lattice_data(root / "temperature_data" / "data" / f"sample_set_L{L}_T_{T:.2f}.npz", L=L)

    X = jnp.array(data["X"])

    if len(X) < N:
        raise ValueError(f"Need {N} samples at T={T}, found {len(X)}")

    maj_pred = np.array([0.,0.])

    for i in range(N):
        x = X[i]
        probabilities = mlp.classifier(params, x)
        prediction = mlp.max_unit_vector(probabilities)
        #print(prediction)
        maj_pred = maj_pred + prediction
        #print(maj_pred)

    ordered_fraction = maj_pred[1] / N
    ordered_fraction_arr.append(ordered_fraction)

    if (mlp.max_unit_vector(maj_pred) == np.array([1.0,0.0])).all():
        phase_arr.append(0)
    else:
        phase_arr.append(1)

    print(f"the NN predicts {phase_arr[-1]} at T = {T}!")
    print(f"The ordered fraction of the samples was {ordered_fraction}")
prediction_dict = {"L": L, "temperatures": temps,
                    "phase": phase_arr,
                    "ordered fraction": ordered_fraction_arr}

df = pd.DataFrame(prediction_dict)
df.to_csv(root / "eval" / f"prediction_L{L}.csv", index = False)
