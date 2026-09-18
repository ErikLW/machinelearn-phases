from pathlib import Path

import numpy as np
import machinelearn_phases as mlp

L = 20
output_dir = Path(__file__).resolve().parent

X,Y = mlp.get_training_set(1.8,2.7, L=L)

np.savez_compressed(
    output_dir / f"training_set_L{L}.npz",
    L=L,
    X=np.asarray(X),
    Y=np.asarray(Y)
)

X_vali1,Y_vali1 = mlp.get_training_set(1.5,3, L=L)

np.savez_compressed(
    output_dir / f"validation_set_easy_L{L}.npz",
    L=L,
    X=np.asarray(X_vali1),
    Y=np.asarray(Y_vali1)
)

X_vali2,Y_vali2 = mlp.get_training_set(2,2.5, L=L)

np.savez_compressed(
    output_dir / f"validation_set_hard_L{L}.npz",
    L=L,
    X=np.asarray(X_vali2),
    Y=np.asarray(Y_vali2)
)