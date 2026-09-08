import numpy as np
import machinelearn_phases as mlp

X,Y = mlp.get_training_set(1,3)

np.savez_compressed(
    "training_set.npz",
    X=np.asarray(X),
    Y=np.asarray(Y)
)