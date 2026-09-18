from pathlib import Path

import numpy as np
import machinelearn_phases as mlp

L = 20
output_dir = Path(__file__).resolve().parent / "data"
output_dir.mkdir(parents=True, exist_ok=True)

temps = np.linspace(1.5, 3, 30)

for i in temps:
    print(i)
    X = mlp.get_samples(i,100, L=L)

    np.savez_compressed(
        output_dir / f"sample_set_L{L}_T_{i:.2f}.npz",
        X=np.asarray(X),
        L=L,
        T=i,
    )
