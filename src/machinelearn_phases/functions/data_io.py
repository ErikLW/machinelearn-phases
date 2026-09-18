"""Read lattice datasets and check their dimensions."""

import numpy as np


def load_lattice_data(path, *, L):
    """Load arrays, accepting legacy archives without L if dimensions match."""
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if "L" in data and (data["L"].shape != () or data["L"].item() != L):
        raise ValueError(f"Dataset lattice size does not match L={L}: {path}")
    X = data["X"]
    if X.ndim != 2 or X.shape[1] != L**2:
        raise ValueError(f"Expected samples with {L**2} inputs for L={L}: {path}")
    if "Y" in data and data["Y"].shape != (len(X), 2):
        raise ValueError(f"Expected one two-class target per sample: {path}")
    return data
