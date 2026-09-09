import jax
import jax.numpy as jnp
import numpy as np

import machinelearn_phases as mlp


def test_model_archive_round_trip(tmp_path):
    params = mlp.init_network_params((4, 3, 2), jax.random.key(4))
    path = tmp_path / "data" / "model_params.npz"
    mlp.save_model_params(params, path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"n_layers", "W_0", "W_1", "b_0", "b_1"}
        assert int(archive["n_layers"]) == 2
    restored = mlp.load_model_params(path)
    for original, loaded in zip(jax.tree.leaves(params), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(original, loaded)
    x = jnp.arange(4, dtype=float)
    np.testing.assert_array_equal(
        mlp.classifier(params, x), mlp.classifier(restored, x)
    )
