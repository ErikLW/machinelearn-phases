from pathlib import Path
import runpy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import machinelearn_phases as mlp
from machinelearn_phases.functions import aux_functions as aux


@pytest.mark.parametrize("L", [3, 10, 20])
def test_sampling_size_counts_and_sweeps(monkeypatch, L):
    calls = []

    def simulate(*args, **kwargs):
        calls.append(args)
        return (*([0] * 6), np.ones((args[4], args[0], args[0]), dtype=int))

    monkeypatch.setattr(aux.ising, "run_metropolis", simulate)
    X = mlp.get_samples(2., 3001, L=L)
    assert X.shape == (3001, L**2)
    assert calls[0][0] == L
    assert calls[0][5:7] == (1000 * L**2, 10 * L**2)
    X, Y = mlp.get_training_set(1., 3., L=L, num_samples=4,
                                burn_in_sweeps=2, sweeps_between_samples=3)
    assert X.shape == (8, L**2)
    np.testing.assert_array_equal(Y[:4], np.tile([0, 1], (4, 1)))
    np.testing.assert_array_equal(Y[4:], np.tile([1, 0], (4, 1)))
    assert calls[-1][5:7] == (2 * L**2, 3 * L**2)


@pytest.mark.parametrize("L", [0, -1, 1, 2.5, True])
def test_invalid_lattice_size(L):
    with pytest.raises(ValueError, match="L must"):
        mlp.get_samples(1., 2, L=L)


def test_dataset_metadata_and_dimensions(tmp_path):
    path = tmp_path / "samples.npz"
    np.savez(path, X=np.ones((2, 9)), Y=np.eye(2), L=3)
    assert mlp.load_lattice_data(path, L=3)["X"].shape == (2, 9)
    with pytest.raises(ValueError, match="lattice size"):
        mlp.load_lattice_data(path, L=4)
    np.savez(path, X=np.ones((2, 4)), L=3)
    with pytest.raises(ValueError, match="9 inputs"):
        mlp.load_lattice_data(path, L=3)
    np.savez(path, X=np.ones((2, 9)), Y=np.ones((1, 2)), L=3)
    with pytest.raises(ValueError, match="target"):
        mlp.load_lattice_data(path, L=3)
    np.savez(path, X=np.ones((2, 9)))
    assert mlp.load_lattice_data(path, L=3)["X"].shape == (2, 9)


@pytest.mark.parametrize("L", [3, 20])
def test_model_size_check_and_round_trip(tmp_path, L):
    params = mlp.init_network_params((L**2, 4, 2), jax.random.key(1))
    path = tmp_path / "model.npz"
    mlp.save_model_params(params, path)
    restored = mlp.load_model_params(path, L=L)
    assert restored[0][0].shape == (4, L**2)
    np.testing.assert_allclose(mlp.classifier(restored, jnp.ones(L**2)).sum(), 1.)
    with pytest.raises(ValueError, match="Model input width"):
        mlp.load_model_params(path, L=L+1)


def test_size_specific_script_workflow(tmp_path, monkeypatch, capsys):
    """Exercise generation, two real optimizer steps, loading and evaluation."""
    repo = Path(__file__).resolve().parents[1]
    scripts = [
        "training_data/generate_training_data.py",
        "temperature_data/generate_temp_samples.py",
        "eval/run_nn_optim.py", "eval/run_nn_optim_velocity.py",
        "eval/validate.py", "eval/predict_phase_diagram.py",
    ]
    for script in scripts:
        target = tmp_path / script
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((repo / script).read_text().replace("L = 10", "L = 3"))

    def samples(T, num_samples, *, L=10, **kwargs):
        return jnp.ones((num_samples, L**2), dtype=int)

    original_training_set = mlp.get_training_set

    def training_set(T1, T2, *, L):
        return original_training_set(T1, T2, L=L, num_samples=32)

    monkeypatch.setattr(aux, "get_samples", samples)
    monkeypatch.setattr(mlp, "get_samples", samples)
    monkeypatch.setattr(mlp, "get_training_set", training_set)
    # Launch from a different working directory to check script-relative paths.
    monkeypatch.chdir(tmp_path)
    for script in scripts[:2]:
        runpy.run_path(str(tmp_path / script))
    for script in scripts[2:4]:
        runpy.run_path(str(tmp_path / script),
                      init_globals={"range": lambda n: range(min(n, 2))})
    for stem in ["model_params", "model_params_velo_fast"]:
        params = mlp.load_model_params(tmp_path / "data" / f"{stem}_L3.npz", L=3)
        assert params[0][0].shape[1] == 9
    for stem in ["results", "results_velo_fast"]:
        assert len(pd.read_csv(tmp_path / "eval" / f"{stem}_L3.csv")) == 2
    monkeypatch.setattr(mlp, "classifier", lambda params, x: jnp.array([0., 1.]))
    capsys.readouterr()
    runpy.run_path(str(tmp_path / "eval/validate.py"))
    assert "50.0%" in capsys.readouterr().out
    runpy.run_path(str(tmp_path / "eval/predict_phase_diagram.py"))
    predictions = pd.read_csv(tmp_path / "eval/prediction_L3.csv")
    assert (predictions["L"] == 3).all()
    assert (predictions["phase"] == 1).all()
    assert (predictions["ordered fraction"] == 1.).all()
