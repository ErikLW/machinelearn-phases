import jax
import jax.numpy as jnp
import numpy as np
import pytest

import machinelearn_phases as mlp


def example_params():
    return (
        [jnp.eye(2), jnp.array([[1., -1.], [-1., 1.]])],
        [jnp.zeros(2), jnp.zeros(2)],
    )


def test_initialization_uses_supplied_key():
    a = mlp.init_network_params((4, 3, 2), jax.random.key(1))
    b = mlp.init_network_params((4, 3, 2), jax.random.key(2))
    assert not np.array_equal(a[0][0], b[0][0])


@pytest.mark.parametrize("zero_bias", [False, True])
def test_layer_width_scaling(zero_bias):
    sizes = (4, 3, 2)
    key = jax.random.key(3)
    raw = mlp.init_network_params(sizes, key, ini_bias_zero=zero_bias)
    scaled = mlp.init_network_params(
        sizes, key, ini_bias_zero=zero_bias, scale_by_layer_width=True
    )
    for i, (W, b) in enumerate(zip(*scaled)):
        np.testing.assert_allclose(W, raw[0][i] / np.sqrt(sizes[i]), rtol=1e-6)
        np.testing.assert_array_equal(b, raw[1][i])


def test_single_weight_layer_warns_and_returns_probabilities():
    params = ([jnp.eye(2)], [jnp.zeros(2)])
    with pytest.warns(UserWarning, match="no hidden layer"):
        result = mlp.classifier(params, jnp.array([-1., 1.]))
    np.testing.assert_allclose(result, jax.nn.softmax(jnp.array([-1., 1.])))
    np.testing.assert_allclose(result.sum(), 1.)


def test_classifier_matches_expected_forward_pass():
    # Hidden activations: [2, 0]; output logits: [2, -2].
    result = mlp.classifier(example_params(), jnp.array([2., -1.]))
    np.testing.assert_allclose(result, jax.nn.softmax(jnp.array([2., -2.])))


@pytest.mark.parametrize("loss", ["squared_error", "cross_entropy"])
def test_loss_values_and_minibatch_selection(loss):
    params = example_params()
    X = jnp.array([[2., -1.], [0., 1.], [1., 1.]])
    Y = jnp.array([[1., 0.], [0., 1.], [1., 0.]])
    probabilities = np.stack([np.asarray(mlp.classifier(params, x)) for x in X])
    expected = (
        np.sum((probabilities - np.asarray(Y))**2, axis=1)
        if loss == "squared_error"
        else -np.sum(np.asarray(Y) * np.log(probabilities), axis=1)
    )
    np.testing.assert_allclose(mlp.cost(params, X[0], Y[0], loss=loss),
                               expected[0], rtol=1e-5)
    np.testing.assert_allclose(mlp.batch_cost(params, X, Y, loss=loss),
                               expected.mean(), rtol=1e-5)
    key = jax.random.key(9)
    idx = jax.random.choice(key, len(X), shape=(2,), replace=False)
    np.testing.assert_allclose(
        mlp.batch_cost(params, X, Y, batch_size=2, rand_key=key, loss=loss),
        expected[np.asarray(idx)].mean(), rtol=1e-5,
    )


def test_squared_error_remains_default():
    params = example_params()
    x, y = jnp.array([2., -1.]), jnp.array([1., 0.])
    assert mlp.cost(params, x, y) == mlp.cost(params, x, y, loss="squared_error")


def test_cross_entropy_extreme_logits_has_finite_useful_gradient():
    params = (
        [jnp.eye(2), jnp.array([[1000., 0.], [-1000., 0.]])],
        [jnp.zeros(2), jnp.zeros(2)],
    )
    value, grads = jax.value_and_grad(mlp.batch_cost)(
        params, jnp.array([[1., 0.]]), jnp.array([[0., 1.]]),
        loss="cross_entropy",
    )
    np.testing.assert_allclose(value, 2000.)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    np.testing.assert_allclose(grads[1][-1], [1., -1.])


@pytest.mark.parametrize("loss", ["squared_error", "cross_entropy"])
def test_gradient_update_reduces_loss(loss):
    params = example_params()
    X = jnp.array([[1., 0.]])
    Y = jnp.array([[0., 1.]])
    value, grads = jax.value_and_grad(mlp.batch_cost)(params, X, Y, loss=loss)
    updated = mlp.update(params, grads, 0.01)
    assert mlp.batch_cost(updated, X, Y, loss=loss) < value


def test_invalid_loss_raises():
    with pytest.raises(ValueError, match="loss must be"):
        mlp.cost(example_params(), jnp.ones(2), jnp.ones(2), loss="typo")
