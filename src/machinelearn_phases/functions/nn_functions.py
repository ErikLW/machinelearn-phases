#import numpy as np
import warnings

import jax
import jax.numpy as jnp
import numpy as np

def init_network_params(sizes, key, ini_bias_zero=False, *, scale_by_layer_width=False):
    """Initialize weights uniformly; optionally divide each by sqrt(input width).

    Bias initialization is unaffected by layer-width scaling.
    """
    #rg = np.random.default_rng(1)  # create instance of default random number generator
    W_arr = []
    b_arr = []
    for i in range(len(sizes)-1):
        key, key_W, key_b = jax.random.split(key,3)

        W = jax.random.uniform(
        key_W,
        shape=(sizes[i+1], sizes[i]),
        minval=-1.0,
        maxval=1.0
        )
        if scale_by_layer_width:
            W = W / jnp.sqrt(sizes[i])
        W_arr.append(W)

        if ini_bias_zero:
            b = jnp.zeros(shape=sizes[i+1])
            b_arr.append(b)
        else:
            b = jax.random.uniform(
                key_b,
                shape=sizes[i+1],
                minval=-0.01,
                maxval=0.01)
            b_arr.append(b)

        #W_arr.append(
        #    jnp.asarray(
        #        rg.uniform(-1,1,(sizes[i+1], sizes[i]))
        #        )
        #    )
        #b_arr.append(
        #    jnp.asarray(
        #        rg.uniform(-1,1,sizes[i+1])
        #        )
        #    )
    return W_arr, b_arr

def ReLU(x):
    return jnp.maximum(0, x)

def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x))
    return exp_x / jnp.sum(exp_x)

activation = ReLU

def _logits(params, x_in):
    W_arr, b_arr = params
    if not b_arr:
        raise ValueError("The network must contain at least one weight layer.")
    if len(b_arr) == 1:
        warnings.warn(
            "The network has no hidden layer; applying softmax directly "
            "to the single affine output.",
            UserWarning,
            stacklevel=3,
        )

    hidden = x_in
    for W, b in zip(W_arr[:-1], b_arr[:-1]):
        hidden = activation(W @ hidden + b)
    return W_arr[-1] @ hidden + b_arr[-1]


def classifier(params, x_in):
    """Return class probabilities for a single input."""
    return softmax(_logits(params, x_in))


def cost(params, x, y_target, *, loss="squared_error"):
    """Compute squared-error or categorical cross-entropy loss.

    Targets are one-hot vectors (or probability distributions).
    Cross-entropy uses log-softmax of logits to avoid log(0) underflow.
    """
    if loss == "squared_error":
        y_out = classifier(params, x)
        return jnp.sum((y_out - y_target)**2)
    if loss == "cross_entropy":
        return -jnp.sum(y_target * jax.nn.log_softmax(_logits(params, x)))
    raise ValueError(
        "loss must be 'squared_error' or 'cross_entropy', "
        f"got {loss!r}"
    )


batched_classifier_eval = jax.vmap(classifier, in_axes=(None, 0))


def batch_cost(params, X, Y, batch_size=None, rand_key=None, *,
               loss="squared_error"):
    """Average the selected loss over all examples or a random minibatch."""
    if batch_size is not None:
        idx = jax.random.choice(
            rand_key, X.shape[0], shape=(batch_size,), replace=False
        )
        X = X[idx]
        Y = Y[idx]

    losses = jax.vmap(
        lambda x, y: cost(params, x, y, loss=loss)
    )(X, Y)
    return jnp.mean(losses)

def max_unit_vector(a):
    e = np.zeros_like(a, dtype=float)
    e[np.argmax(a)] = 1.0
    return e