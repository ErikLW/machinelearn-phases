#import numpy as np
import jax
import jax.numpy as jnp

def init_network_params(sizes, key, ini_bias_zero = False):
    #rg = np.random.default_rng(1)  # create instance of default random number generator
    key = jax.random.key(1)
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
        W_arr.append(W)

        if ini_bias_zero:
            b = jnp.zeros(shape=sizes[i+1])
            b_arr.append(b)
        else:
            b = jax.random.uniform(
                key_b,
                shape=sizes[i+1],
                minval=-1.0,
                maxval=1.0)
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

def classifier(params, x_in):

    W_arr, b_arr = params

    depth_nn = len(b_arr)
    layer_vec_arr = [] #use a list which then contains the np.arrays of the hidden & output layer

    for i in range(depth_nn):
        if i == 0:
            y = activation(W_arr[i] @ x_in + b_arr[i])
            layer_vec_arr.append(y)
        elif i == range(depth_nn)[-1]:
            y_out = softmax(W_arr[i] @ layer_vec_arr[i-1] + b_arr[i])
            layer_vec_arr.append(y_out)
        else:
            y = activation(W_arr[i] @ layer_vec_arr[i-1] + b_arr[i])
            layer_vec_arr.append(y)

    return layer_vec_arr[-1]

#may want to pass params as a dictionary? 
#may want to define a cost function with only one input?



def cost(params, x, y_target):
    y_out = classifier(params, x)
    return jnp.sum((y_out - y_target)**2)

batched_classifier_eval = jax.vmap(classifier,  in_axes = (None, 0))

def batch_cost(params, X, Y, batch_size = None, rand_key = None):

    if batch_size != None:
        idx = jax.random.choice(
                rand_key,
                X.shape[0],
                shape=(batch_size,),
                replace=False
            )

        X_batch = X[idx]
        Y_batch = Y[idx]

        batch_output = batched_classifier_eval(params, X_batch)

        mean_batch_cost = jnp.mean(
            jnp.sum((batch_output - Y_batch)**2, axis=1)
        )
        return mean_batch_cost

    else:
        batch_output = batched_classifier_eval(params, X)
        mean_batch_cost = jnp.mean(jnp.sum((batch_output - Y)**2, axis = 1))
        return mean_batch_cost