import numpy as np
import ising_mc as ising

def init_network_params(sizes):
    rg = np.random.default_rng(1)  # create instance of default random number generator
    W_arr = []
    b_arr = []
    for i in range(len(sizes)-1):
        W_arr.append(rg.uniform(-1,1,(sizes[i+1], sizes[i])))
        b_arr.append(rg.uniform(-1,1,sizes[i+1]))
    return W_arr, b_arr

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def classifier(x_in, W_arr, b_arr, activation):

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

def ReLU(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)