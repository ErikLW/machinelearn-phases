import numpy as np
import ising_mc as ising
import machinelearn_phases as mlp

temp, energy, vare, m, varm, absm, configs = ising.run_metropolis(10, 1, 1, 0, 1000, 1000, 100, return_configurations=True)
x_in = configs[10].ravel()
W_arr, b_arr = mlp.init_network_params((100,25,6,2))
prob = mlp.classifier(x_in, W_arr, b_arr, mlp.ReLU)
print(prob)