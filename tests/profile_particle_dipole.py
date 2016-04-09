import pandas as pd
import numpy as np

from src.calrissian.particle_dipole_network import ParticleDipoleNetwork
from src.calrissian.layers.particle_dipole import ParticleDipole
from src.calrissian.layers.particle_dipole import ParticleDipoleInput

from src.calrissian.particle_dipole_tree_network import ParticleDipoleTreeNetwork
from src.calrissian.layers.particle_dipole_tree import ParticleDipoleTree
from src.calrissian.layers.particle_dipole_tree import ParticleDipoleTreeInput

import time

# Seed random for reproducibility
n_seed = 100
np.random.seed(n_seed)
state = np.random.get_state()

# MNIST data
raw_data_train = pd.read_csv("/Users/alange/programming/MNIST/data/mnist_train.csv", header=None)
print("data loaded")

# Prepare data
X = np.asarray(raw_data_train.ix[:, 1:] / 255.0)  # scaled values in range [0-1]
# length ten categorical vector
Y = []
for val in raw_data_train.ix[:, 0]:
    y = np.zeros(10)
    y[val] = 1.0
    Y.append(y)
Y = np.asarray(Y)

# Data subset
n_sub = 333
X_sub = X[:n_sub, :]
Y_sub = Y[:n_sub, :]

s = 4.0
cut = 3.0
max_level = 5
mac = 0.0
n = 128
n_min = 10

net = ParticleDipoleNetwork(cost="categorical_cross_entropy", particle_input=ParticleDipoleInput(784, s=s))
net.append(ParticleDipole(784, n, activation="sigmoid", s=s))
net.append(ParticleDipole(n, 10, activation="softmax", s=s))

print("starting predict")
times = []
nt = 3
for _ in range(nt):
    ts = time.time()
    c = net.cost(X_sub, Y_sub)
    # c = 1.0
    # net.cost_gradient(X_sub, Y_sub)
    t = time.time() - ts
    print("Cost: {} time: {}".format(c, t))
    times.append(t)
print("Mean: " + str(sum(times)/nt))

net2 = ParticleDipoleTreeNetwork(cost="categorical_cross_entropy", particle_input=ParticleDipoleTreeInput(784, s=s, cut=cut, max_level=max_level, mac=mac))
net2.append(ParticleDipoleTree(784, n, activation="sigmoid", s=s, cut=cut, max_level=max_level, mac=mac, n_particle_min=n_min))
net2.append(ParticleDipoleTree(n, 10, activation="softmax", s=s, cut=cut, max_level=max_level, mac=mac, n_particle_min=n_min))

# Make sure we have the same coordinates and charges
net2.particle_input.copy_pos_neg_positions(net.particle_input.rx_pos, net.particle_input.ry_pos, net.particle_input.rz_pos,
                                           net.particle_input.rx_neg, net.particle_input.ry_neg, net.particle_input.rz_neg)
for l in range(len(net.layers)):
    net2.layers[l].copy_pos_neg_positions(net.layers[l].q, net.layers[l].b,
                                          net.layers[l].rx_pos, net.layers[l].ry_pos, net.layers[l].rz_pos,
                                          net.layers[l].rx_neg, net.layers[l].ry_neg, net.layers[l].rz_neg)

print("starting predict")
times = []
nt = 3
for __ in range(nt):
    ts = time.time()
    c = net2.cost(X_sub, Y_sub)
    # c = 1.0
    # net.cost_gradient(X_sub, Y_sub)
    t = time.time() - ts
    print("Cost: {} time: {}".format(c, t))
    times.append(t)
print("Mean: " + str(sum(times)/nt))
