import pandas as pd
import numpy as np

# BrainSparks imports
from src.calrissian.particle_network import ParticleNetwork
from src.calrissian.layers.particle import Particle
from src.calrissian.layers.particle import ParticleInput

import time

# Seed random for reproducibility
n_seed = 100
np.random.seed(n_seed)
state = np.random.get_state()

# MNIST data
raw_data_train = pd.read_csv("/Users/alange/programming/MNIST/data/mnist_train.csv", header=None)
print("data loaded")

# Prepare data
X = np.asarray(raw_data_train.ix[:, 1:] / 256.0)  # scaled values in range [0-1]
# length ten categorical vector
Y = []
for val in raw_data_train.ix[:, 0]:
    y = np.zeros(10)
    y[val] = 1.0
    Y.append(y)
Y = np.asarray(Y)

# Data subset
n_sub = 1000
X_sub = X[:n_sub, :]
Y_sub = Y[:n_sub, :]

net = ParticleNetwork(cost="categorical_cross_entropy", particle_input=ParticleInput(784))
net.append(Particle(784, 64, activation="sigmoid"))
net.append(Particle(64, 10, activation="softmax"))

print("starting predict")
times = []
nt = 3
for _ in range(nt):
    ts = time.time()
    # c = net.cost(X, Y)
    c = 1.0
    # net.cost_gradient(X_sub, Y_sub)
    net.cost_gradient2(X_sub, Y_sub)
    t = time.time() - ts
    print("Cost: {} time: {}".format(c, t))
    times.append(t)
print("Mean: " + str(sum(times)/nt))
