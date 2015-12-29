import pandas as pd
import numpy as np

# BrainSparks imports
from src.calrissian.particle_network import ParticleNetwork
from src.calrissian.layers.particle import Particle
from src.calrissian.layers.particle import ParticleInput
from src.calrissian.optimizers.particle_rprop import ParticleRPROP

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
n_sub = 200
X_sub = X[:n_sub, :]
Y_sub = Y[:n_sub, :]

net = ParticleNetwork(cost="categorical_cross_entropy", particle_input=ParticleInput(784))
net.append(Particle(784, 128, activation="sigmoid"))
net.append(Particle(128, 10, activation="softmax"))


rprop = ParticleRPROP(n_epochs=10, verbosity=2, cost_freq=25)
rprop.optimize(net, X_sub, Y_sub)
