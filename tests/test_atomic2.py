"""
Script entry point
"""

from src.calrissian.atomic_network import AtomicNetwork
from src.calrissian.layers.atomic import Atomic
from src.calrissian.layers.atomic import AtomicInput
from src.calrissian.optimizers.atomic_sgd import AtomicSGD

import numpy as np


def main():

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = AtomicNetwork(cost="mse", atomic_input=AtomicInput(2))
    net.append(Atomic(2, 5, activation="sigmoid"))
    net.append(Atomic(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    print(net.cost_gradient(train_X, train_Y))

    net.fit(train_X, train_Y, AtomicSGD(mini_batch_size=1, n_epochs=10))

if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    main()
