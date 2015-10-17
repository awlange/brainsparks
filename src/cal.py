"""
Script entry point
"""

from src.calrissian.network import Network
from src.calrissian.layers.dense import Dense

import numpy as np


def main():

    net = Network(cost="quadratic")
    net.append(Dense(2, 5))
    net.append(Dense(5, 3))

    train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    print(net.cost(train_X, train_Y))
    print(net.cost_gradient(train_X, train_Y))


if __name__ == "__main__":
    main()
