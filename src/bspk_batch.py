"""
Script entry point
"""

from src.sandbox.network import Network
from src.sandbox.batch_dense import BatchDense

import numpy as np


def main():

    net = Network(cost="np_quadratic")
    net.add(BatchDense(2, 5))
    net.add(BatchDense(5, 3))

    train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    print(net.batch_full_cost(train_X, train_Y))
    print(net.gradient_batch(train_X, train_Y))


if __name__ == "__main__":
    main()
