"""
Script entry point
"""

from src.calrissian.network import Network
from src.calrissian.layers.dense import Dense
from src.calrissian.layers.convolution_2d import Convolution2D

import numpy as np


def main():
    net = Network(cost="mse")
    net.append(Convolution2D(input_size=(3, 2), n_filters=1, filter_size=(2, 2), stride=(1, 1), activation="sigmoid"))

    X = np.asarray([
        [[0.11, -0.99, 0.222], [-0.30, 0.4, 0.945]],
        [[0.01, 0.050, 1.005], [-0.22, 0.7, 0.019]],
        [[0.07, 0.250, -0.56], [-0.24, 0.8, 0.145]],
        [[0.07, 0.250, -0.56], [-0.24, 0.8, 0.145]]
    ]).reshape((4, 3, 2))

    a = net.feed_to_layer(X, end_layer=0)
    print(a)


def fd():
    pass

if __name__ == "__main__":
    main()
    # main3()
    # main4()
    # fd()
