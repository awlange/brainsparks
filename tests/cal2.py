"""
Script entry point
"""

from src.calrissian.network import Network
from src.calrissian.layers.dense import Dense
from src.calrissian.layers.convolution_1d import Convolution1D
from src.calrissian.layers.flatten import Flatten
from src.calrissian.optimizers.sgd import SGD

import numpy as np


def main():
    net = Network(cost="mse")
    net.append(Convolution1D(input_size=8, filter_size=3, n_filters=3, stride_length=2, activation="sigmoid"))
    net.append(Flatten())
    net.append(Dense(9, 3))

    # train_X = np.asarray([[0.2, -0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9],
                          [-0.1, -0.5, -0.5, 0.4, 0.4, 0.1, 0.2, 0.2]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    sigma = net.predict(train_X)
    print(sigma)
    print(net.cost(train_X, train_Y))


def fd():
    net = Network(cost="mse")
    net.append(Convolution1D(input_size=4, filter_size=2, n_filters=3, stride_length=1,
                             activation="linear", flatten_output=True))
    net.append(Dense(9, 2))

    # train_X = np.asarray([[0.2, -0.3, 0.5, 0.6, 0.2]])
    # train_Y = np.asarray([[[0.0, 0.0, 1.0]]])
    # train_Y = np.asarray([[0.0, 0.0, 1.0]])

    # train_Y = np.asarray([[[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]])
    # train_Y = np.asarray([[[0.0, 0.0, 1.0, 0.5, 0.5, 0.5]]])

    train_X = np.asarray([[0.2, 0.3, 0.4, 0.7]])
    train_Y = np.asarray([[0.0, 1.0]])


    # Finite difference checking

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))

    db, dw = net.cost_gradient(train_X, train_Y)

    h = 0.001

    print("analytic b")
    print(db)

    fd_b = []
    for l in range(len(net.layers)):
        lb = []
        for c in range(len(net.layers[l].b)):
            for b in range(len(net.layers[l].b[c])):
                orig = net.layers[l].b[c][b]
                net.layers[l].b[c][b] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].b[c][b] -= 2*h
                fm = net.cost(train_X, train_Y)
                lb.append((fp - fm) / (2*h))
                net.layers[l].b[c][b] = orig

            fd_b.append(lb)
    print("numerical b")
    print(fd_b)

    print("analytic w")
    for i, x in enumerate(dw):
        print(i)
        for n in x:
            print(n)

    fd_w = []
    for l in range(len(net.layers)):
        lw = []
        for w in range(len(net.layers[l].w)):
            ww = []
            for i in range(len(net.layers[l].w[w])):
                orig = net.layers[l].w[w][i]
                net.layers[l].w[w][i] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].w[w][i] -= 2*h
                fm = net.cost(train_X, train_Y)
                ww.append((fp - fm) / (2*h))
                net.layers[l].w[w][i] = orig
            lw.append(ww)
        fd_w.append(lw)

    print("numerical w")
    for i, x in enumerate(fd_w):
        print(i)
        for n in x:
            print(n)

if __name__ == "__main__":
    # main()
    fd()
