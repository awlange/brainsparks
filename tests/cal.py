"""
Script entry point
"""

from src.calrissian.network import Network
from src.calrissian.layers.dense import Dense
from src.calrissian.optimizers.sgd import SGD

import numpy as np


def main():

    net = Network(cost="quadratic")
    net.append(Dense(2, 5))
    net.append(Dense(5, 3))

    # train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    print(net.cost(train_X, train_Y))
    print(net.cost_gradient(train_X, train_Y))


def main2():
    net = Network(cost="quadratic")
    net.append(Dense(2, 5))
    net.append(Dense(5, 3))

    train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    sgd = SGD(alpha=0.1, n_epochs=10, mini_batch_size=3)
    net.fit(train_X, train_Y, sgd)


def main3():
    net = Network(cost="categorical_cross_entropy")
    net.append(Dense(2, 5, activation="sigmoid"))
    net.append(Dense(5, 3, activation="softmax"))

    train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    sgd = SGD(alpha=0.05, n_epochs=20, mini_batch_size=4, verbosity=1)
    net.fit(train_X, train_Y, sgd)


def fd():
    net = Network(cost="categorical_cross_entropy")
    net.append(Dense(2, 5, activation="tanh"))
    net.append(Dense(5, 3, activation="softmax"))

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.6, -0.2], [0.8, 0.9], [0.1, 0.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    # Finite difference checking

    net.cost(train_X, train_Y)

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
    for x in dw:
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
    for x in fd_w:
        for n in x:
            print(n)

if __name__ == "__main__":
    main()
    # main3()
    # fd()
