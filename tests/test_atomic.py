"""
Script entry point
"""

from src.calrissian.atomic_network import AtomicNetwork
from src.calrissian.layers.atomic import Atomic
from src.calrissian.layers.atomic import AtomicInput

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


def fd():

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = AtomicNetwork(cost="mse", atomic_input=AtomicInput(2))
    net.append(Atomic(2, 5, activation="sigmoid"))
    net.append(Atomic(5, 3, activation="sigmoid"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq = net.cost_gradient(train_X, train_Y)

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

    print("analytic q")
    for x in dq:
        print(x)

    fd_q = []
    for l in range(len(net.layers)):
        lq = []
        for i in range(len(net.layers[l].q)):
            orig = net.layers[l].q[i]
            net.layers[l].q[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].q[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lq.append((fp - fm) / (2*h))
            net.layers[l].q[i] = orig
        fd_q.append(lq)

    print("numerical q")
    for x in fd_q:
        print(x)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
