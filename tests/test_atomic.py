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

    net = AtomicNetwork(cost="categorical_cross_entropy", atomic_input=AtomicInput(2))
    net.append(Atomic(2, 5, activation="sigmoid"))
    net.append(Atomic(5, 4, activation="tanh"))
    net.append(Atomic(4, 3, activation="atomic_softmax"))

    # net = AtomicNetwork(cost="mse", atomic_input=AtomicInput(2))
    # net.append(Atomic(2, 5, activation="linear"))
    # net.append(Atomic(5, 3, activation="linear"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dr = net.cost_gradient(train_X, train_Y)

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

    print("analytic r")
    for x in dr:
        print(x)

    fd_r_x = []
    fd_r_y = []
    fd_r_z = []

    # input first
    layer = net.atomic_input
    lr_x = []
    lr_y = []
    lr_z = []
    for i in range(len(layer.r)):
        # x
        orig = layer.r[i].x
        layer.r[i].x += h
        fp = net.cost(train_X, train_Y)
        layer.r[i].x -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_x.append((fp - fm) / (2*h))
        layer.r[i].x = orig

        # y
        orig = layer.r[i].y
        layer.r[i].y += h
        fp = net.cost(train_X, train_Y)
        layer.r[i].y -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_y.append((fp - fm) / (2*h))
        layer.r[i].y = orig

        # z
        orig = layer.r[i].z
        layer.r[i].z += h
        fp = net.cost(train_X, train_Y)
        layer.r[i].z -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_z.append((fp - fm) / (2*h))
        layer.r[i].z = orig

    fd_r_x.append(lr_x)
    fd_r_y.append(lr_y)
    fd_r_z.append(lr_z)

    # layers
    for layer in net.layers:
        lr_x = []
        lr_y = []
        lr_z = []
        for i in range(len(layer.r)):
            # x
            orig = layer.r[i].x
            layer.r[i].x += h
            fp = net.cost(train_X, train_Y)
            layer.r[i].x -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.r[i].x = orig

            # y
            orig = layer.r[i].y
            layer.r[i].y += h
            fp = net.cost(train_X, train_Y)
            layer.r[i].y -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.r[i].y = orig

            # z
            orig = layer.r[i].z
            layer.r[i].z += h
            fp = net.cost(train_X, train_Y)
            layer.r[i].z -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.r[i].z = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("numerical r x")
    for f in fd_r_x:
        print(f)
    print("numerical r y")
    for f in fd_r_y:
        print(f)
    print("numerical r z")
    for f in fd_r_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
