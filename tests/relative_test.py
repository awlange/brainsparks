"""
Script entry point
"""

from src.calrissian.relative_network import RelativeNetwork
from src.calrissian.layers.relative import Relative
from src.calrissian.layers.relative import RelativeInput
from src.calrissian.optimizers.relative_sgd import RelativeSGD

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = RelativeNetwork(cost="mse", relative_input=RelativeInput(2))
    net.append(Relative(2, 5, activation="sigmoid"))
    net.append(Relative(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def fd():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # net = ParticleNetwork(cost="categorical_cross_entropy", particle_input=ParticleInput(2))
    net = RelativeNetwork(cost="categorical_cross_entropy", relative_input=RelativeInput(2))
    net.append(Relative(2, 5, activation="sigmoid"))
    net.append(Relative(5, 4, activation="sigmoid"))
    net.append(Relative(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dx = net.cost_gradient(train_X, train_Y)

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

    print("analytic x")
    for x in dx:
        print(x)

    # input layer
    fd_x = []
    layer = net.relative_input
    lt = []
    for i in range(len(layer.x)):
        orig = layer.x[i]
        layer.x[i] += h
        fp = net.cost(train_X, train_Y)
        layer.x[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lt.append((fp - fm) / (2*h))
        layer.x[i] = orig
    fd_x.append(lt)

    # layers
    for l in range(len(net.layers)):
        lt = []
        for i in range(len(net.layers[l].x)):
            orig = net.layers[l].x[i]
            net.layers[l].x[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].x[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lt.append((fp - fm) / (2*h))
            net.layers[l].x[i] = orig
        fd_x.append(lt)

    print("numerical x")
    for x in fd_x:
        print(x)



if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    main()
    # main2()
    # main3()
    # main4()
    fd()
