"""
Script entry point
"""

from src.calrissian.particle_conv_network import ParticleConvNetwork
from src.calrissian.layers.particle_conv import ParticleConv
from src.calrissian.layers.particle_conv import ParticleConvInput

import numpy as np


def main():

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0], [0.0]])

    net = ParticleConvNetwork(cost="mse", particle_input=ParticleConvInput(2, nr=3))
    net.append(ParticleConv(2, 3, activation="sigmoid", nr=3, nc=4))
    net.append(ParticleConv(3, 1, activation="sigmoid", nr=3, nc=4))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def main2():

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0], [0.0]])

    net = ParticleConvNetwork(cost="mse", particle_input=ParticleConvInput(2, nr=3))
    net.append(ParticleConv(2, 3, activation="sigmoid", nr=3, nc=4))
    net.append(ParticleConv(3, 1, activation="sigmoid", nr=3, nc=4))

    print(net.cost_gradient(train_X, train_Y))


def fd():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    nc = 4
    nr = 3

    net = ParticleConvNetwork(cost="categorical_cross_entropy", particle_input=ParticleConvInput(2, nr=nr))
    net.append(ParticleConv(5, activation="sigmoid", nr=nr, nc=nc))
    net.append(ParticleConv(4, activation="sigmoid", nr=nr, nc=nc))
    net.append(ParticleConv(3, activation="softmax", nr=nr, nc=nc))

    db, dq, drb, dr = net.cost_gradient(train_X, train_Y)

    h = 0.001

    print("analytic b")
    print(db)

    fd_b = []
    for l in range(len(net.layers)):
        lb = np.zeros_like(db[l])
        for c in range(len(net.layers[l].b)):
            for b in range(len(net.layers[l].b[c])):
                orig = net.layers[l].b[c][b]
                net.layers[l].b[c][b] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].b[c][b] -= 2*h
                fm = net.cost(train_X, train_Y)
                lb[c][b] = (fp - fm) / (2*h)
                net.layers[l].b[c][b] = orig
        fd_b.append(lb)
    print("numerical b")
    print(fd_b)

    print("analytic q")
    for x in dq:
        print(x)

    fd_q = []
    for l in range(len(net.layers)):
        lq = np.zeros_like(dq[l])
        for i in range(len(net.layers[l].q)):
            for c in range(nc):
                orig = net.layers[l].q[i][c]
                net.layers[l].q[i][c] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].q[i][c] -= 2*h
                fm = net.cost(train_X, train_Y)
                lq[i][c] = (fp - fm) / (2*h)
                net.layers[l].q[i][c] = orig
        fd_q.append(lq)

    print("numerical q")
    for x in fd_q:
        print(x)

    print("analytic rb")
    for x in drb:
        print(x)

    fd_rb = []
    for l in range(len(net.layers)):
        lrb = np.zeros_like(drb[l])
        for i in range(len(net.layers[l].rb)):
            for c in range(nc):
                for r in range(nr):
                    orig = net.layers[l].rb[i][c][r]
                    net.layers[l].rb[i][c][r] += h
                    fp = net.cost(train_X, train_Y)
                    net.layers[l].rb[i][c][r] -= 2*h
                    fm = net.cost(train_X, train_Y)
                    lrb[i][c][r] = (fp - fm) / (2*h)
                    net.layers[l].rb[i][c][r] = orig
        fd_rb.append(lrb)

    print("numerical rb")
    for x in fd_rb:
        print(x)

    print("analytic r")
    for x in dr:
        print(x)

    fd_r = []
    for l in range(len(dr)):
        lr = np.zeros_like(dr[l])
        layer = net.layers[l-1] if l > 0 else net.particle_input
        for i in range(len(layer.r)):
            for r in range(nr):
                orig = layer.r[i][r]
                layer.r[i][r] += h
                fp = net.cost(train_X, train_Y)
                layer.r[i][r] -= 2*h
                fm = net.cost(train_X, train_Y)
                lr[i][r] = (fp - fm) / (2*h)
                layer.r[i][r] = orig
        fd_r.append(lr)

    print("numerical r")
    for x in fd_r:
        print(x)

    diff_b = np.sum([np.sum(np.abs(fd_b[i] - db[i])) for i in range(len(db))])
    print("diff b: {}".format(diff_b))

    diff_q = np.sum([np.sum(np.abs(fd_q[i] - dq[i])) for i in range(len(dq))])
    print("diff q: {}".format(diff_q))

    diff_rb = np.sum([np.sum(np.abs(fd_rb[i] - drb[i])) for i in range(len(drb))])
    print("diff rb: {}".format(diff_rb))

    diff_r = np.sum([np.sum(np.abs(fd_r[i] - dr[i])) for i in range(len(dr))])
    print("diff r: {}".format(diff_r))


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # main2()
    fd()
