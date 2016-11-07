"""
Script entry point
"""

from src.calrissian.particle_sparse_network import ParticleSparseNetwork
from src.calrissian.layers.particle_sparse import ParticleSparse
from src.calrissian.layers.particle_sparse import ParticleSparseInput

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([{0: 0.45, 1: 3.33}, {1: 2.22}])
    train_Y = np.asarray([{1: 1.0}, {2: 1.0}])

    net = ParticleSparseNetwork(cost="mse_sparse", particle_input=ParticleSparseInput(2))
    net.append(ParticleSparse(2, 3))
    # net.append(ParticleSparse(5, 3))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    print(net.cost_gradient(train_X, train_Y))


def fd():

    train_X = np.asarray([{0: 0.45, 1: 3.33}, {1: 2.22}, {0: -0.9}])
    train_Y = np.asarray([{1: 1.0}, {2: 1.0}, {0: 1.0}])

    net = ParticleSparseNetwork(cost="mse_sparse", particle_input=ParticleSparseInput(2))
    net.append(ParticleSparse(2, 3))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dr, dt = net.cost_gradient(train_X, train_Y)

    h = 0.00001

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

    print("analytic theta")
    for x in dt:
        print(x)

    # input layer
    fd_t = []
    layer = net.particle_input
    lt = []
    for i in range(len(layer.theta)):
        orig = layer.theta[i]
        layer.theta[i] += h
        fp = net.cost(train_X, train_Y)
        layer.theta[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lt.append((fp - fm) / (2*h))
        layer.theta[i] = orig
    fd_t.append(lt)

    # layers
    for l in range(len(net.layers)):
        lt = []
        for i in range(len(net.layers[l].theta)):
            orig = net.layers[l].theta[i]
            net.layers[l].theta[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].theta[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lt.append((fp - fm) / (2*h))
            net.layers[l].theta[i] = orig
        fd_t.append(lt)

    print("numerical theta")
    for x in fd_t:
        print(x)

    fd_r_x = []
    fd_r_y = []
    fd_r_z = []

    # input first
    layer = net.particle_input
    lr_x = []
    lr_y = []
    lr_z = []
    for i in range(layer.output_size):
        # x
        orig = layer.rx[i]
        layer.rx[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rx[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_x.append((fp - fm) / (2*h))
        layer.rx[i] = orig

        # y
        orig = layer.ry[i]
        layer.ry[i] += h
        fp = net.cost(train_X, train_Y)
        layer.ry[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_y.append((fp - fm) / (2*h))
        layer.ry[i] = orig

        # z
        orig = layer.rz[i]
        layer.rz[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rz[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_z.append((fp - fm) / (2*h))
        layer.rz[i] = orig

    fd_r_x.append(lr_x)
    fd_r_y.append(lr_y)
    fd_r_z.append(lr_z)

    # layers
    for layer in net.layers:
        lr_x = []
        lr_y = []
        lr_z = []
        for i in range(layer.output_size):
            # x
            orig = layer.rx[i]
            layer.rx[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx[i] = orig

            # y
            orig = layer.ry[i]
            layer.ry[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry[i] = orig

            # z
            orig = layer.rz[i]
            layer.rz[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.rz[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("analytic x")
    for layer in dr[0]:
        print(layer)
    print("numerical r x")
    for f in fd_r_x:
        print(f)

    print("analytic y")
    for layer in dr[1]:
        print(layer)
    print("numerical r y")
    for f in fd_r_y:
        print(f)

    print("analytic z")
    for layer in dr[2]:
        print(layer)
    print("numerical r z")
    for f in fd_r_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # main2()
    # main3()
    # main4()
    fd()
