"""
Script entry point
"""

from src.calrissian.particle5_network import Particle5Network
from src.calrissian.layers.particle5 import Particle5
from src.calrissian.layers.particle5 import Particle5Input
from src.calrissian.optimizers.particle5_sgd import Particle5SGD

import numpy as np


def main():

    # train_X = np.asarray([{0: 0.45, 1: 3.33}, {1: 2.22}])
    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle5Network(cost="mse", particle_input=Particle5Input(2, phase_enabled=True))
    net.append(Particle5(2, 3, activation="sigmoid", phase_enabled=True))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def main2():

    # sgd = Particle5SGD(alpha=0.2, n_epochs=3, mini_batch_size=1, verbosity=2, weight_update="sd")
    sgd = Particle5SGD(alpha=0.2, n_epochs=3, mini_batch_size=1, verbosity=2, weight_update="rmsprop", gamma=0.9)

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = Particle5Network(cost="mse", particle_input=Particle5Input(2))
    net.append(Particle5(2, 5, activation="sigmoid"))
    net.append(Particle5(5, 3, activation="sigmoid"))

    sgd.optimize(net, train_X, train_Y)


def fd():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    net = Particle5Network(cost="categorical_cross_entropy", particle_input=Particle5Input(2))
    net.append(Particle5(2, 5, activation="sigmoid"))
    net.append(Particle5(5, 4, activation="sigmoid"))
    net.append(Particle5(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dq2, dr, dr2, dt, dt2 = net.cost_gradient(train_X, train_Y)

    h = 0.0001

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

    print("analytic q2")
    for x in dq2:
        print(x)

    fd_q = []
    for l in range(len(net.layers)):
        lq = []
        for i in range(len(net.layers[l].q2)):
            orig = net.layers[l].q2[i]
            net.layers[l].q2[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].q2[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lq.append((fp - fm) / (2*h))
            net.layers[l].q2[i] = orig
        fd_q.append(lq)

    print("numerical q2")
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

    print("analytic theta2")
    for x in dt2:
        print(x)

    # input layer
    fd_t = []
    layer = net.particle_input
    lt = []
    for i in range(len(layer.theta2)):
        orig = layer.theta2[i]
        layer.theta2[i] += h
        fp = net.cost(train_X, train_Y)
        layer.theta2[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lt.append((fp - fm) / (2*h))
        layer.theta2[i] = orig
    fd_t.append(lt)

    # layers
    for l in range(len(net.layers)):
        lt = []
        for i in range(len(net.layers[l].theta2)):
            orig = net.layers[l].theta2[i]
            net.layers[l].theta2[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].theta2[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lt.append((fp - fm) / (2*h))
            net.layers[l].theta2[i] = orig
        fd_t.append(lt)

    print("numerical theta2")
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
        orig = layer.rx2[i]
        layer.rx2[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rx2[i] -= 2 * h
        fm = net.cost(train_X, train_Y)
        lr_x.append((fp - fm) / (2 * h))
        layer.rx2[i] = orig

        # y
        orig = layer.ry2[i]
        layer.ry2[i] += h
        fp = net.cost(train_X, train_Y)
        layer.ry2[i] -= 2 * h
        fm = net.cost(train_X, train_Y)
        lr_y.append((fp - fm) / (2 * h))
        layer.ry2[i] = orig

        # z
        orig = layer.rz2[i]
        layer.rz2[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rz2[i] -= 2 * h
        fm = net.cost(train_X, train_Y)
        lr_z.append((fp - fm) / (2 * h))
        layer.rz2[i] = orig

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
            orig = layer.rx2[i]
            layer.rx2[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx2[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2 * h))
            layer.rx2[i] = orig

            # y
            orig = layer.ry2[i]
            layer.ry2[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry2[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2 * h))
            layer.ry2[i] = orig

            # z
            orig = layer.rz2[i]
            layer.rz2[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz2[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2 * h))
            layer.rz2[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("analytic x2")
    for layer in dr2[0]:
        print(layer)
    print("numerical r x2")
    for f in fd_r_x:
        print(f)

    print("analytic y2")
    for layer in dr2[1]:
        print(layer)
    print("numerical r y2")
    for f in fd_r_y:
        print(f)

    print("analytic z2")
    for layer in dr2[2]:
        print(layer)
    print("numerical r z2")
    for f in fd_r_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # main2()
    fd()
