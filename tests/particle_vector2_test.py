"""
Script entry point
"""

from src.calrissian.particle_vector2_network import ParticleVector2Network
from src.calrissian.layers.particle_vector2 import ParticleVector2
from src.calrissian.layers.particle_vector2 import ParticleVector2Input

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    net = ParticleVector2Network(cost="mse", particle_input=ParticleVector2Input(2))
    net.append(ParticleVector2(2, 3, activation="sigmoid"))
    net.append(ParticleVector2(3, 2, activation="sigmoid"))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def fd():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    net = ParticleVector2Network(cost="categorical_cross_entropy", particle_input=ParticleVector2Input(2))
    net.append(ParticleVector2(2, 5, activation="sigmoid"))
    net.append(ParticleVector2(5, 4, activation="sigmoid"))
    net.append(ParticleVector2(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dr, dn, dm = net.cost_gradient(train_X, train_Y)

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
        
    fd_n_x = []
    fd_n_y = []
    fd_n_z = []

    # input first
    layer = net.particle_input
    ln_x = []
    ln_y = []
    ln_z = []
    for i in range(layer.output_size):
        # x
        orig_x = layer.nx[i]
        orig_y = layer.ny[i]
        orig_z = layer.nz[i]

        layer.nx[i] = orig_x + h
        layer.ny[i] = orig_y
        layer.nz[i] = orig_z
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fp = net.cost(train_X, train_Y)

        layer.nx[i] = orig_x - h
        layer.ny[i] = orig_y
        layer.nz[i] = orig_z
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fm = net.cost(train_X, train_Y)

        ln_x.append((fp - fm) / (2 * h))

        # y
        layer.nx[i] = orig_x
        layer.ny[i] = orig_y + h
        layer.nz[i] = orig_z
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fp = net.cost(train_X, train_Y)

        layer.nx[i] = orig_x
        layer.ny[i] = orig_y - h
        layer.nz[i] = orig_z
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fm = net.cost(train_X, train_Y)

        ln_y.append((fp - fm) / (2 * h))

        # z
        layer.nx[i] = orig_x
        layer.ny[i] = orig_y
        layer.nz[i] = orig_z + h
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fp = net.cost(train_X, train_Y)

        layer.nx[i] = orig_x
        layer.ny[i] = orig_y
        layer.nz[i] = orig_z - h
        # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
        # layer.nx[i] /= d
        # layer.ny[i] /= d
        # layer.nz[i] /= d
        fm = net.cost(train_X, train_Y)

        ln_z.append((fp - fm) / (2 * h))
        layer.nx[i] = orig_x
        layer.ny[i] = orig_y
        layer.nz[i] = orig_z

    fd_n_x.append(ln_x)
    fd_n_y.append(ln_y)
    fd_n_z.append(ln_z)

    # layers
    for layer in net.layers:
        ln_x = []
        ln_y = []
        ln_z = []
        for i in range(layer.output_size):
            # x
            orig_x = layer.nx[i]
            orig_y = layer.ny[i]
            orig_z = layer.nz[i]

            layer.nx[i] = orig_x + h
            layer.ny[i] = orig_y
            layer.nz[i] = orig_z
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fp = net.cost(train_X, train_Y)

            layer.nx[i] = orig_x - h
            layer.ny[i] = orig_y
            layer.nz[i] = orig_z
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fm = net.cost(train_X, train_Y)

            ln_x.append((fp - fm) / (2 * h))

            # y
            layer.nx[i] = orig_x
            layer.ny[i] = orig_y + h
            layer.nz[i] = orig_z
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fp = net.cost(train_X, train_Y)

            layer.nx[i] = orig_x
            layer.ny[i] = orig_y - h
            layer.nz[i] = orig_z
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fm = net.cost(train_X, train_Y)

            ln_y.append((fp - fm) / (2 * h))

            # z
            layer.nx[i] = orig_x
            layer.ny[i] = orig_y
            layer.nz[i] = orig_z + h
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fp = net.cost(train_X, train_Y)

            layer.nx[i] = orig_x
            layer.ny[i] = orig_y
            layer.nz[i] = orig_z - h
            # d = np.sqrt(layer.nx[i] ** 2 + layer.ny[i] ** 2 + layer.nz[i] ** 2)
            # layer.nx[i] /= d
            # layer.ny[i] /= d
            # layer.nz[i] /= d
            fm = net.cost(train_X, train_Y)

            ln_z.append((fp - fm) / (2 * h))
            layer.nx[i] = orig_x
            layer.ny[i] = orig_y
            layer.nz[i] = orig_z

        fd_n_x.append(ln_x)
        fd_n_y.append(ln_y)
        fd_n_z.append(ln_z)

    print("analytic nx")
    for layer in dn[0]:
        print(layer)
    print("numerical n x")
    for f in fd_n_x:
        print(f)

    print("analytic ny")
    for layer in dn[1]:
        print(layer)
    print("numerical n y")
    for f in fd_n_y:
        print(f)

    print("analytic nz")
    for layer in dn[2]:
        print(layer)
    print("numerical n z")
    for f in fd_n_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
