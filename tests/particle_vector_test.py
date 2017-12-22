"""
Script entry point
"""

from src.calrissian.particle_vector_network import ParticleVectorNetwork
from src.calrissian.layers.particle_vector import ParticleVector
from src.calrissian.layers.particle_vector import ParticleVectorInput

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    net = ParticleVectorNetwork(cost="mse", particle_input=ParticleVectorInput(2))
    net.append(ParticleVector(2, 3, activation="sigmoid"))
    net.append(ParticleVector(3, 2, activation="sigmoid"))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def fd():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    net = ParticleVectorNetwork(cost="categorical_cross_entropy", particle_input=ParticleVectorInput(2))
    net.append(ParticleVector(2, 5, activation="sigmoid"))
    net.append(ParticleVector(5, 4, activation="sigmoid"))
    net.append(ParticleVector(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dr, dn, dzeta = net.cost_gradient(train_X, train_Y)

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

    fd_r_x = []
    fd_r_y = []
    fd_r_z = []
    fd_r_w = []

    # input first
    layer = net.particle_input
    lr_x = []
    lr_y = []
    lr_z = []
    lr_w = []
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

        # w
        orig = layer.rw[i]
        layer.rw[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rw[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_w.append((fp - fm) / (2*h))
        layer.rw[i] = orig

    fd_r_x.append(lr_x)
    fd_r_y.append(lr_y)
    fd_r_z.append(lr_z)
    fd_r_w.append(lr_w)

    # layers
    for layer in net.layers:
        lr_x = []
        lr_y = []
        lr_z = []
        lr_w = []
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

            # w
            orig = layer.rw[i]
            layer.rw[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rw[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_w.append((fp - fm) / (2 * h))
            layer.rw[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)
        fd_r_w.append(lr_w)

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

    print("analytic w")
    for layer in dr[3]:
        print(layer)
    print("numerical r w")
    for f in fd_r_w:
        print(f)
        
    fd_n_x = []
    fd_n_y = []
    fd_n_z = []
    fd_n_w = []

    # input first
    layer = net.particle_input
    ln_x = []
    ln_y = []
    ln_z = []
    ln_w = []
    for i in range(layer.output_size):
        # x
        orig_x = layer.nx[i]
        orig_y = layer.ny[i]
        orig_z = layer.nz[i]
        orig_w = layer.nw[i]

        layer.nx[i] = orig_x + h
        fp = net.cost(train_X, train_Y)
        layer.nx[i] = orig_x - h
        fm = net.cost(train_X, train_Y)
        layer.nx[i] = orig_x
        ln_x.append((fp - fm) / (2 * h))

        # y
        layer.ny[i] = orig_y + h
        fp = net.cost(train_X, train_Y)
        layer.ny[i] = orig_y - h
        fm = net.cost(train_X, train_Y)
        layer.ny[i] = orig_y
        ln_y.append((fp - fm) / (2 * h))

        # z
        layer.nz[i] = orig_z + h
        fp = net.cost(train_X, train_Y)
        layer.nz[i] = orig_z - h
        fm = net.cost(train_X, train_Y)
        layer.nz[i] = orig_z
        ln_z.append((fp - fm) / (2 * h))

        # w
        layer.nw[i] = orig_w + h
        fp = net.cost(train_X, train_Y)
        layer.nw[i] = orig_w - h
        fm = net.cost(train_X, train_Y)
        layer.nw[i] = orig_w
        ln_w.append((fp - fm) / (2 * h))

    fd_n_x.append(ln_x)
    fd_n_y.append(ln_y)
    fd_n_z.append(ln_z)
    fd_n_w.append(ln_w)

    # layers
    for layer in net.layers:
        ln_x = []
        ln_y = []
        ln_z = []
        ln_w = []
        for i in range(layer.output_size):
            # x
            orig_x = layer.nx[i]
            orig_y = layer.ny[i]
            orig_z = layer.nz[i]
            orig_w = layer.nw[i]

            layer.nx[i] = orig_x + h
            fp = net.cost(train_X, train_Y)
            layer.nx[i] = orig_x - h
            fm = net.cost(train_X, train_Y)
            layer.nx[i] = orig_x
            ln_x.append((fp - fm) / (2 * h))

            # y
            layer.ny[i] = orig_y + h
            fp = net.cost(train_X, train_Y)
            layer.ny[i] = orig_y - h
            fm = net.cost(train_X, train_Y)
            layer.ny[i] = orig_y
            ln_y.append((fp - fm) / (2 * h))

            # z
            layer.nz[i] = orig_z + h
            fp = net.cost(train_X, train_Y)
            layer.nz[i] = orig_z - h
            fm = net.cost(train_X, train_Y)
            layer.nz[i] = orig_z
            ln_z.append((fp - fm) / (2 * h))

            # w
            layer.nw[i] = orig_w + h
            fp = net.cost(train_X, train_Y)
            layer.nw[i] = orig_w - h
            fm = net.cost(train_X, train_Y)
            layer.nw[i] = orig_w
            ln_w.append((fp - fm) / (2 * h))

        fd_n_x.append(ln_x)
        fd_n_y.append(ln_y)
        fd_n_z.append(ln_z)
        fd_n_w.append(ln_w)

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

    print("analytic nw")
    for layer in dn[3]:
        print(layer)
    print("numerical n w")
    for f in fd_n_w:
        print(f)

    fd_zeta = []

    # input first
    layer = net.particle_input
    lr_zeta = []
    for i in range(layer.output_size):
        orig = layer.zeta[i]
        layer.zeta[i] += h
        fp = net.cost(train_X, train_Y)
        layer.zeta[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_zeta.append((fp - fm) / (2*h))
        layer.zeta[i] = orig
    fd_zeta.append(lr_zeta)

    # layers
    for layer in net.layers:
        lr_zeta = []
        for i in range(layer.output_size):
            # x
            orig = layer.zeta[i]
            layer.zeta[i] += h
            fp = net.cost(train_X, train_Y)
            layer.zeta[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_zeta.append((fp - fm) / (2 * h))
            layer.zeta[i] = orig
        fd_zeta.append(lr_zeta)

    print("analytic zeta")
    for layer in dzeta:
        print(layer)
    print("numerical zeta")
    for f in fd_zeta:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
