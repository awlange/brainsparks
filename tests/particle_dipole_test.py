"""
Script entry point
"""

from src.calrissian.particle_dipole_network import ParticleDipoleNetwork
from src.calrissian.layers.particle_dipole import ParticleDipole
from src.calrissian.layers.particle_dipole import ParticleDipoleInput

import numpy as np


def main():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = ParticleDipoleNetwork(cost="mse", particle_input=ParticleDipoleInput(2))
    net.append(ParticleDipole(2, 5, activation="sigmoid", k_eq=0.1))
    net.append(ParticleDipole(5, 3, activation="sigmoid", k_eq=0.1))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))


def fd():
    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = ParticleDipoleNetwork(cost="mse", particle_input=ParticleDipoleInput(2))
    net.append(ParticleDipole(2, 5, activation="sigmoid", k_eq=0.1, k_bond=10.0))
    net.append(ParticleDipole(5, 3, activation="sigmoid", k_eq=0.1, k_bond=10.0))

    db, dq, drx_pos, dry_pos, drz_pos, drx_neg, dry_neg, drz_neg = net.cost_gradient(train_X, train_Y)

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

    print("analytic x pos")
    for layer in drx_pos:
        print(layer)
    print("analytic y pos")
    for layer in dry_pos:
        print(layer)
    print("analytic z pos")
    for layer in drz_pos:
        print(layer)

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
        orig = layer.rx_pos[i]
        layer.rx_pos[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rx_pos[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_x.append((fp - fm) / (2*h))
        layer.rx_pos[i] = orig

        # y
        orig = layer.ry_pos[i]
        layer.ry_pos[i] += h
        fp = net.cost(train_X, train_Y)
        layer.ry_pos[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_y.append((fp - fm) / (2*h))
        layer.ry_pos[i] = orig

        # z
        orig = layer.rz_pos[i]
        layer.rz_pos[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rz_pos[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_z.append((fp - fm) / (2*h))
        layer.rz_pos[i] = orig

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
            orig = layer.rx_pos[i]
            layer.rx_pos[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_pos[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx_pos[i] = orig

            # y
            orig = layer.ry_pos[i]
            layer.ry_pos[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_pos[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry_pos[i] = orig

            # z
            orig = layer.rz_pos[i]
            layer.rz_pos[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz_pos[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.rz_pos[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("numerical r x pos")
    for f in fd_r_x:
        print(f)
    print("numerical r y pos")
    for f in fd_r_y:
        print(f)
    print("numerical r z pos")
    for f in fd_r_z:
        print(f)

    print("analytic x neg")
    for layer in drx_neg:
        print(layer)
    print("analytic y neg")
    for layer in dry_neg:
        print(layer)
    print("analytic z neg")
    for layer in drz_neg:
        print(layer)

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
        orig = layer.rx_neg[i]
        layer.rx_neg[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rx_neg[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_x.append((fp - fm) / (2*h))
        layer.rx_neg[i] = orig

        # y
        orig = layer.ry_neg[i]
        layer.ry_neg[i] += h
        fp = net.cost(train_X, train_Y)
        layer.ry_neg[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_y.append((fp - fm) / (2*h))
        layer.ry_neg[i] = orig

        # z
        orig = layer.rz_neg[i]
        layer.rz_neg[i] += h
        fp = net.cost(train_X, train_Y)
        layer.rz_neg[i] -= 2*h
        fm = net.cost(train_X, train_Y)
        lr_z.append((fp - fm) / (2*h))
        layer.rz_neg[i] = orig

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
            orig = layer.rx_neg[i]
            layer.rx_neg[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_neg[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx_neg[i] = orig

            # y
            orig = layer.ry_neg[i]
            layer.ry_neg[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_neg[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry_neg[i] = orig

            # z
            orig = layer.rz_neg[i]
            layer.rz_neg[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz_neg[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.rz_neg[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("numerical r x neg")
    for f in fd_r_x:
        print(f)
    print("numerical r y neg")
    for f in fd_r_y:
        print(f)
    print("numerical r z neg")
    for f in fd_r_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
