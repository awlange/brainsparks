"""
Script entry point
"""

from src.calrissian.particle_vector_n_network_local_conv4 import ParticleVectorNLocalConvolution4Network
from src.calrissian.layers.particle_vector_n_local_conv4 import ParticleVectorNLocalConvolution4
from src.calrissian.layers.particle_vector_n_local_conv4 import ParticleVectorNLocalConvolution4Input

import numpy as np
import time


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    net = ParticleVectorNLocalConvolution4Network(cost="mse", particle_input=ParticleVectorNLocalConvolution4Input(2))
    net.append(ParticleVectorNLocalConvolution4(2, 3, activation="sigmoid", apply_convolution=True, delta_r=0.1,
                                                n_steps=[-1, 0, 1],
                                                pool_size=[2, 2, 1], pool_stride=[0.1, 0.1, 0.1]))
    net.append(ParticleVectorNLocalConvolution4(3, 2, activation="sigmoid"))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def timer():

    nr = 3
    nv = 3
    nw = 3

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05], [0.01, 0.01], [0.03, 0.04]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    net = ParticleVectorNLocalConvolution4Network(cost="categorical_cross_entropy",
                                                  particle_input=ParticleVectorNLocalConvolution4Input(2))
    net.append(ParticleVectorNLocalConvolution4(2, 5, activation="tanh", potential="gaussian",
                                                apply_convolution=True, delta_r=0.4,
                                                pool_size=[4, 4, 1], pool_stride=[0.1, 0.1, 0.1]))
    net.append(ParticleVectorNLocalConvolution4(5, 4, activation="tanh", srl=[0.1, 0.5, 0.5]))
    net.append(ParticleVectorNLocalConvolution4(4, 3, activation="softmax"))

    # Finite difference checking

    ts = time.time()

    total = 0.0
    for i in range(50):
        db, dr, dn, dm = net.cost_gradient(train_X, train_Y)
        total += i
    print("Time for {}: {}".format(i+1, time.time() - ts))


def fd():

    nr = 3
    nv = 3
    nw = 3

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05], [0.01, 0.01], [0.03, 0.04]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    net = ParticleVectorNLocalConvolution4Network(cost="categorical_cross_entropy",
                                                  particle_input=ParticleVectorNLocalConvolution4Input(2))
    net.append(ParticleVectorNLocalConvolution4(2, 5, activation="tanh", potential="gaussian",
                                                apply_convolution=True, delta_r=0.4,
                                                pool_size=[3, 3, 1], pool_stride=[0.1, 0.1, 0.1]))
    net.append(ParticleVectorNLocalConvolution4(5, 4, activation="tanh", srl=[0.1, 0.5, 0.5]))
    net.append(ParticleVectorNLocalConvolution4(4, 3, activation="softmax"))

    # Finite difference checking

    ts = time.time()

    net.cost(train_X, train_Y)

    db, dr, dn, dm = net.cost_gradient(train_X, train_Y)

    print("Time: {}".format(time.time() - ts))

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

    for r in range(nr):
        fd_r_x = []

        # input first
        layer = net.particle_input
        lr_x = []
        for i in range(layer.output_size):
            # x
            orig = layer.positions[r][i]
            layer.positions[r][i] += h
            fp = net.cost(train_X, train_Y)
            layer.positions[r][i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.positions[r][i] = orig

        fd_r_x.append(lr_x)

        # layers
        for layer in net.layers:
            lr_x = []
            for i in range(layer.output_size):
                # x
                orig = layer.positions[r][i]
                layer.positions[r][i] += h
                fp = net.cost(train_X, train_Y)
                layer.positions[r][i] -= 2*h
                fm = net.cost(train_X, train_Y)
                lr_x.append((fp - fm) / (2*h))
                layer.positions[r][i] = orig

            fd_r_x.append(lr_x)

        print("analytic r: " + str(r))
        for layer in dr:
            print(layer[r])
        print("numerical r: " + str(r))
        for f in fd_r_x:
            print(f)

    for v in range(nv):
        fd_n_x = []

        # input first
        layer = net.particle_input
        ln_x = []
        for i in range(layer.output_size):
            # x
            orig_x = layer.nvectors[v][i]

            layer.nvectors[v][i] = orig_x + h
            fp = net.cost(train_X, train_Y)
            layer.nvectors[v][i] = orig_x - h
            fm = net.cost(train_X, train_Y)
            layer.nvectors[v][i] = orig_x
            ln_x.append((fp - fm) / (2 * h))

        fd_n_x.append(ln_x)

        # layers
        for layer in net.layers:
            ln_x = []
            for i in range(layer.output_size):
                # x
                orig_x = layer.nvectors[v][i]

                layer.nvectors[v][i] = orig_x + h
                fp = net.cost(train_X, train_Y)
                layer.nvectors[v][i] = orig_x - h
                fm = net.cost(train_X, train_Y)
                layer.nvectors[v][i] = orig_x
                ln_x.append((fp - fm) / (2 * h))

            fd_n_x.append(ln_x)

        print("analytic n: " + str(v))
        for layer in dn:
            print(layer[v])
        print("numerical n: " + str(v))
        for f in fd_n_x:
            print(f)

    for w in range(nw):
        fd_n_x = []

        # input first
        layer = net.particle_input
        ln_x = []
        for i in range(layer.output_size):
            # x
            orig_x = layer.nwectors[w][i]

            layer.nwectors[w][i] = orig_x + h
            fp = net.cost(train_X, train_Y)
            layer.nwectors[w][i] = orig_x - h
            fm = net.cost(train_X, train_Y)
            layer.nwectors[w][i] = orig_x
            ln_x.append((fp - fm) / (2 * h))

        fd_n_x.append(ln_x)

        # layers
        for layer in net.layers:
            ln_x = []
            for i in range(layer.output_size):
                # x
                orig_x = layer.nwectors[w][i]

                layer.nwectors[w][i] = orig_x + h
                fp = net.cost(train_X, train_Y)
                layer.nwectors[w][i] = orig_x - h
                fm = net.cost(train_X, train_Y)
                layer.nwectors[w][i] = orig_x
                ln_x.append((fp - fm) / (2 * h))

            fd_n_x.append(ln_x)

        print("analytic m: " + str(w))
        for layer in dm:
            print(layer[w])
        print("numerical m: " + str(w))
        for f in fd_n_x:
            print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # fd()
    timer()
