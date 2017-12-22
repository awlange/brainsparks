"""
Script entry point
"""

from src.calrissian.particle_vector_n_network_local_conv import ParticleVectorNLocalConvolutionNetwork
from src.calrissian.layers.particle_vector_n_local_conv import ParticleVectorNLocalConvolution
from src.calrissian.layers.particle_vector_n_local_conv import ParticleVectorNLocalConvolutionInput

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    net = ParticleVectorNLocalConvolutionNetwork(cost="mse", particle_input=ParticleVectorNLocalConvolutionInput(2))
    net.append(ParticleVectorNLocalConvolution(2, 3, activation="sigmoid", apply_convolution=True, delta_r=0.1))
    net.append(ParticleVectorNLocalConvolution(3, 2, activation="sigmoid"))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def fd():

    nr = 3
    nv = 3

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    net = ParticleVectorNLocalConvolutionNetwork(cost="categorical_cross_entropy", particle_input=ParticleVectorNLocalConvolutionInput(2))
    net.append(ParticleVectorNLocalConvolution(2, 5, activation="tanh", apply_convolution=True, delta_r=0.1))
    net.append(ParticleVectorNLocalConvolution(5, 4, activation="tanh"))
    net.append(ParticleVectorNLocalConvolution(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dr, dn = net.cost_gradient(train_X, train_Y)

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


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    fd()
