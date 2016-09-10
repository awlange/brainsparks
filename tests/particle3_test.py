"""
Script entry point
"""

from src.calrissian.particle3_network import Particle3Network
from src.calrissian.layers.particle3 import Particle3
from src.calrissian.optimizers.particle3_sgd import Particle3SGD

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle3Network(cost="mse")
    net.append(Particle3(2, 5, activation="sigmoid"))
    net.append(Particle3(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def main2():

    sgd = Particle3SGD(alpha=0.2, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="momentum", beta=0.5)

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = Particle3Network(cost="mse")
    net.append(Particle3(2, 5, activation="sigmoid"))
    net.append(Particle3(5, 3, activation="sigmoid"))

    sgd.optimize(net, train_X, train_Y)


def main3():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle3Network(cost="mse")
    net.append(Particle3(2, 5, activation="sigmoid"))
    net.append(Particle3(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    #
    # with open("/Users/alange/network.json", "w") as f:
    #     net.write_to_json(f)
    #
    # with open("/Users/alange/network.json", "r") as f:
    #     new_net = Particle2Network.read_from_json(f)
    #     print(new_net.predict(train_X))


def fd():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle3Network(cost="categorical_cross_entropy")
    net.append(Particle3(2, 5, activation="sigmoid"))
    net.append(Particle3(5, 4, activation="sigmoid"))
    net.append(Particle3(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, drx_inp, dry_inp, drx_pos_out, dry_pos_out, drx_neg_out, dry_neg_out = net.cost_gradient(train_X, train_Y)

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

    print("analytic x input")
    for layer in drx_inp:
        print(layer)

    print("analytic y input")
    for layer in dry_inp:
        print(layer)

    fd_r_x = []
    fd_r_y = []

    for layer in net.layers:
        lr_x = []
        lr_y = []
        for i in range(layer.input_size):
            # x
            orig = layer.rx_inp[i]
            layer.rx_inp[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_inp[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx_inp[i] = orig

            # y
            orig = layer.ry_inp[i]
            layer.ry_inp[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_inp[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry_inp[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)

    print("numerical r x input")
    for f in fd_r_x:
        print(f)
    print("numerical r y input")
    for f in fd_r_y:
        print(f)

    print("analytic pos x output")
    for layer in drx_pos_out:
        print(layer)
    print("analytic pos y output")
    for layer in dry_pos_out:
        print(layer)

    fd_r_x = []
    fd_r_y = []

    for layer in net.layers:
        lr_x = []
        lr_y = []
        for i in range(layer.output_size):
            # x
            orig = layer.rx_pos_out[i]
            layer.rx_pos_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_pos_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx_pos_out[i] = orig

            # y
            orig = layer.ry_pos_out[i]
            layer.ry_pos_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_pos_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry_pos_out[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)

    print("numerical pos r x output")
    for f in fd_r_x:
        print(f)
    print("numerical pos r y output")
    for f in fd_r_y:
        print(f)

    print("analytic neg x output")
    for layer in drx_neg_out:
        print(layer)
    print("analytic neg y output")
    for layer in dry_neg_out:
        print(layer)

    fd_r_x = []
    fd_r_y = []

    for layer in net.layers:
        lr_x = []
        lr_y = []
        for i in range(layer.output_size):
            # x
            orig = layer.rx_neg_out[i]
            layer.rx_neg_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_neg_out[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2 * h))
            layer.rx_neg_out[i] = orig

            # y
            orig = layer.ry_neg_out[i]
            layer.ry_neg_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_neg_out[i] -= 2 * h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2 * h))
            layer.ry_neg_out[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)

    print("numerical neg r x output")
    for f in fd_r_x:
        print(f)
    print("numerical neg r y output")
    for f in fd_r_y:
        print(f)

if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    main2()
    # main3()
    # fd()
