"""
Script entry point
"""

from src.calrissian.particle2_network import Particle2Network
from src.calrissian.layers.particle2 import Particle2
from src.calrissian.optimizers.particle2_sgd import Particle2SGD

import numpy as np


def main():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle2Network(cost="mse")
    net.append(Particle2(2, 5, activation="sigmoid"))
    net.append(Particle2(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    # print(net.cost_gradient(train_X, train_Y))


def main2():

    sgd = Particle2SGD(alpha=0.2, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="momentum", beta=0.5)
    # sgd = Particle2SGD(alpha=0.2, n_epochs=1, mini_batch_size=1, verbosity=2)

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = Particle2Network(cost="mse")
    net.append(Particle2(2, 5, activation="sigmoid"))
    net.append(Particle2(5, 3, activation="sigmoid"))

    sgd.optimize(net, train_X, train_Y)


def main3():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle2Network(cost="mse")
    net.append(Particle2(2, 5, activation="sigmoid"))
    net.append(Particle2(5, 3, activation="sigmoid"))

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

    net = Particle2Network(cost="categorical_cross_entropy")
    net.append(Particle2(2, 5, activation="sigmoid"))
    net.append(Particle2(5, 4, activation="sigmoid"))
    net.append(Particle2(4, 3, activation="softmax"))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dr_inp, dr_out, dt_inp, dt_out = net.cost_gradient(train_X, train_Y)

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

    print("analytic theta input")
    for x in dt_inp:
        print(x)

    fd_t = []
    for l in range(len(net.layers)):
        lt = []
        for i in range(len(net.layers[l].theta_inp)):
            orig = net.layers[l].theta_inp[i]
            net.layers[l].theta_inp[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].theta_inp[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lt.append((fp - fm) / (2*h))
            net.layers[l].theta_inp[i] = orig
        fd_t.append(lt)

    print("numerical theta input")
    for x in fd_t:
        print(x)

    print("analytic theta output")
    for x in dt_out:
        print(x)

    fd_t = []
    for l in range(len(net.layers)):
        lt = []
        for i in range(len(net.layers[l].theta_out)):
            orig = net.layers[l].theta_out[i]
            net.layers[l].theta_out[i] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].theta_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lt.append((fp - fm) / (2*h))
            net.layers[l].theta_out[i] = orig
        fd_t.append(lt)

    print("numerical theta output")
    for x in fd_t:
        print(x)

    print("analytic x input")
    for layer in dr_inp[0]:
        print(layer)
    print("analytic y input")
    for layer in dr_inp[1]:
        print(layer)
    print("analytic z input")
    for layer in dr_inp[2]:
        print(layer)

    fd_r_x = []
    fd_r_y = []
    fd_r_z = []

    for layer in net.layers:
        lr_x = []
        lr_y = []
        lr_z = []
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

            # z
            orig = layer.rz_inp[i]
            layer.rz_inp[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz_inp[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.rz_inp[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("numerical r x input")
    for f in fd_r_x:
        print(f)
    print("numerical r y input")
    for f in fd_r_y:
        print(f)
    print("numerical r z input")
    for f in fd_r_z:
        print(f)


    print("analytic x output")
    for layer in dr_out[0]:
        print(layer)
    print("analytic y output")
    for layer in dr_out[1]:
        print(layer)
    print("analytic z output")
    for layer in dr_out[2]:
        print(layer)

    fd_r_x = []
    fd_r_y = []
    fd_r_z = []

    for layer in net.layers:
        lr_x = []
        lr_y = []
        lr_z = []
        for i in range(layer.output_size):
            # x
            orig = layer.rx_out[i]
            layer.rx_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rx_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_x.append((fp - fm) / (2*h))
            layer.rx_out[i] = orig

            # y
            orig = layer.ry_out[i]
            layer.ry_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.ry_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_y.append((fp - fm) / (2*h))
            layer.ry_out[i] = orig

            # z
            orig = layer.rz_out[i]
            layer.rz_out[i] += h
            fp = net.cost(train_X, train_Y)
            layer.rz_out[i] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr_z.append((fp - fm) / (2*h))
            layer.rz_out[i] = orig

        fd_r_x.append(lr_x)
        fd_r_y.append(lr_y)
        fd_r_z.append(lr_z)

    print("numerical r x output")
    for f in fd_r_x:
        print(f)
    print("numerical r y output")
    for f in fd_r_y:
        print(f)
    print("numerical r z ouput")
    for f in fd_r_z:
        print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # main2()
    # main3()
    fd()
