"""
Script entry point
"""

from src.calrissian.particle4_network import Particle4Network
from src.calrissian.layers.particle4 import Particle4
from src.calrissian.layers.particle4 import Particle4Input
from src.calrissian.optimizers.particle4_sgd import Particle4SGD

import numpy as np


def main():

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle4Network(cost="mse", particle_input=Particle4Input(2, phase_enabled=True))
    net.append(Particle4(2, 3, activation="sigmoid", phase_enabled=True))

    print(net.particle_input.get_rxyz())

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))


def main2():

    sgd = Particle4SGD(alpha=0.2, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="momentum", beta=0.5)

    train_X = np.asarray([[0.2, -0.3]])
    train_Y = np.asarray([[0.0, 1.0, 0.0]])

    net = Particle4Network(cost="mse", particle_input=Particle4Input(2))
    net.append(Particle4(2, 5, activation="sigmoid"))
    net.append(Particle4(5, 3, activation="sigmoid"))

    sgd.optimize(net, train_X, train_Y)


def main3():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    net = Particle4Network(cost="mse", particle_input=Particle4Input(2))
    net.append(Particle4(2, 5, activation="sigmoid"))
    net.append(Particle4(5, 3, activation="sigmoid"))

    print(net.predict(train_X))

    with open("/Users/adrianlange/network.json", "w") as f:
        net.write_to_json(f)

    with open("/Users/adrianlange/network.json", "r") as f:
        new_net = Particle4Network.read_from_json(f)
        print(new_net.predict(train_X))


def fd():

    # train_X = np.asarray([[0.2, -0.3]])
    # train_Y = np.asarray([[0.0, 1.0, 0.0]])

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05], [0.33, -0.9], [0.44, -1.1]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    r_dim = 4

    net = Particle4Network(cost="categorical_cross_entropy", particle_input=Particle4Input(2, r_dim=r_dim), r_dim=4)
    net.append(Particle4(2, 5, activation="sigmoid", r_dim=r_dim))
    net.append(Particle4(5, 4, activation="sigmoid", r_dim=r_dim))
    net.append(Particle4(4, 3, activation="softmax", r_dim=r_dim))

    # Finite difference checking

    net.cost(train_X, train_Y)

    db, dq, dr, dt = net.cost_gradient(train_X, train_Y)

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

    for dim in range(net.r_dim):
        fd_r = []

        # input first
        layer = net.particle_input
        lr = []
        for i in range(layer.output_size):
            orig = layer.r[i][dim]
            layer.r[i][dim] += h
            fp = net.cost(train_X, train_Y)
            layer.r[i][dim] -= 2*h
            fm = net.cost(train_X, train_Y)
            lr.append((fp - fm) / (2*h))
            layer.r[i][dim] = orig
        fd_r.append(lr)

        # layers
        for layer in net.layers:
            lr = []
            for i in range(layer.output_size):
                orig = layer.r[i][dim]
                layer.r[i][dim] += h
                fp = net.cost(train_X, train_Y)
                layer.r[i][dim] -= 2*h
                fm = net.cost(train_X, train_Y)
                lr.append((fp - fm) / (2*h))
                layer.r[i][dim] = orig
            fd_r.append(lr)

        print("analytic r {}".format(dim))
        for lr in dr:
            print(lr.transpose()[dim])
        print("numerical r x")
        for f in fd_r:
            print(f)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    main2()
    # main3()
    # main4()
    # fd()
