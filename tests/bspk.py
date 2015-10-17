"""
Script entry point
"""

from src.sandbox.network import Network
from src.sandbox.dense import Dense

import src.sandbox.linalg as linalg


def main():

    net = Network()
    net.add(Dense(2, 5))
    net.add(Dense(5, 3))

    # input_vector = [0.6 for _ in range(10)]
    # a_L = net.forward_pass(input_vector)
    # print(a_L)

    train_X = [[0.2, -0.3]]
    train_Y = [[0.0, 1.0, 0.0]]

    print(net.full_cost(train_X, train_Y))

    # TODO: technically need to be divided by N, but since there is only one example, it is implicitly there
    db, dw = net.gradient_single(train_X[0], train_Y[0])

    # Finite difference checking

    h = 0.001

    print(db)

    fd_b = []
    for l in range(len(net.layers)):
        lb = []
        for b in range(len(net.layers[l].b)):
            orig = net.layers[l].b[b]
            net.layers[l].b[b] += h
            fp = net.full_cost(train_X, train_Y)
            net.layers[l].b[b] -= 2*h
            fm = net.full_cost(train_X, train_Y)
            lb.append((fp - fm) / (2*h))
            net.layers[l].b[b] = orig
        fd_b.append(lb)
    print(fd_b)

    for x in dw:
        for n in x:
            print(n)

    fd_w = []
    for l in range(len(net.layers)):
        lw = []
        for w in range(len(net.layers[l].w)):
            ww = []
            for i in range(len(net.layers[l].w[w])):
                orig = net.layers[l].w[w][i]
                net.layers[l].w[w][i] += h
                fp = net.full_cost(train_X, train_Y)
                net.layers[l].w[w][i] -= 2*h
                fm = net.full_cost(train_X, train_Y)
                ww.append((fp - fm) / (2*h))
                net.layers[l].w[w][i] = orig
            lw.append(ww)
        fd_w.append(lw)

    for x in fd_w:
        for n in x:
            print(n)

if __name__ == "__main__":
    main()
