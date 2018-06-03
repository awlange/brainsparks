"""
Script entry point
"""

from src.calrissian.particle333_network import Particle333Network
from src.calrissian.layers.particle333 import Particle333
from src.calrissian.optimizers.particle333_sgd import Particle333SGD
from multiprocessing import Pool

import numpy as np
import time
import pandas as pd
import pickle


def main():

    train_X = np.asarray([[0.45, 3.33], [0.0, 2.22], [0.45, -0.54]])
    train_Y = np.asarray([[1.0], [0.0], [0.0]])

    net = Particle333Network(cost="mse")
    net.append(Particle333(2, 5, activation="sigmoid", nr=4, nc=6))
    net.append(Particle333(5, 1, activation="sigmoid", nr=4, nc=6))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))
    print(net.cost_gradient(train_X, train_Y))


def main2():

    train_X = np.random.normal(0.0, 0.1, (3, 4*4))
    train_Y = np.random.normal(0.0, 0.1, (3, 1))

    nr = 3
    nc = 3

    net = Particle333Network(cost="mse")
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(4, 4, 1),
                           output_shape=(2, 2, 3),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5)))
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(2, 2, 3),
                           output_shape=(2, 2, 1),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5)))
    net.append(Particle333(4, 1, activation="sigmoid", nr=nr, nc=nc))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))


def main3():

    train_X = np.random.normal(0.0, 0.1, (3, 4*4))
    train_Y = np.random.normal(0.0, 0.1, (3, 1))

    nr = 3
    nc = 3

    net = Particle333Network(cost="mse")
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(4, 4, 1),
                           output_shape=(2, 2, 3),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5),
                           output_pool_shape=(2, 2, 1),
                           output_pool_delta=(0.1, 0.1, 0.1)
                           ))
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(2, 2, 3),
                           output_shape=(2, 2, 1),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5)))
    net.append(Particle333(4, 1, activation="sigmoid", nr=nr, nc=nc))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))


def main4():

    train_X = np.random.normal(0.0, 0.1, (1, 4*4))
    train_Y = np.random.normal(0.0, 0.1, (1, 1))

    nr = 3
    nc = 1

    net = Particle333Network(cost="mse")
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(4, 4, 1),
                           output_shape=(1, 1, 1),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5)))
    net.append(Particle333(1, 1, activation="sigmoid", nr=nr, nc=nc))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))


def fd():

    ts = time.time()

    train_X = None
    train_Y = None
    nc = None
    nr = None
    net = None

    if False:
        train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.1, 0.05], [0.2, -0.3], [0.1, -0.9], [0.1, 0.05]])
        train_Y = np.asarray(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        nc = 4
        nr = 3

        net = Particle333Network(cost="categorical_cross_entropy")
        net.append(Particle333(2, 5, activation="sigmoid", nr=nr, nc=nc))
        net.append(Particle333(5, 6, activation="sigmoid", nr=nr, nc=nc))
        net.append(Particle333(6, 3, activation="softmax", nr=nr, nc=nc))
    else:
        train_X = np.random.normal(0.0, 1.0, (3, 4 * 4))
        train_Y = np.random.choice([0.0, 1.0], (3, 1))

        nr = 3
        nc = 2

        # working!
        # net = Particle333Network(cost="mse")
        # net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
        #                        apply_convolution=True,
        #                        input_shape=(4, 4, 1),
        #                        output_shape=(3, 3, 2),
        #                        input_delta=(0.5, 0.5, 0.5),
        #                        output_delta=(0.5, 0.5, 0.5),
        #                        output_pool_shape=(2, 3, 1),
        #                        output_pool_delta=(0.1, 0.1, 0.1)
        #                        ))
        # net.append(Particle333(3*3*2, 1, activation="sigmoid", nr=nr, nc=nc))

        # working too!
        # net = Particle333Network(cost="mse")
        net = Particle333Network(cost="mse", regularizer="l2", lam=0.01)
        net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                               apply_convolution=True,
                               rand="normal",
                               input_shape=(4, 4, 1),
                               output_shape=(3, 3, 2),
                               input_delta=(0.5, 0.5, 0.5),
                               output_delta=(0.5, 0.5, 0.5),
                               output_pool_shape=(2, 3, 1),
                               output_pool_delta=(0.1, 0.1, 0.1)
                               ))
        net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                               apply_convolution=True,
                               rand="normal",
                               input_shape=(3, 3, 2),
                               output_shape=(3, 3, 1),
                               input_delta=(0.5, 0.5, 0.5),
                               output_delta=(0.2, 0.2, 0.2),
                               output_pool_shape=(1, 1, 1),
                               output_pool_delta=(0.1, 0.1, 0.1)
                               ))
        net.append(Particle333(3*3*1, 4, activation="sigmoid", rand="normal", nr=nr, nc=nc))
        net.append(Particle333(4, 1, activation="sigmoid", rand="normal", nr=nr, nc=nc))

    db, dq, dz, dr_inp, dr_out = net.cost_gradient(train_X, train_Y)

    h = 0.0001

    print("analytic b")
    print(db)

    fd_b = []
    for l in range(len(net.layers)):
        lb = np.zeros_like(db[l])
        for b in range(len(net.layers[l].b[0])):
            orig = net.layers[l].b[0][b]
            net.layers[l].b[0][b] += h
            fp = net.cost(train_X, train_Y)
            net.layers[l].b[0][b] -= 2*h
            fm = net.cost(train_X, train_Y)
            lb[0][b] = (fp - fm) / (2*h)
            net.layers[l].b[0][b] = orig
        fd_b.append(lb)
    print("numerical b")
    print(fd_b)

    print("analytic q")
    for x in dq:
        print(x)

    fd_q = []
    for l in range(len(net.layers)):
        lq = np.zeros_like(dq[l])
        for i in range(len(net.layers[l].q)):
            for c in range(nc):
                orig = net.layers[l].q[i][c]
                net.layers[l].q[i][c] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].q[i][c] -= 2*h
                fm = net.cost(train_X, train_Y)
                lq[i][c] = (fp - fm) / (2*h)
                net.layers[l].q[i][c] = orig
        fd_q.append(lq)

    print("numerical q")
    for x in fd_q:
        print(x)
        
    print("analytic zeta")
    for x in dz:
        print(x)

    fd_zeta = []
    for l in range(len(net.layers)):
        lzeta = np.zeros_like(dz[l])
        for i in range(len(net.layers[l].zeta)):
            for c in range(nc):
                orig = net.layers[l].zeta[i][c]
                net.layers[l].zeta[i][c] += h
                fp = net.cost(train_X, train_Y)
                net.layers[l].zeta[i][c] -= 2*h
                fm = net.cost(train_X, train_Y)
                lzeta[i][c] = (fp - fm) / (2*h)
                net.layers[l].zeta[i][c] = orig
        fd_zeta.append(lzeta)

    print("numerical zeta")
    for x in fd_zeta:
        print(x)

    print("analytic r_inp")
    for x in dr_inp:
        print(x)

    fd_r_inp = []
    for l in range(len(dr_inp)):
        lr = np.zeros_like(dr_inp[l])
        layer = net.layers[l]
        for i in range(len(layer.r_inp)):
            for r in range(nr):
                orig = layer.r_inp[i][r]
                layer.r_inp[i][r] += h
                fp = net.cost(train_X, train_Y)
                layer.r_inp[i][r] -= 2*h
                fm = net.cost(train_X, train_Y)
                lr[i][r] = (fp - fm) / (2*h)
                layer.r_inp[i][r] = orig
        fd_r_inp.append(lr)

    print("numerical r")
    for x in fd_r_inp:
        print(x)

    print("analytic r_out")
    for x in dr_out:
        print(x)

    fd_r_out = []
    for l in range(len(net.layers)):
        lr_out = np.zeros_like(dr_out[l])
        for i in range(len(net.layers[l].r_out)):
            for c in range(nc):
                for r in range(nr):
                    orig = net.layers[l].r_out[i][c][r]
                    net.layers[l].r_out[i][c][r] += h
                    fp = net.cost(train_X, train_Y)
                    net.layers[l].r_out[i][c][r] -= 2*h
                    fm = net.cost(train_X, train_Y)
                    lr_out[i][c][r] = (fp - fm) / (2*h)
                    net.layers[l].r_out[i][c][r] = orig
        fd_r_out.append(lr_out)

    print("numerical r_out")
    for x in fd_r_out:
        print(x)

    diff_b = np.sum([np.sum(np.abs(fd_b[i] - db[i])) for i in range(len(db))])
    print("diff b: {}".format(diff_b))

    diff_q = np.sum([np.sum(np.abs(fd_q[i] - dq[i])) for i in range(len(dq))])
    print("diff q: {}".format(diff_q))
    
    diff_zeta = np.sum([np.sum(np.abs(fd_zeta[i] - dz[i])) for i in range(len(dz))])
    print("diff zeta: {}".format(diff_zeta))

    diff_r_inp = np.sum([np.sum(np.abs(fd_r_inp[i] - dr_inp[i])) for i in range(len(dr_inp))])
    print("diff r_inp: {}".format(diff_r_inp))

    diff_r = np.sum([np.sum(np.abs(fd_r_out[i] - dr_out[i])) for i in range(len(dr_out))])
    print("diff r_out: {}".format(diff_r))

    print("time: {}".format(time.time() - ts))


def sgd(pool=None):

    ts = time.time()

    ndata = 500
    train_X = np.random.normal(0.0, 1.0, (ndata, 12 * 12))
    train_Y = np.random.choice([0.0, 1.0], (ndata, 2))

    nr = 3
    nc = 2

    net = Particle333Network(cost="categorical_cross_entropy")
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(12, 12, 1),
                           output_shape=(6, 6, 4),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.5, 0.5, 0.5),
                           output_pool_shape=(2, 3, 1),
                           output_pool_delta=(0.1, 0.1, 0.1)
                           ))
    net.append(Particle333(activation="sigmoid", nr=nr, nc=nc,
                           apply_convolution=True,
                           input_shape=(6, 6, 4),
                           output_shape=(3, 3, 2),
                           input_delta=(0.5, 0.5, 0.5),
                           output_delta=(0.2, 0.2, 0.2),
                           output_pool_shape=(1, 1, 1),
                           output_pool_delta=(0.1, 0.1, 0.1)
                           ))
    net.append(Particle333(3 * 3 * 2, 10, activation="sigmoid", nr=nr, nc=nc))
    net.append(Particle333(10, 2, activation="softmax", nr=nr, nc=nc))

    mbs = 20
    nt = 2
    cs = mbs // nt

    sgd = Particle333SGD(n_epochs=1, mini_batch_size=mbs, verbosity=1, weight_update="sd",
                         alpha=0.01, beta=0.8, gamma=0.95,
                         cost_freq=1, fixed_input=True,
                         n_threads=nt, chunk_size=cs
                         )

    sgd.optimize(net, train_X, train_Y, pool=pool)

    print("time: {}".format(time.time() - ts))


def mnist(pool=None):
    raw_data_train = pd.read_csv("/Users/adrianlange/programming/MNIST/data/mnist_train.csv", header=None)
    raw_data_test = pd.read_csv("/Users/adrianlange/programming/MNIST/data/mnist_test.csv", header=None)

    # In[7]:

    # Prepare data
    denom = 255.0
    # denom = 1.0
    X = np.asarray(raw_data_train.iloc[:,1:] / denom)  # scaled values in range [0:1]
    X_test = np.asarray(raw_data_test.iloc[:,1:] / denom)

    # Z-score centering
    m = X.mean()
    s = X.std()
    X = (X - m) / s
    X_test = (X_test - m) / s

    # length ten categorical vector
    Y = []
    for val in raw_data_train.iloc[:,0]:
        y = np.zeros(10)
        y[val] = 1.0
        Y.append(y)
    Y = np.asarray(Y)

    Y_test = []
    for val in raw_data_test.iloc[:,0]:
        y = np.zeros(10)
        y[val] = 1.0
        Y_test.append(y)
    Y_test = np.asarray(Y_test)

    # Data subset
    # n_sub = len(X)
    n_sub = 100
    X_sub = X[:n_sub, :]
    Y_sub = Y[:n_sub, :]
    X_other = X[n_sub:, :]
    Y_other = Y[n_sub:, :]

    s = 0.1
    q = 0.1
    b = 0.1
    z = 0.1
    p = "gaussian"
    rand = "normal"

    nr = 3
    nc = 4

    net = Particle333Network(cost="categorical_cross_entropy")
    net.append(Particle333(activation="tanh", nr=nr, nc=nc,
                           s=0.1, q=0.1, b=b,
                           z=z, zoff=4.0,
                           potential=p, rand=rand,
                           apply_convolution=True,
                           input_shape=(28, 28, 1),
                           output_shape=(10, 10, 10),
                           input_delta=(0.1, 0.1, 0.1),
                           output_delta=(0.2, 0.2, 0.2),
                           output_pool_shape=(2, 2, 1),
                           output_pool_delta=(0.1, 0.1, 0.1)
                           ))
    net.append(Particle333(activation="tanh", nr=nr, nc=nc,
                           s=0.1, q=0.1, b=b,
                           z=z, zoff=4.0,
                           potential=p, rand=rand,
                           apply_convolution=True,
                           input_shape=(10, 10, 10),
                           output_shape=(7, 7, 10),
                           input_delta=(0.1, 0.1, 0.1),
                           output_delta=(0.2, 0.2, 0.2),
                           output_pool_shape=(2, 2, 1),
                           output_pool_delta=(0.1, 0.1, 0.1)
                           ))
    net.append(Particle333(7*7*10, 100, activation="tanh", potential=p, rand=rand, s=1.0, q=0.1, b=b, z=z, zoff=1.0, nr=nr, nc=nc))
    net.append(Particle333(100, 10, activation="softmax", potential=p, rand=rand, s=1.0, q=0.1, b=b, z=z, zoff=1.0, nr=nr, nc=nc))

    # input position to be fixed at the origin
    net.layers[0].r_inp = np.zeros_like(net.layers[0].r_inp)

    n = 0
    mbs = 4
    nt = 1
    cs = mbs // nt

    sgd = Particle333SGD(n_epochs=1, mini_batch_size=mbs, verbosity=1, weight_update="rmsprop",
                         alpha=0.01, beta=0.8, gamma=0.95,
                         cost_freq=1, fixed_input=True,
                         n_threads=nt, chunk_size=cs
                        )

    sgd.optimize(net, X_sub, Y_sub, pool=pool)


if __name__ == "__main__":

    # Ensure same seed
    np.random.seed(100)

    # main()
    # main2()
    # main3()
    # main4()
    fd()

    # sgd(pool=Pool(processes=2))
    # mnist(pool=Pool(processes=2))
