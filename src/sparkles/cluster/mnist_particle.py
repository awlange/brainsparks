from pyspark import SparkContext

import numpy

"""
Script for fitting MNIST using Particle Networks
"""


def main():
    sc = SparkContext(appName="test")

    XY_train_rdd, XY_test_rdd = prepare_data(sc, n_workers=3)

    total = XY_train_rdd.map(lambda x: len(x[0])).reduce(lambda a, b: a + b)
    print(total)

    sc.stop()


def prepare_data(sc, n_workers=3):
    train_data_filename = "/home/pi/data/mnist/mnist_train.csv"
    test_data_filename = "/home/pi/data/mnist/mnist_test.csv"

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    with open(train_data_filename) as f:
        for line in f:
            tmp = [int(s) for s in line.split(",")]
            X_train.append([float(x) / 255.0 for x in tmp[1:]])
            ytmp = [0.0 for _ in range(10)]
            ytmp[tmp[0]] = 1.0
            Y_train.append(ytmp)

    with open(test_data_filename) as f:
        for line in f:
            tmp = [int(s) for s in line.split(",")]
            X_test.append([float(x) / 255.0 for x in tmp[1:]])
            ytmp = [0.0 for _ in range(10)]
            ytmp[tmp[0]] = 1.0
            Y_test.append(ytmp)

    XY_train_rdd = sc.parallelize(list(zip(chunks(X_train, 3), chunks(Y_train, 3))))
    XY_test_rdd = sc.parallelize(list(zip(chunks(X_test, 3), chunks(Y_test, 3))))

    return XY_train_rdd, XY_test_rdd


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


if __name__ == "__main__":
    main()
