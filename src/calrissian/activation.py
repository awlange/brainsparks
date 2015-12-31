import numpy as np


class Activation(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation().sigmoid
        if nl == "linear":
            return Activation().linear
        if nl == "relu":
            return Activation().relu
        if nl == "leaky_relu":
            return Activation().leaky_relu
        if nl == "tanh":
            return Activation().tanh
        if nl == "softmax":
            return Activation().softmax
        if nl == "softplus":
            return Activation().softplus
        if nl == "atomic_softmax":
            return Activation().atomic_softmax

        raise NameError("Activation name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation().d_sigmoid
        if nl == "linear":
            return Activation().d_linear
        if nl == "relu":
            return Activation().d_relu
        if nl == "leaky_relu":
            return Activation().d_leaky_relu
        if nl == "tanh":
            return Activation().d_tanh
        if nl == "softmax" or "atomic_softmax":
            return Activation().d_softmax
        if nl == "softplus":
            return Activation().d_softplus

        raise NameError("Activation name {} is not implemented.".format(name))

    # Sigmoid

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def d_sigmoid(self, x):
        return Activation().sigmoid(x) * (1.0 - Activation().sigmoid(x))

    # Linear

    def linear(self, x):
        return x

    def d_linear(self, x):
        return np.ones(x.shape)

    # ReLU

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return np.piecewise(x, [x < 0.0, x >= 0], [0.0, 1.0])

    # LeakyReLU

    def leaky_relu(self, x):
        return np.maximum(0.01*x, x)

    def d_leaky_relu(self, x):
        return np.piecewise(x, [x < 0.0, x >= 0], [0.01, 1.0])

    # tanh

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        tx = np.tanh(x)
        return 1.0 - tx*tx

    # SoftMax

    def softmax(self, x):
        ex = np.exp(x)
        denoms = np.sum(ex, axis=1)
        for i in range(len(ex)):
            ex[i] /= denoms[i]
        return ex

    def d_softmax(self, x):
        # Note: not actually used. So, just echo input to maintain interface.
        return x

    def atomic_softmax(self, x):
        ex = np.exp(x)
        denom = np.sum(ex)
        return ex / denom

    # Softplus

    def softplus(self, x):
        return np.log(1.0 + np.exp(x))

    def d_softplus(self, x):
        return Activation.sigmoid(x)
