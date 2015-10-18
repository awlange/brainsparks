import numpy as np


class Activation(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation.sigmoid
        if nl == "linear":
            return Activation.linear
        if nl == "relu":
            return Activation.relu
        if nl == "tanh":
            return Activation.tanh
        if nl == "softmax":
            return Activation.softmax
        if nl == "softplus":
            return Activation.softplus

        raise NameError("Activation name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation.d_sigmoid
        if nl == "linear":
            return Activation.d_linear
        if nl == "relu":
            return Activation.d_relu
        if nl == "tanh":
            return Activation.d_tanh
        if nl == "softmax":
            return Activation.d_softmax
        if nl == "softplus":
            return Activation.d_softplus

        raise NameError("Activation name {} is not implemented.".format(name))

    # Sigmoid

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return Activation.sigmoid(x) * (1.0 - Activation.sigmoid(x))

    # Linear

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def d_linear(x):
        return 1.0

    # ReLU

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def d_relu(x):
        return np.piecewise(x, [x < 0.0, x >= 0], [0.0, 1.0])

    # tanh

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        tx = np.tanh(x)
        return 1.0 - tx*tx

    # SoftMax

    @staticmethod
    def softmax(x):
        ex = np.exp(x)
        denoms = np.sum(ex, axis=1)
        for i in range(len(ex)):
            ex[i] /= denoms[i]
        return ex

    @staticmethod
    def d_softmax(x):
        # Note: not actually used. So, just echo input to maintain interface.
        return x

    # Softplus

    @staticmethod
    def softplus(x):
        return np.log(1.0 + np.exp(x))

    @staticmethod
    def d_softplus(x):
        return Activation.sigmoid(x)
