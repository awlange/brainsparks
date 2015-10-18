import numpy as np


class Activation(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation.sigmoid
        elif nl == "linear":
            return Activation.linear
        elif nl == "relu":
            return Activation.relu
        elif nl == "tanh":
            return Activation.tanh

        raise NameError("Activation name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "sigmoid":
            return Activation.d_sigmoid
        elif nl == "linear":
            return Activation.d_linear
        elif nl == "relu":
            return Activation.d_relu
        elif nl == "tanh":
            return Activation.d_tanh

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
        return np.piecewise(x, [x < 0.0, x >= 0.0], [0.0, lambda z: z])

    @staticmethod
    def d_relu(x):
        return np.piecewise(x, [x < 0.0, x >= 0], [0.0, 1.0])

    # tanh

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1.0 - np.tanh(x)*np.tanh(x)
