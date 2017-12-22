import numpy as np

MAX_FLOAT = np.finfo(0.0).max


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
        if nl == "exp":
            return Activation().exp
        if nl == "relu":
            return Activation().relu
        if nl == "relu2":
            return Activation().relu2
        if nl == "relu3":
            return Activation().relu3
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
        if nl == "exp":
            return Activation().d_exp
        if nl == "relu":
            return Activation().d_relu
        if nl == "relu2":
            return Activation().d_relu2
        if nl == "relu3":
            return Activation().d_relu3
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

    # Exponential

    def exp(self, x):
        return np.exp(x)

    def d_exp(self, x):
        return np.exp(x)

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

    # ReLU2: ReLU half pos, half neg

    def relu2(self, x):
        xt = x.transpose()
        xs = np.array_split(xt, 2, axis=1)
        spos = np.maximum(0, xs[0])
        sneg = np.minimum(0, xs[1])
        return np.concatenate((spos, sneg), axis=1).transpose()

    def d_relu2(self, x):
        xt = x.transpose()
        xs = np.array_split(xt, 2, axis=1)
        spos = np.piecewise(xs[0], [xs[0] < 0.0, xs[0] >= 0], [0.0, 1.0])
        sneg = np.piecewise(xs[1], [xs[1] < 0.0, xs[1] >= 0], [1.0, 0.0])
        return np.concatenate((spos, sneg), axis=1).transpose()

    # ReLU

    def relu3(self, x):
        return np.maximum(0, x + 0.5*x*x)

    def d_relu3(self, x):
        return np.piecewise(x, [x < 0.0, x >= 0], [0.0, 1.0 + x])

    # tanh

    def tanh(self, x):
        return np.tanh(x)

    def d_tanh(self, x):
        tx = np.tanh(x)
        return 1.0 - tx*tx

    # SoftMax

    def softmax(self, x):
        # ex = np.exp(x)
        # denoms = np.sum(ex, axis=1)
        # for i in range(len(ex)):
        #     ex[i] /= denoms[i]
        # return ex
        # Code borrowed from Theano for numerical stability:
        # http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
        e_x = np.exp(x - x.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out

    def d_softmax(self, x):
        # Note: not actually used. So, just echo input to maintain interface.
        return x

    def atomic_softmax(self, x):
        ex = np.exp(x)
        denom = np.sum(ex)
        return ex / denom

    # Softplus

    def softplus(self, x):
        # return np.log(1.0 + np.exp(x))
        # Prevent overflow
        return np.piecewise(x, [x < 10.0, x >= 10.0], [lambda x: np.log(1.0 + np.exp(x)), lambda x: x])

    def d_softplus(self, x):
        return Activation().sigmoid(x)
