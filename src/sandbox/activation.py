import math
import numpy as np


class Activation(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        if name == "sigmoid":
            return Activation.sigmoid
        if name == "np_sigmoid":
            return Activation.np_sigmoid
        return None

    @staticmethod
    def get_d(name):
        if name == "sigmoid":
            return Activation.d_sigmoid
        if name == "np_sigmoid":
            return Activation.np_d_sigmoid
        return None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def d_sigmoid(x):
        return Activation.sigmoid(x) * (1.0 - Activation.sigmoid(x))

    @staticmethod
    def np_sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def np_d_sigmoid(x):
        return Activation.np_sigmoid(x) * (1.0 - Activation.np_sigmoid(x))
