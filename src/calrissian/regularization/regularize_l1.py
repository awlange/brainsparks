import numpy as np


class RegularizeL1(object):
    """
    L1 regularizer for weights and biases
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, w):
        return self.coeff_lambda * np.sum(np.abs(w))

    def cost_gradient(self, w):
        return self.coeff_lambda * np.piecewise(w, [w < 0, w >= 0], [-1, 1])
