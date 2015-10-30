import numpy as np


class RegularizeL2(object):
    """
    L2 regularizer for weights and biases
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, w):
        return self.coeff_lambda * np.sum(w * w)

    def cost_gradient(self, w):
        return 2.0 * self.coeff_lambda * w
