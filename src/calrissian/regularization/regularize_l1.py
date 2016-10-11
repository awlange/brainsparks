import numpy as np


class RegularizeL1(object):
    """
    L1 regularizer for weights
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        c = 0.0
        for layer in layers:
            c += np.sum(np.abs(layer.w))
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        for l, layer in enumerate(layers):
            dc_dw[l] += self.coeff_lambda * np.piecewise(layer.w, [layer.w < 0, layer.w >= 0], [-1, 1])
        return dc_db, dc_dw