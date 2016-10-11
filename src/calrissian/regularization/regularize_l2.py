import numpy as np


class RegularizeL2(object):
    """
    L2 plus regularizer for weights
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        c = 0.0
        for layer in layers:
            # c += np.sum(layer.b * layer.b) + np.sum(layer.w * layer.w)
            c += np.sum(layer.w * layer.w)
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        two_lambda = 2.0 * self.coeff_lambda
        for l, layer in enumerate(layers):
            # dc_db[l] += two_lambda * layer.b
            dc_dw[l] += two_lambda * layer.w
        return dc_db, dc_dw
