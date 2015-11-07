import numpy as np


class ParticleRegularize(object):
    """
    L2 regularizer for charges
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        c = 0.0
        for layer in layers:
            c += np.sum(layer.q * layer.q)
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_dq):
        two_lambda = 2.0 * self.coeff_lambda
        for l, layer in enumerate(layers):
            dc_dq[l] += two_lambda * layer.q
        return dc_dq
