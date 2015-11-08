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
            # c += np.sum(layer.q * layer.q) + np.sum(layer.b * layer.b)
            c += np.sum(layer.q * layer.q)
        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db):
        two_lambda = 2.0 * self.coeff_lambda
        dc_dq[0] += two_lambda * particle_input.q
        for l, layer in enumerate(layers):
            dc_dq[l+1] += two_lambda * layer.q
            # dc_db[l] += two_lambda * layer.b
        return dc_dq, dc_db
