import numpy as np


class ParticleRegularizeL2Charge(object):
    """
    L2 regularizer for charges
    """

    def __init__(self, coeff_lambda=0.0, zeta=8.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, particle_input, layers):
        c = 0.0
        for layer in layers:
            c += np.sum(layer.q * layer.q)
        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db, dc_dr):
        two_lambda = 2.0 * self.coeff_lambda
        for l, layer in enumerate(layers):
            dc_dq[l] += two_lambda * layer.q
        return dc_dq, dc_db, dc_dr
