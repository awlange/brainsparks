import numpy as np


class ParticleRegularize(object):
    """
    L2 regularizer for charges
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, particle_input, layers):
        # c = np.sum(particle_input.q * particle_input.q)
        # c = np.sum(particle_input.r * particle_input.r)
        c = 0.0
        for layer in layers:
            # c += np.sum(layer.q * layer.q) + np.sum(layer.b * layer.b)
            c += np.sum(layer.q * layer.q)
            # c += np.sum(layer.r * layer.r)
        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db, dc_dr):
        two_lambda = 2.0 * self.coeff_lambda
        # dc_dq[0] += two_lambda * particle_input.q
        # dc_dr[0] += two_lambda * particle_input.r
        for l, layer in enumerate(layers):
            dc_dq[l] += two_lambda * layer.q
            # dc_db[l] += two_lambda * layer.b
            # dc_dr[l+1] += two_lambda * layer.r
        return dc_dq, dc_db, dc_dr
