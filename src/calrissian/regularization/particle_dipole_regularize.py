import numpy as np


class ParticleDipoleRegularize(object):
    """
    L2 regularizer for charges
    """

    def __init__(self, coeff_lambda=0.0, zeta=8.0):
        self.coeff_lambda = coeff_lambda
        self.zeta = zeta

        self.n = 1

    def cost(self, particle_input, layers):
        # c = 0.0
        # for layer in layers:
        #     c += np.sum(layer.q * layer.q)
        # return self.coeff_lambda * c

        # Compute the matrices
        c = 0.0
        r = particle_input.get_rxyz()
        for i, layer in enumerate(layers):
            w = layer.compute_w(r)
            c += np.sum(w * w)
            # c += np.sum(np.abs(w))
            r = layer.get_rxyz()

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq):
        # two_lambda = 2.0 * self.coeff_lambda
        # for l, layer in enumerate(layers):
        #     dc_dq[l] += two_lambda * layer.q
        return dc_dq
