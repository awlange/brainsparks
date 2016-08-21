import numpy as np


class ParticleRegularizeL2Plus(object):
    """
    L2 plus regularizer
    """

    def __init__(self, coeff_lambda=0.0, zeta=8.0):
        self.coeff_lambda = coeff_lambda
        self.zeta = zeta
        self.n = 1

    def cost(self, particle_input, layers):
        c = 0.0

        # Compute the matrices
        r = particle_input.get_rxyz()
        for l, layer in enumerate(layers):
            w = layer.compute_w(r)

            wt = w.transpose()
            for j in range(layer.output_size):
                for k in range(j, layer.output_size):
                    c += np.abs(wt[j].dot(wt[k]))
            r = layer.get_rxyz()

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db, dc_dr):
        return dc_dq, dc_db, dc_dr
