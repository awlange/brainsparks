import numpy as np


class ParticleRegularizeOrthogonal(object):
    """
    Orthogonal regularizer
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, particle_input, layers):
        c = 0.0

        # Compute the matrices
        r = particle_input.get_rxyz()
        for l, layer in enumerate(layers):
            w = layer.compute_w(r)
            wt = w.transpose()
            for j in range(layer.output_size):
                wtj = wt[j] / np.sqrt(wt[j].dot(wt[j]))
                for k in range(layer.output_size):
                    if j == k:
                        continue
                    wtk = wt[k] / np.sqrt(wt[k].dot(wt[k]))
                    c += np.abs(wtj.dot(wtk))
            r = layer.get_rxyz()

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db, dc_dr):
        return dc_dq, dc_db, dc_dr
