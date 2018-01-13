import numpy as np


class ParticleVectorRegularizeVector(object):
    """
    L2 regularizer for the vector components
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, particle_input, layers):
        c = 0.0

        for l, layer in enumerate(layers):
            c += np.sum(layer.b**2)
            for v in range(layer.nv):
                c += np.sum(layer.nvectors[v]**2)

        for v in range(particle_input.nv):
            c += np.sum(particle_input.nvectors[v]**2)

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dv, dc_db):
        two_lambda = 2.0 * self.coeff_lambda

        for l, layer in enumerate(layers):
            dc_db[l] += two_lambda * layer.b
            for v in range(layer.nv):
                dc_dv[l+1][v] += two_lambda * layer.nvectors[v]

        for v in range(particle_input.nv):
            dc_dv[0][v] += two_lambda * particle_input.nvectors[v]

        return dc_dv, dc_db
