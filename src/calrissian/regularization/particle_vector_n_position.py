import numpy as np


class ParticleVectorRegularizeDistance(object):
    """
    L2 regularizer by particle distance
    """

    def __init__(self, coeff_lambda=0.0, zeta=8.0):
        self.coeff_lambda = coeff_lambda
        self.zeta = zeta

    def cost(self, particle_input, layers):
        c = 0.0

        for l, layer in enumerate(layers):
            if l != len(layers)-1:
                # Output layer only!
                continue

            # Layer inter-particle repulsion
            for i in range(layer.output_size):
                for j in range(i+1, layer.output_size):
                    d2 = 0.0
                    for d in range(layer.nr):
                        dr = layer.positions[d][j] - layer.positions[d][i]
                        d2 += dr*dr
                    c += np.exp(-self.zeta * d2)

        # # Input layer inter-particle repulsion
        # for i in range(particle_input.output_size):
        #     for j in range(i+1, particle_input.output_size):
        #         d2 = 0.0
        #         for d in range(particle_input.nr):
        #             dr = particle_input.positions[d][j] - r_i
        #             d2 += np.sum(dr*dr)
        #         c += np.exp(-self.zeta * d2)

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dr):
        two_lambda = 2.0 * self.coeff_lambda

        for l, layer in enumerate(layers):
            if l != len(layers)-1:
                # Output layer only!
                continue

            for i in range(layer.output_size):
                for j in range(i+1, layer.output_size):
                    drr = []
                    d2 = 0.0
                    for d in range(layer.nr):
                        dr = layer.positions[d][j] - layer.positions[d][i]
                        d2 += dr*dr
                        drr.append(dr)
                    tmp = two_lambda * self.zeta * np.exp(-self.zeta * d2)

                    for d in range(layer.nr):
                        dc_dr[l+1][d][i] += tmp * drr[d]
                        dc_dr[l+1][d][j] -= tmp * drr[d]

        # for i in range(particle_input.output_size):
        #     for j in range(i+1, particle_input.output_size):
        #         drr = []
        #         d2 = 0.0
        #         for d in range(layer.nr):
        #             dr = particle_input.positions[j] - particle_input.positions[d][i]
        #             d2 += dr*dr
        #             drr.append(dr)
        #         tmp = two_lambda * self.zeta * np.exp(-self.zeta * d2)
        #
        #         or d in range(layer.nr):
        #             dc_dr[0][d][i] += tmp * dr[d]
        #             dc_dr[0][d][j] -= tmp * dr[d]

        return dc_dr
