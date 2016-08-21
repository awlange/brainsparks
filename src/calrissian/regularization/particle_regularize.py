import numpy as np


class ParticleRegularize(object):
    """
    L2 regularizer for charges
    """

    def __init__(self, coeff_lambda=0.0, zeta=8.0):
        self.coeff_lambda = coeff_lambda
        self.zeta = zeta
        self.n = 1

    def cost(self, particle_input, layers):
        c = 0.0

        # c = np.sum(particle_input.q * particle_input.q)
        # # c = np.sum(particle_input.rx * particle_input.rx + particle_input.ry * particle_input.ry + particle_input.rz * particle_input.rz)

        # for layer in layers:
        #     # c += np.sum(layer.q * layer.q) + np.sum(layer.b * layer.b)
        #     # c += np.sum(layer.q * layer.q)
        #     # c += np.sum(layer.rx * layer.rx + layer.ry * layer.ry + layer.rz * layer.rz)
        #
        #     # Layer inter-particle repulsion
        #     for i in range(layer.output_size):
        #         rx_i = layer.rx[i]
        #         ry_i = layer.ry[i]
        #         rz_i = layer.rz[i]
        #         for j in range(i+1, layer.output_size):
        #             dx = layer.rx[j] - rx_i
        #             dy = layer.ry[j] - ry_i
        #             dz = layer.rz[j] - rz_i
        #             d2 = dx*dx + dy*dy + dz*dz
        #             c += np.exp(-self.zeta * d2)
        #
        #     n = layer.output_size
        #     c /= (n * (n-1)) / 2

        # # Input layer inter-particle repulsion
        # for i in range(particle_input.output_size):
        #     rx_i = particle_input.rx[i]
        #     ry_i = particle_input.ry[i]
        #     rz_i = particle_input.rz[i]
        #     for j in range(i+1, particle_input.output_size):
        #         dx = particle_input.rx[j] - rx_i
        #         dy = particle_input.ry[j] - ry_i
        #         dz = particle_input.rz[j] - rz_i
        #         d2 = dx*dx + dy*dy + dz*dz
        #         c += np.exp(-self.zeta * d2)

        # c /= n

        # Compute the matrices
        r = particle_input.get_rxyz()
        for i, layer in enumerate(layers):
            w = layer.compute_w(r)
            c += np.sum(w * w)
            r = layer.get_rxyz()

        return self.coeff_lambda * c

    def cost_gradient(self, particle_input, layers, dc_dq, dc_db, dc_dr):
        # dc_dr_x = dc_dr[0]
        # dc_dr_y = dc_dr[1]
        # dc_dr_z = dc_dr[2]
        #
        # two_lambda = 2.0 * self.coeff_lambda
        #
        # # # dc_dq[0] += two_lambda * particle_input.q
        # # # dc_dr_x[0] += two_lambda * particle_input.rx
        # # # dc_dr_y[0] += two_lambda * particle_input.ry
        # # # dc_dr_z[0] += two_lambda * particle_input.rz
        # for l, layer in enumerate(layers):
        #     # dc_dq[l] += two_lambda * layer.q
        #     # dc_db[l] += two_lambda * layer.b
        #     # dc_dr_x[l+1] += two_lambda * layer.rx
        #     # dc_dr_y[l+1] += two_lambda * layer.ry
        #     # dc_dr_z[l+1] += two_lambda * layer.rz
        #
        #     n = layer.output_size
        #     n = (n * (n-1)) / 2
        #     for i in range(layer.output_size):
        #         rx_i = layer.rx[i]
        #         ry_i = layer.ry[i]
        #         rz_i = layer.rz[i]
        #         for j in range(i+1, layer.output_size):
        #             dx = layer.rx[j] - rx_i
        #             dy = layer.ry[j] - ry_i
        #             dz = layer.rz[j] - rz_i
        #             d2 = dx*dx + dy*dy + dz*dz
        #             # tmp = two_lambda * self.zeta * np.exp(-self.zeta * d2)
        #             tmp = two_lambda * self.zeta * np.exp(-self.zeta * d2) / n
        #             tx = tmp * dx
        #             ty = tmp * dy
        #             tz = tmp * dz
        #
        #             dc_dr_x[l+1][i] += tx
        #             dc_dr_y[l+1][i] += ty
        #             dc_dr_z[l+1][i] += tz
        #             dc_dr_x[l+1][j] -= tx
        #             dc_dr_y[l+1][j] -= ty
        #             dc_dr_z[l+1][j] -= tz

        # #
        # # # for i in range(particle_input.output_size):
        # # #     rx_i = particle_input.rx[i]
        # # #     ry_i = particle_input.ry[i]
        # # #     rz_i = particle_input.rz[i]
        # # #     for j in range(i+1, particle_input.output_size):
        # # #         dx = particle_input.rx[j] - rx_i
        # # #         dy = particle_input.ry[j] - ry_i
        # # #         dz = particle_input.rz[j] - rz_i
        # # #         d2 = dx*dx + dy*dy + dz*dz
        # # #         tmp = two_lambda * self.zeta * np.exp(-self.zeta * d2)
        # # #         tx = tmp * dx
        # # #         ty = tmp * dy
        # # #         tz = tmp * dz
        # # #
        # # #         dc_dr_x[0][i] += tx
        # # #         dc_dr_y[0][i] += ty
        # # #         dc_dr_z[0][i] += tz
        # # #         dc_dr_x[0][j] -= tx
        # # #         dc_dr_y[0][j] -= ty
        # # #         dc_dr_z[0][j] -= tz
        #


        # dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z)

        return dc_dq, dc_db, dc_dr
