from .cost import Cost

from .layers.particle_dipole import ParticleDipole
from .layers.particle_dipole import ParticleDipoleInput

import numpy as np
import json


class ParticleDipoleNetwork(object):

    def __init__(self, particle_input=None, cost="mse", regularizer=None):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.particle_input = particle_input

    def append(self, layer):
        """
        Appends a layer to the network

        :param layer:
        :return:
        """
        self.layers.append(layer)

    def build(self):
        """
        Handle networks layer dimensions checks, other possible initializations

        Release build lock

        :return:
        """
        # TODO

        self.lock_built = True

    def predict(self, data_X):
        """
        Pass given input through network to compute the output prediction

        :param data_X:
        :return:
        """
        a, r = self.particle_input.feed_forward(data_X)
        for layer in self.layers:
            a, r = layer.feed_forward(a, r)
        return a

    def cost(self, data_X, data_Y):
        """
        Compute the cost for all input data corresponding to expected output

        :param data_X:
        :param data_Y:
        :return:
        """
        c = self.cost_function(data_Y, self.predict(data_X))
        # Add cost of bonds
        # c += self.particle_input.compute_bond_cost()
        # for layer in self.layers:
        #     c += layer.compute_bond_cost()

        if self.regularizer is not None:
            c += self.regularizer.cost(self.particle_input, self.layers)

        return c

    def cost_gradient_thread(self, data_XY):
        """
        Wrapper for multithreaded call
        :param data_XY:
        :return:
        """
        return self.cost_gradient(data_XY[0], data_XY[1])

    def cost_gradient(self, data_X, data_Y):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dq = []
        dc_drx_pos = [np.zeros(self.particle_input.output_size)]
        dc_dry_pos = [np.zeros(self.particle_input.output_size)]
        dc_drz_pos = [np.zeros(self.particle_input.output_size)]
        dc_drx_neg = [np.zeros(self.particle_input.output_size)]
        dc_dry_neg = [np.zeros(self.particle_input.output_size)]
        dc_drz_neg = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_drx_pos.append(np.zeros(len(layer.q)))
            dc_dry_pos.append(np.zeros(len(layer.q)))
            dc_drz_pos.append(np.zeros(len(layer.q)))
            dc_drx_neg.append(np.zeros(len(layer.q)))
            dc_dry_neg.append(np.zeros(len(layer.q)))
            dc_drz_neg.append(np.zeros(len(layer.q)))

        sigma_Z = []
        A_scaled, _ = self.particle_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        prev_layer_rr = self.particle_input.get_rxyz()
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l], prev_layer_rr)
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))
            prev_layer_rr = layer.get_rxyz()

        delta_L = self.cost_d_function(data_Y, A[-1], sigma_Z[-1])

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((prev_layer.output_size, len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            qj = layer.q[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            dx_pos_pos = (prev_layer.rx_pos - layer.rx_pos[j]).reshape((prev_layer.output_size, 1))
            dy_pos_pos = (prev_layer.ry_pos - layer.ry_pos[j]).reshape((prev_layer.output_size, 1))
            dz_pos_pos = (prev_layer.rz_pos - layer.rz_pos[j]).reshape((prev_layer.output_size, 1))
            # tmp = 1.0/np.sqrt(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2)
            # tmp3 = tmp*tmp*tmp
            # dx_pos_pos *= tmp3
            # dy_pos_pos *= tmp3
            # dz_pos_pos *= tmp3
            tmp = np.exp(-(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2))
            dx_pos_pos *= tmp
            dy_pos_pos *= tmp
            dz_pos_pos *= tmp
            potential = tmp

            dx_pos_neg = (prev_layer.rx_pos - layer.rx_neg[j]).reshape((prev_layer.output_size, 1))
            dy_pos_neg = (prev_layer.ry_pos - layer.ry_neg[j]).reshape((prev_layer.output_size, 1))
            dz_pos_neg = (prev_layer.rz_pos - layer.rz_neg[j]).reshape((prev_layer.output_size, 1))
            # tmp = -1.0/np.sqrt(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2)
            # tmp3 = tmp*tmp*tmp
            # dx_pos_neg *= tmp3
            # dy_pos_neg *= tmp3
            # dz_pos_neg *= tmp3
            tmp = -np.exp(-(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2))
            dx_pos_neg *= tmp
            dy_pos_neg *= tmp
            dz_pos_neg *= tmp
            potential += tmp

            dx_neg_pos = (prev_layer.rx_neg - layer.rx_pos[j]).reshape((prev_layer.output_size, 1))
            dy_neg_pos = (prev_layer.ry_neg - layer.ry_pos[j]).reshape((prev_layer.output_size, 1))
            dz_neg_pos = (prev_layer.rz_neg - layer.rz_pos[j]).reshape((prev_layer.output_size, 1))
            # tmp = -1.0/np.sqrt(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2)
            # tmp3 = tmp*tmp*tmp
            # dx_neg_pos *= tmp3
            # dy_neg_pos *= tmp3
            # dz_neg_pos *= tmp3
            tmp = -np.exp(-(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2))
            dx_neg_pos *= tmp
            dy_neg_pos *= tmp
            dz_neg_pos *= tmp
            potential += tmp

            dx_neg_neg = (prev_layer.rx_neg - layer.rx_neg[j]).reshape((prev_layer.output_size, 1))
            dy_neg_neg = (prev_layer.ry_neg - layer.ry_neg[j]).reshape((prev_layer.output_size, 1))
            dz_neg_neg = (prev_layer.rz_neg - layer.rz_neg[j]).reshape((prev_layer.output_size, 1))
            # tmp = 1.0/np.sqrt(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2)
            # tmp3 = tmp*tmp*tmp
            # dx_neg_neg *= tmp3
            # dy_neg_neg *= tmp3
            # dz_neg_neg *= tmp3
            tmp = np.exp(-(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2))
            dx_neg_neg *= tmp
            dy_neg_neg *= tmp
            dz_neg_neg *= tmp
            potential += tmp

            # Next delta
            next_delta += (qj * trans_delta_L_j) * potential * trans_sigma_Z_l

            # Charge gradient
            dq = potential * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            # tmp = qj * dq / potential
            tmp = 2 * qj * (Al_trans * trans_delta_L_j)

            dc_drx_pos[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
            dc_dry_pos[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
            dc_drz_pos[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

            dc_drx_pos[l-1] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
            dc_dry_pos[l-1] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
            dc_drz_pos[l-1] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

            dc_drx_neg[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
            dc_dry_neg[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
            dc_drz_neg[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

            dc_drx_neg[l-1] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
            dc_dry_neg[l-1] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
            dc_drz_neg[l-1] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

            if self.regularizer is not None:
                # ----- L2 regularized w_ij by position
                coeff_lambda = self.regularizer.coeff_lambda
                w_ij = qj * potential

                # Charge gradient
                dq = 2 * coeff_lambda * w_ij * potential
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                tmp = 2 * qj * (2 * coeff_lambda * w_ij)

                dc_drx_pos[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
                dc_dry_pos[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
                dc_drz_pos[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

                dc_drx_pos[l-1] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
                dc_dry_pos[l-1] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
                dc_drz_pos[l-1] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

                dc_drx_neg[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
                dc_dry_neg[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
                dc_drz_neg[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

                dc_drx_neg[l-1] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
                dc_dry_neg[l-1] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
                dc_drz_neg[l-1] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((prev_layer.output_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Position gradient
            for j in range(layer.output_size):
                qj = layer.q[j]
                this_delta_j = this_delta[j]

                dx_pos_pos = (prev_layer.rx_pos - layer.rx_pos[j]).reshape((prev_layer.output_size, 1))
                dy_pos_pos = (prev_layer.ry_pos - layer.ry_pos[j]).reshape((prev_layer.output_size, 1))
                dz_pos_pos = (prev_layer.rz_pos - layer.rz_pos[j]).reshape((prev_layer.output_size, 1))
                # tmp = 1.0/np.sqrt(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2)
                # tmp3 = tmp*tmp*tmp
                # dx_pos_pos *= tmp3
                # dy_pos_pos *= tmp3
                # dz_pos_pos *= tmp3
                tmp = np.exp(-(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2))
                dx_pos_pos *= tmp
                dy_pos_pos *= tmp
                dz_pos_pos *= tmp
                potential = tmp

                dx_pos_neg = (prev_layer.rx_pos - layer.rx_neg[j]).reshape((prev_layer.output_size, 1))
                dy_pos_neg = (prev_layer.ry_pos - layer.ry_neg[j]).reshape((prev_layer.output_size, 1))
                dz_pos_neg = (prev_layer.rz_pos - layer.rz_neg[j]).reshape((prev_layer.output_size, 1))
                # tmp = -1.0/np.sqrt(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2)
                # tmp3 = tmp*tmp*tmp
                # dx_pos_neg *= tmp3
                # dy_pos_neg *= tmp3
                # dz_pos_neg *= tmp3
                tmp = -np.exp(-(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2))
                dx_pos_neg *= tmp
                dy_pos_neg *= tmp
                dz_pos_neg *= tmp
                potential += tmp

                dx_neg_pos = (prev_layer.rx_neg - layer.rx_pos[j]).reshape((prev_layer.output_size, 1))
                dy_neg_pos = (prev_layer.ry_neg - layer.ry_pos[j]).reshape((prev_layer.output_size, 1))
                dz_neg_pos = (prev_layer.rz_neg - layer.rz_pos[j]).reshape((prev_layer.output_size, 1))
                # tmp = -1.0/np.sqrt(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2)
                # tmp3 = tmp*tmp*tmp
                # dx_neg_pos *= tmp3
                # dy_neg_pos *= tmp3
                # dz_neg_pos *= tmp3
                tmp = -np.exp(-(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2))
                dx_neg_pos *= tmp
                dy_neg_pos *= tmp
                dz_neg_pos *= tmp
                potential += tmp

                dx_neg_neg = (prev_layer.rx_neg - layer.rx_neg[j]).reshape((prev_layer.output_size, 1))
                dy_neg_neg = (prev_layer.ry_neg - layer.ry_neg[j]).reshape((prev_layer.output_size, 1))
                dz_neg_neg = (prev_layer.rz_neg - layer.rz_neg[j]).reshape((prev_layer.output_size, 1))
                # tmp = 1.0/np.sqrt(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2)
                # tmp3 = tmp*tmp*tmp
                # dx_neg_neg *= tmp3
                # dy_neg_neg *= tmp3
                # dz_neg_neg *= tmp3
                tmp = np.exp(-(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2))
                dx_neg_neg *= tmp
                dy_neg_neg *= tmp
                dz_neg_neg *= tmp
                potential += tmp

                # Next delta
                next_delta += (qj * this_delta_j) * potential * trans_sigma_Z_l

                # Charge gradient
                dq = potential * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                # tmp = qj * dq / potential
                tmp = 2 * qj * (Al_trans * this_delta_j)

                dc_drx_pos[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
                dc_dry_pos[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
                dc_drz_pos[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

                dc_drx_pos[l-1] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
                dc_dry_pos[l-1] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
                dc_drz_pos[l-1] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

                dc_drx_neg[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
                dc_dry_neg[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
                dc_drz_neg[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

                dc_drx_neg[l-1] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
                dc_dry_neg[l-1] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
                dc_drz_neg[l-1] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

                if self.regularizer is not None:
                    # ----- L2 regularized w_ij by position
                    coeff_lambda = self.regularizer.coeff_lambda
                    w_ij = qj * potential

                    # Charge gradient
                    dq = 2 * coeff_lambda * w_ij * potential
                    dc_dq[l][j] += np.sum(dq)

                    # Position gradient
                    tmp = 2 * qj * (2 * coeff_lambda * w_ij)

                    dc_drx_pos[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
                    dc_dry_pos[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
                    dc_drz_pos[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

                    dc_drx_pos[l-1] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
                    dc_dry_pos[l-1] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
                    dc_drz_pos[l-1] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

                    dc_drx_neg[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
                    dc_dry_neg[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
                    dc_drz_neg[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

                    dc_drx_neg[l-1] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
                    dc_dry_neg[l-1] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
                    dc_drz_neg[l-1] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

        # Compute gradient of bonds
        # tx, ty, tz = self.particle_input.compute_bond_cost_gradient()
        # dc_drx_pos[0] += tx
        # dc_dry_pos[0] += ty
        # dc_drz_pos[0] += tz
        # dc_drx_neg[0] -= tx
        # dc_dry_neg[0] -= ty
        # dc_drz_neg[0] -= tz
        # for l, layer in enumerate(self.layers):
        #     tx, ty, tz = layer.compute_bond_cost_gradient()
        #     dc_drx_pos[l+1] += tx
        #     dc_dry_pos[l+1] += ty
        #     dc_drz_pos[l+1] += tz
        #     dc_drx_neg[l+1] -= tx
        #     dc_dry_neg[l+1] -= ty
        #     dc_drz_neg[l+1] -= tz

        # Perform charge regularization if needed
        # if self.regularizer is not None:
        #     dc_dq = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dq)

        return dc_db, dc_dq, dc_drx_pos, dc_dry_pos, dc_drz_pos, dc_drx_neg, dc_dry_neg, dc_drz_neg

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    # def write_to_json(self, file):
    #     """
    #     Write network data to file in JSON format
    #     :param file: a file open for writing
    #     :return:
    #     """
    #
    #     network = {"particle_input": {}, "layers": [], "cost_name": self.cost_name}
    #
    #     p_inp = {"rx": [], "ry": [], "rz": []}
    #     for i in range(self.particle_input.output_size):
    #         p_inp["rx"].append(self.particle_input.rx[i])
    #         p_inp["ry"].append(self.particle_input.ry[i])
    #         p_inp["rz"].append(self.particle_input.rz[i])
    #     network["particle_input"] = p_inp
    #
    #     for layer in self.layers:
    #         l_data = {"rx": [], "ry": [], "rz": [], "q": [], "b": [], "activation_name": layer.activation_name}
    #         for i in range(layer.output_size):
    #             l_data["q"].append(layer.q[i])
    #             l_data["b"].append(layer.b[0][i])
    #             l_data["rx"].append(layer.rx[i])
    #             l_data["ry"].append(layer.ry[i])
    #             l_data["rz"].append(layer.rz[i])
    #         network["layers"].append(l_data)
    #
    #     json.dump(network, file)
    #
    # @staticmethod
    # def read_from_json(file):
    #     """
    #     Read network data from file in JSON format, return new ParticleNetwork
    #     :param file: a file open for reading
    #     :return:
    #     """
    #
    #     data = json.load(file)
    #
    #     network = ParticleNetwork(cost=data.get("cost_name"))
    #
    #     data_p_inp = data.get("particle_input")
    #     particle_input = ParticleInput(len(data_p_inp.get("rx")))
    #     for i, r in enumerate(data_p_inp.get("rx")):
    #         particle_input.rx[i] = r
    #     for i, r in enumerate(data_p_inp.get("ry")):
    #         particle_input.ry[i] = r
    #     for i, r in enumerate(data_p_inp.get("rz")):
    #         particle_input.rz[i] = r
    #     network.particle_input = particle_input
    #
    #     data_layers = data.get("layers")
    #     n_input = len(data_p_inp.get("rx"))
    #     for d_layer in data_layers:
    #         particle = Particle(input_size=n_input, output_size=len(d_layer.get("r")),
    #                             activation=d_layer.get("activation_name"))
    #         for i, r in enumerate(d_layer.get("rx")):
    #             particle.q[i] = d_layer.get("q")[i]
    #             particle.b[0][i] = d_layer.get("b")[i]
    #             for i, r in enumerate(data_p_inp.get("rx")):
    #                 particle.rx[i] = r
    #             for i, r in enumerate(data_p_inp.get("ry")):
    #                 particle.ry[i] = r
    #             for i, r in enumerate(data_p_inp.get("rz")):
    #                 particle.rz[i] = r
    #         network.layers.append(particle)
    #
    #     return network

