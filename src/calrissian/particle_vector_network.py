from .cost import Cost

from .layers.particle_vector import ParticleVector
from .layers.particle_vector import ParticleVectorInput

import numpy as np
import json


class ParticleVectorNetwork(object):

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

    def predict_single(self, data_X):
        """
        Same as predict, but only one sample
        """
        return self.predict(data_X.reshape((1, len(data_X))))

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

    def feed_to_layer(self, data_X, end_layer=0):
        """
        Feed data forward until given end layer. Return the resulting activation

        :param data_X: input data
        :param end_layer: the index of the ending layer
        :return: resulting activation at end layer
        """
        if len(self.layers) <= end_layer < 0:
            return None

        a, r = self.particle_input.feed_forward(data_X)
        for l, layer in enumerate(self.layers):
            a, r = layer.feed_forward(a, r)
            if l == end_layer:
                return a

        return None

    def cost(self, data_X, data_Y):
        """
        Compute the cost for all input data corresponding to expected output

        :param data_X:
        :param data_Y:
        :return:
        """
        c = self.cost_function(data_Y, self.predict(data_X))

        if self.regularizer is not None:
            c += self.regularizer.cost(self.particle_input, self.layers)

        return c

    def cost_gradient_thread(self, data_XYt):
        """
        Wrapper for multithreaded call
        :param data_XY:
        :return:
        """
        return self.cost_gradient(data_XYt[0], data_XYt[1], thread_scale=data_XYt[2])

    def cost_gradient(self, data_X, data_Y, thread_scale=1):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dr_x = [np.zeros(self.particle_input.output_size)]
        dc_dr_y = [np.zeros(self.particle_input.output_size)]
        dc_dr_z = [np.zeros(self.particle_input.output_size)]
        dc_dr_w = [np.zeros(self.particle_input.output_size)]
        dc_dn_x = [np.zeros(self.particle_input.output_size)]
        dc_dn_y = [np.zeros(self.particle_input.output_size)]
        dc_dn_z = [np.zeros(self.particle_input.output_size)]
        dc_dn_w = [np.zeros(self.particle_input.output_size)]
        dc_dzeta = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dr_x.append(np.zeros(layer.output_size))
            dc_dr_y.append(np.zeros(layer.output_size))
            dc_dr_z.append(np.zeros(layer.output_size))
            dc_dr_w.append(np.zeros(layer.output_size))
            dc_dn_x.append(np.zeros(layer.output_size))
            dc_dn_y.append(np.zeros(layer.output_size))
            dc_dn_z.append(np.zeros(layer.output_size))
            dc_dn_w.append(np.zeros(layer.output_size))
            dc_dzeta.append(np.zeros(layer.output_size))

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

        # IMPORTANT:
        # For threaded calls, we need to divide the cost gradient by the number threads to account for the mean being
        # taken in the cost function. When data is split, the mean is off by a factor of the number of threads.
        if thread_scale > 1:
            delta_L /= thread_scale

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        # Reshape positions
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, 1))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, 1))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, 1))
        self.particle_input.rw = self.particle_input.rw.reshape((self.particle_input.output_size, 1))
        self.particle_input.nx = self.particle_input.nx.reshape((self.particle_input.output_size, 1))
        self.particle_input.ny = self.particle_input.ny.reshape((self.particle_input.output_size, 1))
        self.particle_input.nz = self.particle_input.nz.reshape((self.particle_input.output_size, 1))
        self.particle_input.nw = self.particle_input.nw.reshape((self.particle_input.output_size, 1))
        self.particle_input.zeta = self.particle_input.zeta.reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, 1))
            layer.ry = layer.ry.reshape((layer.output_size, 1))
            layer.rz = layer.rz.reshape((layer.output_size, 1))
            layer.rw = layer.rw.reshape((layer.output_size, 1))
            layer.nx = layer.nx.reshape((layer.output_size, 1))
            layer.ny = layer.ny.reshape((layer.output_size, 1))
            layer.nz = layer.nz.reshape((layer.output_size, 1))
            layer.nw = layer.nw.reshape((layer.output_size, 1))
            layer.zeta = layer.zeta.reshape((layer.output_size, 1))

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((len(prev_layer.rx), len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            dx = (prev_layer.rx - layer.rx[j])
            dy = (prev_layer.ry - layer.ry[j])
            dz = (prev_layer.rz - layer.rz[j])
            dw = (prev_layer.rw - layer.rw[j])
            d2 = dx**2 + dy**2 + dz**2 + dw**2
            r = np.sqrt(d2)
            zeta_ij = np.sqrt(np.abs(prev_layer.zeta * layer.zeta[j]))
            r *= zeta_ij
            exp_dij = layer.potential(r)
            dot = prev_layer.nx * layer.nx[j] + prev_layer.ny * layer.ny[j] + prev_layer.nz * layer.nz[j] + prev_layer.nw * layer.nw[j]
            exp_dij *= dot

            # Next delta
            next_delta += trans_delta_L_j * exp_dij * trans_sigma_Z_l
            atj = Al_trans * trans_delta_L_j
            dq = exp_dij * atj

            # Position gradient
            tmp = -dot * atj * layer.d_potential(r) / r
            tmp *= zeta_ij
            tx = dx * tmp
            ty = dy * tmp
            tz = dz * tmp
            tw = dw * tmp

            dc_dr_x[l][j] += np.sum(tx)
            dc_dr_y[l][j] += np.sum(ty)
            dc_dr_z[l][j] += np.sum(tz)
            dc_dr_w[l][j] += np.sum(tw)

            dc_dr_x[l-1] -= np.sum(tx, axis=1)
            dc_dr_y[l-1] -= np.sum(ty, axis=1)
            dc_dr_z[l-1] -= np.sum(tz, axis=1)
            dc_dr_w[l-1] -= np.sum(tw, axis=1)

            # Width gradient
            tmp = 0.5 * dot * atj * layer.d_potential(r) * np.sqrt(d2) / zeta_ij
            dc_dzeta[l][j] += np.sum(tmp * np.abs(prev_layer.zeta) * np.sign(layer.zeta[j]))
            dc_dzeta[l-1] += np.sum(tmp * np.abs(layer.zeta[j]) * np.sign(prev_layer.zeta), axis=1)

            # Vector gradient
            tmp = dq / dot
            tx = tmp * prev_layer.nx
            ty = tmp * prev_layer.ny
            tz = tmp * prev_layer.nz
            tw = tmp * prev_layer.nw

            dc_dn_x[l][j] += np.sum(tx)
            dc_dn_y[l][j] += np.sum(ty)
            dc_dn_z[l][j] += np.sum(tz)
            dc_dn_w[l][j] += np.sum(tw)

            tx = tmp * layer.nx[j]
            ty = tmp * layer.ny[j]
            tz = tmp * layer.nz[j]
            tw = tmp * layer.nw[j]

            dc_dn_x[l-1] += np.sum(tx, axis=1)
            dc_dn_y[l-1] += np.sum(ty, axis=1)
            dc_dn_z[l-1] += np.sum(tz, axis=1)
            dc_dn_w[l-1] += np.sum(tw, axis=1)

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
                this_delta_j = this_delta[j]

                dx = (prev_layer.rx - layer.rx[j])
                dy = (prev_layer.ry - layer.ry[j])
                dz = (prev_layer.rz - layer.rz[j])
                dw = (prev_layer.rw - layer.rw[j])
                d2 = dx**2 + dy**2 + dz**2 + dw**2
                r = np.sqrt(d2)
                zeta_ij = np.sqrt(np.abs(prev_layer.zeta * layer.zeta[j]))
                r *= zeta_ij
                exp_dij = layer.potential(r)
                dot = prev_layer.nx * layer.nx[j] + prev_layer.ny * layer.ny[j] + prev_layer.nz * layer.nz[j] + prev_layer.nw * layer.nw[j]
                exp_dij *= dot

                # Next delta
                next_delta += this_delta_j * exp_dij * trans_sigma_Z_l

                # Charge gradient
                atj = Al_trans * this_delta_j
                dq = exp_dij * atj

                # Position gradient
                tmp = -dot * atj * layer.d_potential(r) / r
                tmp *= zeta_ij
                tx = dx * tmp
                ty = dy * tmp
                tz = dz * tmp
                tw = dw * tmp

                dc_dr_x[l][j] += np.sum(tx)
                dc_dr_y[l][j] += np.sum(ty)
                dc_dr_z[l][j] += np.sum(tz)
                dc_dr_w[l][j] += np.sum(tw)

                dc_dr_x[l-1] -= np.sum(tx, axis=1)
                dc_dr_y[l-1] -= np.sum(ty, axis=1)
                dc_dr_z[l-1] -= np.sum(tz, axis=1)
                dc_dr_w[l-1] -= np.sum(tw, axis=1)

                # Width gradient
                tmp = 0.5 * dot * atj * layer.d_potential(r) * np.sqrt(d2) / zeta_ij
                dc_dzeta[l][j] += np.sum(tmp * np.abs(prev_layer.zeta) * np.sign(layer.zeta[j]))
                dc_dzeta[l - 1] += np.sum(tmp * np.abs(layer.zeta[j]) * np.sign(prev_layer.zeta), axis=1)

                # Vector gradient
                tmp = dq / dot
                tx = tmp * prev_layer.nx
                ty = tmp * prev_layer.ny
                tz = tmp * prev_layer.nz
                tw = tmp * prev_layer.nw

                dc_dn_x[l][j] += np.sum(tx)
                dc_dn_y[l][j] += np.sum(ty)
                dc_dn_z[l][j] += np.sum(tz)
                dc_dn_w[l][j] += np.sum(tw)

                tx = tmp * layer.nx[j]
                ty = tmp * layer.ny[j]
                tz = tmp * layer.nz[j]
                tw = tmp * layer.nw[j]

                dc_dn_x[l - 1] += np.sum(tx, axis=1)
                dc_dn_y[l - 1] += np.sum(ty, axis=1)
                dc_dn_z[l - 1] += np.sum(tz, axis=1)
                dc_dn_w[l - 1] += np.sum(tw, axis=1)

        # Position gradient list
        dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z, dc_dr_w)
        dc_dn = (dc_dn_x, dc_dn_y, dc_dn_z, dc_dn_w)

        # Restore shapes
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, ))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, ))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, ))
        self.particle_input.rw = self.particle_input.rw.reshape((self.particle_input.output_size, ))
        self.particle_input.nx = self.particle_input.nx.reshape((self.particle_input.output_size, ))
        self.particle_input.ny = self.particle_input.ny.reshape((self.particle_input.output_size, ))
        self.particle_input.nz = self.particle_input.nz.reshape((self.particle_input.output_size, ))
        self.particle_input.nw = self.particle_input.nw.reshape((self.particle_input.output_size, ))
        self.particle_input.zeta = self.particle_input.zeta.reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, ))
            layer.ry = layer.ry.reshape((layer.output_size, ))
            layer.rz = layer.rz.reshape((layer.output_size, ))
            layer.rw = layer.rw.reshape((layer.output_size, ))
            layer.nx = layer.nx.reshape((layer.output_size, ))
            layer.ny = layer.ny.reshape((layer.output_size, ))
            layer.nz = layer.nz.reshape((layer.output_size, ))
            layer.nw = layer.nw.reshape((layer.output_size, ))
            layer.zeta = layer.zeta.reshape((layer.output_size, ))

        return dc_db, dc_dr, dc_dn, dc_dzeta

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    def write_to_json(self, file=None):
        """
        Write network data to file in JSON format
        :param file: a file open for writing
        :return:
        """

        network = {"particle_input": {}, "layers": [], "cost_name": self.cost_name}

        p_inp = {"rx": [], "ry": [], "rz": [], "nx": [], "ny": [], "nz": []}
        for i in range(self.particle_input.output_size):
            p_inp["rx"].append(self.particle_input.rx[i])
            p_inp["ry"].append(self.particle_input.ry[i])
            p_inp["rz"].append(self.particle_input.rz[i])
            p_inp["nx"].append(self.particle_input.nx[i])
            p_inp["ny"].append(self.particle_input.ny[i])
            p_inp["nz"].append(self.particle_input.nz[i])
        network["particle_input"] = p_inp

        for layer in self.layers:
            l_data = {"rx": [], "ry": [], "rz": [], "nx": [], "ny": [], "nz": [],
                      "b": [], "activation_name": layer.activation_name}
            for i in range(layer.output_size):
                l_data["b"].append(layer.b[0][i])
                l_data["rx"].append(layer.rx[i])
                l_data["ry"].append(layer.ry[i])
                l_data["rz"].append(layer.rz[i])
                l_data["nx"].append(layer.nx[i])
                l_data["ny"].append(layer.ny[i])
                l_data["nz"].append(layer.nz[i])
            network["layers"].append(l_data)

        if file is not None:
            json.dump(network, file)
        else:
            return json.dumps(network)
