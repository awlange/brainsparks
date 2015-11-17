from .cost import Cost

from .layers.particle import Particle
from .layers.particle import ParticleInput

import numpy as np
import math
import json


class ParticleNetwork(object):

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

        if self.regularizer is not None:
            c += self.regularizer.cost(self.particle_input, self.layers)

        return c

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
        dc_dr_x = [np.zeros(self.particle_input.output_size)]
        dc_dr_y = [np.zeros(self.particle_input.output_size)]
        dc_dr_z = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr_x.append(np.zeros(len(layer.q)))
            dc_dr_y.append(np.zeros(len(layer.q)))
            dc_dr_z.append(np.zeros(len(layer.q)))

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
        prev_layer = self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((len(prev_layer.rx), len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            qj = layer.q[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1]

            dx = (prev_layer.rx - layer.rx[j]).reshape((prev_layer.output_size, 1))
            dy = (prev_layer.ry - layer.ry[j]).reshape((prev_layer.output_size, 1))
            dz = (prev_layer.rz - layer.rz[j]).reshape((prev_layer.output_size, 1))
            d2 = dx**2 + dy**2 + dz**2
            exp_dij = np.exp(-d2)

            # Next delta
            next_delta += (qj * trans_delta_L_j) * exp_dij * trans_sigma_Z_l

            # Charge gradient
            dq = exp_dij * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            tmp = 2.0 * qj * dq
            tx = dx * tmp
            ty = dy * tmp
            tz = dz * tmp

            dc_dr_x[l][j] += np.sum(tx)
            dc_dr_y[l][j] += np.sum(ty)
            dc_dr_z[l][j] += np.sum(tz)

            dc_dr_x[l-1] -= np.sum(tx, axis=1)
            dc_dr_y[l-1] -= np.sum(ty, axis=1)
            dc_dr_z[l-1] -= np.sum(tz, axis=1)

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

                dx = (prev_layer.rx - layer.rx[j]).reshape((prev_layer.output_size, 1))
                dy = (prev_layer.ry - layer.ry[j]).reshape((prev_layer.output_size, 1))
                dz = (prev_layer.rz - layer.rz[j]).reshape((prev_layer.output_size, 1))
                d2 = dx**2 + dy**2 + dz**2
                exp_dij = np.exp(-d2)

                # Next delta
                next_delta += (qj * this_delta_j) * exp_dij * trans_sigma_Z_l

                # Charge gradient
                dq = exp_dij * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                tmp = 2.0 * qj * dq
                tx = dx * tmp
                ty = dy * tmp
                tz = dz * tmp

                dc_dr_x[l][j] += np.sum(tx)
                dc_dr_y[l][j] += np.sum(ty)
                dc_dr_z[l][j] += np.sum(tz)

                dc_dr_x[l-1] -= np.sum(tx, axis=1)
                dc_dr_y[l-1] -= np.sum(ty, axis=1)
                dc_dr_z[l-1] -= np.sum(tz, axis=1)

        # Position gradient list
        dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z)

        # Perform charge regularization if needed
        if self.regularizer is not None:
            dc_dq, dc_db, dc_dr = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dq, dc_db, dc_dr)

        return dc_db, dc_dq, dc_dr

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    def write_to_json(self, file):
        """
        Write network data to file in JSON format
        :param file: a file open for writing
        :return:
        """

        network = {"particle_input": {}, "layers": [], "cost_name": self.cost_name}

        p_inp = {"r": [], "q": []}
        for i, r in enumerate(self.particle_input.r):
            p_inp["q"].append(self.particle_input.q[i])
            p_inp["r"].append([r[0], r[1], r[2]])
        network["particle_input"] = p_inp

        for layer in self.layers:
            l_data = {"r": [], "q": [], "b": [], "activation_name": layer.activation_name}
            for i, r in enumerate(layer.r):
                l_data["q"].append(layer.q[i])
                l_data["b"].append(layer.b[0][i])
                l_data["r"].append([r[0], r[1], r[2]])
            network["layers"].append(l_data)

        json.dump(network, file)

    @staticmethod
    def read_from_json(file):
        """
        Read network data from file in JSON format, return new ParticleNetwork
        :param file: a file open for reading
        :return:
        """

        data = json.load(file)

        network = ParticleNetwork(cost=data.get("cost_name"))

        data_p_inp = data.get("particle_input")
        particle_input = ParticleInput(len(data_p_inp.get("r")))
        for i, r in enumerate(data_p_inp.get("r")):
            particle_input.q[i] = data_p_inp.get("q")[i]
            particle_input.r[i][0] = r[0]
            particle_input.r[i][1] = r[1]
            particle_input.r[i][2] = r[2]
        network.particle_input = particle_input

        data_layers = data.get("layers")
        n_input = len(data_p_inp.get("r"))
        for d_layer in data_layers:
            particle = Particle(input_size=n_input, output_size=len(d_layer.get("r")),
                                activation=d_layer.get("activation_name"))
            for i, r in enumerate(d_layer.get("r")):
                particle.q[i] = d_layer.get("q")[i]
                particle.b[0][i] = d_layer.get("b")[i]
                particle.r[i][0] = r[0]
                particle.r[i][1] = r[1]
                particle.r[i][2] = r[2]
            network.layers.append(particle)

        return network

