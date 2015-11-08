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
        dc_dq = [np.zeros(self.particle_input.q.shape)]  # charge gradient
        dc_dr = [np.zeros((len(self.particle_input.r), 3))]  # position gradient, a bit trickier

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr.append(np.zeros((len(layer.q), 3)))

        sigma_Z = []
        A_scaled, _ = self.particle_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        R = [self.particle_input.r]
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l], R[l])
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))
            R.append(layer.r)

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

        next_delta = np.zeros((len(prev_layer.r), len(data_X)))

        # Position gradient
        for j in range(len(layer.r)):
            rj_x = layer.r[j][0]
            rj_y = layer.r[j][1]
            rj_z = layer.r[j][2]
            qj = layer.q[j]
            qj2 = 2.0 * qj

            dc_dr_lj_x = 0.0
            dc_dr_lj_y = 0.0
            dc_dr_lj_z = 0.0

            trans_delta_L_j = trans_delta_L[j]

            for i in range(len(prev_layer.r)):
                dx = prev_layer.r[i][0] - rj_x
                dy = prev_layer.r[i][1] - rj_y
                dz = prev_layer.r[i][2] - rj_z
                d2 = dx*dx + dy*dy + dz*dz
                exp_dij = math.exp(-d2)

                # Next delta
                w_ij = qj * exp_dij
                trans_sigma_Z_i = trans_sigma_Z[l-1][i]
                next_delta[i] += w_ij * trans_delta_L_j * trans_sigma_Z_i

                # Charge gradient
                dq = exp_dij * Al_trans[i] * trans_delta_L_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                tmp = qj2 * dq
                tx = np.sum(tmp * dx)
                ty = np.sum(tmp * dy)
                tz = np.sum(tmp * dz)
                dc_dr_lj_x += tx
                dc_dr_lj_y += ty
                dc_dr_lj_z += tz

                dc_dr[l-1][i][0] -= tx
                dc_dr[l-1][i][1] -= ty
                dc_dr[l-1][i][2] -= tz
            dc_dr[l][j][0] += dc_dr_lj_x
            dc_dr[l][j][1] += dc_dr_lj_y
            dc_dr[l][j][2] += dc_dr_lj_z

        data_ones = np.ones(len(data_X))

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((len(prev_layer.r), len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else None

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Position gradient
            for j in range(len(layer.r)):
                rj_x = layer.r[j][0]
                rj_y = layer.r[j][1]
                rj_z = layer.r[j][2]
                qj = layer.q[j]
                qj2 = 2.0 * qj

                dc_dr_lj_x = 0.0
                dc_dr_lj_y = 0.0
                dc_dr_lj_z = 0.0

                this_delta_j = this_delta[j]

                for i in range(len(prev_layer.r)):
                    dx = prev_layer.r[i][0] - rj_x
                    dy = prev_layer.r[i][1] - rj_y
                    dz = prev_layer.r[i][2] - rj_z
                    d2 = dx*dx + dy*dy + dz*dz
                    exp_dij = math.exp(-d2)

                    # Next delta
                    w_ij = qj * exp_dij
                    trans_sigma_Z_i = trans_sigma_Z_l[i] if trans_sigma_Z_l is not None else data_ones
                    next_delta[i] += w_ij * this_delta_j * trans_sigma_Z_i

                    # Charge gradient
                    dq = exp_dij * Al_trans[i] * this_delta_j
                    dc_dq[l][j] += np.sum(dq)

                    # Position gradient
                    tmp = qj2 * dq
                    tx = np.sum(tmp * dx)
                    ty = np.sum(tmp * dy)
                    tz = np.sum(tmp * dz)
                    dc_dr_lj_x += tx
                    dc_dr_lj_y += ty
                    dc_dr_lj_z += tz

                    dc_dr[l-1][i][0] -= tx
                    dc_dr[l-1][i][1] -= ty
                    dc_dr[l-1][i][2] -= tz

                dc_dr[l][j][0] += dc_dr_lj_x
                dc_dr[l][j][1] += dc_dr_lj_y
                dc_dr[l][j][2] += dc_dr_lj_z

        # # Input layer charge gradient
        # Al_trans = data_X.transpose()
        # for j in range(len(self.particle_input.r)):
        #     Al_trans_i = Al_trans[j]
        #     for di in range(len(data_X)):
        #         dc_dq[0][j] += Al_trans_i[di] * deltas[0][di][j]

        # Perform charge regularization if needed
        if self.regularizer is not None:
            dc_dq, dc_db = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dq, dc_db)

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

