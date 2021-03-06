from .cost import Cost

from .layers.particle_sparse import ParticleSparse
from .layers.particle_sparse import ParticleSparseInput

import numpy as np
import json


class ParticleSparseNetwork(object):

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
        dc_dq = []
        dc_dr_x = [np.zeros(self.particle_input.output_size)]
        dc_dr_y = [np.zeros(self.particle_input.output_size)]
        dc_dr_z = [np.zeros(self.particle_input.output_size)]
        dc_dt = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr_x.append(np.zeros(len(layer.q)))
            dc_dr_y.append(np.zeros(len(layer.q)))
            dc_dr_z.append(np.zeros(len(layer.q)))
            dc_dt.append(np.zeros(layer.theta.shape))

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
        # if thread_scale > 1:
        #     delta_L /= thread_scale

        # For each piece of data
        for di, data in enumerate(data_X):
            for k, v in delta_L[di].items():
                dc_db[-1][0][k] += v * sigma_Z[-1][di].get(k, 0.0) / thread_scale

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        next_delta = [{} for _ in range(len(data_X))]

        for di, data in enumerate(data_X):
            j_set = set(delta_L[di].keys())
            for j in j_set:
                qj = layer.q[j]
                trans_delta_L_j = delta_L[di].get(j, 0.0) / thread_scale
                for i in Al[di].keys():
                    # for i in range(prev_layer.output_size):
                    #     if i not in Al[di].keys():
                    #         continue

                    dx = (prev_layer.rx[i] - layer.rx[j])
                    dy = (prev_layer.ry[i] - layer.ry[j])
                    dz = (prev_layer.rz[i] - layer.rz[j])
                    d2 = dx**2 + dy**2 + dz**2
                    exp_dij = np.exp(-d2)

                    dt = (prev_layer.theta[i] - layer.theta[j])
                    exp_dij *= np.cos(dt)

                    # Next delta
                    factor = sigma_Z[l][di].get(j, 0.0)
                    if -(l - 1) <= len(self.layers):
                        next_delta[di][i] = (qj * trans_delta_L_j) * exp_dij * sigma_Z[l][di].get(j, 0.0) + next_delta[di].get(i, 0.0)
                    else:
                        next_delta[di][i] = (qj * trans_delta_L_j) * exp_dij + next_delta[di].get(i, 0.0)

                    # Charge gradient
                    dq = exp_dij * Al[di].get(i, 0.0) * trans_delta_L_j * factor
                    dc_dq[l][j] += dq

                    # Position gradient
                    tmp = 2.0 * qj * dq
                    tx = dx * tmp
                    ty = dy * tmp
                    tz = dz * tmp

                    dc_dr_x[l][j] += tx
                    dc_dr_y[l][j] += ty
                    dc_dr_z[l][j] += tz

                    dc_dr_x[l-1][i] -= tx
                    dc_dr_y[l-1][i] -= ty
                    dc_dr_z[l-1][i] -= tz

                    # Phase gradient
                    dq *= -np.tan(dt)
                    tmp = qj * dq
                    dc_dt[l][j] -= tmp
                    dc_dt[l-1][i] += tmp

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            this_delta = next_delta
            next_delta = [{} for _ in range(len(data_X))]

            # Bias gradient
            for di, data in enumerate(data_X):
                for k, v in this_delta[di].items():
                    dc_db[l][0][k] += v * sigma_Z[l][di].get(k, 0.0)

            for di, data in enumerate(data_X):
                j_set = set(this_delta[di].keys())
                for j in j_set:
                    qj = layer.q[j]
                    trans_delta_j = this_delta[di].get(j)
                    for i in Al[di].keys():
                        # for i in range(prev_layer.output_size):
                        dx = (prev_layer.rx[i] - layer.rx[j])
                        dy = (prev_layer.ry[i] - layer.ry[j])
                        dz = (prev_layer.rz[i] - layer.rz[j])
                        d2 = dx ** 2 + dy ** 2 + dz ** 2
                        exp_dij = np.exp(-d2)

                        dt = (prev_layer.theta[i] - layer.theta[j])
                        exp_dij *= np.cos(dt)

                        # Next delta
                        factor = sigma_Z[l][di].get(j, 0.0)
                        if -(l - 1) <= len(self.layers):
                            next_delta[di][i] = (qj * trans_delta_j) * exp_dij * sigma_Z[l][di].get(j, 0.0) + next_delta[di].get(i, 0.0)
                        else:
                            next_delta[di][i] = (qj * trans_delta_j) * exp_dij + next_delta[di].get(i, 0.0)

                        # Charge gradient
                        dq = exp_dij * Al[di].get(i, 0.0) * trans_delta_j * factor
                        dc_dq[l][j] += dq

                        # Position gradient
                        tmp = 2.0 * qj * dq
                        tx = dx * tmp
                        ty = dy * tmp
                        tz = dz * tmp

                        dc_dr_x[l][j] += tx
                        dc_dr_y[l][j] += ty
                        dc_dr_z[l][j] += tz

                        dc_dr_x[l - 1][i] -= tx
                        dc_dr_y[l - 1][i] -= ty
                        dc_dr_z[l - 1][i] -= tz

                        # Phase gradient
                        dq *= -np.tan(dt)
                        tmp = qj * dq
                        dc_dt[l][j] -= tmp
                        dc_dt[l - 1][i] += tmp

        # Position gradient list
        dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z)
        return dc_db, dc_dq, dc_dr, dc_dt

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
    #     p_inp = {"rx": [], "ry": [], "rz": [], "theta": []}
    #     for i in range(self.particle_input.output_size):
    #         p_inp["rx"].append(self.particle_input.rx[i])
    #         p_inp["ry"].append(self.particle_input.ry[i])
    #         p_inp["rz"].append(self.particle_input.rz[i])
    #         p_inp["theta"].append(self.particle_input.theta[i])
    #     network["particle_input"] = p_inp
    #
    #     for layer in self.layers:
    #         l_data = {"rx": [], "ry": [], "rz": [], "q": [], "b": [], "theta": [],
    #                   "activation_name": layer.activation_name}
    #         for i in range(layer.output_size):
    #             l_data["q"].append(layer.q[i])
    #             l_data["b"].append(layer.b[0][i])
    #             l_data["rx"].append(layer.rx[i])
    #             l_data["ry"].append(layer.ry[i])
    #             l_data["rz"].append(layer.rz[i])
    #             l_data["theta"].append(layer.theta[i])
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
    #     for i, t in enumerate(data_p_inp.get("theta")):
    #         particle_input.theta[i] = t
    #     network.particle_input = particle_input
    #
    #     data_layers = data.get("layers")
    #     n_input = len(data_p_inp.get("rx"))
    #     for d_layer in data_layers:
    #         particle = Particle(input_size=n_input, output_size=len(d_layer.get("rx")),
    #                             activation=d_layer.get("activation_name"))
    #         for j, _ in enumerate(d_layer.get("rx")):
    #             particle.q[j] = d_layer.get("q")[j]
    #             particle.b[0][j] = d_layer.get("b")[j]
    #             for i, r in enumerate(d_layer.get("rx")):
    #                 particle.rx[i] = r
    #             for i, r in enumerate(d_layer.get("ry")):
    #                 particle.ry[i] = r
    #             for i, r in enumerate(d_layer.get("rz")):
    #                 particle.rz[i] = r
    #             for i, t in enumerate(d_layer.get("theta")):
    #                 particle.theta[i] = t
    #         network.layers.append(particle)
    #         n_input = len(d_layer.get("rx"))
    #
    #     return network

