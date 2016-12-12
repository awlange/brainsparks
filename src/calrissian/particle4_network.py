from .cost import Cost

from .layers.particle import Particle
from .layers.particle import ParticleInput
from .regularization.particle_regularize_l2 import ParticleRegularizeL2
from .regularization.particle_regularize_l2plus import ParticleRegularizeL2Plus
from .regularization.particle_regularize_orthogonal import ParticleRegularizeOrthogonal

import numpy as np
import json


class Particle4Network(object):

    def __init__(self, particle_input=None, cost="mse", regularizer=None, r_dim=3):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.particle_input = particle_input
        self.r_dim = r_dim

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
        dc_dq = []
        dc_dr = [np.zeros((self.r_dim, self.particle_input.output_size))]
        dc_dt = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr.append(np.zeros((layer.r_dim, layer.output_size)))
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
        if thread_scale > 1:
            delta_L /= thread_scale

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        # Reshape positions
        # self.particle_input.r = self.particle_input.r.reshape((self.particle_input.output_size, self.r_dim, 1))
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            # layer.r = layer.r.reshape((layer.output_size, self.r_dim, 1))
            layer.theta = layer.theta.reshape((layer.output_size, 1))

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

            dr = prev_layer.r - layer.r[j]
            d2 = np.sum(dr * dr, axis=1).reshape((prev_layer.output_size, 1))
            dt = (prev_layer.theta - layer.theta[j])
            exp_dij = np.exp(-d2) * np.cos(dt)

            # Next delta
            next_delta += (qj * trans_delta_L_j) * exp_dij * trans_sigma_Z_l

            # Charge gradient
            dq = exp_dij * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            tmp = 2.0 * qj * dq
            for dim in range(layer.r_dim):
                tr = dr.transpose()[dim].reshape(len(tmp), 1) * tmp
                dc_dr[l][dim][j] += np.sum(tr)
                dc_dr[l-1][dim] -= np.sum(tr, axis=1)

            # Phase gradient
            dq *= -np.tan(dt)
            tmp = qj * dq
            dc_dt[l][j] -= np.sum(tmp)
            dc_dt[l-1] += np.sum(tmp, axis=1)

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

                dr = prev_layer.r - layer.r[j]
                d2 = np.sum(dr * dr, axis=1).reshape((prev_layer.output_size, 1))
                dt = (prev_layer.theta - layer.theta[j])
                exp_dij = np.exp(-d2) * np.cos(dt)

                # Next delta
                next_delta += (qj * this_delta_j) * exp_dij * trans_sigma_Z_l

                # Charge gradient
                dq = exp_dij * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                tmp = 2.0 * qj * dq
                for dim in range(layer.r_dim):
                    tr = dr.transpose()[dim].reshape(len(tmp), 1) * tmp
                    dc_dr[l][dim][j] += np.sum(tr)
                    dc_dr[l - 1][dim] -= np.sum(tr, axis=1)

                # Phase gradient
                dq *= -np.tan(dt)
                tmp = qj * dq
                dc_dt[l][j] -= np.sum(tmp)
                dc_dt[l - 1] += np.sum(tmp, axis=1)

        # Restore shapes
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            layer.theta = layer.theta.reshape((layer.output_size, ))
        for l in range(len(dc_dr)):
            dc_dr[l] = dc_dr[l].transpose()

        return dc_db, dc_dq, dc_dr, dc_dt

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    # def write_to_json(self, file=None):
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
    #     if file is not None:
    #         json.dump(network, file)
    #     else:
    #         return json.dumps(network)
    #
    # @staticmethod
    # def read_from_json(file, from_string=None):
    #     """
    #     Read network data from file in JSON format, return new ParticleNetwork
    #     :param file: a file open for reading
    #     :return:
    #     """
    #
    #     data = None
    #     if from_string is None:
    #         data = json.load(file)
    #     else:
    #         data = json.loads(from_string)
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
