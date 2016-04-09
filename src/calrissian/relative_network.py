from .cost import Cost

from .layers.relative import Relative
from .layers.relative import RelativeInput

import numpy as np
import json


class RelativeNetwork(object):

    def __init__(self, relative_input=None, cost="mse", regularizer=None):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.regularizer = regularizer
        self.relative_input = relative_input

    def append(self, layer):
        """
        Appends a layer to the network

        :param layer:
        :return:
        """
        self.layers.append(layer)

    def predict(self, data_X):
        """
        Pass given input through network to compute the output prediction

        :param data_X:
        :return:
        """
        a, v = self.relative_input.feed_forward(data_X)
        for layer in self.layers:
            a, v = layer.feed_forward(a, v)
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
            c += self.regularizer.cost(self.relative_input, self.layers)

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
        dc_dx = [np.zeros(self.relative_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dx.append(np.zeros(layer.x.shape))

        sigma_Z = []
        A_scaled, _ = self.relative_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        prev_layer_vars = self.relative_input.get_vars()
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l], prev_layer_vars)
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))
            prev_layer_vars = layer.get_vars()

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

        next_delta = np.zeros((len(prev_layer.x), len(data_X)))

        for j in range(layer.output_size):
            qj = layer.q[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1]

            dx = (prev_layer.x - layer.x[j]).reshape((prev_layer.output_size, 1))

            # Next delta
            next_delta += qj * trans_delta_L_j * dx * trans_sigma_Z_l

            dq = Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq * dx)

            dq *= qj
            dc_dx[l][j] -= np.sum(dq)
            dc_dx[l-1] += np.sum(dq, axis=1)

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.relative_input if -(l - 1) > len(self.layers) else self.layers[l - 1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((prev_layer.output_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            for j in range(layer.output_size):
                qj = layer.q[j]
                this_delta_j = this_delta[j]

                dx = (prev_layer.x - layer.x[j]).reshape((prev_layer.output_size, 1))

                # Next delta
                next_delta += qj * this_delta_j * dx * trans_sigma_Z_l

                dq = Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq * dx)

                dq *= qj
                dc_dx[l][j] -= np.sum(dq)
                dc_dx[l-1] += np.sum(dq, axis=1)

        return dc_db, dc_dq, dc_dx

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

        p_inp = {"x": []}
        for i in range(self.relative_input.output_size):
            p_inp["x"].append(self.relative_input.x[i])
        network["relative_input"] = p_inp

        for layer in self.layers:
            l_data = {"x": [], "b": [], "activation_name": layer.activation_name}
            for i in range(layer.output_size):
                l_data["b"].append(layer.b[0][i])
                l_data["x"].append(layer.x[i])
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

        network = RelativeNetwork(cost=data.get("cost_name"))

        data_p_inp = data.get("relative_input")
        relative_input = RelativeInput(len(data_p_inp.get("x")))
        for i, r in enumerate(data_p_inp.get("x")):
            relative_input.x[i] = r
        network.particle_input = relative_input

        data_layers = data.get("layers")
        n_input = len(data_p_inp.get("x"))
        for d_layer in data_layers:
            relative = Relative(input_size=n_input, output_size=len(d_layer.get("r")),
                                activation=d_layer.get("activation_name"))
            for i, r in enumerate(d_layer.get("rx")):
                relative.b[0][i] = d_layer.get("b")[i]
                for i, r in enumerate(data_p_inp.get("x")):
                    relative.x[i] = r
            network.layers.append(relative)

        return network

