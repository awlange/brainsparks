from src.calrissian.cost import Cost

import numpy as np


class Network(object):

    def __init__(self, cost="quadratic"):
        self.layers = []
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)

    def append(self, layer):
        """
        Appends a layer to the network

        :param layer:
        :return:
        """
        self.layers.append(layer)

    def predict(self, input_data):
        """
        Pass given input through network to compute the output prediction

        :param input_data:
        :return:
        """
        a = input_data
        for layer in self.layers:
            a = layer.feed_forward(a)
        return a

    def cost(self, input_data, output_expected):
        """
        Compute the cost for all input data corresponding to expected output

        :param input_data:
        :param output_expected:
        :return:
        """
        return self.cost_function(output_expected, self.predict(input_data))

    def cost_gradient(self, input_data, output_expected):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param input_data:
        :param output_expected:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dw = []

        sigma_Z = []
        A = [input_data]
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            A.append(layer.compute_a(z))
            sigma_Z.append(layer.compute_da(z))

            # Initialize
            dc_db.append(np.zeros(layer.b.shape))
            dc_dw.append(np.zeros(layer.w.shape))

        delta_L = self.cost_d_function(output_expected, A[-1]) * sigma_Z[-1]

        # For each training case
        for i in range(len(input_data)):
            prev_delta = delta_L[i]
            dc_db[-1] += prev_delta
            dc_dw[-1] += np.outer(A[-2][i], prev_delta)

            for l in range(len(self.layers)-2, -1, -1):
                layer = self.layers[l+1]
                dc_db_l, dc_dw_l = layer.compute_gradient(prev_delta, sigma_Z[l][i], A[l][i])
                dc_db[l] += dc_db_l
                dc_dw[l] += dc_dw_l
                prev_delta = dc_db_l

        return dc_db, dc_dw

