from .cost import Cost

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

    def predict(self, data_X):
        """
        Pass given input through network to compute the output prediction

        :param data_X:
        :return:
        """
        a = data_X
        for layer in self.layers:
            a = layer.feed_forward(a)
        return a

    def cost(self, data_X, data_Y):
        """
        Compute the cost for all input data corresponding to expected output

        :param data_X:
        :param data_Y:
        :return:
        """
        return self.cost_function(data_Y, self.predict(data_X))

    def cost_gradient(self, data_X, data_Y):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dw = []

        sigma_Z = []
        A = [data_X]
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            A.append(layer.compute_a(z))
            sigma_Z.append(layer.compute_da(z))

            # Initialize
            dc_db.append(np.zeros(layer.b.shape))
            dc_dw.append(np.zeros(layer.w.shape))

        delta_L = self.cost_d_function(data_Y, A[-1], sigma_Z[-1])

        # For each training case
        for i in range(len(data_X)):
            prev_delta = delta_L[i]
            dc_db_l = None
            dc_dw_l = None

            if self.layers[-1].has_gradient:
                dc_db_l, dc_dw_l = self.layers[-1].compute_gradient_final_layer(prev_delta, A[-2][i])
                dc_db_l, dc_dw_l = self.layers[-1].compute_gradient_update(dc_db_l, dc_dw_l)
                dc_db[-1] += dc_db_l
                dc_dw[-1] += dc_dw_l

            for l in range(len(self.layers)-2, -1, -1):
                layer = self.layers[l+1]
                if layer.has_gradient:
                    dc_db_l, dc_dw_l = layer.compute_gradient(prev_delta, sigma_Z[l][i], A[l][i])

                if self.layers[l].has_gradient:
                    # Note: if false, these get sent back until needed
                    dc_db_l, dc_dw_l = self.layers[l].compute_gradient_update(dc_db_l, dc_dw_l)
                    dc_db[l] += dc_db_l
                    dc_dw[l] += dc_dw_l
                    prev_delta = dc_db_l

        return dc_db, dc_dw

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)
