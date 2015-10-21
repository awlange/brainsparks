from .cost import Cost

import numpy as np


class Network(object):

    def __init__(self, cost="quadratic"):
        self.layers = []
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False

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

    def cost_gradient_hold(self, data_X, data_Y):
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

        final_l = -1
        while not self.layers[final_l].has_gradient:
            final_l -= 1

        delta_L = self.cost_d_function(data_Y, A[final_l], sigma_Z[final_l])

        # For each training case
        for i in range(len(data_X)):
            prev_delta = delta_L[i]
            dc_db_l = None
            dc_dw_l = None

            dc_db_l, dc_dw_l = self.layers[final_l].compute_gradient_final_layer(prev_delta, A[final_l-1][i])
            dc_db_l, dc_dw_l = self.layers[final_l].compute_gradient_update(dc_db_l, dc_dw_l)
            dc_db[final_l] += dc_db_l
            dc_dw[final_l] += dc_dw_l

            l = final_l - 1
            while -l <= len(self.layers):
                layer = self.layers[l+1]
                if layer.has_gradient:
                    dc_db_l, dc_dw_l = layer.compute_gradient(prev_delta, sigma_Z[l][i], A[l][i])

                if self.layers[l].has_gradient:
                    # Note: if false, these get sent back until needed

                    dc_db_l, dc_dw_l = self.layers[l].compute_gradient(dc_db_l, sigma_Z[l][i], A[l-1][i])

                    dc_db_l, dc_dw_l = self.layers[l].compute_gradient_update(dc_db_l, dc_dw_l)
                    dc_db[l] += dc_db_l
                    dc_dw[l] += dc_dw_l
                    prev_delta = dc_db_l

                l -= 1

        return dc_db, dc_dw

    def cost_gradient(self, data_X, data_Y):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        For now, raise an error if the last layer does not have a gradient

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dw = []

        sigma_Z = []
        A = [data_X]  # Note: A has one more element than sigma_Z
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
            l = -1
            dc_db_l, dc_dw_l = self.layers[l].compute_gradient(delta_L[i], A[l-1][i])
            dc_db_l, dc_dw_l = self.layers[l].compute_gradient_update(dc_db_l, dc_dw_l, convolve=False)
            dc_db[l] += dc_db_l
            dc_dw[l] += dc_dw_l

            while -l < len(self.layers):
                l -= 1

                # Convolve A ?
                Ai = A[l-1][i]
                # if self.layers[l].type == "Convolution1D":
                #     Ai = self.layers[l].convolve_input(Ai)

                dc_db_l, dc_dw_l = self.layers[l+1].compute_gradient(dc_db_l, Ai, sigma_Z[l][i], dc_dw_l)
                dc_db_l, dc_dw_l = self.layers[l].compute_gradient_update(dc_db_l, dc_dw_l)
                dc_db[l] += dc_db_l
                dc_dw[l] += dc_dw_l

        return dc_db, dc_dw

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)
