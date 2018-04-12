from .cost import Cost

from .layers.particle3 import Particle3

import numpy as np
import json


class Particle333Network(object):

    def __init__(self, cost="mse", regularizer=None):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer

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

    def feed_to_layer(self, data_X, end_layer=0):
        """
        Feed data forward until given end layer. Return the resulting activation

        :param data_X: input data
        :param end_layer: the index of the ending layer
        :return: resulting activation at end layer
        """
        if len(self.layers) <= end_layer < 0:
            return None

        a = data_X
        for l, layer in enumerate(self.layers):
            a = layer.feed_forward(a)
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
            c += self.regularizer.cost(self.layers)

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
        dc_dr_inp = []
        dc_dr_out = []

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr_inp.append(np.zeros_like(layer.r_inp))
            dc_dr_out.append(np.zeros_like(layer.r_out))

        sigma_Z = []
        A = [data_X]  # Note: A has one more element than sigma_Z
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))

        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        # Gradient backpropagating through layers
        next_delta = None
        l = 0
        while -l < len(self.layers):
            l -= 1
            layer = self.layers[l]
            Al_trans = A[l-1].transpose()

            this_delta = next_delta
            if l == -1:
                this_delta = self.cost_d_function(data_Y, A[-1], sigma_Z[-1]).transpose()

            next_delta = np.zeros((layer.input_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((layer.input_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Interaction gradient
            for j in range(layer.output_size):
                qj = layer.q[j]
                this_delta_j = this_delta[j]

                sum_atj = np.sum(Al_trans * this_delta_j, axis=1).reshape((-1, 1))

                for c in range(layer.nc):
                    delta_r = layer.r_inp - layer.r_out[j][c]
                    r = np.sqrt(np.sum(delta_r * delta_r, axis=1)).reshape((-1, 1))
                    potential = layer.potential(r)

                    # Next delta
                    next_delta += (qj[c] * this_delta_j) * potential * trans_sigma_Z_l

                    # Charge gradient
                    dc_dq[l][j][c] += np.sum(potential * sum_atj)

                    # Position gradient
                    dx = delta_r * layer.d_potential(r) / r
                    tmp = -qj[c] * sum_atj * dx

                    dc_dr_out[l][j][c] += np.sum(tmp, axis=0)
                    dc_dr_inp[l] -= tmp

        return dc_db, dc_dq, dc_dr_inp, dc_dr_out

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)


