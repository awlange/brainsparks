from .cost import Cost

import numpy as np


class AtomicNetwork(object):

    def __init__(self, atomic_input=None, cost="mse", regularizer=None):
        self.layers = []
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.atomic_input = atomic_input

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
        result = []
        for data in data_X:
            a = data
            r = self.atomic_input.r
            for layer in self.layers:
                a, r = layer.feed_forward(a, r)
            result.append(a)
        return np.asarray(result)

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

    def cost_gradient(self, data_X, data_Y):
        """
        Computes the gradient of the cost with respect to each weight and bias in the network

        :param data_X:
        :param data_Y:
        :return:
        """

        # Output gradients
        dc_db = []
        dc_dq = []  # charge gradient
        dc_dr = []  # position gradient, a bit trickier

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr.append(np.zeros((len(layer.q), 3)))

        # For each piece of data
        for i, data in enumerate(data_X):
            sigma_Z = []
            A = [data]  # Note: A has one more element than sigma_Z
            R = [self.atomic_input.r]
            for l, layer in enumerate(self.layers):
                z = layer.compute_z(A[l], R[l])
                a = layer.compute_a(z)
                A.append(a)
                R.append(layer.r)
                sigma_Z.append(layer.compute_da(z))

            l = -1
            layer = self.layers[l]

            # Delta and bias gradient
            delta = self.cost_d_function(data_Y, A[l], sigma_Z[l])

            # Charge gradient
            K = layer.build_kernel_matrix(R[l-1])
            KdotA = K.dot(A[l-1])
            dc_dq_l = KdotA * delta

            # Position gradient
            D = layer.build_distance_matrix(R[l-1])

            da_dx = 0.0
            rj_x = layer.r[0].x
            for i in range(len(D[0])):
                tmp = layer.q[0] * A[l-1][i] * np.exp(-D[0][i]) / D[0][i]
                diff_x = self.layers[l-1].r[i].x - rj_x
                da_dx += tmp * diff_x
            da_dx *= delta[0][0]
            

            dc_db[l] += delta[0]
            dc_dq[l] += dc_dq_l[0]

            while -l < len(self.layers):
                l -= 1

                # Gradient computation
                prev_delta = delta
                next_layer = self.layers[l+1]
                layer = self.layers[l]

                # Delta and bias gradient
                w = next_layer.build_weight_matrix(R[l])
                delta = prev_delta.dot(w) * sigma_Z[l]

                # Charge gradient
                K = layer.build_kernel_matrix(R[l-1])
                dc_dq_l = K.dot(A[l-1]) * delta

                # Update
                dc_db[l] += delta[0]
                dc_dq[l] += dc_dq_l[0]



        return dc_db, dc_dq

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)
