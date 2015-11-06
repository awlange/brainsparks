from .cost import Cost

import numpy as np
import math


class ParticleNetwork(object):

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
        a = data_X
        r = self.atomic_input.r
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
        dc_dr = [np.zeros((len(self.atomic_input.r), 3))]  # position gradient, a bit trickier

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr.append(np.zeros((len(layer.q), 3)))

        sigma_Z = []
        A = [data_X]  # Note: A has one more element than sigma_Z
        R = [self.atomic_input.r]
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

        # Position gradient
        for j in range(len(layer.r)):
            rj_x = layer.r[j][0]
            rj_y = layer.r[j][1]
            rj_z = layer.r[j][2]
            qj = layer.q[j]
            for i in range(len(prev_layer.r)):
                dx = prev_layer.r[i][0] - rj_x
                dy = prev_layer.r[i][1] - rj_y
                dz = prev_layer.r[i][2] - rj_z
                dij = math.sqrt(dx*dx + dy*dy + dz*dz)
                exp_dij = math.exp(-dij)
                tmp_qj_over_dij = qj / dij

                for di in range(len(data_X)):
                    delta = delta_L[di]
                    dq = A[l-1][di][i] * exp_dij * delta[j]

                    # Charge
                    dc_dq[l][j] += dq

                    # Position
                    tmp = tmp_qj_over_dij * dq
                    dc_dr[l-1][i][0] -= tmp * dx
                    dc_dr[l-1][i][1] -= tmp * dy
                    dc_dr[l-1][i][2] -= tmp * dz
                    dc_dr[l][j][0] += tmp * dx
                    dc_dr[l][j][1] += tmp * dy
                    dc_dr[l][j][2] += tmp * dz

        for di, data in enumerate(data_X):
            delta = delta_L[di]
            l = -1
            while -l < len(self.layers):
                l -= 1

                # Gradient computation
                prev_delta = delta
                next_layer = self.layers[l+1]
                layer = self.layers[l]
                prev_layer = self.atomic_input if -(l-1) > len(self.layers) else self.layers[l-1]

                # TODO: probably can combine this with below and push the data loop to innermost
                # Delta and bias gradient
                w = next_layer.build_weight_matrix(R[l])
                delta = prev_delta.dot(w) * sigma_Z[l][di]
                dc_db[l] += delta

                # Position gradient
                for j in range(len(layer.r)):
                    rj_x = layer.r[j][0]
                    rj_y = layer.r[j][1]
                    rj_z = layer.r[j][2]
                    qj = layer.q[j]
                    for i in range(len(prev_layer.r)):
                        dx = prev_layer.r[i][0] - rj_x
                        dy = prev_layer.r[i][1] - rj_y
                        dz = prev_layer.r[i][2] - rj_z
                        dij = np.sqrt(dx*dx + dy*dy + dz*dz)
                        exp_dij = math.exp(-dij)

                        # Charge
                        dc_dq[l][j] += A[l-1][di][i] * exp_dij * delta[j]

                        # Position
                        tmp = (qj * A[l-1][di][i] * exp_dij / dij) * delta[j]
                        dc_dr[l-1][i][0] += -tmp * dx
                        dc_dr[l-1][i][1] += -tmp * dy
                        dc_dr[l-1][i][2] += -tmp * dz
                        dc_dr[l][j][0] += tmp * dx
                        dc_dr[l][j][1] += tmp * dy
                        dc_dr[l][j][2] += tmp * dz

        return dc_db, dc_dq, dc_dr

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)
