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
        dc_dp = []  # position gradient, a bit trickier
        da_dr = []  # position gradient, a bit trickier

        # Initialize
        dc_dr.append(np.zeros((len(self.atomic_input.r), 3)))
        dc_dp.append(np.zeros((len(self.atomic_input.r), 3)))
        da_dr.append(np.zeros((len(self.atomic_input.r), 3)))

        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr.append(np.zeros((len(layer.q), 3)))
            dc_dp.append(np.zeros((len(layer.q), 3)))
            da_dr.append(np.zeros((len(layer.q), 3)))

        # For each piece of data
        for di, data in enumerate(data_X):
            sigma_Z = []
            A = [data]  # Note: A has one more element than sigma_Z
            R = [self.atomic_input.r]
            for l, layer in enumerate(self.layers):
                z = layer.compute_z(A[l], R[l])
                a = layer.compute_a(z)
                A.append(a)
                R.append(layer.r)
                sigma_Z.append(layer.compute_da(z))

            # Keep deltas per layer
            deltas = [[] for _ in range(len(self.layers))]

            l = -1
            layer = self.layers[l]
            prev_layer = self.layers[l-1]

            # Delta and bias gradient
            delta = self.cost_d_function(data_Y, A[l], sigma_Z[l])
            deltas[l] = delta
            dc_db[l] += delta[0]

            # Charge gradient
            K = layer.build_kernel_matrix(R[l-1])
            dc_dq_l = K.dot(A[l-1]) * delta
            dc_dq[l] += dc_dq_l[0]

            # Position gradient
            for j in range(len(layer.r)):
                rj_x = layer.r[j].x
                rj_y = layer.r[j].y
                rj_z = layer.r[j].z
                qj = layer.q[j]
                for i in range(len(prev_layer.r)):
                    ri_x = prev_layer.r[i].x
                    ri_y = prev_layer.r[i].y
                    ri_z = prev_layer.r[i].z
                    dx = ri_x - rj_x
                    dy = ri_y - rj_y
                    dz = ri_z - rj_z
                    dij = np.sqrt(dx*dx + dy*dy + dz*dz)
                    tmp = (qj * A[l-1][i] * np.exp(-dij) / dij)

                    # For product rule correction
                    da_dr[l][j][0] += tmp * dx
                    da_dr[l][j][1] += tmp * dy
                    da_dr[l][j][2] += tmp * dz

                    tmp *= delta[0][j]
                    dc_dr[l-1][i][0] += -tmp * dx
                    dc_dr[l-1][i][1] += -tmp * dy
                    dc_dr[l-1][i][2] += -tmp * dz
                    dc_dr[l][j][0] += tmp * dx
                    dc_dr[l][j][1] += tmp * dy
                    dc_dr[l][j][2] += tmp * dz

            while -l < len(self.layers):
                l -= 1

                # Gradient computation
                prev_delta = delta
                next_layer = self.layers[l+1]
                layer = self.layers[l]
                prev_layer = self.atomic_input if l-1 < 0 else self.layers[l-1]

                # Delta and bias gradient
                w = next_layer.build_weight_matrix(R[l])
                delta = prev_delta.dot(w) * sigma_Z[l]
                deltas[l] = delta
                dc_db[l] += delta[0]

                # Charge gradient
                K = layer.build_kernel_matrix(R[l-1])
                dc_dq_l = K.dot(A[l-1]) * delta
                dc_dq[l] += dc_dq_l[0]

                # Position gradient
                for j in range(len(layer.r)):
                    rj_x = layer.r[j].x
                    rj_y = layer.r[j].y
                    rj_z = layer.r[j].z
                    qj = layer.q[j]
                    for i in range(len(prev_layer.r)):
                        ri_x = prev_layer.r[i].x
                        ri_y = prev_layer.r[i].y
                        ri_z = prev_layer.r[i].z
                        dx = ri_x - rj_x
                        dy = ri_y - rj_y
                        dz = ri_z - rj_z
                        dij = np.sqrt(dx*dx + dy*dy + dz*dz)
                        tmp = (qj * A[l-1][i] * np.exp(-dij) / dij)

                        # For product rule correction
                        da_dr[l][j][0] += tmp * dx
                        da_dr[l][j][1] += tmp * dy
                        da_dr[l][j][2] += tmp * dz

                        tmp *= delta[0][j]
                        dc_dr[l-1][i][0] += -tmp * dx
                        dc_dr[l-1][i][1] += -tmp * dy
                        dc_dr[l-1][i][2] += -tmp * dz
                        dc_dr[l][j][0] += tmp * dx
                        dc_dr[l][j][1] += tmp * dy
                        dc_dr[l][j][2] += tmp * dz

            # # Product rule corrections
            # l = 0
            # while l < len(self.layers):
            #     layer = self.layers[l]
            #     prev_layer = self.atomic_input if l == 0 else self.layers[l-1]
            #
            #     for i in range(len(prev_layer.r)):
            #         for j in range(len(layer.r)):
            #             dx = prev_layer.r[i].x - layer.r[j].x
            #             dy = prev_layer.r[i].y - layer.r[j].y
            #             dz = prev_layer.r[i].z - layer.r[j].z
            #             dij = np.sqrt(dx*dx + dy*dy + dz*dz)
            #
            #             sz = 1.0 if l == 0 else sigma_Z[l-1][i]
            #             # tmp = layer.q[j] * np.exp(-dij) * sz * deltas[l][0][j]
            #             tmp = layer.q[j] * np.exp(-dij) * sz
            #
            #             # dc_dr[l][i][0] += tmp * da_dr[l][i][0]
            #
            #             dc_dp[l+1][j][0] += tmp * da_dr[l][i][0]
            #
            #             # dc_dp[l+1][j][1] += -tmp * dc_dr[l][i][1]
            #             # dc_dp[l+1][j][2] += -tmp * dc_dr[l][i][2]
            #
            #     l += 1


        return dc_db, dc_dq, dc_dr, dc_dp

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)
