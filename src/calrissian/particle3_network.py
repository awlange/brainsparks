from .cost import Cost

from .layers.particle3 import Particle3

import numpy as np
import json


class Particle3Network(object):

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
        dc_drx_inp = []
        dc_dry_inp = []
        dc_drx_pos_out = []
        dc_dry_pos_out = []
        dc_drx_neg_out = []
        dc_dry_neg_out = []

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_drx_inp.append(np.zeros(layer.input_size))
            dc_dry_inp.append(np.zeros(layer.input_size))
            dc_drx_pos_out.append(np.zeros(layer.output_size))
            dc_dry_pos_out.append(np.zeros(layer.output_size))
            dc_drx_neg_out.append(np.zeros(layer.output_size))
            dc_dry_neg_out.append(np.zeros(layer.output_size))

        sigma_Z = []
        A = [data_X]  # Note: A has one more element than sigma_Z
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))

        delta_L = self.cost_d_function(data_Y, A[-1], sigma_Z[-1])

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        # Reshape positions
        for layer in self.layers:
            layer.rx_inp = layer.rx_inp.reshape((layer.input_size, 1))
            layer.ry_inp = layer.ry_inp.reshape((layer.input_size, 1))
            layer.rx_pos_out = layer.rx_pos_out.reshape((layer.output_size, 1))
            layer.ry_pos_out = layer.ry_pos_out.reshape((layer.output_size, 1))
            layer.rx_neg_out = layer.rx_neg_out.reshape((layer.output_size, 1))
            layer.ry_neg_out = layer.ry_neg_out.reshape((layer.output_size, 1))

        l = -1
        layer = self.layers[l]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((layer.input_size, len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            qj = layer.q[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((layer.input_size, len(data_X)))

            dx_pos = (layer.rx_inp - layer.rx_pos_out[j])
            dy_pos = (layer.ry_inp - layer.ry_pos_out[j])
            # tmp = np.exp(-(dx_pos**2 + dy_pos**2))
            r = np.sqrt(dx_pos**2 + dy_pos**2)
            potential = layer.potential(r)
            tmp = layer.d_potential(r) / r
            dx_pos *= tmp
            dy_pos *= tmp

            dx_neg = (layer.rx_inp - layer.rx_neg_out[j])
            dy_neg = (layer.ry_inp - layer.ry_neg_out[j])
            # tmp = -np.exp(-(dx_neg**2 + dy_neg**2))
            # dx_neg *= tmp
            # dy_neg *= tmp
            # potential += tmp
            r = np.sqrt(dx_neg ** 2 + dy_neg ** 2)
            potential += -layer.potential(r)
            tmp = -layer.d_potential(r) / r
            dx_neg *= tmp
            dy_neg *= tmp

            # Next delta
            next_delta += (qj * trans_delta_L_j) * potential * trans_sigma_Z_l

            # Charge gradient
            atj = Al_trans * trans_delta_L_j
            dq = potential * atj
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            # tmp = 2.0 * qj * atj
            tmp = -qj * atj

            dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
            dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)

            dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
            dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)

            dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
            dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)

            if self.regularizer is not None:
                # # ----- L2 regularized w_ij
                # coeff_lambda = self.regularizer.coeff_lambda
                # w_ij = qj * potential
                #
                # # Charge gradient
                # dq = 2.0 * coeff_lambda * w_ij * potential
                # dc_dq[l][j] += np.sum(dq)
                #
                # # Position gradient
                # tmp = 2.0 * qj * (2.0 * coeff_lambda * w_ij)
                #
                # dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
                # dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)
                #
                # dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
                # dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)
                #
                # dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
                # dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)

                coeff_lambda = self.regularizer.coeff_lambda
                # Should be computed from before
                wt = layer.w.transpose()

                for kk in range(layer.output_size):
                    if j == kk:
                        continue

                    # s = np.sign(wt[j].dot(wt[kk]))
                    s = 2 * wt[j].dot(wt[kk])
                    dq = 2 * coeff_lambda * s * wt[kk].reshape((layer.input_size, 1)) * potential
                    dc_dq[l][j] += np.sum(dq)

                    # Position gradient
                    tmp = -qj * dq / potential

                    dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
                    dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)

                    dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
                    dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)

                    dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
                    dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)


        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((layer.input_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((layer.input_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Position gradient
            for j in range(layer.output_size):
                qj = layer.q[j]
                this_delta_j = this_delta[j]

                dx_pos = (layer.rx_inp - layer.rx_pos_out[j])
                dy_pos = (layer.ry_inp - layer.ry_pos_out[j])
                # tmp = np.exp(-(dx_pos**2 + dy_pos**2))
                r = np.sqrt(dx_pos ** 2 + dy_pos ** 2)
                potential = layer.potential(r)
                tmp = layer.d_potential(r) / r
                dx_pos *= tmp
                dy_pos *= tmp

                dx_neg = (layer.rx_inp - layer.rx_neg_out[j])
                dy_neg = (layer.ry_inp - layer.ry_neg_out[j])
                # tmp = -np.exp(-(dx_neg ** 2 + dy_neg ** 2))
                # dx_neg *= tmp
                # dy_neg *= tmp
                # potential += tmp
                r = np.sqrt(dx_neg ** 2 + dy_neg ** 2)
                potential += -layer.potential(r)
                tmp = -layer.d_potential(r) / r
                dx_neg *= tmp
                dy_neg *= tmp

                # Next delta
                next_delta += (qj * this_delta_j) * potential * trans_sigma_Z_l

                # Charge gradient
                atj = Al_trans * this_delta_j
                dq = potential * atj
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                # tmp = 2.0 * qj * atj
                tmp = -qj * atj

                dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
                dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)

                dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
                dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)

                dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
                dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)

                if self.regularizer is not None:
                    #     # ----- L2 regularized w_ij
                    #     coeff_lambda = self.regularizer.coeff_lambda
                    #     w_ij = qj * potential
                    #
                    #     # Charge gradient
                    #     dq = 2.0 * coeff_lambda * w_ij * potential
                    #     dc_dq[l][j] += np.sum(dq)
                    #
                    #     # Position gradient
                    #     tmp = 2.0 * qj * (2.0 * coeff_lambda * w_ij)
                    #
                    #     dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
                    #     dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)
                    #
                    #     dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
                    #     dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)
                    #
                    #     dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
                    #     dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)

                    coeff_lambda = self.regularizer.coeff_lambda
                    # Should be computed from before
                    wt = layer.w.transpose()

                    for kk in range(layer.output_size):
                        if j == kk:
                            continue

                        # s = np.sign(wt[j].dot(wt[kk]))
                        s = 2 * wt[j].dot(wt[kk])
                        dq = 2 * coeff_lambda * s * wt[kk].reshape((layer.input_size, 1)) * potential
                        dc_dq[l][j] += np.sum(dq)

                        # Position gradient
                        tmp = -qj * dq / potential

                        dc_drx_pos_out[l][j] += np.sum(dx_pos * tmp)
                        dc_dry_pos_out[l][j] += np.sum(dy_pos * tmp)

                        dc_drx_inp[l] -= np.sum((dx_pos + dx_neg) * tmp, axis=1)
                        dc_dry_inp[l] -= np.sum((dy_pos + dy_neg) * tmp, axis=1)

                        dc_drx_neg_out[l][j] += np.sum(dx_neg * tmp)
                        dc_dry_neg_out[l][j] += np.sum(dy_neg * tmp)

        # Reshape positions
        for layer in self.layers:
            layer.rx_inp = layer.rx_inp.reshape((layer.input_size, ))
            layer.ry_inp = layer.ry_inp.reshape((layer.input_size, ))
            layer.rx_pos_out = layer.rx_pos_out.reshape((layer.output_size, ))
            layer.ry_pos_out = layer.ry_pos_out.reshape((layer.output_size, ))
            layer.rx_neg_out = layer.rx_neg_out.reshape((layer.output_size, ))
            layer.ry_neg_out = layer.ry_neg_out.reshape((layer.output_size, ))

        if self.regularizer is not None:
            dc_dq = self.regularizer.cost_gradient(self.layers, dc_dq)

        return dc_db, dc_dq, \
               dc_drx_inp, dc_dry_inp, \
               dc_drx_pos_out, dc_dry_pos_out, \
               dc_drx_neg_out, dc_dry_neg_out

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)


