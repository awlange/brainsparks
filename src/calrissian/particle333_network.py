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
        dc_dz = []
        dc_dr_inp = []
        dc_dr_out = []

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dz.append(np.zeros(layer.zeta.shape))
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
            if layer.apply_convolution:
                # need to do it for each particle in the filter grid
                next_delta = np.zeros((layer.input_shape[2]*layer.input_shape[1]*layer.input_shape[0], len(data_X)))

            trans_sigma_Z_l = None
            if -(l - 1) <= len(self.layers):
                trans_sigma_Z_l = trans_sigma_Z[l - 1]
            else:
                if layer.apply_convolution:
                    trans_sigma_Z_l = np.ones((layer.input_shape[2]*layer.input_shape[1]*layer.input_shape[0], len(data_X)))
                else:
                    trans_sigma_Z_l = np.ones((layer.input_size, len(data_X)))

            if not layer.apply_convolution:
                # Bias gradient
                trans_delta = this_delta.transpose()
                for di in range(len(data_X)):
                    dc_db[l] += trans_delta[di]

                # Interaction gradient - no convolution
                for j in range(layer.output_size):
                    qj = layer.q[j]
                    zj = layer.zeta[j]
                    this_delta_j = this_delta[j]

                    sum_atj = np.sum(Al_trans * this_delta_j, axis=1).reshape((-1, 1))

                    for c in range(layer.nc):
                        delta_r = layer.r_inp - layer.r_out[j][c]
                        r = np.sqrt(np.sum(delta_r * delta_r, axis=1)).reshape((-1, 1))
                        potential = layer.potential(r, zeta=zj[c])

                        # Next delta
                        next_delta += (qj[c] * this_delta_j) * potential * trans_sigma_Z_l

                        # Charge gradient
                        dc_dq[l][j][c] += np.sum(potential * sum_atj)

                        # Width gradient
                        tmp = qj[c] * sum_atj * layer.dz_potential(r, zeta=zj[c])
                        dc_dz[l][j][c] += np.sum(tmp, axis=0)

                        # Position gradient
                        tmp = -qj[c] * sum_atj * delta_r * layer.d_potential(r, zeta=zj[c]) / r
                        dc_dr_out[l][j][c] += np.sum(tmp, axis=0)
                        dc_dr_inp[l] -= tmp

            else:
                # Interaction gradient for convolution layer
                trans_delta = this_delta.transpose()
                len_data = len(data_X)

                for j in range(layer.output_size):
                    qj = layer.q[j]
                    zj = layer.zeta[j]

                    # x fast index, y slow index
                    for jy in range(layer.output_shape[1]):
                        for jx in range(layer.output_shape[0]):

                            # off set for j-th filter
                            joff = j*layer.output_shape[1]*layer.output_shape[0] + jy*layer.output_shape[0] + jx

                            # pooling offsets
                            j_pool_offsets = layer.z_pool_max_cache[j][jy][jx]

                            this_delta_j = this_delta[joff]
                            atj = Al_trans * this_delta_j

                            # TODO: can this loop be swapped into th innermost so that we don't recompute the potential?
                            for di in range(len_data):
                                # Bias gradient
                                dc_db[l][0][j] += trans_delta[di][joff]

                                # yes, its reversed from what you might expect, but its in the ordered from left to
                                # right as slowest to fastest index
                                j_pool_offset_y, j_pool_offset_x = np.unravel_index(j_pool_offsets[di],
                                                                                    (layer.output_pool_shape[1],
                                                                                     layer.output_pool_shape[0]))

                                for c in range(layer.nc):
                                    rjc = layer.r_out[j][c]
                                    rjc_x = rjc[0] + jx * layer.output_delta[0] + j_pool_offset_x * layer.output_pool_delta[0]
                                    rjc_y = rjc[1] + jy * layer.output_delta[1] + j_pool_offset_y * layer.output_pool_delta[1]

                                    for i in range(layer.input_size):
                                        for iy in range(layer.input_shape[1]):
                                            ri_y = layer.r_inp[i][1] + iy * layer.input_delta[1]
                                            for ix in range(layer.input_shape[0]):
                                                ri_x = layer.r_inp[i][0] + ix * layer.input_delta[0]

                                                ioff = i*layer.input_shape[1]*layer.input_shape[0] + iy*layer.input_shape[0] + ix

                                                dx = ri_x - rjc_x
                                                dy = ri_y - rjc_y
                                                dz = layer.r_inp[i][2] - rjc[2]
                                                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                                                potential = layer.potential(r, zeta=zj[c])

                                                # Next delta
                                                next_delta[ioff][di] += (qj[c] * this_delta_j[di]) * potential * trans_sigma_Z_l[ioff][di]

                                                # Charge gradient
                                                dc_dq[l][j][c] += potential * atj[ioff][di]

                                                # Width gradient
                                                tmp = qj[c] * atj[ioff][di] * layer.dz_potential(r, zeta=zj[c])
                                                dc_dz[l][j][c] += np.sum(tmp, axis=0)

                                                # Position gradient
                                                tmp = -qj[c] * atj[ioff][di] * layer.d_potential(r, zeta=zj[c]) / r
                                                tdx = tmp * dx
                                                tdy = tmp * dy
                                                tdz = tmp * dz

                                                dc_dr_out[l][j][c][0] += tdx
                                                dc_dr_out[l][j][c][1] += tdy
                                                dc_dr_out[l][j][c][2] += tdz
                                                dc_dr_inp[l][i][0] -= tdx
                                                dc_dr_inp[l][i][1] -= tdy
                                                dc_dr_inp[l][i][2] -= tdz

        return dc_db, dc_dq, dc_dz, dc_dr_inp, dc_dr_out

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)


