from .cost import Cost

import numpy as np
import json
import os


class Particle333Network(object):

    def __init__(self, cost="mse", regularizer=None, lam=0.0):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.lam = lam

        # references to data
        self.ref_data_X = None
        self.ref_data_Y = None

        # thread partitioned gradients
        self.thread_dc_db = None
        self.thread_dc_dq = None
        self.thread_dc_dz = None
        self.thread_dc_dr_inp = None
        self.thread_dc_dr_out = None

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
        cost = self.cost_function(data_Y, self.predict(data_X))

        # Apply L2 regularization
        if self.regularizer == "l2":

            reg_total = 0.0
            for l, layer in enumerate(self.layers):
                for j in range(layer.output_size):
                    for c in range(layer.nc):
                        reg_total += layer.q[j][c] * layer.q[j][c]
                        reg_total += 1.0 / (layer.zeta[j][c] * layer.zeta[j][c])

                        for i in range(layer.input_size):
                            dx = layer.r_inp[i][0] - layer.r_out[j][c][0]
                            dy = layer.r_inp[i][1] - layer.r_out[j][c][1]
                            dz = layer.r_inp[i][2] - layer.r_out[j][c][2]
                            dd = dx*dx + dy*dy + dz*dz
                            reg_total += 1.0 / dd

            cost += reg_total * self.lam

        return cost

    def cost_gradient(self, data_X, data_Y, thread_scale=1):
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

                # IMPORTANT:
                # For threaded calls, we need to divide the cost gradient by the number threads to account for the mean being
                # taken in the cost function. When data is split, the mean denominator is off by a factor of the number of threads.
                if thread_scale > 1:
                    this_delta /= thread_scale

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

            elif False:
                """
                Original working version
                """

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

                            # TODO: can this loop be swapped into the innermost so that we don't recompute the potential?
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

            else:
                """
                Vectorized version
                """
                # Interaction gradient for convolution layer
                len_data = len(data_X)

                # recompute potentials like in forward propagation
                pout_size = layer.output_size * layer.nc * layer.output_shape[1] * layer.output_shape[0] * layer.output_pool_shape[1] * layer.output_pool_shape[0]
                pin_size = layer.input_size * layer.input_shape[1] * layer.input_shape[0]
                positions_output = np.zeros((layer.nr, 1, pout_size))
                positions_input = np.zeros((layer.nr, 1, pin_size))
                zeta_matrix = np.ones((pout_size, pin_size))

                joff = -1
                for j in range(layer.output_size):
                    for jy in range(layer.output_shape[1]):
                        for jx in range(layer.output_shape[0]):
                            for pool_jy in range(layer.output_pool_shape[1]):
                                for pool_jx in range(layer.output_pool_shape[0]):
                                    for c in range(layer.nc):
                                        rjc = layer.r_out[j][c]
                                        zjc = layer.zeta[j][c]

                                        joff += 1
                                        positions_output[0][0][joff] = rjc[0] + jx * layer.output_delta[0] + pool_jx * layer.output_pool_delta[0]
                                        positions_output[1][0][joff] = rjc[1] + jy * layer.output_delta[1] + pool_jy * layer.output_pool_delta[1]
                                        positions_output[2][0][joff] = rjc[2]
                                        zeta_matrix[joff] *= zjc

                ioff = -1
                for i in range(layer.input_size):
                    for iy in range(layer.input_shape[1]):
                        for ix in range(layer.input_shape[0]):
                            ioff += 1
                            positions_input[0][0][ioff] = layer.r_inp[i][0] + ix * layer.input_delta[0]
                            positions_input[1][0][ioff] = layer.r_inp[i][1] + iy * layer.input_delta[1]
                            positions_input[2][0][ioff] = layer.r_inp[i][2]

                matrix_dx = positions_output[0].transpose() - positions_input[0]
                matrix_dy = positions_output[1].transpose() - positions_input[1]
                matrix_dz = positions_output[2].transpose() - positions_input[2]
                r_matrix = np.sqrt(matrix_dx ** 2 + matrix_dy ** 2 + matrix_dz ** 2)
                potential_matrix = layer.potential(r_matrix, zeta=zeta_matrix)

                # # Use cached data
                # matrix_dx = layer.matrix_dx
                # matrix_dy = layer.matrix_dy
                # matrix_dz = layer.matrix_dz
                # r_matrix = layer.r_matrix
                # potential_matrix = layer.potential_matrix
                # zeta_matrix = layer.zeta_matrix

                dz_potential_matrix = layer.dz_potential(r_matrix, zeta=zeta_matrix)
                d_potential_matrix = layer.d_potential(r_matrix, zeta=zeta_matrix) / r_matrix  # divide d_potential_matrix by distances r_matrix for convenience and speed here

                # bias gradient
                chunk_size = layer.output_shape[1]*layer.output_shape[0]
                for j in range(layer.output_size):
                    dc_db[l][0][j] = this_delta[j*chunk_size:(j+1)*chunk_size].sum()

                # Helper arrays for this layer
                dc_dr_inp_dx = np.zeros(layer.input_size)
                dc_dr_inp_dy = np.zeros(layer.input_size)
                dc_dr_inp_dz = np.zeros(layer.input_size)

                trans_next_delta = next_delta.transpose()
                sigma_Z_l = trans_sigma_Z_l.transpose()

                joff = -1
                pool_size = layer.output_pool_shape[1] * layer.output_pool_shape[0]
                for j in range(layer.output_size):
                    qj = layer.q[j]
                    for jy in range(layer.output_shape[1]):
                        for jx in range(layer.output_shape[0]):

                            joff += 1
                            this_delta_j = this_delta[joff]
                            atj = Al_trans * this_delta_j
                            trans_atj = atj.transpose()
                            j_pool_offsets = layer.z_pool_max_cache[j][jy][jx]

                            for di in range(len_data):
                                offset = joff * pool_size * layer.nc + j_pool_offsets[di] * layer.nc
                                delta_sigma = this_delta_j[di] * sigma_Z_l[di]

                                for c in range(layer.nc):
                                    # Next layer delta - easier to do in transpose
                                    # trans_next_delta[di] += (qj[c] * this_delta_j[di]) * potential_matrix[offset + c] * sigma_Z_l[di]
                                    trans_next_delta[di] += (qj[c] * potential_matrix[offset + c]) * delta_sigma

                                    # Charge gradient
                                    dc_dq[l][j][c] += potential_matrix[offset + c].dot(trans_atj[di])

                                    # Width gradient
                                    dc_dz[l][j][c] += qj[c] * (dz_potential_matrix[offset + c].dot(trans_atj[di]))

                                    # Position gradient
                                    tmp = qj[c] * d_potential_matrix[offset + c] * trans_atj[di]
                                    tdx = tmp * matrix_dx[offset + c]
                                    tdy = tmp * matrix_dy[offset + c]
                                    tdz = tmp * matrix_dz[offset + c]

                                    # tdx = qj[c] * ((d_potential_matrix[offset + c] * matrix_dx[offset + c]) * (trans_atj[di]))
                                    # tdy = qj[c] * ((d_potential_matrix[offset + c] * matrix_dy[offset + c]) * (trans_atj[di]))
                                    # tdz = qj[c] * ((d_potential_matrix[offset + c] * matrix_dz[offset + c]) * (trans_atj[di]))

                                    dc_dr_out[l][j][c][0] += tdx.sum()
                                    dc_dr_out[l][j][c][1] += tdy.sum()
                                    dc_dr_out[l][j][c][2] += tdz.sum()
                                    dc_dr_inp_dx -= tdx.reshape((layer.input_size, -1)).sum(axis=1)
                                    dc_dr_inp_dy -= tdy.reshape((layer.input_size, -1)).sum(axis=1)
                                    dc_dr_inp_dz -= tdz.reshape((layer.input_size, -1)).sum(axis=1)

                for i in range(layer.input_size):
                    dc_dr_inp[l][i][0] += dc_dr_inp_dx[i]
                    dc_dr_inp[l][i][1] += dc_dr_inp_dy[i]
                    dc_dr_inp[l][i][2] += dc_dr_inp_dz[i]

                next_delta = trans_next_delta.transpose()

        # Apply L2 regularization
        if self.regularizer == "l2":
            two_lam = 2.0 * self.lam
            for l, layer in enumerate(self.layers):
                for j in range(layer.output_size):
                    for c in range(layer.nc):
                        dc_dq[l][j][c] += two_lam * layer.q[j][c]
                        dc_dz[l][j][c] += -two_lam / (layer.zeta[j][c] * layer.zeta[j][c] * layer.zeta[j][c])

                        for i in range(layer.input_size):
                            dx = layer.r_inp[i][0] - layer.r_out[j][c][0]
                            dy = layer.r_inp[i][1] - layer.r_out[j][c][1]
                            dz = layer.r_inp[i][2] - layer.r_out[j][c][2]
                            dd = dx*dx + dy*dy + dz*dz
                            tmp = two_lam / (dd * dd)

                            dc_dr_out[l][j][c][0] += dx * tmp
                            dc_dr_out[l][j][c][1] += dy * tmp
                            dc_dr_out[l][j][c][2] += dz * tmp
                            dc_dr_inp[l][i][0] -= dx * tmp
                            dc_dr_inp[l][i][1] -= dy * tmp
                            dc_dr_inp[l][i][2] -= dz * tmp

        return dc_db, dc_dq, dc_dz, dc_dr_inp, dc_dr_out

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)


