from .cost import Cost

from .layers.particle5 import Particle5
from .layers.particle5 import Particle5Input

import numpy as np
import json


class Particle5Network(object):

    def __init__(self, particle_input=None, cost="mse", regularizer=None):
        self.layers = []
        self.cost_name = cost
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)
        self.lock_built = False
        self.regularizer = regularizer
        self.particle_input = particle_input

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
        a, r = self.particle_input.feed_forward(data_X)
        for layer in self.layers:
            a, r = layer.feed_forward(a, r)
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

        a, r = self.particle_input.feed_forward(data_X)
        for l, layer in enumerate(self.layers):
            a, r = layer.feed_forward(a, r)
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
            c += self.regularizer.cost(self.particle_input, self.layers)

        return c

    def cost_gradient_thread(self, data_XYt):
        """
        Wrapper for multithreaded call
        :param data_XY:
        :return:
        """
        return self.cost_gradient(data_XYt[0], data_XYt[1], thread_scale=data_XYt[2])

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
        dc_dr_x = [np.zeros(self.particle_input.output_size)]
        dc_dr_y = [np.zeros(self.particle_input.output_size)]
        dc_dr_z = [np.zeros(self.particle_input.output_size)]
        dc_dq2 = []
        dc_dr_x2 = [np.zeros(self.particle_input.output_size)]
        dc_dr_y2 = [np.zeros(self.particle_input.output_size)]
        dc_dr_z2 = [np.zeros(self.particle_input.output_size)]
        dc_dt = [np.zeros(self.particle_input.output_size)]
        dc_dt2 = [np.zeros(self.particle_input.output_size)]

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_dr_x.append(np.zeros(len(layer.q)))
            dc_dr_y.append(np.zeros(len(layer.q)))
            dc_dr_z.append(np.zeros(len(layer.q)))
            dc_dq2.append(np.zeros(layer.q.shape))
            dc_dr_x2.append(np.zeros(len(layer.q)))
            dc_dr_y2.append(np.zeros(len(layer.q)))
            dc_dr_z2.append(np.zeros(len(layer.q)))
            dc_dt.append(np.zeros(layer.theta.shape))
            dc_dt2.append(np.zeros(layer.theta2.shape))

        sigma_Z = []
        A_scaled, _ = self.particle_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        prev_layer_rr = self.particle_input.get_rxyz()
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l], prev_layer_rr)
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))
            prev_layer_rr = layer.get_rxyz()

        delta_L = self.cost_d_function(data_Y, A[-1], sigma_Z[-1])

        # IMPORTANT:
        # For threaded calls, we need to divide the cost gradient by the number threads to account for the mean being
        # taken in the cost function. When data is split, the mean is off by a factor of the number of threads.
        if thread_scale > 1:
            delta_L /= thread_scale

        # For each piece of data
        for di, data in enumerate(data_X):
            dc_db[-1] += delta_L[di]

        # Reshape positions
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, 1))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, 1))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, 1))
        self.particle_input.rx2 = self.particle_input.rx2.reshape((self.particle_input.output_size, 1))
        self.particle_input.ry2 = self.particle_input.ry2.reshape((self.particle_input.output_size, 1))
        self.particle_input.rz2 = self.particle_input.rz2.reshape((self.particle_input.output_size, 1))
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, 1))
        self.particle_input.theta2 = self.particle_input.theta2.reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, 1))
            layer.ry = layer.ry.reshape((layer.output_size, 1))
            layer.rz = layer.rz.reshape((layer.output_size, 1))
            layer.rx2 = layer.rx2.reshape((layer.output_size, 1))
            layer.ry2 = layer.ry2.reshape((layer.output_size, 1))
            layer.rz2 = layer.rz2.reshape((layer.output_size, 1))
            layer.theta = layer.theta.reshape((layer.output_size, 1))
            layer.theta2 = layer.theta2.reshape((layer.output_size, 1))

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((len(prev_layer.rx), len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            qj = layer.q[j]
            qj2 = layer.q2[j]
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            dx = (prev_layer.rx - layer.rx[j])
            dy = (prev_layer.ry - layer.ry[j])
            dz = (prev_layer.rz - layer.rz[j])
            dt = (prev_layer.theta - layer.theta[j])
            exp_dij = np.exp(-(dx**2 + dy**2 + dz**2)) * np.cos(dt)

            dx2 = (prev_layer.rx2 - layer.rx2[j])
            dy2 = (prev_layer.ry2 - layer.ry2[j])
            dz2 = (prev_layer.rz2 - layer.rz2[j])
            dt2 = (prev_layer.theta2 - layer.theta2[j])
            exp_dij2 = np.exp(-(dx2**2 + dy2**2 + dz2**2)) * np.cos(dt2)

            # Next delta
            next_delta += trans_delta_L_j * (qj * exp_dij + qj2 * exp_dij2) * trans_sigma_Z_l

            # Charge gradient
            dq = exp_dij * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)
            dq2 = exp_dij2 * Al_trans * trans_delta_L_j
            dc_dq2[l][j] += np.sum(dq2)

            # Position gradient
            tmp = 2.0 * qj * dq
            tx = dx * tmp
            ty = dy * tmp
            tz = dz * tmp

            dc_dr_x[l][j] += np.sum(tx)
            dc_dr_y[l][j] += np.sum(ty)
            dc_dr_z[l][j] += np.sum(tz)

            dc_dr_x[l-1] -= np.sum(tx, axis=1)
            dc_dr_y[l-1] -= np.sum(ty, axis=1)
            dc_dr_z[l-1] -= np.sum(tz, axis=1)

            tmp = 2.0 * qj2 * dq2
            tx = dx2 * tmp
            ty = dy2 * tmp
            tz = dz2 * tmp

            dc_dr_x2[l][j] += np.sum(tx)
            dc_dr_y2[l][j] += np.sum(ty)
            dc_dr_z2[l][j] += np.sum(tz)

            dc_dr_x2[l-1] -= np.sum(tx, axis=1)
            dc_dr_y2[l-1] -= np.sum(ty, axis=1)
            dc_dr_z2[l-1] -= np.sum(tz, axis=1)

            # Phase gradient
            tmp = -qj * dq * np.tan(dt)
            dc_dt[l][j] -= np.sum(tmp)
            dc_dt[l-1] += np.sum(tmp, axis=1)

            tmp = -qj2 * dq2 * np.tan(dt2)
            dc_dt2[l][j] -= np.sum(tmp)
            dc_dt2[l-1] += np.sum(tmp, axis=1)

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            next_delta = np.zeros((prev_layer.output_size, len(data_X)))
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            # Bias gradient
            trans_delta = this_delta.transpose()
            for di, data in enumerate(data_X):
                dc_db[l] += trans_delta[di]

            # Position gradient
            for j in range(layer.output_size):
                qj = layer.q[j]
                qj2 = layer.q2[j]
                this_delta_j = this_delta[j]

                dx = (prev_layer.rx - layer.rx[j])
                dy = (prev_layer.ry - layer.ry[j])
                dz = (prev_layer.rz - layer.rz[j])
                dt = (prev_layer.theta - layer.theta[j])
                exp_dij = np.exp(-(dx ** 2 + dy ** 2 + dz ** 2)) * np.cos(dt)

                dx2 = (prev_layer.rx2 - layer.rx2[j])
                dy2 = (prev_layer.ry2 - layer.ry2[j])
                dz2 = (prev_layer.rz2 - layer.rz2[j])
                dt2 = (prev_layer.theta2 - layer.theta2[j])
                exp_dij2 = np.exp(-(dx2 ** 2 + dy2 ** 2 + dz2 ** 2)) * np.cos(dt2)

                # Next delta
                next_delta += this_delta_j * (qj * exp_dij + qj2 * exp_dij2) * trans_sigma_Z_l

                # Charge gradient
                dq = exp_dij * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)
                dq2 = exp_dij2 * Al_trans * this_delta_j
                dc_dq2[l][j] += np.sum(dq2)

                # Position gradient
                tmp = 2.0 * qj * dq
                tx = dx * tmp
                ty = dy * tmp
                tz = dz * tmp

                dc_dr_x[l][j] += np.sum(tx)
                dc_dr_y[l][j] += np.sum(ty)
                dc_dr_z[l][j] += np.sum(tz)

                dc_dr_x[l - 1] -= np.sum(tx, axis=1)
                dc_dr_y[l - 1] -= np.sum(ty, axis=1)
                dc_dr_z[l - 1] -= np.sum(tz, axis=1)

                tmp = 2.0 * qj2 * dq2
                tx = dx2 * tmp
                ty = dy2 * tmp
                tz = dz2 * tmp

                dc_dr_x2[l][j] += np.sum(tx)
                dc_dr_y2[l][j] += np.sum(ty)
                dc_dr_z2[l][j] += np.sum(tz)

                dc_dr_x2[l - 1] -= np.sum(tx, axis=1)
                dc_dr_y2[l - 1] -= np.sum(ty, axis=1)
                dc_dr_z2[l - 1] -= np.sum(tz, axis=1)

                # Phase gradient
                tmp = -qj * dq * np.tan(dt)
                dc_dt[l][j] -= np.sum(tmp)
                dc_dt[l - 1] += np.sum(tmp, axis=1)

                tmp = -qj2 * dq2 * np.tan(dt2)
                dc_dt2[l][j] -= np.sum(tmp)
                dc_dt2[l - 1] += np.sum(tmp, axis=1)

        # Position gradient list
        dc_dr = (dc_dr_x, dc_dr_y, dc_dr_z)
        dc_dr2 = (dc_dr_x2, dc_dr_y2, dc_dr_z2)

        # Restore shapes
        self.particle_input.rx = self.particle_input.rx.reshape((self.particle_input.output_size, ))
        self.particle_input.ry = self.particle_input.ry.reshape((self.particle_input.output_size, ))
        self.particle_input.rz = self.particle_input.rz.reshape((self.particle_input.output_size, ))
        self.particle_input.rx2 = self.particle_input.rx2.reshape((self.particle_input.output_size, ))
        self.particle_input.ry2 = self.particle_input.ry2.reshape((self.particle_input.output_size, ))
        self.particle_input.rz2 = self.particle_input.rz2.reshape((self.particle_input.output_size, ))
        self.particle_input.theta = self.particle_input.theta.reshape((self.particle_input.output_size, ))
        self.particle_input.theta2 = self.particle_input.theta2.reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            layer.rx = layer.rx.reshape((layer.output_size, ))
            layer.ry = layer.ry.reshape((layer.output_size, ))
            layer.rz = layer.rz.reshape((layer.output_size, ))
            layer.rx2 = layer.rx2.reshape((layer.output_size, ))
            layer.ry2 = layer.ry2.reshape((layer.output_size, ))
            layer.rz2 = layer.rz2.reshape((layer.output_size, ))
            layer.theta = layer.theta.reshape((layer.output_size, ))
            layer.theta2 = layer.theta2.reshape((layer.output_size, ))

        return dc_db, dc_dq, dc_dq2, dc_dr, dc_dr2, dc_dt, dc_dt2

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

