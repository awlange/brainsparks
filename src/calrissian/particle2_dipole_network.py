from .cost import Cost

from .layers.particle2_dipole import Particle2Dipole

import numpy as np
import json


class Particle2DipoleNetwork(object):

    def __init__(self, particle_input=None, cost="mse", regularizer=None):
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
        dc_drx_pos_inp = []
        dc_dry_pos_inp = []
        dc_drz_pos_inp = []
        dc_drx_neg_inp = []
        dc_dry_neg_inp = []
        dc_drz_neg_inp = []
        dc_drx_pos_out = []
        dc_dry_pos_out = []
        dc_drz_pos_out = []
        dc_drx_neg_out = []
        dc_dry_neg_out = []
        dc_drz_neg_out = []

        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dq.append(np.zeros(layer.q.shape))
            dc_drx_pos_inp.append(np.zeros(layer.input_size))
            dc_dry_pos_inp.append(np.zeros(layer.input_size))
            dc_drz_pos_inp.append(np.zeros(layer.input_size))
            dc_drx_neg_inp.append(np.zeros(layer.input_size))
            dc_dry_neg_inp.append(np.zeros(layer.input_size))
            dc_drz_neg_inp.append(np.zeros(layer.input_size))
            dc_drx_pos_out.append(np.zeros(layer.output_size))
            dc_dry_pos_out.append(np.zeros(layer.output_size))
            dc_drz_pos_out.append(np.zeros(layer.output_size))
            dc_drx_neg_out.append(np.zeros(layer.output_size))
            dc_dry_neg_out.append(np.zeros(layer.output_size))
            dc_drz_neg_out.append(np.zeros(layer.output_size))

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

        l = -1
        layer = self.layers[l]
        # layer = self.layers[l-1]

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
            trans_sigma_Z_l = trans_sigma_Z[l-1]

            dx_pos_pos = (layer.rx_pos_inp - layer.rx_pos_out[j]).reshape((layer.input_size, 1))
            dy_pos_pos = (layer.ry_pos_inp - layer.ry_pos_out[j]).reshape((layer.input_size, 1))
            dz_pos_pos = (layer.rz_pos_inp - layer.rz_pos_out[j]).reshape((layer.input_size, 1))
            tmp = np.exp(-(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2))
            dx_pos_pos *= tmp
            dy_pos_pos *= tmp
            dz_pos_pos *= tmp
            potential = tmp

            dx_pos_neg = (layer.rx_pos_inp - layer.rx_neg_out[j]).reshape((layer.input_size, 1))
            dy_pos_neg = (layer.ry_pos_inp - layer.ry_neg_out[j]).reshape((layer.input_size, 1))
            dz_pos_neg = (layer.rz_pos_inp - layer.rz_neg_out[j]).reshape((layer.input_size, 1))
            tmp = -np.exp(-(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2))
            dx_pos_neg *= tmp
            dy_pos_neg *= tmp
            dz_pos_neg *= tmp
            potential += tmp

            dx_neg_pos = (layer.rx_neg_inp - layer.rx_pos_out[j]).reshape((layer.input_size, 1))
            dy_neg_pos = (layer.ry_neg_inp - layer.ry_pos_out[j]).reshape((layer.input_size, 1))
            dz_neg_pos = (layer.rz_neg_inp - layer.rz_pos_out[j]).reshape((layer.input_size, 1))
            tmp = -np.exp(-(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2))
            dx_neg_pos *= tmp
            dy_neg_pos *= tmp
            dz_neg_pos *= tmp
            potential += tmp

            dx_neg_neg = (layer.rx_neg_inp - layer.rx_neg_out[j]).reshape((layer.input_size, 1))
            dy_neg_neg = (layer.ry_neg_inp - layer.ry_neg_out[j]).reshape((layer.input_size, 1))
            dz_neg_neg = (layer.rz_neg_inp - layer.rz_neg_out[j]).reshape((layer.input_size, 1))
            tmp = np.exp(-(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2))
            dx_neg_neg *= tmp
            dy_neg_neg *= tmp
            dz_neg_neg *= tmp
            potential += tmp

            # Next delta
            next_delta += (qj * trans_delta_L_j) * potential * trans_sigma_Z_l

            # Charge gradient
            dq = potential * Al_trans * trans_delta_L_j
            dc_dq[l][j] += np.sum(dq)

            # Position gradient
            # tmp = qj * dq / potential
            tmp = 2 * qj * (Al_trans * trans_delta_L_j)

            dc_drx_pos_out[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
            dc_dry_pos_out[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
            dc_drz_pos_out[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

            dc_drx_pos_inp[l] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
            dc_dry_pos_inp[l] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
            dc_drz_pos_inp[l] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

            dc_drx_neg_out[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
            dc_dry_neg_out[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
            dc_drz_neg_out[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

            dc_drx_neg_inp[l] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
            dc_dry_neg_inp[l] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
            dc_drz_neg_inp[l] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

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

                dx_pos_pos = (layer.rx_pos_inp - layer.rx_pos_out[j]).reshape((layer.input_size, 1))
                dy_pos_pos = (layer.ry_pos_inp - layer.ry_pos_out[j]).reshape((layer.input_size, 1))
                dz_pos_pos = (layer.rz_pos_inp - layer.rz_pos_out[j]).reshape((layer.input_size, 1))
                tmp = np.exp(-(dx_pos_pos**2 + dy_pos_pos**2 + dz_pos_pos**2))
                dx_pos_pos *= tmp
                dy_pos_pos *= tmp
                dz_pos_pos *= tmp
                potential = tmp

                dx_pos_neg = (layer.rx_pos_inp - layer.rx_neg_out[j]).reshape((layer.input_size, 1))
                dy_pos_neg = (layer.ry_pos_inp - layer.ry_neg_out[j]).reshape((layer.input_size, 1))
                dz_pos_neg = (layer.rz_pos_inp - layer.rz_neg_out[j]).reshape((layer.input_size, 1))
                tmp = -np.exp(-(dx_pos_neg**2 + dy_pos_neg**2 + dz_pos_neg**2))
                dx_pos_neg *= tmp
                dy_pos_neg *= tmp
                dz_pos_neg *= tmp
                potential += tmp

                dx_neg_pos = (layer.rx_neg_inp - layer.rx_pos_out[j]).reshape((layer.input_size, 1))
                dy_neg_pos = (layer.ry_neg_inp - layer.ry_pos_out[j]).reshape((layer.input_size, 1))
                dz_neg_pos = (layer.rz_neg_inp - layer.rz_pos_out[j]).reshape((layer.input_size, 1))
                tmp = -np.exp(-(dx_neg_pos**2 + dy_neg_pos**2 + dz_neg_pos**2))
                dx_neg_pos *= tmp
                dy_neg_pos *= tmp
                dz_neg_pos *= tmp
                potential += tmp

                dx_neg_neg = (layer.rx_neg_inp - layer.rx_neg_out[j]).reshape((layer.input_size, 1))
                dy_neg_neg = (layer.ry_neg_inp - layer.ry_neg_out[j]).reshape((layer.input_size, 1))
                dz_neg_neg = (layer.rz_neg_inp - layer.rz_neg_out[j]).reshape((layer.input_size, 1))
                tmp = np.exp(-(dx_neg_neg**2 + dy_neg_neg**2 + dz_neg_neg**2))
                dx_neg_neg *= tmp
                dy_neg_neg *= tmp
                dz_neg_neg *= tmp
                potential += tmp

                # Next delta
                next_delta += (qj * this_delta_j) * potential * trans_sigma_Z_l

                # Charge gradient
                dq = potential * Al_trans * this_delta_j
                dc_dq[l][j] += np.sum(dq)

                # Position gradient
                # tmp = qj * dq / potential
                tmp = 2 * qj * (Al_trans * this_delta_j)

                dc_drx_pos_out[l][j] += np.sum((dx_pos_pos + dx_neg_pos) * tmp)
                dc_dry_pos_out[l][j] += np.sum((dy_pos_pos + dy_neg_pos) * tmp)
                dc_drz_pos_out[l][j] += np.sum((dz_pos_pos + dz_neg_pos) * tmp)

                dc_drx_pos_inp[l] -= np.sum((dx_pos_pos + dx_pos_neg) * tmp, axis=1)
                dc_dry_pos_inp[l] -= np.sum((dy_pos_pos + dy_pos_neg) * tmp, axis=1)
                dc_drz_pos_inp[l] -= np.sum((dz_pos_pos + dz_pos_neg) * tmp, axis=1)

                dc_drx_neg_out[l][j] += np.sum((dx_pos_neg + dx_neg_neg) * tmp)
                dc_dry_neg_out[l][j] += np.sum((dy_pos_neg + dy_neg_neg) * tmp)
                dc_drz_neg_out[l][j] += np.sum((dz_pos_neg + dz_neg_neg) * tmp)

                dc_drx_neg_inp[l] -= np.sum((dx_neg_pos + dx_neg_neg) * tmp, axis=1)
                dc_dry_neg_inp[l] -= np.sum((dy_neg_pos + dy_neg_neg) * tmp, axis=1)
                dc_drz_neg_inp[l] -= np.sum((dz_neg_pos + dz_neg_neg) * tmp, axis=1)

        return dc_db, dc_dq, \
               dc_drx_pos_inp, dc_dry_pos_inp, dc_drz_pos_inp, dc_drx_neg_inp, dc_dry_neg_inp, dc_drz_neg_inp, \
               dc_drx_pos_out, dc_dry_pos_out, dc_drz_pos_out, dc_drx_neg_out, dc_dry_neg_out, dc_drz_neg_out

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

