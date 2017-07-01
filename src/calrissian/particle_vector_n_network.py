from .cost import Cost

from .layers.particle_vector_n import ParticleVectorN
from .layers.particle_vector_n import ParticleVectorNInput

import numpy as np
import json


class ParticleVectorNNetwork(object):

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

    def predict_single(self, data_X):
        """
        Same as predict, but only one sample
        """
        return self.predict(data_X.reshape((1, len(data_X))))

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
        dc_dr = [[np.zeros(self.particle_input.output_size) for _ in range(self.particle_input.nr)]]
        dc_dn = [[np.zeros(self.particle_input.output_size) for _ in range(self.particle_input.nv)]]
        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dr.append([np.zeros(layer.output_size) for _ in range(self.particle_input.nr)])
            dc_dn.append([np.zeros(layer.output_size) for _ in range(self.particle_input.nv)])

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

        # Reshape
        for r in range(self.particle_input.nr):
            self.particle_input.positions[r] = self.particle_input.positions[r].reshape((self.particle_input.output_size, 1))
        for v in range(self.particle_input.nv):
            self.particle_input.nvectors[v] = self.particle_input.nvectors[v].reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            for r in range(layer.nr):
                layer.positions[r] = layer.positions[r].reshape((layer.output_size, 1))
            for v in range(layer.nv):
                layer.nvectors[v] = layer.nvectors[v].reshape((layer.output_size, 1))

        l = -1
        layer = self.layers[l]
        prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

        Al = A[l-1]
        Al_trans = Al.transpose()
        trans_delta_L = delta_L.transpose()
        trans_sigma_Z = []
        for sz in sigma_Z:
            trans_sigma_Z.append(np.asarray(sz).transpose())

        next_delta = np.zeros((len(prev_layer.positions[0]), len(data_X)))

        # Position gradient
        for j in range(layer.output_size):
            trans_delta_L_j = trans_delta_L[j]
            trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            dr = []
            d2 = np.zeros_like(prev_layer.positions[r])
            for r in range(layer.nr):
                dtmp = prev_layer.positions[r] - layer.positions[r][j]
                d2 += dtmp**2
                dr.append(dtmp)
            d = np.sqrt(d2)
            dot = 0.0
            for v in range(layer.nv):
                dot += prev_layer.nvectors[v] * layer.nvectors[v][j]
            exp_dij = layer.potential(d) * dot

            # Next delta
            next_delta += trans_delta_L_j * exp_dij * trans_sigma_Z_l
            atj = Al_trans * trans_delta_L_j
            dq = exp_dij * atj

            # Position gradient
            tmp = -dot * atj * layer.d_potential(d) / d
            for r in range(layer.nr):
                tr = dr[r] * tmp
                dc_dr[l][r][j] += np.sum(tr)
                dc_dr[l-1][r] -= np.sum(tr, axis=1)

            # Vector gradient
            tmp = dq / dot
            for v in range(layer.nv):
                tv = tmp * prev_layer.nvectors[v]
                dc_dn[l][v][j] += np.sum(tv)
                tv = tmp * layer.nvectors[v][j]
                dc_dn[l-1][v] += np.sum(tv, axis=1)

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
                this_delta_j = this_delta[j]

                dr = []
                d2 = np.zeros_like(prev_layer.positions[r])
                for r in range(layer.nr):
                    dtmp = prev_layer.positions[r] - layer.positions[r][j]
                    d2 += dtmp ** 2
                    dr.append(dtmp)
                d = np.sqrt(d2)
                dot = 0.0
                for v in range(layer.nv):
                    dot += prev_layer.nvectors[v] * layer.nvectors[v][j]
                exp_dij = layer.potential(d) * dot

                # Next delta
                next_delta += this_delta_j * exp_dij * trans_sigma_Z_l

                # Charge gradient
                atj = Al_trans * this_delta_j
                dq = exp_dij * atj

                # Position gradient
                tmp = -dot * atj * layer.d_potential(d) / d
                for r in range(layer.nr):
                    tr = dr[r] * tmp
                    dc_dr[l][r][j] += np.sum(tr)
                    dc_dr[l - 1][r] -= np.sum(tr, axis=1)

                # Vector gradient
                tmp = dq / dot
                for v in range(layer.nv):
                    tv = tmp * prev_layer.nvectors[v]
                    dc_dn[l][v][j] += np.sum(tv)
                    tv = tmp * layer.nvectors[v][j]
                    dc_dn[l - 1][v] += np.sum(tv, axis=1)

        # Restore shapes
        for r in range(self.particle_input.nr):
            self.particle_input.positions[r] = self.particle_input.positions[r].reshape((self.particle_input.output_size, ))
        for v in range(self.particle_input.nv):
            self.particle_input.nvectors[v] = self.particle_input.nvectors[v].reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            for r in range(layer.nr):
                layer.positions[r] = layer.positions[r].reshape((layer.output_size, ))
            for v in range(layer.nv):
                layer.nvectors[v] = layer.nvectors[v].reshape((layer.output_size, ))

        return dc_db, dc_dr, dc_dn

    def fit(self, data_X, data_Y, optimizer):
        """
        Run the optimizer for specified number of epochs

        :param data_X:
        :param data_Y:
        :return:
        """

        return optimizer.optimize(self, data_X, data_Y)

    def write_to_json(self, file=None):
        """
        Write network data to file in JSON format
        :param file: a file open for writing
        :return:
        """
        pass
