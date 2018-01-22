from .cost import Cost

import numpy as np
import json


class ParticleVectorNLocalConvolution4Network(object):

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
        dc_dm = [[np.zeros(self.particle_input.output_size) for _ in range(self.particle_input.nw)]]
        # Initialize
        for l, layer in enumerate(self.layers):
            dc_db.append(np.zeros(layer.b.shape))
            dc_dr.append([np.zeros(layer.output_size) for _ in range(self.particle_input.nr)])
            dc_dn.append([np.zeros(layer.output_size) for _ in range(self.particle_input.nv)])
            dc_dm.append([np.zeros(layer.output_size) for _ in range(self.particle_input.nw)])

        sigma_Z = []
        A_scaled, _ = self.particle_input.feed_forward(data_X)
        A = [A_scaled]  # Note: A has one more element than sigma_Z
        prev_layer_rr = self.particle_input.get_rxyz()
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l], prev_layer_rr, apply_input_noise=(l == 0))
            a = layer.compute_a(z, apply_dropout=True)
            A.append(a)
            sigma_Z.append(layer.compute_da(z, apply_dropout=True))
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
        for w in range(self.particle_input.nw):
            self.particle_input.nwectors[w] = self.particle_input.nwectors[w].reshape((self.particle_input.output_size, 1))
        for layer in self.layers:
            for r in range(layer.nr):
                layer.positions[r] = layer.positions[r].reshape((layer.output_size, 1))
            for v in range(layer.nv):
                layer.nvectors[v] = layer.nvectors[v].reshape((layer.output_size, 1))
            for w in range(layer.nw):
                layer.nwectors[w] = layer.nwectors[w].reshape((layer.output_size, 1))

            if layer.apply_convolution:
                layer.positions_cache = layer.positions_cache.reshape((layer.nr, len(layer.positions_cache[0]), 1))
                layer.nvectors_cache = layer.nvectors_cache.reshape((layer.nv, len(layer.nvectors_cache[0]), 1))
                layer.nwectors_cache = layer.nwectors_cache.reshape((layer.nw, len(layer.nwectors_cache[0]), 1))

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

            d2 = None
            lpos = None
            if layer.apply_convolution:
                d2 = np.zeros((len(prev_layer.positions[0]), len(data_X)))  # this is amazing stuff! numpy is the best!
                lpos = layer.positions_cache

            else:
                d2 = np.zeros_like(prev_layer.positions[0])
                lpos = layer.positions

            dr = []
            for r in range(layer.nr):
                dtmp = prev_layer.positions[r] - lpos[r][j]
                d2 += dtmp ** 2
                dr.append(dtmp)

            d = np.sqrt(d2)
            dot = 0.0
            for v in range(layer.nv):
                dot += prev_layer.nvectors[v] * layer.nwectors[v][j]
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
                dc_dr[l - 1][r] -= np.sum(tr, axis=1)

            # Vector gradient
            tmp = dq / dot
            for v in range(layer.nv):
                tv = tmp * prev_layer.nvectors[v]
                dc_dm[l][v][j] += np.sum(tv)
                tv = tmp * layer.nwectors[v][j]
                dc_dn[l - 1][v] += np.sum(tv, axis=1)

        l = -1
        while -l < len(self.layers):
            l -= 1
            # Gradient computation
            layer = self.layers[l]
            prev_layer = self.particle_input if -(l-1) > len(self.layers) else self.layers[l-1]

            Al = A[l-1]
            Al_trans = Al.transpose()

            this_delta = next_delta
            if layer.apply_convolution:
                this_delta = this_delta.reshape((layer.output_size, layer.n_convolution, -1))
                # Bias gradient
                trans_delta = np.sum(this_delta, axis=1).transpose()
                for di, data in enumerate(data_X):
                    dc_db[l] += trans_delta[di]
            else:
                # Bias gradient
                trans_delta = this_delta.transpose()
                for di, data in enumerate(data_X):
                    dc_db[l] += trans_delta[di]

            if prev_layer.apply_convolution:
                next_delta = np.zeros((len(prev_layer.positions_cache[0]), len(data_X)))
                trans_sigma_Z_l = trans_sigma_Z[l - 1] if -(l - 1) <= len(self.layers) else np.ones((len(prev_layer.positions_cache[0]), len(data_X)))

            else:
                next_delta = np.zeros((prev_layer.output_size, len(data_X)))
                trans_sigma_Z_l = trans_sigma_Z[l-1] if -(l-1) <= len(self.layers) else np.ones((prev_layer.output_size, len(data_X)))

            # Position gradient
            for j in range(layer.output_size):
                this_delta_j = this_delta[j]

                d2 = None
                lpos = None
                lvec = None
                dot = 0.0
                dr = []

                if layer.apply_convolution:
                    # use the non-flattened caches
                    lpos = layer.positions_cache2.reshape((layer.nr, layer.output_size, layer.n_convolution, len(data_X), 1))  # use the max pool contributor position
                    lvec = layer.nwectors_cache.reshape((layer.nw, layer.output_size, layer.n_convolution, 1))

                    if prev_layer.apply_convolution:
                        # todo
                        pass

                    else:
                        d2 = np.zeros((len(data_X), layer.n_convolution, prev_layer.output_size, 1))
                        dr = np.zeros((len(data_X), layer.n_convolution, layer.nr, prev_layer.output_size, 1))
                        for a in range(len(data_X)):
                            for c in range(layer.n_convolution):
                                for r in range(layer.nr):
                                    dtmp = prev_layer.positions[r] - lpos[r][j][c][a]
                                    d2[a][c] += dtmp ** 2
                                    dr[a][c][r] += dtmp
                                for w in range(layer.nw):
                                    dot += prev_layer.nvectors[w] * lvec[w][j][c]
                        d = np.sqrt(d2)

                else:
                    lpos = layer.positions
                    lvec = layer.nwectors
                    if prev_layer.apply_convolution:
                        d2 = np.zeros_like(prev_layer.positions_cache[0])
                        for r in range(layer.nr):
                            dtmp = prev_layer.positions_cache[r] - lpos[r][j]
                            d2 += dtmp ** 2
                            dr.append(dtmp)
                        d = np.sqrt(d2)
                        for w in range(layer.nw):
                            dot += prev_layer.nvectors_cache[w] * lvec[w][j]
                    else:
                        d2 = np.zeros_like(prev_layer.positions[0])
                        for r in range(layer.nr):
                            dtmp = prev_layer.positions[r] - lpos[r][j]
                            d2 += dtmp ** 2
                            dr.append(dtmp)
                        d = np.sqrt(d2)
                        for w in range(layer.nw):
                            dot += prev_layer.nvectors[w] * lvec[w][j]

                exp_dij = layer.potential(d) * dot

                # Next delta
                if layer.apply_convolution:
                    # Loop through each j-th convolution
                    ld_pot = layer.d_potential(d) / d

                    next_delta = next_delta.transpose()
                    sigma_Z_l = trans_sigma_Z_l.transpose()

                    for a in range(len(data_X)):
                        for c in range(layer.n_convolution):
                            next_delta[a] += this_delta_j[c][a] * exp_dij[a][c].flatten() * sigma_Z_l[a]
                            atj = Al[a] * this_delta_j[c][a]
                            dq = exp_dij[a][c].flatten() * atj

                            jcdot = 0.0
                            for w in range(layer.nw):
                                jcdot += prev_layer.nvectors[w] * lvec[w][j][c]
                            p_tmp = -jcdot.flatten() * atj * ld_pot[a][c].flatten()  # only the dot for this j-th conv?
                            v_tmp = dq.flatten() / dot.flatten()

                            if prev_layer.apply_convolution:
                                pass
                            else:
                                for r in range(layer.nr):
                                    tr = dr[a][c][r].flatten() * p_tmp
                                    dc_dr[l][r][j] += np.sum(tr)
                                    dc_dr[l - 1][r] -= tr

                                for w in range(layer.nw):
                                    tv = v_tmp * prev_layer.nvectors[w].flatten()
                                    dc_dm[l][w][j] += np.sum(tv)
                                    tv = v_tmp * layer.nwectors[w][j]
                                    dc_dn[l - 1][w] += tv

                    next_delta = next_delta.transpose()

                else:
                    exp_dij = exp_dij.reshape((-1, 1))

                    next_delta += this_delta_j * exp_dij * trans_sigma_Z_l
                    atj = Al_trans * this_delta_j
                    dq = exp_dij * atj

                    p_tmp = -dot * atj * layer.d_potential(d) / d
                    v_tmp = dq / dot

                    if prev_layer.apply_convolution:
                        for r in range(layer.nr):
                            tr = dr[r] * p_tmp
                            dc_dr[l][r][j] += np.sum(tr)
                            dc_dr[l - 1][r] -= np.sum(np.sum(tr, axis=1).reshape((len(prev_layer.positions[0]), -1)), axis=1)

                        for w in range(layer.nw):
                            tv = v_tmp * prev_layer.nvectors_cache[w]
                            dc_dm[l][w][j] += np.sum(tv)
                            tv = v_tmp * layer.nwectors[w][j]
                            dc_dn[l - 1][w] += np.sum(np.sum(tv, axis=1).reshape((len(prev_layer.nvectors[0]), -1)), axis=1)
                    else:
                        for r in range(layer.nr):
                            tr = dr[r] * p_tmp
                            dc_dr[l][r][j] += np.sum(tr)
                            dc_dr[l - 1][r] -= np.sum(tr, axis=1)

                        for w in range(layer.nw):
                            tv = v_tmp * prev_layer.nvectors[w]
                            dc_dm[l][w][j] += np.sum(tv)
                            tv = v_tmp * layer.nwectors[w][j]
                            dc_dn[l - 1][w] += np.sum(tv, axis=1)

        # Restore shapes
        for r in range(self.particle_input.nr):
            self.particle_input.positions[r] = self.particle_input.positions[r].reshape((self.particle_input.output_size, ))
        for v in range(self.particle_input.nv):
            self.particle_input.nvectors[v] = self.particle_input.nvectors[v].reshape((self.particle_input.output_size, ))
        for w in range(self.particle_input.nw):
            self.particle_input.nwectors[w] = self.particle_input.nwectors[w].reshape((self.particle_input.output_size, ))
        for layer in self.layers:
            for r in range(layer.nr):
                layer.positions[r] = layer.positions[r].reshape((layer.output_size, ))
            for v in range(layer.nv):
                layer.nvectors[v] = layer.nvectors[v].reshape((layer.output_size, ))
            for w in range(layer.nw):
                layer.nwectors[w] = layer.nwectors[w].reshape((layer.output_size, ))

            if layer.apply_convolution:
                layer.positions_cache = layer.positions_cache.reshape((layer.nr, len(layer.positions_cache[0]), ))
                layer.nvectors_cache = layer.nvectors_cache.reshape((layer.nv, len(layer.nvectors_cache[0]), ))
                layer.nwectors_cache = layer.nwectors_cache.reshape((layer.nw, len(layer.nwectors_cache[0]), ))

        # Regularizer
        if self.regularizer is not None:
            # dc_dr = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dr)
            dc_dn, dc_db = self.regularizer.cost_gradient(self.particle_input, self.layers, dc_dn, dc_db)

        return dc_db, dc_dr, dc_dn, dc_dm

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
