from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleCG(Optimizer):
    """
    Conjugate gradient for particle networks

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, cost_freq=2, alpha=0.1, n_threads=1, chunk_size=100):
        super().__init__()
        self.n_epochs = n_epochs
        self.verbosity = verbosity
        self.cost_freq = cost_freq

        # Constant learning rate (line search param)
        self.alpha = alpha

        # Weight gradients, to keep around for a step
        self.prev_dc_db = None
        self.prev_dc_dq = None
        self.prev_dc_dr = None
        self.prev_dc_dt = None
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_dt = None

        # Conjugate directions
        self.s_b = None
        self.s_q = None
        self.s_rx = None
        self.s_ry = None
        self.s_rz = None
        self.s_t = None

        # Run params
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.pool = None

    def get_pool(self):
        if self.pool is None:
            self.pool = Pool(self.n_threads)
        return self.pool

    def cost_gradient_parallel(self, network, data_X, data_Y):
        offset = 0
        chunks = len(data_X) / self.chunk_size
        while offset < len(data_X):
            data_X_sub = data_X[offset:(offset+self.chunk_size), :]
            data_Y_sub = data_Y[offset:(offset+self.chunk_size), :]
            data_X_split = np.array_split(data_X_sub, self.n_threads)
            data_Y_split = np.array_split(data_Y_sub, self.n_threads)
            data_XY_list = [(data_X_split[i], data_Y_split[i], self.n_threads * chunks) for i in range(self.n_threads)]

            result = self.get_pool().map(network.cost_gradient_thread, data_XY_list)

            for t, result_t in enumerate(result):
                tmp_dc_db = result_t[0]
                tmp_dc_dq = result_t[1]
                tmp_dc_dr = result_t[2]
                tmp_dc_dt = result_t[3]

                if t == 0 and offset == 0:
                    self.dc_db = tmp_dc_db
                    self.dc_dq = tmp_dc_dq
                    self.dc_dr = tmp_dc_dr
                    self.dc_dt = tmp_dc_dt
                else:
                    for l, tmp_b in enumerate(tmp_dc_db):
                        self.dc_db[l] += tmp_b
                    for l, tmp_q in enumerate(tmp_dc_dq):
                        self.dc_dq[l] += tmp_q
                    for l, tmp_t in enumerate(tmp_dc_dt):
                        self.dc_dt[l] += tmp_t
                    for i in range(3):
                        for l, tmp_r in enumerate(tmp_dc_dr[i]):
                            self.dc_dr[i][l] += tmp_r

            offset += self.chunk_size

    def optimize(self, network, data_X, data_Y):
        """
        :return: optimized network
        """
        optimize_start_time = time.time()

        if self.verbosity > 0:
            c = network.cost(data_X, data_Y)
            print("Cost before epochs: {}".format(c))

        # Epoch loop
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()

            if self.n_threads > 1:
                self.cost_gradient_parallel(network, data_X, data_Y)
            else:
                # Full batch gradient
                self.dc_db, self.dc_dq, self.dc_dr, self.dc_dt = network.cost_gradient(data_X, data_Y)

            # Update weights and biases
            self.weight_update(network)

            if self.verbosity > 0:
                c = network.cost(data_X, data_Y)
                print("Cost after epoch {}: {:g}".format(epoch, c))
                print("Epoch time: {:g} s".format(time.time() - epoch_start_time))

        if self.verbosity > 0:
            c = network.cost(data_X, data_Y)
            print("\n\nCost after optimize run: {:g}".format(c))
            print("Optimize run time: {:g} s".format(time.time() - optimize_start_time))

        return network

    def full_dot(self, network, dc_db, dc_dq, dc_dr, dc_dt):
        dot = 0.0
        for l, layer in enumerate(network.layers):
            dot += np.sum(dc_db[l] * dc_db[l])
            dot += np.dot(dc_dq[l].transpose(), dc_dq[l])
        for l in range(len(network.layers) + 1):
            dot += np.dot(dc_dt[l].transpose(), dc_dt[l])
            dot += np.dot(dc_dr[0][l].transpose(), dc_dr[0][l])
            dot += np.dot(dc_dr[1][l].transpose(), dc_dr[1][l])
            dot += np.dot(dc_dr[2][l].transpose(), dc_dr[2][l])
        return dot

    def weight_update(self, network):
        """
        Update weights and biases according to Conjugate Gradient
        """
        first_iter = False
        if self.s_b is None or self.s_q is None or self.s_rx is None or self.s_ry is None or self.s_rz is None:
            # Initial iteration
            first_iter = True

            self.s_b = []
            self.s_q = []
            self.s_rx = [np.zeros(network.particle_input.output_size)]
            self.s_ry = [np.zeros(network.particle_input.output_size)]
            self.s_rz = [np.zeros(network.particle_input.output_size)]
            self.s_t = [np.zeros(network.particle_input.output_size)]

            self.prev_dc_db = []
            self.prev_dc_dq = []
            self.prev_dc_dr = [[np.zeros(network.particle_input.output_size)] for _ in range(3)]
            self.prev_dc_dt = [np.zeros(network.particle_input.output_size)]

            for l, layer in enumerate(network.layers):
                self.s_b.append(np.zeros(layer.b.shape))
                self.s_q.append(np.zeros(layer.q.shape))
                self.s_rx.append(np.zeros(layer.output_size))
                self.s_ry.append(np.zeros(layer.output_size))
                self.s_rz.append(np.zeros(layer.output_size))
                self.s_t.append(np.zeros(layer.theta.shape))

                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dq.append(np.zeros_like(self.dc_dq[l]))
                self.prev_dc_dr[0].append(np.zeros_like(self.dc_dr[0][l+1]))
                self.prev_dc_dr[1].append(np.zeros_like(self.dc_dr[1][l+1]))
                self.prev_dc_dr[2].append(np.zeros_like(self.dc_dr[2][l+1]))
                self.prev_dc_dt.append(np.zeros_like(self.dc_dt[l+1]))

        # Compute beta according to Fletcher-Reeves
        # Need to do the dot products for all parameters
        beta = 0.0
        if not first_iter:
            numerator = self.full_dot(network, self.dc_db, self.dc_dq, self.dc_dr, self.dc_dt)
            denominator = self.full_dot(network, self.prev_dc_db, self.prev_dc_dq, self.prev_dc_dr, self.prev_dc_dt)
            beta = numerator / denominator
        if np.isnan(beta):
            beta = 0.0

        # Update the conjugate directions
        for l, layer in enumerate(network.layers):
            self.s_b[l] = self.dc_db[l] + beta * self.s_b[l]
            self.s_q[l] = self.dc_dq[l] + beta * self.s_q[l]
            self.s_rx[l+1] = self.dc_dr[0][l+1] + beta * self.s_rx[l+1]
            self.s_ry[l+1] = self.dc_dr[1][l+1] + beta * self.s_ry[l+1]
            self.s_rz[l+1] = self.dc_dr[2][l+1] + beta * self.s_rz[l+1]
            self.s_t[l+1] = self.dc_dt[l+1] + beta * self.s_t[l+1]
        self.s_rx[0] = self.dc_dr[0][0] + beta * self.s_rx[0]
        self.s_ry[0] = self.dc_dr[1][0] + beta * self.s_ry[0]
        self.s_rz[0] = self.dc_dr[2][0] + beta * self.s_rz[0]
        self.s_t[0] = self.dc_dt[0] + beta * self.s_t[0]

        # Take step
        for l, layer in enumerate(network.layers):
            layer.b -= self.alpha * self.s_b[l]
            layer.q -= self.alpha * self.s_q[l]
            layer.theta -= self.alpha * self.s_t[l+1]
            layer.rx -= self.alpha * self.s_rx[l+1]
            layer.ry -= self.alpha * self.s_ry[1][l+1]
            layer.rz -= self.alpha * self.s_rz[2][l+1]

        network.particle_input.theta -= self.alpha * self.s_t[0]
        network.particle_input.rx -= self.alpha * self.s_rx[0][0]
        network.particle_input.ry -= self.alpha * self.s_ry[1][0]
        network.particle_input.rz -= self.alpha * self.s_rz[2][0]

        # Copy params for next iter
        self.prev_dc_db = self.dc_db
        self.prev_dc_dq = self.dc_dq
        self.prev_dc_dr = self.dc_dr
        self.prev_dc_dt = self.dc_dt