from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleRPROP(Optimizer):
    """
    RPROP for particle networks

    Technically, iRPROP-

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, cost_freq=2, init_delta=0.1, eta_plus=1.2, eta_minus=0.5,
                 delta_min=1e-6, delta_max=50.0, manhattan=True, n_threads=1, chunk_size=100):
        """
        rprop
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbosity = verbosity
        self.cost_freq = cost_freq

        # RPROP params
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.init_delta = init_delta
        self.delta_max = delta_max
        self.delta_min = delta_min

        # Weight gradients, to keep around for a step
        self.prev_dc_db = None
        self.prev_dc_dq = None
        self.prev_dc_dr = None
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_dt = None

        # Deltas
        self.delta_b = None
        self.delta_q = None
        self.delta_rx = None
        self.delta_ry = None
        self.delta_rz = None
        self.delta_t = None

        # Run params
        self.manhattan = manhattan
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.pool = None

    def get_pool(self):
        if self.pool is None:
            self.pool = Pool(self.n_threads)
        return self.pool

    def cost_gradient_parallel(self, network, data_X, data_Y):
        offset = 0
        while offset < len(data_X):
            data_X_sub = data_X[offset:(offset+self.chunk_size), :]
            data_Y_sub = data_Y[offset:(offset+self.chunk_size), :]
            data_X_split = np.array_split(data_X_sub, self.n_threads)
            data_Y_split = np.array_split(data_Y_sub, self.n_threads)
            data_XY_list = [(data_X_split[i], data_Y_split[i]) for i in range(self.n_threads)]

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

    def get_delta(self, prod, delta, dc):
        if prod > 0:
            delta = min(delta * self.eta_plus, self.delta_max)
        elif prod < 0:
            delta = max(delta * self.eta_minus, self.delta_min)
            dc = 0.0
        return delta, dc

    def weight_update(self, network):
        """
        Update weights and biases according to RPROP
        """
        if self.delta_b is None or self.delta_q is None or self.delta_rx is None or self.delta_ry is None or self.delta_rz is None:
            # Initial iteration

            self.delta_b = []
            self.delta_q = []
            self.delta_rx = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_ry = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_rz = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_t = [np.ones(network.particle_input.output_size) * self.init_delta]

            self.prev_dc_db = []
            self.prev_dc_dq = []
            self.prev_dc_dr = [[np.zeros(network.particle_input.output_size)] for _ in range(3)]
            self.prev_dc_dt = [np.zeros(network.particle_input.output_size)]

            for l, layer in enumerate(network.layers):
                self.delta_b.append(np.ones(layer.b.shape) * self.init_delta)
                self.delta_q.append(np.ones(layer.q.shape) * self.init_delta)
                self.delta_rx.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_ry.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_rz.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_t.append(np.ones(layer.theta.shape) * self.init_delta)

                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dq.append(np.zeros_like(self.dc_dq[l]))
                self.prev_dc_dr[0].append(np.zeros_like(self.dc_dr[0][l+1]))
                self.prev_dc_dr[1].append(np.zeros_like(self.dc_dr[1][l+1]))
                self.prev_dc_dr[2].append(np.zeros_like(self.dc_dr[2][l+1]))
                self.prev_dc_dt.append(np.zeros_like(self.dc_dt[l+1]))

        for l, layer in enumerate(network.layers):

            # Biases
            prod = self.prev_dc_db[l] * self.dc_db[l]
            for i, b in enumerate(layer.b[0]):
                self.delta_b[l][0][i], self.dc_db[l][0][i] = self.get_delta(prod[0][i], self.delta_b[l][0][i], self.dc_db[l][0][i])
                layer.b[0][i] -= np.sign(self.dc_db[l][0][i]) * self.delta_b[l][0][i]
                self.prev_dc_db[l][0][i] = self.dc_db[l][0][i]

            # Charges
            prod = self.prev_dc_dq[l] * self.dc_dq[l]
            for i, q in enumerate(layer.q):
                self.delta_q[l][i], self.dc_dq[l][i] = self.get_delta(prod[i], self.delta_q[l][i], self.dc_dq[l][i])
                layer.q[i] -= np.sign(self.dc_dq[l][i]) * self.delta_q[l][i]
                self.prev_dc_dq[l][i] = self.dc_dq[l][i]

            # Phases
            prod = self.prev_dc_dt[l+1] * self.dc_dt[l+1]
            for i, t in enumerate(layer.theta):
                self.delta_t[l+1][i], self.dc_dt[l+1][i] = self.get_delta(prod[i], self.delta_t[l+1][i], self.dc_dt[l+1][i])
                layer.theta[i] -= np.sign(self.dc_dt[l+1][i]) * self.delta_t[l+1][i]
                self.prev_dc_dt[l+1][i] = self.dc_dt[l+1][i]

            # Positions

            if self.manhattan:
                # X
                prod = self.prev_dc_dr[0][l+1] * self.dc_dr[0][l+1]
                for i, rx in enumerate(layer.rx):
                    self.delta_rx[l+1][i], self.dc_dr[0][l+1][i] = self.get_delta(prod[i], self.delta_rx[l+1][i], self.dc_dr[0][l+1][i])
                    layer.rx[i] -= np.sign(self.dc_dr[0][l+1][i]) * self.delta_rx[l+1][i]
                    self.prev_dc_dr[0][l+1][i] = self.dc_dr[0][l+1][i]

                # Y
                prod = self.prev_dc_dr[1][l+1] * self.dc_dr[1][l+1]
                for i, ry in enumerate(layer.ry):
                    self.delta_ry[l+1][i], self.dc_dr[1][l+1][i] = self.get_delta(prod[i], self.delta_ry[l+1][i], self.dc_dr[1][l+1][i])
                    layer.ry[i] -= np.sign(self.dc_dr[1][l+1][i]) * self.delta_ry[l+1][i]
                    self.prev_dc_dr[1][l+1][i] = self.dc_dr[1][l+1][i]

                # Z
                prod = self.prev_dc_dr[2][l+1] * self.dc_dr[2][l+1]
                for i, rz in enumerate(layer.rz):
                    self.delta_rz[l+1][i], self.dc_dr[2][l+1][i] = self.get_delta(prod[i], self.delta_rz[l+1][i], self.dc_dr[2][l+1][i])
                    layer.rz[i] -= np.sign(self.dc_dr[2][l+1][i]) * self.delta_rz[l+1][i]
                    self.prev_dc_dr[2][l+1][i] = self.dc_dr[2][l+1][i]
            else:
                # R, dot product
                prod = self.prev_dc_dr[0][l+1] * self.dc_dr[0][l+1] \
                     + self.prev_dc_dr[1][l+1] * self.dc_dr[1][l+1] \
                     + self.prev_dc_dr[2][l+1] * self.dc_dr[2][l+1]
                for i, rx in enumerate(layer.rx):
                    delta, dc = self.get_delta(prod[i], self.delta_rx[l+1][i], 1.0)
                    self.dc_dr[0][l+1][i] *= dc
                    self.dc_dr[1][l+1][i] *= dc
                    self.dc_dr[2][l+1][i] *= dc
                    self.delta_rx[l+1][i] = delta
                    self.delta_ry[l+1][i] = delta
                    self.delta_rz[l+1][i] = delta
                    layer.rx[i] -= np.sign(self.dc_dr[0][l+1][i]) * self.delta_rx[l+1][i]
                    layer.ry[i] -= np.sign(self.dc_dr[1][l+1][i]) * self.delta_ry[l+1][i]
                    layer.rz[i] -= np.sign(self.dc_dr[2][l+1][i]) * self.delta_rz[l+1][i]
                    self.prev_dc_dr[0][l+1][i] = self.dc_dr[0][l+1][i]
                    self.prev_dc_dr[1][l+1][i] = self.dc_dr[1][l+1][i]
                    self.prev_dc_dr[2][l+1][i] = self.dc_dr[2][l+1][i]

        # Input layer phase
        prod = self.prev_dc_dt[0] * self.dc_dt[0]
        for i, t in enumerate(layer.theta):
            self.delta_t[0][i], self.dc_dt[0][i] = self.get_delta(prod[i], self.delta_t[0][i], self.dc_dt[0][i])
            layer.theta[i] -= np.sign(self.dc_dt[0][i]) * self.delta_t[0][i]
            self.prev_dc_dt[0][i] = self.dc_dt[0][i]

        # Input layer position

        if self.manhattan:
            # X
            prod = self.prev_dc_dr[0][0] * self.dc_dr[0][0]
            for i, rx in enumerate(network.particle_input.rx):
                self.delta_rx[0][i], self.dc_dr[0][0][i] = self.get_delta(prod[i], self.delta_rx[0][i], self.dc_dr[0][0][i])
                network.particle_input.rx[i] -= np.sign(self.dc_dr[0][0][i]) * self.delta_rx[0][i]
                self.prev_dc_dr[0][0][i] = self.dc_dr[0][0][i]

            # Y
            prod = self.prev_dc_dr[1][0] * self.dc_dr[1][0]
            for i, ry in enumerate(network.particle_input.ry):
                self.delta_ry[0][i], self.dc_dr[1][0][i] = self.get_delta(prod[i], self.delta_ry[0][i], self.dc_dr[1][0][i])
                network.particle_input.ry[i] -= np.sign(self.dc_dr[1][0][i]) * self.delta_ry[0][i]
                self.prev_dc_dr[1][0][i] = self.dc_dr[1][0][i]

            # Z
            prod = self.prev_dc_dr[2][0] * self.dc_dr[2][0]
            for i, rz in enumerate(network.particle_input.rz):
                self.delta_rz[0][i], self.dc_dr[2][0][i] = self.get_delta(prod[i], self.delta_rz[0][i], self.dc_dr[2][0][i])
                network.particle_input.rz[i] -= np.sign(self.dc_dr[2][0][i]) * self.delta_rz[0][i]
                self.prev_dc_dr[2][0][i] = self.dc_dr[2][0][i]
        else:
            # R, dot product
            prod = self.prev_dc_dr[0][0] * self.dc_dr[0][0] \
                 + self.prev_dc_dr[1][0] * self.dc_dr[1][0] \
                 + self.prev_dc_dr[2][0] * self.dc_dr[2][0]
            for i, rx in enumerate(network.particle_input.rx):
                delta, dc = self.get_delta(prod[i], self.delta_rx[0][i], 1.0)
                self.dc_dr[0][0][i] *= dc
                self.dc_dr[1][0][i] *= dc
                self.dc_dr[2][0][i] *= dc
                self.delta_rx[0][i] = delta
                self.delta_ry[0][i] = delta
                self.delta_rz[0][i] = delta
                network.particle_input.rx[i] -= np.sign(self.dc_dr[0][0][i]) * self.delta_rx[0][i]
                network.particle_input.ry[i] -= np.sign(self.dc_dr[1][0][i]) * self.delta_ry[0][i]
                network.particle_input.rz[i] -= np.sign(self.dc_dr[2][0][i]) * self.delta_rz[0][i]
                self.prev_dc_dr[0][0][i] = self.dc_dr[0][0][i]
                self.prev_dc_dr[1][0][i] = self.dc_dr[1][0][i]
                self.prev_dc_dr[2][0][i] = self.dc_dr[2][0][i]
