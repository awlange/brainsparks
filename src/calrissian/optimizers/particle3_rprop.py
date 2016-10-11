from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class Particle3RPROP(Optimizer):
    """
    RPROP for particle3 networks

    Technically, iRPROP-

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, cost_freq=2, init_delta=0.1, eta_plus=1.2, eta_minus=0.5,
                 delta_min=1e-6, delta_max=50.0, n_threads=1, chunk_size=1):
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
        self.prev_dc_drx_inp = None
        self.prev_dc_dry_inp = None
        self.prev_dc_drx_pos_out = None
        self.prev_dc_dry_pos_out = None
        self.prev_dc_drx_neg_out = None
        self.prev_dc_dry_neg_out = None

        self.dc_db = None
        self.dc_dq = None
        self.dc_drx_inp = None
        self.dc_dry_inp = None
        self.dc_drx_pos_out = None
        self.dc_dry_pos_out = None
        self.dc_drx_neg_out = None
        self.dc_dry_neg_out = None

        # Deltas
        self.delta_b = None
        self.delta_q = None
        self.delta_rx_inp = None
        self.delta_ry_inp = None
        self.delta_rx_pos_out = None
        self.delta_ry_pos_out = None
        self.delta_rx_neg_out = None
        self.delta_ry_neg_out = None

        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.pool = None

    def get_pool(self):
        if self.pool is None:
            self.pool = Pool(processes=self.n_threads)
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

            for t, gradients in enumerate(result):
                if t == 0 and offset == 0:
                    self.dc_db = gradients[0]
                    self.dc_dq = gradients[1]
                    self.dc_drx_inp = gradients[2]
                    self.dc_dry_inp = gradients[3]
                    self.dc_drx_pos_out = gradients[4]
                    self.dc_dry_pos_out = gradients[5]
                    self.dc_drx_neg_out = gradients[6]
                    self.dc_dry_neg_out = gradients[7]
                else:
                    tmp_dc_db = gradients[0]
                    tmp_dc_dq = gradients[1]
                    tmp_dc_drx_inp = gradients[2]
                    tmp_dc_dry_inp = gradients[3]
                    tmp_dc_drx_pos_out = gradients[4]
                    tmp_dc_dry_pos_out = gradients[5]
                    tmp_dc_drx_neg_out = gradients[6]
                    tmp_dc_dry_neg_out = gradients[7]

                    for l, tmp in enumerate(tmp_dc_db): self.dc_db[l] += tmp
                    for l, tmp in enumerate(tmp_dc_dq): self.dc_dq[l] += tmp
                    for l, tmp in enumerate(tmp_dc_drx_inp): self.dc_drx_inp[l] += tmp
                    for l, tmp in enumerate(tmp_dc_dry_inp): self.dc_dry_inp[l] += tmp
                    for l, tmp in enumerate(tmp_dc_drx_pos_out): self.dc_drx_pos_out[l] += tmp
                    for l, tmp in enumerate(tmp_dc_dry_pos_out): self.dc_dry_pos_out[l] += tmp
                    for l, tmp in enumerate(tmp_dc_drx_neg_out): self.dc_drx_neg_out[l] += tmp
                    for l, tmp in enumerate(tmp_dc_dry_neg_out): self.dc_dry_neg_out[l] += tmp

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

            # Full batch gradient
            if self.n_threads > 1:
                self.cost_gradient_parallel(network, data_X, data_Y)
            else:
                gradients = network.cost_gradient(data_X, data_Y)

                self.dc_db = gradients[0]
                self.dc_dq = gradients[1]
                self.dc_drx_inp = gradients[2]
                self.dc_dry_inp = gradients[3]
                self.dc_drx_pos_out = gradients[4]
                self.dc_dry_pos_out = gradients[5]
                self.dc_drx_neg_out = gradients[6]
                self.dc_dry_neg_out = gradients[7]

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

    def compute_position(self, dc, prev_dc, delta, layer_r, l):
        # TODO: return stuff since pass by value
        prod = prev_dc[l] * dc[l]
        for i, r in enumerate(layer_r):
            delta[l][i], dc[l][i] = self.get_delta(prod[i], delta[l][i], dc[l][i])
            layer_r[i] -= np.sign(dc[l][i]) * delta[l][i]
            prev_dc[l][i] = dc[l][i]
        return dc, prev_dc, delta, layer_r

    def weight_update(self, network):
        """
        Update weights and biases according to RPROP

        TODO: Oof... this code needs to be improved!
        """
        if self.delta_b is None or self.delta_q is None:
            # Initial iteration

            self.delta_b = []
            self.delta_q = []
            self.delta_rx_inp = []
            self.delta_ry_inp = []
            self.delta_rx_pos_out = []
            self.delta_ry_pos_out = []
            self.delta_rx_neg_out = []
            self.delta_ry_neg_out = []

            self.prev_dc_db = []
            self.prev_dc_dq = []
            self.prev_dc_drx_inp = []
            self.prev_dc_dry_inp = []
            self.prev_dc_drx_pos_out = []
            self.prev_dc_dry_pos_out = []
            self.prev_dc_drx_neg_out = []
            self.prev_dc_dry_neg_out = []

            for l, layer in enumerate(network.layers):
                self.delta_b.append(np.random.uniform(0.0, self.init_delta, layer.b.shape))
                self.delta_q.append(np.random.uniform(0.0, self.init_delta, layer.q.shape))
                self.delta_rx_inp.append(np.random.uniform(0.0, self.init_delta, layer.input_size))
                self.delta_ry_inp.append(np.random.uniform(0.0, self.init_delta, layer.input_size))
                self.delta_rx_pos_out.append(np.random.uniform(0.0, self.init_delta, layer.output_size))
                self.delta_ry_pos_out.append(np.random.uniform(0.0, self.init_delta, layer.output_size))
                self.delta_rx_neg_out.append(np.random.uniform(0.0, self.init_delta, layer.output_size))
                self.delta_ry_neg_out.append(np.random.uniform(0.0, self.init_delta, layer.output_size))

                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dq.append(np.zeros_like(self.dc_dq[l]))
                self.prev_dc_drx_inp.append(np.zeros_like(self.dc_drx_inp[l]))
                self.prev_dc_dry_inp.append(np.zeros_like(self.dc_dry_inp[l]))
                self.prev_dc_drx_pos_out.append(np.zeros_like(self.dc_drx_pos_out[l]))
                self.prev_dc_dry_pos_out.append(np.zeros_like(self.dc_dry_pos_out[l]))
                self.prev_dc_drx_neg_out.append(np.zeros_like(self.dc_drx_neg_out[l]))
                self.prev_dc_dry_neg_out.append(np.zeros_like(self.dc_dry_neg_out[l]))

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

            # Positions
            self.dc_drx_inp, self.prev_dc_drx_inp, self.delta_rx_inp, layer.rx_inp = self.compute_position(self.dc_drx_inp, self.prev_dc_drx_inp, self.delta_rx_inp, layer.rx_inp, l)
            self.dc_dry_inp, self.prev_dc_dry_inp, self.delta_ry_inp, layer.ry_inp = self.compute_position(self.dc_dry_inp, self.prev_dc_dry_inp, self.delta_ry_inp, layer.ry_inp, l)
            self.dc_drx_pos_out, self.prev_dc_drx_pos_out, self.delta_rx_pos_out, layer.rx_pos_out = self.compute_position(self.dc_drx_pos_out, self.prev_dc_drx_pos_out, self.delta_rx_pos_out, layer.rx_pos_out, l)
            self.dc_dry_pos_out, self.prev_dc_dry_pos_out, self.delta_ry_pos_out, layer.ry_pos_out = self.compute_position(self.dc_dry_pos_out, self.prev_dc_dry_pos_out, self.delta_ry_pos_out, layer.ry_pos_out, l)
            self.dc_drx_neg_out, self.prev_dc_drx_neg_out, self.delta_rx_neg_out, layer.rx_neg_out = self.compute_position(self.dc_drx_neg_out, self.prev_dc_drx_neg_out, self.delta_rx_neg_out, layer.rx_neg_out, l)
            self.dc_dry_neg_out, self.prev_dc_dry_neg_out, self.delta_ry_neg_out, layer.ry_neg_out = self.compute_position(self.dc_dry_neg_out, self.prev_dc_dry_neg_out, self.delta_ry_neg_out, layer.ry_neg_out, l)

