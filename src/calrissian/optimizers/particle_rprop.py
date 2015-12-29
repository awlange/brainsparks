from .optimizer import Optimizer

import numpy as np
import time


class ParticleRPROP(Optimizer):
    """
    RPROP for particle networks

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, cost_freq=2, init_delta=0.1, eta_plus=1.2, eta_minus=0.5):
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

        # Weight gradients, to keep around for a step
        self.prev_dc_db = None
        self.prev_dc_dq = None
        self.prev_dc_dr = None
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None

        # Deltas
        self.init_delta = init_delta
        self.delta_max = 4.0
        self.delta_min = 1e-6
        self.delta_b = None
        self.delta_q = None
        self.delta_rx = None
        self.delta_ry = None
        self.delta_rz = None

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
            self.dc_db, self.dc_dq, self.dc_dr = network.cost_gradient(data_X, data_Y)

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

            self.prev_dc_db = []
            self.prev_dc_dq = []
            self.prev_dc_dr = [[np.zeros(network.particle_input.output_size)] for _ in range(3)]

            for l, layer in enumerate(network.layers):
                self.delta_b.append(np.ones(layer.b.shape) * self.init_delta)
                self.delta_q.append(np.ones(layer.q.shape) * self.init_delta)
                self.delta_rx.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_ry.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_rz.append(np.ones(layer.output_size) * self.init_delta)

                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dq.append(np.zeros_like(self.dc_dq[l]))
                self.prev_dc_dr[0].append(np.zeros_like(self.dc_dr[0][l+1]))
                self.prev_dc_dr[1].append(np.zeros_like(self.dc_dr[1][l+1]))
                self.prev_dc_dr[2].append(np.zeros_like(self.dc_dr[2][l+1]))

        for l, layer in enumerate(network.layers):

            # Biases
            prod = self.prev_dc_db[l] * self.dc_db[l]
            for i, b in enumerate(layer.b):
                if prod[0][i] > 0:
                    self.delta_b[l][0][i] = min(self.delta_b[l][0][i] * self.eta_plus, self.delta_max)
                    layer.b[i] -= np.sign(self.dc_db[l][i]) * self.delta_b[l][i]
                    self.prev_dc_db[l][0][i] = self.dc_db[l][0][i]
                elif prod[0][i] < 0:
                    self.delta_b[l][0][i] = max(self.delta_b[l][0][i] * self.eta_minus, self.delta_min)
                    self.prev_dc_db[l][0][i] = 0.0
                else:
                    layer.b[i] -= np.sign(self.dc_db[l][0][i]) * self.delta_b[l][0][i]
                    self.prev_dc_db[l][0][i] = self.dc_db[l][0][i]

            # Charges
            prod = self.prev_dc_dq[l] * self.dc_dq[l]
            for i, q in enumerate(layer.q):
                if prod[i] > 0:
                    self.delta_q[l][i] = min(self.delta_q[l][i] * self.eta_plus, self.delta_max)
                    layer.q[i] -= np.sign(self.dc_dq[l][i]) * self.delta_q[l][i]
                    self.prev_dc_dq[l][i] = self.dc_dq[l][i]
                elif prod[i] < 0:
                    self.delta_q[l][i] = max(self.delta_q[l][i] * self.eta_minus, self.delta_min)
                    self.prev_dc_dq[l][i] = 0.0
                else:
                    layer.q[i] -= np.sign(self.dc_dq[l][i]) * self.delta_q[l][i]
                    self.prev_dc_dq[l][i] = self.dc_dq[l][i]

            # Positions

            # X
            prod = self.prev_dc_dr[0][l+1] * self.dc_dr[0][l+1]
            for i, rx in enumerate(layer.rx):
                if prod[i] > 0:
                    self.delta_rx[l+1][i] = min(self.delta_rx[l+1][i] * self.eta_plus, self.delta_max)
                    layer.rx[i] -= np.sign(self.dc_dr[0][l+1][i]) * self.delta_rx[l+1][i]
                    self.prev_dc_dr[0][l+1][i] = self.dc_dr[0][l+1][i]
                elif prod[i] < 0:
                    self.delta_rx[l+1][i] = max(self.delta_rx[l+1][i] * self.eta_minus, self.delta_min)
                    self.prev_dc_dr[0][l+1][i] = 0.0
                else:
                    layer.rx[i] -= np.sign(self.dc_dr[0][l+1][i]) * self.delta_rx[l+1][i]
                    self.prev_dc_dr[0][l+1][i] = self.dc_dr[0][l+1][i]

            # Y
            prod = self.prev_dc_dr[1][l+1] * self.dc_dr[1][l+1]
            for i, ry in enumerate(layer.ry):
                if prod[i] > 0:
                    self.delta_ry[l+1][i] = min(self.delta_ry[l+1][i] * self.eta_plus, self.delta_max)
                    layer.ry[i] -= np.sign(self.dc_dr[1][l+1][i]) * self.delta_ry[l+1][i]
                    self.prev_dc_dr[1][l+1][i] = self.dc_dr[1][l+1][i]
                elif prod[i] < 0:
                    self.delta_ry[l+1][i] = max(self.delta_ry[l+1][i] * self.eta_minus, self.delta_min)
                    self.prev_dc_dr[1][l+1][i] = 0.0
                else:
                    layer.ry[i] -= np.sign(self.dc_dr[1][l+1][i]) * self.delta_ry[l+1][i]
                    self.prev_dc_dr[1][l+1][i] = self.dc_dr[1][l+1][i]

            # Z
            prod = self.prev_dc_dr[2][l+1] * self.dc_dr[2][l+1]
            for i, rz in enumerate(layer.rz):
                if prod[i] > 0:
                    self.delta_rz[l+1][i] = min(self.delta_rz[l+1][i] * self.eta_plus, self.delta_max)
                    layer.rz[i] -= np.sign(self.dc_dr[2][l+1][i]) * self.delta_rz[l+1][i]
                    self.prev_dc_dr[2][l+1][i] = self.dc_dr[2][l+1][i]
                elif prod[i] < 0:
                    self.delta_rz[l+1][i] = max(self.delta_rz[l+1][i] * self.eta_minus, self.delta_min)
                    self.prev_dc_dr[2][l+1][i] = 0.0
                else:
                    layer.rz[i] -= np.sign(self.dc_dr[2][l+1][i]) * self.delta_rz[l+1][i]
                    self.prev_dc_dr[2][l+1][i] = self.dc_dr[2][l+1][i]

        # Input layer position

        # X
        prod = self.prev_dc_dr[0][0] * self.dc_dr[0][0]
        for i, rx in enumerate(network.particle_input.rx):
            if prod[i] > 0:
                self.delta_rx[0][i] = min(self.delta_rx[0][i] * self.eta_plus, self.delta_max)
                network.particle_input.rx[i] -= np.sign(self.dc_dr[0][0][i]) * self.delta_rx[0][i]
                self.prev_dc_dr[0][0][i] = self.dc_dr[0][0][i]
            elif prod[i] < 0:
                self.delta_rx[0][i] = max(self.delta_rx[0][i] * self.eta_minus, self.delta_min)
                self.prev_dc_dr[0][0][i] = 0.0
            else:
                network.particle_input.rx[i] -= np.sign(self.dc_dr[0][0][i]) * self.delta_rx[0][i]
                self.prev_dc_dr[0][0][i] = self.dc_dr[0][0][i]

        # Y
        prod = self.prev_dc_dr[1][0] * self.dc_dr[1][0]
        for i, ry in enumerate(network.particle_input.ry):
            if prod[i] > 0:
                self.delta_ry[0][i] = min(self.delta_ry[0][i] * self.eta_plus, self.delta_max)
                network.particle_input.ry[i] -= np.sign(self.dc_dr[1][0][i]) * self.delta_ry[0][i]
                self.prev_dc_dr[1][0][i] = self.dc_dr[1][0][i]
            elif prod[i] < 0:
                self.delta_ry[0][i] = max(self.delta_ry[0][i] * self.eta_minus, self.delta_min)
                self.prev_dc_dr[1][0][i] = 0.0
            else:
                network.particle_input.ry[i] -= np.sign(self.dc_dr[1][0][i]) * self.delta_ry[0][i]
                self.prev_dc_dr[1][0][i] = self.dc_dr[1][0][i]

        # Z
        prod = self.prev_dc_dr[2][0] * self.dc_dr[2][0]
        for i, rz in enumerate(network.particle_input.rz):
            if prod[i] > 0:
                self.delta_rz[0][i] = min(self.delta_rz[0][i] * self.eta_plus, self.delta_max)
                network.particle_input.rz[i] -= np.sign(self.dc_dr[2][0][i]) * self.delta_rz[0][i]
                self.prev_dc_dr[2][0][i] = self.dc_dr[2][0][i]
            elif prod[i] < 0:
                self.delta_rz[0][i] = max(self.delta_rz[0][i] * self.eta_minus, self.delta_min)
                self.prev_dc_dr[2][0][i] = 0.0
            else:
                network.particle_input.rz[i] -= np.sign(self.dc_dr[2][0][i]) * self.delta_rz[0][i]
                self.prev_dc_dr[2][0][i] = self.dc_dr[2][0][i]

