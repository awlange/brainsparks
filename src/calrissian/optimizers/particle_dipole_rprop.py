from .optimizer import Optimizer

import numpy as np
import time


class ParticleDipoleRPROP(Optimizer):
    """
    RPROP for particle networks

    Technically, iRPROP-

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, cost_freq=2, init_delta=0.1, eta_plus=1.2, eta_minus=0.5,
                 delta_min=1e-6, delta_max=50.0):
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
        self.prev_dc_drx_pos = None
        self.prev_dc_dry_pos = None
        self.prev_dc_drz_pos = None
        self.prev_dc_drx_neg = None
        self.prev_dc_dry_neg = None
        self.prev_dc_drz_neg = None

        self.dc_db = None
        self.dc_dq = None
        self.dc_drx_pos = None
        self.dc_dry_pos = None
        self.dc_drz_pos = None
        self.dc_drx_neg = None
        self.dc_dry_neg = None
        self.dc_drz_neg = None

        # Deltas
        self.delta_b = None
        self.delta_q = None
        self.delta_rx_pos = None
        self.delta_ry_pos = None
        self.delta_rz_pos = None
        self.delta_rx_neg = None
        self.delta_ry_neg = None
        self.delta_rz_neg = None

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
            gradients = network.cost_gradient(data_X, data_Y)

            self.dc_db = gradients[0]
            self.dc_dq = gradients[1]
            self.dc_drx_pos = gradients[2]
            self.dc_dry_pos = gradients[3]
            self.dc_drz_pos = gradients[4]
            self.dc_drx_neg = gradients[5]
            self.dc_dry_neg = gradients[6]
            self.dc_drz_neg = gradients[7]

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
        # TODO: return stuff since pas by value
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
            self.delta_rx_pos = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_ry_pos = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_rz_pos = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_rx_neg = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_ry_neg = [np.ones(network.particle_input.output_size) * self.init_delta]
            self.delta_rz_neg = [np.ones(network.particle_input.output_size) * self.init_delta]

            self.prev_dc_db = []
            self.prev_dc_dq = []
            self.prev_dc_drx_pos = [np.zeros(network.particle_input.output_size)]
            self.prev_dc_dry_pos = [np.zeros(network.particle_input.output_size)]
            self.prev_dc_drz_pos = [np.zeros(network.particle_input.output_size)]
            self.prev_dc_drx_neg = [np.zeros(network.particle_input.output_size)]
            self.prev_dc_dry_neg = [np.zeros(network.particle_input.output_size)]
            self.prev_dc_drz_neg = [np.zeros(network.particle_input.output_size)]

            for l, layer in enumerate(network.layers):
                self.delta_b.append(np.ones(layer.b.shape) * self.init_delta)
                self.delta_q.append(np.ones(layer.q.shape) * self.init_delta)
                self.delta_rx_pos.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_ry_pos.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_rz_pos.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_rx_neg.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_ry_neg.append(np.ones(layer.output_size) * self.init_delta)
                self.delta_rz_neg.append(np.ones(layer.output_size) * self.init_delta)

                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dq.append(np.zeros_like(self.dc_dq[l]))
                self.prev_dc_drx_pos.append(np.zeros_like(self.dc_drx_pos[l+1]))
                self.prev_dc_dry_pos.append(np.zeros_like(self.dc_dry_pos[l+1]))
                self.prev_dc_drz_pos.append(np.zeros_like(self.dc_drz_pos[l+1]))
                self.prev_dc_drx_neg.append(np.zeros_like(self.dc_drx_neg[l+1]))
                self.prev_dc_dry_neg.append(np.zeros_like(self.dc_dry_neg[l+1]))
                self.prev_dc_drz_neg.append(np.zeros_like(self.dc_drz_neg[l+1]))

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

            self.dc_drx_pos, self.prev_dc_drx_pos, self.delta_rx_pos, layer.rx_pos = self.compute_position(self.dc_drx_pos, self.prev_dc_drx_pos, self.delta_rx_pos, layer.rx_pos, l+1)
            self.dc_dry_pos, self.prev_dc_dry_pos, self.delta_ry_pos, layer.ry_pos = self.compute_position(self.dc_dry_pos, self.prev_dc_dry_pos, self.delta_ry_pos, layer.ry_pos, l+1)
            self.dc_drz_pos, self.prev_dc_drz_pos, self.delta_rz_pos, layer.rz_pos = self.compute_position(self.dc_drz_pos, self.prev_dc_drz_pos, self.delta_rz_pos, layer.rz_pos, l+1)
            self.dc_drx_neg, self.prev_dc_drx_neg, self.delta_rx_neg, layer.rx_neg = self.compute_position(self.dc_drx_neg, self.prev_dc_drx_neg, self.delta_rx_neg, layer.rx_neg, l+1)
            self.dc_dry_neg, self.prev_dc_dry_neg, self.delta_ry_neg, layer.ry_neg = self.compute_position(self.dc_dry_neg, self.prev_dc_dry_neg, self.delta_ry_neg, layer.ry_neg, l+1)
            self.dc_drz_neg, self.prev_dc_drz_neg, self.delta_rz_neg, layer.rz_neg = self.compute_position(self.dc_drz_neg, self.prev_dc_drz_neg, self.delta_rz_neg, layer.rz_neg, l+1)

        # Input layer position

        self.dc_drx_pos, self.prev_dc_drx_pos, self.delta_rx_pos, network.particle_input.rx_pos = self.compute_position(self.dc_drx_pos, self.prev_dc_drx_pos, self.delta_rx_pos, network.particle_input.rx_pos, 0)
        self.dc_dry_pos, self.prev_dc_dry_pos, self.delta_ry_pos, network.particle_input.ry_pos = self.compute_position(self.dc_dry_pos, self.prev_dc_dry_pos, self.delta_ry_pos, network.particle_input.ry_pos, 0)
        self.dc_drz_pos, self.prev_dc_drz_pos, self.delta_rz_pos, network.particle_input.rz_pos = self.compute_position(self.dc_drz_pos, self.prev_dc_drz_pos, self.delta_rz_pos, network.particle_input.rz_pos, 0)
        self.dc_drx_neg, self.prev_dc_drx_neg, self.delta_rx_neg, network.particle_input.rx_neg = self.compute_position(self.dc_drx_neg, self.prev_dc_drx_neg, self.delta_rx_neg, network.particle_input.rx_neg, 0)
        self.dc_dry_neg, self.prev_dc_dry_neg, self.delta_ry_neg, network.particle_input.ry_neg = self.compute_position(self.dc_dry_neg, self.prev_dc_dry_neg, self.delta_ry_neg, network.particle_input.ry_neg, 0)
        self.dc_drz_neg, self.prev_dc_drz_neg, self.delta_rz_neg, network.particle_input.rz_neg = self.compute_position(self.dc_drz_neg, self.prev_dc_drz_neg, self.delta_rz_neg, network.particle_input.rz_neg, 0)

