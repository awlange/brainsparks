from .optimizer import Optimizer

import numpy as np
import time


class RPROP(Optimizer):
    """
    RPROP

    Technically, iRPROP-

    Full-batch, not mini-batch
    """

    def __init__(self, n_epochs=1, verbosity=2, init_delta=0.1, eta_plus=1.2, eta_minus=0.5,
                 delta_min=1e-6, delta_max=50.0):
        """
        :param alpha: learning rate
        :param beta: momentum damping (viscosity)
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbosity = verbosity

        # RPROP params
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.init_delta = init_delta
        self.delta_max = delta_max
        self.delta_min = delta_min

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dw = None
        self.prev_dc_db = None
        self.prev_dc_dw = None

        # Deltas
        self.delta_b = None
        self.delta_w = None

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
            self.dc_db, self.dc_dw = network.cost_gradient(data_X, data_Y)

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
        if self.delta_b is None or self.delta_w:
            # Initial iteration
            self.delta_b = []
            self.delta_w = []
            self.prev_dc_db = []
            self.prev_dc_dw = []
            for l, layer in enumerate(network.layers):
                self.delta_b.append(np.ones(layer.b.shape) * self.init_delta)
                self.delta_w.append(np.ones(layer.w.shape) * self.init_delta)
                self.prev_dc_db.append(np.zeros_like(self.dc_db[l]))
                self.prev_dc_dw.append(np.zeros_like(self.dc_dw[l]))

        for l, layer in enumerate(network.layers):
            # Biases
            prod = self.prev_dc_db[l] * self.dc_db[l]
            for i, b in enumerate(layer.b[0]):
                self.delta_b[l][0][i], self.dc_db[l][0][i] = self.get_delta(prod[0][i], self.delta_b[l][0][i], self.dc_db[l][0][i])
                layer.b[0][i] -= np.sign(self.dc_db[l][0][i]) * self.delta_b[l][0][i]
                self.prev_dc_db[l][0][i] = self.dc_db[l][0][i]

            # Weights
            prod = self.prev_dc_dw[l] * self.dc_dw[l]
            for i, q in enumerate(layer.w):
                self.delta_w[l][i], self.dc_dw[l][i] = self.get_delta(prod[i], self.delta_w[l][i], self.dc_dw[l][i])
                layer.q[i] -= np.sign(self.dc_dw[l][i]) * self.delta_w[l][i]
                self.prev_dc_dw[l][i] = self.dc_dw[l][i]
