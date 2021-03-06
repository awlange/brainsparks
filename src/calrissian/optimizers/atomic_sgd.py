from .optimizer import Optimizer

import numpy as np
import time


class AtomicSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Atomic layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd"):
        """
        :param alpha: learning rate
        :param beta: momentum damping (viscosity)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.verbosity = verbosity

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_r = None

    def optimize(self, network, data_X, data_Y):
        """
        :return: optimized network
        """
        optimize_start_time = time.time()

        indexes = np.arange(len(data_X))

        if self.verbosity > 0:
            c = network.cost(data_X, data_Y)
            print("Cost before epochs: {}".format(c))

        # Epoch loop
        for epoch in range(self.n_epochs):
            epoch_start_time = time.time()

            # Shuffle data by index
            np.random.shuffle(indexes)  # in-place shuffle
            shuffled_mini_batch_indexes = np.array_split(indexes, len(data_X) // self.mini_batch_size)

            # Split into mini-batches
            for m, mini_batch_indexes in enumerate(shuffled_mini_batch_indexes):
                mini_X = np.asarray([data_X[i] for i in mini_batch_indexes])
                mini_Y = np.asarray([data_Y[i] for i in mini_batch_indexes])

                # Compute gradient for mini-batch
                self.dc_db, self.dc_dq, self.dc_dr = network.cost_gradient(mini_X, mini_Y)

                # Update weights and biases
                self.weight_update_func(network)

                if self.verbosity > 1:
                    c = network.cost(data_X, data_Y)
                    print("Cost at epoch {} mini-batch {}: {:g}".format(epoch, m, c))
                    # TODO: could output projected time left based on mini-batch times

            if self.verbosity > 0:
                c = network.cost(data_X, data_Y)
                print("Cost after epoch {}: {:g}".format(epoch, c))
                print("Epoch time: {:g} s".format(time.time() - epoch_start_time))

        if self.verbosity > 0:
            c = network.cost(data_X, data_Y)
            print("\n\nCost after optimize run: {:g}".format(c))
            print("Optimize run time: {:g} s".format(time.time() - optimize_start_time))

        return network

    def weight_update_steepest_descent(self, network):
        """
        Update weights and biases according to steepest descent
        """
        for l, layer in enumerate(network.layers):
            layer.b -= self.alpha * self.dc_db[l]
            layer.q -= self.alpha * self.dc_dq[l]
            for i in range(len(layer.r)):
                layer.r[i].x -= self.alpha * self.dc_dr[l+1][i][0]
                layer.r[i].y -= self.alpha * self.dc_dr[l+1][i][1]
                layer.r[i].z -= self.alpha * self.dc_dr[l+1][i][2]
        for i in range(len(network.atomic_input.r)):
            network.atomic_input.r[i].x -= self.alpha * self.dc_dr[0][i][0]
            network.atomic_input.r[i].y -= self.alpha * self.dc_dr[0][i][1]
            network.atomic_input.r[i].z -= self.alpha * self.dc_dr[0][i][2]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        TODO
        """
        # # Initialize velocities to zero for momentum
        # if self.vel_b is None or self.vel_w is None:
        #     self.vel_b = []
        #     self.vel_w = []
        #     for l, layer in enumerate(network.layers):
        #         self.vel_b.append(np.zeros(layer.b.shape))
        #         self.vel_w.append(np.zeros(layer.w.shape))
        #
        # for l, layer in enumerate(network.layers):
        #     self.vel_b[l] = -self.alpha * dc_db[l] + self.beta * self.vel_b[l]
        #     self.vel_w[l] = -self.alpha * dc_dw[l] + self.beta * self.vel_w[l]
        #     layer.b += self.vel_b[l]
        #     layer.w += self.vel_w[l]
