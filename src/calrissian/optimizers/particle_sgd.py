from .optimizer import Optimizer

import numpy as np
import time


class ParticleSGD(Optimizer):
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

            # TODO: Doubles memory usage of data by having a copy. Figure out how to shuffle data_X with data_Y
            # Shuffle data by index
            np.random.shuffle(indexes)  # in-place shuffle
            shuffle_X = np.asarray([data_X[i] for i in indexes])
            shuffle_Y = np.asarray([data_Y[i] for i in indexes])

            # Split into mini-batches
            for m in range(len(data_X) // self.mini_batch_size):  # not guaranteed to divide perfectly, might miss a few
                mini_X = shuffle_X[m:(m+self.mini_batch_size)]
                mini_Y = shuffle_Y[m:(m+self.mini_batch_size)]

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
            layer.q -= self.alpha * self.dc_dq[l+1]
            for i in range(len(layer.r)):
                layer.r[i][0] -= self.alpha * self.dc_dr[l+1][i][0]
                layer.r[i][1] -= self.alpha * self.dc_dr[l+1][i][1]
                layer.r[i][2] -= self.alpha * self.dc_dr[l+1][i][2]
        for i in range(len(network.particle_input.r)):
            network.particle_input.q[i][0] -= self.alpha * self.dc_dq[0][i][0]
            network.particle_input.r[i][0] -= self.alpha * self.dc_dr[0][i][0]
            network.particle_input.r[i][1] -= self.alpha * self.dc_dr[0][i][1]
            network.particle_input.r[i][2] -= self.alpha * self.dc_dr[0][i][2]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        TODO
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None or self.vel_r is None:
            self.vel_b = []
            self.vel_q = [np.zeros(network.particle_input.q.shape)]
            self.vel_r = [np.zeros(network.particle_input.r.shape)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_r.append(np.zeros(layer.r.shape))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l+1] = -self.alpha * self.dc_dq[l+1] + self.beta * self.vel_q[l+1]
            self.vel_r[l+1] = -self.alpha * self.dc_dr[l+1] + self.beta * self.vel_r[l+1]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l+1]
            layer.r += self.vel_r[l+1]
        self.vel_r[0] = -self.alpha * self.dc_dr[0] + self.beta * self.vel_r[0]
        network.particle_input.r += self.vel_r[0]
        # self.vel_q[0] = -self.alpha * self.dc_dq[0] + self.beta * self.vel_q[0]
        # network.particle_input.q += self.vel_q[0]
