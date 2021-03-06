from .optimizer import Optimizer

import numpy as np
import time
import sys


class ParticleConvSGD(Optimizer):
    """
    Stochastic gradient descent optimization
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, n_threads=1, chunk_size=10, epsilon=10e-8, gamma2=0.1, alpha_decay=None):
        """
        :param alpha: learning rate
        :param beta: momentum damping (viscosity)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma2 = gamma2
        self.epsilon = epsilon
        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.verbosity = verbosity
        self.cost_freq = cost_freq
        self.alpha_decay = alpha_decay

        self.residual = 0.0

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        elif weight_update == "rmsprop":
            self.weight_update_func = self.weight_update_rmsprop

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_drb = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_r = None
        self.vel_rb = None

        # Mean squares
        self.ms_db = None
        self.ms_dq = None
        self.ms_dr = None
        self.ms_drb = None

        self.ms_b = None
        self.ms_q = None
        self.ms_r = None
        self.ms_rb = None

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
            mdiv = len(data_X) // self.mini_batch_size
            for m in range(mdiv):  # not guaranteed to divide perfectly, might miss a few
                mb_start_time = time.time()
                mini_X = shuffle_X[m*self.mini_batch_size:(m+1)*self.mini_batch_size]
                mini_Y = shuffle_Y[m*self.mini_batch_size:(m+1)*self.mini_batch_size]

                # Compute gradient for mini-batch
                self.dc_db, self.dc_dq, self.dc_drb, self.dc_dr = network.cost_gradient(mini_X, mini_Y)
                
                # Update weights and biases
                self.weight_update_func(network)

                # Alpha decay
                if self.alpha_decay is not None:
                    self.alpha *= 1.0 - self.alpha_decay

                if self.verbosity > 1 and m % self.cost_freq == 0:
                    c = network.cost(data_X, data_Y)
                    print("Cost at epoch {} mini-batch {}: {:g} mini-batch time: {}".format(epoch, m, c, time.time() - mb_start_time))
                    sys.stdout.flush()

                elif self.verbosity == 1 and m % self.cost_freq == 0:
                    c = network.cost(mini_X, mini_Y)
                    mt = time.time() - mb_start_time
                    emt = mt * (mdiv-1 - m)
                    print("Estimated cost at epoch {:2d} mini-batch {:4d}: {:.6f} mini-batch time: {:.2f} estimated epoch time remaining: {:.2f}".format(
                        epoch, m, c, mt, emt))
                    sys.stdout.flush()

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
            layer.r -= self.alpha * self.dc_dr[l+1]
            layer.rb -= self.alpha * self.dc_drb[l]

        network.particle_input.rx -= self.alpha * self.dc_dr[0]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent with simple momentum
        """
        # Initialize RMS to zero
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_r = [np.zeros(network.particle_input.r.shape)]
            self.vel_rb = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_r.append(np.zeros(layer.r.shape))
                self.vel_rb.append(np.zeros(layer.rb.shape))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_r[l+1] = -self.alpha * self.dc_dr[l+1] + self.beta * self.vel_r[l+1]
            self.vel_rb[l] = -self.alpha * self.dc_drb[l] + self.beta * self.vel_rb[l]

            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.r += self.vel_r[l+1]
            layer.rb += self.vel_rb[l]

        self.vel_r[0] = -self.alpha * self.dc_dr[0] + self.beta * self.vel_r[0]
        network.particle_input.r += self.vel_r[0]

    def weight_update_rmsprop(self, network):
        """
        Update weights and biases according to RMSProp
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        alpha = self.alpha
        epsilon = self.epsilon  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_dr = [np.zeros(network.particle_input.r.shape)]
            self.ms_drb = []
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_dr.append(np.zeros(layer.r.shape))
                self.ms_drb.append(np.zeros(layer.rb.shape))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma * self.ms_dq[l] + one_m_gamma * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dr[l + 1] = gamma * self.ms_dr[l + 1] + one_m_gamma * (self.dc_dr[l + 1] * self.dc_dr[l + 1])
            self.ms_drb[l] = gamma * self.ms_drb[l] + one_m_gamma * (self.dc_drb[l] * self.dc_drb[l])

            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.q -= alpha * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon)
            layer.r -= alpha * self.dc_dr[l+1] / np.sqrt(self.ms_dr[l + 1] + epsilon)
            layer.rb -= alpha * self.dc_drb[l] / np.sqrt(self.ms_drb[l] + epsilon)

        # Input layer
        self.ms_dr[0] = gamma * self.ms_dr[0] + one_m_gamma * (self.dc_dr[0] * self.dc_dr[0])
        network.particle_input.r -= alpha * self.dc_dr[0] / np.sqrt(self.ms_dr[0] + epsilon)

