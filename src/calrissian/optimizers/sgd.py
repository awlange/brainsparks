from .optimizer import Optimizer

import numpy as np
import time


class SGD(Optimizer):
    """
    Stochastic gradient descent optimization
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=10, verbosity=2, weight_update="sd",
                 cost_freq=1, gamma=0.9):
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
        self.cost_freq = cost_freq
        self.gamma = gamma

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        elif weight_update == "rmsprop":
            self.weight_update_func = self.weight_update_rmsprop
        elif weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dw = None

        # Velocities
        self.vel_b = None
        self.vel_w = None

        # RMS
        self.ms_db = None
        self.ms_dw = None

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
                mini_X = shuffle_X[m*self.mini_batch_size:(m+1)*self.mini_batch_size]
                mini_Y = shuffle_Y[m*self.mini_batch_size:(m+1)*self.mini_batch_size]

                # Compute gradient for mini-batch
                self.dc_db, self.dc_dw = network.cost_gradient(mini_X, mini_Y)

                # Update weights and biases
                self.weight_update_func(network, self.dc_db, self.dc_dw)

                if self.verbosity > 1 and m % self.cost_freq == 0:
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

    def weight_update_steepest_descent(self, network, dc_db, dc_dw):
        """
        Update weights and biases according to steepest descent
        """
        for l, layer in enumerate(network.layers):
            layer.b -= self.alpha * dc_db[l]
            layer.w -= self.alpha * dc_dw[l]

    def weight_update_steepest_descent_with_momentum(self, network, dc_db, dc_dw):
        """
        Update weights and biases according to steepest descent
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_w is None:
            self.vel_b = []
            self.vel_w = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_w.append(np.zeros(layer.w.shape))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha * dc_db[l] + self.beta * self.vel_b[l]
            self.vel_w[l] = -self.alpha * dc_dw[l] + self.beta * self.vel_w[l]
            layer.b += self.vel_b[l]
            layer.w += self.vel_w[l]

    def weight_update_rmsprop(self, network, dc_db, dc_dw):
        """
        Update weights and biases according to RMSProp
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        alpha = self.alpha
        epsilon = 10e-8  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dw is None:
            self.ms_db = []
            self.ms_dw = []
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dw.append(np.zeros(layer.w.shape))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (dc_db[l] * dc_db[l])
            self.ms_dw[l] = gamma * self.ms_dw[l] + one_m_gamma * (dc_dw[l] * dc_dw[l])
            layer.b -= alpha * dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.w -= alpha * dc_dw[l] / np.sqrt(self.ms_dw[l] + epsilon)

    def weight_update_adagrad(self, network, dc_db, dc_dw):
        """
        Update weights and biases according to AdaGrad
        """
        alpha = self.alpha
        epsilon = 10e-8  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dw is None:
            self.ms_db = []
            self.ms_dw = []
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dw.append(np.zeros(layer.w.shape))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] += (dc_db[l] * dc_db[l])
            self.ms_dw[l] += (dc_dw[l] * dc_dw[l])
            layer.b -= alpha * dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.w -= alpha * dc_dw[l] / np.sqrt(self.ms_dw[l] + epsilon)
