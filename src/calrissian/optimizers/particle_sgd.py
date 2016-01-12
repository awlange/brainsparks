from .optimizer import Optimizer

import numpy as np
import time


class ParticleSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Atomic layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=0.01, alpha_r=0.01, alpha_t=0.01):
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
        self.position_grad = position_grad  # Turn off position gradient?

        self.alpha_b = alpha_b
        self.alpha_q = alpha_q
        self.alpha_r = alpha_r
        self.alpha_t = alpha_t

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_dt = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_rx = None
        self.vel_ry = None
        self.vel_rz = None
        self.vel_t = None

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
                self.dc_db, self.dc_dq, self.dc_dr, self.dc_dt = network.cost_gradient(mini_X, mini_Y)

                # Update weights and biases
                self.weight_update_func(network)

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

    def weight_update_steepest_descent(self, network):
        """
        Update weights and biases according to steepest descent
        """
        for l, layer in enumerate(network.layers):
            layer.b -= self.alpha * self.dc_db[l]
            layer.q -= self.alpha * self.dc_dq[l]
            layer.theta -= self.alpha * self.dc_dt[l+1]
            if self.position_grad:
                layer.rx -= self.alpha * self.dc_dr[0][l+1]
                layer.ry -= self.alpha * self.dc_dr[1][l+1]
                layer.rz -= self.alpha * self.dc_dr[2][l+1]

        network.particle_input.theta -= self.alpha * self.dc_dt[0]
        if self.position_grad:
            network.particle_input.rx -= self.alpha * self.dc_dr[0][0]
            network.particle_input.ry -= self.alpha * self.dc_dr[1][0]
            network.particle_input.rz -= self.alpha * self.dc_dr[2][0]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        TODO
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None or self.vel_rx is None or self.vel_ry is None or self.vel_rz is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx = [np.zeros(network.particle_input.output_size)]
            self.vel_ry = [np.zeros(network.particle_input.output_size)]
            self.vel_rz = [np.zeros(network.particle_input.output_size)]
            self.vel_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx.append(np.zeros(layer.output_size))
                self.vel_ry.append(np.zeros(layer.output_size))
                self.vel_rz.append(np.zeros(layer.output_size))
                self.vel_t.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha_b * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha_q * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx[l+1] = -self.alpha_r * self.dc_dr[0][l+1] + self.beta * self.vel_rx[l+1]
            self.vel_ry[l+1] = -self.alpha_r * self.dc_dr[1][l+1] + self.beta * self.vel_ry[l+1]
            self.vel_rz[l+1] = -self.alpha_r * self.dc_dr[2][l+1] + self.beta * self.vel_rz[l+1]
            self.vel_t[l+1] = -self.alpha_t * self.dc_dt[l+1] + self.beta * self.vel_t[l+1]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.theta += self.vel_t[l+1]
            if self.position_grad:
                layer.rx += self.vel_rx[l+1]
                layer.ry += self.vel_ry[l+1]
                layer.rz += self.vel_rz[l+1]

        self.vel_t[0] = -self.alpha_t * self.dc_dt[0][0] + self.beta * self.vel_t[0]
        network.particle_input.theta += self.vel_t[0]
        if self.position_grad:
            self.vel_rx[0] = -self.alpha_r * self.dc_dr[0][0] + self.beta * self.vel_rx[0]
            self.vel_ry[0] = -self.alpha_r * self.dc_dr[1][0] + self.beta * self.vel_ry[0]
            self.vel_rz[0] = -self.alpha_r * self.dc_dr[2][0] + self.beta * self.vel_rz[0]
            network.particle_input.rx += self.vel_rx[0]
            network.particle_input.ry += self.vel_ry[0]
            network.particle_input.rz += self.vel_rz[0]
