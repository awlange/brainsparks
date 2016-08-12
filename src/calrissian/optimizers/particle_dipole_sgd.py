from .optimizer import Optimizer

import numpy as np
import time


class ParticleDipoleSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Particle Dipole layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=0.01, alpha_r=0.01):
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

        self.alpha_b = alpha
        self.alpha_q = alpha
        self.alpha_r = alpha

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        if weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_drx_pos = None
        self.dc_dry_pos = None
        self.dc_drz_pos = None
        self.dc_drx_neg = None
        self.dc_dry_neg = None
        self.dc_drz_neg = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_rx_pos = None
        self.vel_ry_pos = None
        self.vel_rz_pos = None
        self.vel_rx_neg = None
        self.vel_ry_neg = None
        self.vel_rz_neg = None

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
                gradients = network.cost_gradient(mini_X, mini_Y)

                self.dc_db = gradients[0]
                self.dc_dq = gradients[1]
                self.dc_drx_pos = gradients[2]
                self.dc_dry_pos = gradients[3]
                self.dc_drz_pos = gradients[4]
                self.dc_drx_neg = gradients[5]
                self.dc_dry_neg = gradients[6]
                self.dc_drz_neg = gradients[7]

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
            layer.rx_pos -= self.alpha * self.dc_drx_pos[l+1]
            layer.ry_pos -= self.alpha * self.dc_dry_pos[l+1]
            layer.rz_pos -= self.alpha * self.dc_drz_pos[l+1]
            layer.rx_neg -= self.alpha * self.dc_drx_neg[l+1]
            layer.ry_neg -= self.alpha * self.dc_dry_neg[l+1]
            layer.rz_neg -= self.alpha * self.dc_drz_neg[l+1]

        network.particle_input.rx_pos -= self.alpha * self.dc_drx_pos[0]
        network.particle_input.ry_pos -= self.alpha * self.dc_dry_pos[0]
        network.particle_input.rz_pos -= self.alpha * self.dc_drz_pos[0]
        network.particle_input.rx_neg -= self.alpha * self.dc_drx_neg[0]
        network.particle_input.ry_neg -= self.alpha * self.dc_dry_neg[0]
        network.particle_input.rz_neg -= self.alpha * self.dc_drz_neg[0]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_ry_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_rz_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_rx_neg = [np.zeros(network.particle_input.output_size)]
            self.vel_ry_neg = [np.zeros(network.particle_input.output_size)]
            self.vel_rz_neg = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx_pos.append(np.zeros(layer.output_size))
                self.vel_ry_pos.append(np.zeros(layer.output_size))
                self.vel_rz_pos.append(np.zeros(layer.output_size))
                self.vel_rx_neg.append(np.zeros(layer.output_size))
                self.vel_ry_neg.append(np.zeros(layer.output_size))
                self.vel_rz_neg.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha_b * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha_q * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx_pos[l+1] = -self.alpha_r * self.dc_drx_pos[l+1] + self.beta * self.vel_rx_pos[l+1]
            self.vel_ry_pos[l+1] = -self.alpha_r * self.dc_dry_pos[l+1] + self.beta * self.vel_ry_pos[l+1]
            self.vel_rz_pos[l+1] = -self.alpha_r * self.dc_drz_pos[l+1] + self.beta * self.vel_rz_pos[l+1]
            self.vel_rx_neg[l+1] = -self.alpha_r * self.dc_drx_neg[l+1] + self.beta * self.vel_rx_neg[l+1]
            self.vel_ry_neg[l+1] = -self.alpha_r * self.dc_dry_neg[l+1] + self.beta * self.vel_ry_neg[l+1]
            self.vel_rz_neg[l+1] = -self.alpha_r * self.dc_drz_neg[l+1] + self.beta * self.vel_rz_neg[l+1]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.rx_pos += self.vel_rx_pos[l+1]
            layer.ry_pos += self.vel_ry_pos[l+1]
            layer.rz_pos += self.vel_rz_pos[l+1]
            layer.rx_neg += self.vel_rx_neg[l+1]
            layer.ry_neg += self.vel_ry_neg[l+1]
            layer.rz_neg += self.vel_rz_neg[l+1]

        self.vel_rx_pos[0] = -self.alpha_r * self.dc_drx_pos[0] + self.beta * self.vel_rx_pos[0]
        self.vel_ry_pos[0] = -self.alpha_r * self.dc_dry_pos[0] + self.beta * self.vel_ry_pos[0]
        self.vel_rz_pos[0] = -self.alpha_r * self.dc_drz_pos[0] + self.beta * self.vel_rz_pos[0]
        self.vel_rx_neg[0] = -self.alpha_r * self.dc_drx_neg[0] + self.beta * self.vel_rx_neg[0]
        self.vel_ry_neg[0] = -self.alpha_r * self.dc_dry_neg[0] + self.beta * self.vel_ry_neg[0]
        self.vel_rz_neg[0] = -self.alpha_r * self.dc_drz_neg[0] + self.beta * self.vel_rz_neg[0]
        network.particle_input.rx_pos += self.vel_rx_pos[0]
        network.particle_input.ry_pos += self.vel_ry_pos[0]
        network.particle_input.rz_pos += self.vel_rz_pos[0]
        network.particle_input.rx_neg += self.vel_rx_neg[0]
        network.particle_input.ry_neg += self.vel_ry_neg[0]
        network.particle_input.rz_neg += self.vel_rz_neg[0]

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to AdaGrad
        """
        epsilon = 10e-8

        # Initialize velocities/MSE to zero for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_ry_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_rz_pos = [np.zeros(network.particle_input.output_size)]
            self.vel_rx_neg = [np.zeros(network.particle_input.output_size)]
            self.vel_ry_neg = [np.zeros(network.particle_input.output_size)]
            self.vel_rz_neg = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx_pos.append(np.zeros(layer.output_size))
                self.vel_ry_pos.append(np.zeros(layer.output_size))
                self.vel_rz_pos.append(np.zeros(layer.output_size))
                self.vel_rx_neg.append(np.zeros(layer.output_size))
                self.vel_ry_neg.append(np.zeros(layer.output_size))
                self.vel_rz_neg.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] += self.dc_db[l]**2
            self.vel_q[l] += self.dc_dq[l]**2
            self.vel_rx_pos[l+1] += self.dc_drx_pos[l+1]**2
            self.vel_ry_pos[l+1] += self.dc_dry_pos[l+1]**2
            self.vel_rz_pos[l+1] += self.dc_drz_pos[l+1]**2
            self.vel_rx_neg[l+1] += self.dc_drx_neg[l+1]**2
            self.vel_ry_neg[l+1] += self.dc_dry_neg[l+1]**2
            self.vel_rz_neg[l+1] += self.dc_drz_neg[l+1]**2
            layer.b += -self.alpha * self.dc_db[l] / np.sqrt(self.vel_b[l] + epsilon)
            layer.q += -self.alpha * self.dc_dq[l] / np.sqrt(self.vel_q[l] + epsilon)
            layer.rx_pos += -self.alpha * self.dc_drx_pos[l+1] / np.sqrt(self.vel_rx_pos[l+1] + epsilon)
            layer.ry_pos += -self.alpha * self.dc_dry_pos[l+1] / np.sqrt(self.vel_ry_pos[l+1] + epsilon)
            layer.rz_pos += -self.alpha * self.dc_drz_pos[l+1] / np.sqrt(self.vel_rz_pos[l+1] + epsilon)
            layer.rx_neg += -self.alpha * self.dc_drx_neg[l+1] / np.sqrt(self.vel_rx_neg[l+1] + epsilon)
            layer.ry_neg += -self.alpha * self.dc_dry_neg[l+1] / np.sqrt(self.vel_ry_neg[l+1] + epsilon)
            layer.rz_neg += -self.alpha * self.dc_drz_neg[l+1] / np.sqrt(self.vel_rz_neg[l+1] + epsilon)

        layer = network.particle_input
        l = 0
        self.vel_rx_pos[l] += self.dc_drx_pos[l]**2
        self.vel_ry_pos[l] += self.dc_dry_pos[l]**2
        self.vel_rz_pos[l] += self.dc_drz_pos[l]**2
        self.vel_rx_neg[l] += self.dc_drx_neg[l]**2
        self.vel_ry_neg[l] += self.dc_dry_neg[l]**2
        self.vel_rz_neg[l] += self.dc_drz_neg[l]**2
        layer.rx_pos += -self.alpha * self.dc_drx_pos[l] / np.sqrt(self.vel_rx_pos[l] + epsilon)
        layer.ry_pos += -self.alpha * self.dc_dry_pos[l] / np.sqrt(self.vel_ry_pos[l] + epsilon)
        layer.rz_pos += -self.alpha * self.dc_drz_pos[l] / np.sqrt(self.vel_rz_pos[l] + epsilon)
        layer.rx_neg += -self.alpha * self.dc_drx_neg[l] / np.sqrt(self.vel_rx_neg[l] + epsilon)
        layer.ry_neg += -self.alpha * self.dc_dry_neg[l] / np.sqrt(self.vel_ry_neg[l] + epsilon)
        layer.rz_neg += -self.alpha * self.dc_drz_neg[l] / np.sqrt(self.vel_rz_neg[l] + epsilon)
