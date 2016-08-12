from .optimizer import Optimizer

import numpy as np
import time


class Particle2DipoleSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Particle2 Dipole layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=0.01, alpha_r=0.01, gamma=0.9):
        """
        :param alpha: learning rate
        :param beta: momentum damping (viscosity)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
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
        self.dc_drx_pos_inp = None
        self.dc_dry_pos_inp = None
        self.dc_drz_pos_inp = None
        self.dc_drx_neg_inp = None
        self.dc_dry_neg_inp = None
        self.dc_drz_neg_inp = None
        self.dc_drx_pos_out = None
        self.dc_dry_pos_out = None
        self.dc_drz_pos_out = None
        self.dc_drx_neg_out = None
        self.dc_dry_neg_out = None
        self.dc_drz_neg_out = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_rx_pos_inp = None
        self.vel_ry_pos_inp = None
        self.vel_rz_pos_inp = None
        self.vel_rx_neg_inp = None
        self.vel_ry_neg_inp = None
        self.vel_rz_neg_inp = None
        self.vel_rx_pos_out = None
        self.vel_ry_pos_out = None
        self.vel_rz_pos_out = None
        self.vel_rx_neg_out = None
        self.vel_ry_neg_out = None
        self.vel_rz_neg_out = None

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
                self.dc_drx_pos_inp = gradients[2]
                self.dc_dry_pos_inp = gradients[3]
                self.dc_drz_pos_inp = gradients[4]
                self.dc_drx_neg_inp = gradients[5]
                self.dc_dry_neg_inp = gradients[6]
                self.dc_drz_neg_inp = gradients[7]
                self.dc_drx_pos_out = gradients[8]
                self.dc_dry_pos_out = gradients[9]
                self.dc_drz_pos_out = gradients[10]
                self.dc_drx_neg_out = gradients[11]
                self.dc_dry_neg_out = gradients[12]
                self.dc_drz_neg_out = gradients[13]

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
            layer.rx_pos_inp -= self.alpha * self.dc_drx_pos_inp[l]
            layer.ry_pos_inp -= self.alpha * self.dc_dry_pos_inp[l]
            layer.rz_pos_inp -= self.alpha * self.dc_drz_pos_inp[l]
            layer.rx_neg_inp -= self.alpha * self.dc_drx_neg_inp[l]
            layer.ry_neg_inp -= self.alpha * self.dc_dry_neg_inp[l]
            layer.rz_neg_inp -= self.alpha * self.dc_drz_neg_inp[l]
            layer.rx_pos_out -= self.alpha * self.dc_drx_pos_out[l]
            layer.ry_pos_out -= self.alpha * self.dc_dry_pos_out[l]
            layer.rz_pos_out -= self.alpha * self.dc_drz_pos_out[l]
            layer.rx_neg_out -= self.alpha * self.dc_drx_neg_out[l]
            layer.ry_neg_out -= self.alpha * self.dc_dry_neg_out[l]
            layer.rz_neg_out -= self.alpha * self.dc_drz_neg_out[l]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_pos_inp = []
            self.vel_ry_pos_inp = []
            self.vel_rz_pos_inp = []
            self.vel_rx_neg_inp = []
            self.vel_ry_neg_inp = []
            self.vel_rz_neg_inp = []
            self.vel_rx_pos_out = []
            self.vel_ry_pos_out = []
            self.vel_rz_pos_out = []
            self.vel_rx_neg_out = []
            self.vel_ry_neg_out = []
            self.vel_rz_neg_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx_pos_inp.append(np.zeros(layer.input_size))
                self.vel_ry_pos_inp.append(np.zeros(layer.input_size))
                self.vel_rz_pos_inp.append(np.zeros(layer.input_size))
                self.vel_rx_neg_inp.append(np.zeros(layer.input_size))
                self.vel_ry_neg_inp.append(np.zeros(layer.input_size))
                self.vel_rz_neg_inp.append(np.zeros(layer.input_size))
                self.vel_rx_pos_out.append(np.zeros(layer.output_size))
                self.vel_ry_pos_out.append(np.zeros(layer.output_size))
                self.vel_rz_pos_out.append(np.zeros(layer.output_size))
                self.vel_rx_neg_out.append(np.zeros(layer.output_size))
                self.vel_ry_neg_out.append(np.zeros(layer.output_size))
                self.vel_rz_neg_out.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha_b * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha_q * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx_pos_inp[l] = -self.alpha_r * self.dc_drx_pos_inp[l] + self.beta * self.vel_rx_pos_inp[l]
            self.vel_ry_pos_inp[l] = -self.alpha_r * self.dc_dry_pos_inp[l] + self.beta * self.vel_ry_pos_inp[l]
            self.vel_rz_pos_inp[l] = -self.alpha_r * self.dc_drz_pos_inp[l] + self.beta * self.vel_rz_pos_inp[l]
            self.vel_rx_neg_inp[l] = -self.alpha_r * self.dc_drx_neg_inp[l] + self.beta * self.vel_rx_neg_inp[l]
            self.vel_ry_neg_inp[l] = -self.alpha_r * self.dc_dry_neg_inp[l] + self.beta * self.vel_ry_neg_inp[l]
            self.vel_rz_neg_inp[l] = -self.alpha_r * self.dc_drz_neg_inp[l] + self.beta * self.vel_rz_neg_inp[l]

            self.vel_rx_pos_out[l] = -self.alpha_r * self.dc_drx_pos_out[l] + self.beta * self.vel_rx_pos_out[l]
            self.vel_ry_pos_out[l] = -self.alpha_r * self.dc_dry_pos_out[l] + self.beta * self.vel_ry_pos_out[l]
            self.vel_rz_pos_out[l] = -self.alpha_r * self.dc_drz_pos_out[l] + self.beta * self.vel_rz_pos_out[l]
            self.vel_rx_neg_out[l] = -self.alpha_r * self.dc_drx_neg_out[l] + self.beta * self.vel_rx_neg_out[l]
            self.vel_ry_neg_out[l] = -self.alpha_r * self.dc_dry_neg_out[l] + self.beta * self.vel_ry_neg_out[l]
            self.vel_rz_neg_out[l] = -self.alpha_r * self.dc_drz_neg_out[l] + self.beta * self.vel_rz_neg_out[l]

            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]

            layer.rx_pos_inp += self.vel_rx_pos_inp[l]
            layer.ry_pos_inp += self.vel_ry_pos_inp[l]
            layer.rz_pos_inp += self.vel_rz_pos_inp[l]
            layer.rx_neg_inp += self.vel_rx_neg_inp[l]
            layer.ry_neg_inp += self.vel_ry_neg_inp[l]
            layer.rz_neg_inp += self.vel_rz_neg_inp[l]

            layer.rx_pos_out += self.vel_rx_pos_out[l]
            layer.ry_pos_out += self.vel_ry_pos_out[l]
            layer.rz_pos_out += self.vel_rz_pos_out[l]
            layer.rx_neg_out += self.vel_rx_neg_out[l]
            layer.ry_neg_out += self.vel_ry_neg_out[l]
            layer.rz_neg_out += self.vel_rz_neg_out[l]

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to AdaGrad
        """
        epsilon = 10e-8

        # Initialize velocities/MSE to zero for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_pos_inp = []
            self.vel_ry_pos_inp = []
            self.vel_rz_pos_inp = []
            self.vel_rx_neg_inp = []
            self.vel_ry_neg_inp = []
            self.vel_rz_neg_inp = []
            self.vel_rx_pos_out = []
            self.vel_ry_pos_out = []
            self.vel_rz_pos_out = []
            self.vel_rx_neg_out = []
            self.vel_ry_neg_out = []
            self.vel_rz_neg_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx_pos_inp.append(np.zeros(layer.input_size))
                self.vel_ry_pos_inp.append(np.zeros(layer.input_size))
                self.vel_rz_pos_inp.append(np.zeros(layer.input_size))
                self.vel_rx_neg_inp.append(np.zeros(layer.input_size))
                self.vel_ry_neg_inp.append(np.zeros(layer.input_size))
                self.vel_rz_neg_inp.append(np.zeros(layer.input_size))
                self.vel_rx_pos_out.append(np.zeros(layer.output_size))
                self.vel_ry_pos_out.append(np.zeros(layer.output_size))
                self.vel_rz_pos_out.append(np.zeros(layer.output_size))
                self.vel_rx_neg_out.append(np.zeros(layer.output_size))
                self.vel_ry_neg_out.append(np.zeros(layer.output_size))
                self.vel_rz_neg_out.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] += self.dc_db[l]**2
            self.vel_q[l] += self.dc_dq[l]**2

            self.vel_rx_pos_inp[l] += self.dc_drx_pos_inp[l]**2
            self.vel_ry_pos_inp[l] += self.dc_dry_pos_inp[l]**2
            self.vel_rz_pos_inp[l] += self.dc_drz_pos_inp[l]**2
            self.vel_rx_neg_inp[l] += self.dc_drx_neg_inp[l]**2
            self.vel_ry_neg_inp[l] += self.dc_dry_neg_inp[l]**2
            self.vel_rz_neg_inp[l] += self.dc_drz_neg_inp[l]**2

            self.vel_rx_pos_out[l] += self.dc_drx_pos_out[l]**2
            self.vel_ry_pos_out[l] += self.dc_dry_pos_out[l]**2
            self.vel_rz_pos_out[l] += self.dc_drz_pos_out[l]**2
            self.vel_rx_neg_out[l] += self.dc_drx_neg_out[l]**2
            self.vel_ry_neg_out[l] += self.dc_dry_neg_out[l]**2
            self.vel_rz_neg_out[l] += self.dc_drz_neg_out[l]**2

            layer.b += -self.alpha * self.dc_db[l] / np.sqrt(self.vel_b[l] + epsilon)
            layer.q += -self.alpha * self.dc_dq[l] / np.sqrt(self.vel_q[l] + epsilon)

            layer.rx_pos_inp += -self.alpha * self.dc_drx_pos_inp[l] / np.sqrt(self.vel_rx_pos_inp[l] + epsilon)
            layer.ry_pos_inp += -self.alpha * self.dc_dry_pos_inp[l] / np.sqrt(self.vel_ry_pos_inp[l] + epsilon)
            layer.rz_pos_inp += -self.alpha * self.dc_drz_pos_inp[l] / np.sqrt(self.vel_rz_pos_inp[l] + epsilon)
            layer.rx_neg_inp += -self.alpha * self.dc_drx_neg_inp[l] / np.sqrt(self.vel_rx_neg_inp[l] + epsilon)
            layer.ry_neg_inp += -self.alpha * self.dc_dry_neg_inp[l] / np.sqrt(self.vel_ry_neg_inp[l] + epsilon)
            layer.rz_neg_inp += -self.alpha * self.dc_drz_neg_inp[l] / np.sqrt(self.vel_rz_neg_inp[l] + epsilon)

            layer.rx_pos_out += -self.alpha * self.dc_drx_pos_out[l] / np.sqrt(self.vel_rx_pos_out[l] + epsilon)
            layer.ry_pos_out += -self.alpha * self.dc_dry_pos_out[l] / np.sqrt(self.vel_ry_pos_out[l] + epsilon)
            layer.rz_pos_out += -self.alpha * self.dc_drz_pos_out[l] / np.sqrt(self.vel_rz_pos_out[l] + epsilon)
            layer.rx_neg_out += -self.alpha * self.dc_drx_neg_out[l] / np.sqrt(self.vel_rx_neg_out[l] + epsilon)
            layer.ry_neg_out += -self.alpha * self.dc_dry_neg_out[l] / np.sqrt(self.vel_ry_neg_out[l] + epsilon)
            layer.rz_neg_out += -self.alpha * self.dc_drz_neg_out[l] / np.sqrt(self.vel_rz_neg_out[l] + epsilon)

