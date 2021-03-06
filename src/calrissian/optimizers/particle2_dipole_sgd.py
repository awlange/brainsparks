from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class Particle2DipoleSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Particle2 Dipole layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, gamma=0.9, alpha_decay=1.0, n_threads=1, chunk_size=1):
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

        self.alpha_decay = alpha_decay

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        if weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad
        if weight_update == "rmsprop":
            self.weight_update_func = self.weight_update_rmsprop

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

        self.n_threads = n_threads
        self.chunk_size = chunk_size

    def cost_gradient_parallel(self, network, data_X, data_Y):
        with Pool(processes=self.n_threads) as pool:
            offset = 0
            chunks = len(data_X) / self.chunk_size
            while offset < len(data_X):
                data_X_sub = data_X[offset:(offset+self.chunk_size), :]
                data_Y_sub = data_Y[offset:(offset+self.chunk_size), :]
                data_X_split = np.array_split(data_X_sub, self.n_threads)
                data_Y_split = np.array_split(data_Y_sub, self.n_threads)
                data_XY_list = [(data_X_split[i], data_Y_split[i], self.n_threads * chunks) for i in range(self.n_threads)]

                result = pool.map(network.cost_gradient_thread, data_XY_list)

                for t, gradients in enumerate(result):
                    if t == 0 and offset == 0:
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
                    else:
                        tmp_dc_db = gradients[0]
                        tmp_dc_dq = gradients[1]
                        tmp_dc_drx_pos_inp = gradients[2]
                        tmp_dc_dry_pos_inp = gradients[3]
                        tmp_dc_drz_pos_inp = gradients[4]
                        tmp_dc_drx_neg_inp = gradients[5]
                        tmp_dc_dry_neg_inp = gradients[6]
                        tmp_dc_drz_neg_inp = gradients[7]
                        tmp_dc_drx_pos_out = gradients[8]
                        tmp_dc_dry_pos_out = gradients[9]
                        tmp_dc_drz_pos_out = gradients[10]
                        tmp_dc_drx_neg_out = gradients[11]
                        tmp_dc_dry_neg_out = gradients[12]
                        tmp_dc_drz_neg_out = gradients[13]

                        for l, tmp_b in enumerate(tmp_dc_db): self.dc_db[l] += tmp_b
                        for l, tmp_q in enumerate(tmp_dc_dq): self.dc_dq[l] += tmp_q
                        for l, tmp in enumerate(tmp_dc_drx_pos_inp): self.dc_drx_pos_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_dry_pos_inp): self.dc_dry_pos_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drz_pos_inp): self.dc_drz_pos_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drx_neg_inp): self.dc_drx_neg_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_dry_neg_inp): self.dc_dry_neg_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drz_neg_inp): self.dc_drz_neg_inp[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drx_pos_out): self.dc_drx_pos_out[l] += tmp
                        for l, tmp in enumerate(tmp_dc_dry_pos_out): self.dc_dry_pos_out[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drz_pos_out): self.dc_drz_pos_out[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drx_neg_out): self.dc_drx_neg_out[l] += tmp
                        for l, tmp in enumerate(tmp_dc_dry_neg_out): self.dc_dry_neg_out[l] += tmp
                        for l, tmp in enumerate(tmp_dc_drz_neg_out): self.dc_drz_neg_out[l] += tmp

                offset += self.chunk_size

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
                if self.n_threads > 1:
                    self.cost_gradient_parallel(network, mini_X, mini_Y)
                else:
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

                # Decay
                self.alpha *= self.alpha_decay

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
            self.vel_b[l] = -self.alpha * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx_pos_inp[l] = -self.alpha * self.dc_drx_pos_inp[l] + self.beta * self.vel_rx_pos_inp[l]
            self.vel_ry_pos_inp[l] = -self.alpha * self.dc_dry_pos_inp[l] + self.beta * self.vel_ry_pos_inp[l]
            self.vel_rz_pos_inp[l] = -self.alpha * self.dc_drz_pos_inp[l] + self.beta * self.vel_rz_pos_inp[l]
            self.vel_rx_neg_inp[l] = -self.alpha * self.dc_drx_neg_inp[l] + self.beta * self.vel_rx_neg_inp[l]
            self.vel_ry_neg_inp[l] = -self.alpha * self.dc_dry_neg_inp[l] + self.beta * self.vel_ry_neg_inp[l]
            self.vel_rz_neg_inp[l] = -self.alpha * self.dc_drz_neg_inp[l] + self.beta * self.vel_rz_neg_inp[l]

            self.vel_rx_pos_out[l] = -self.alpha * self.dc_drx_pos_out[l] + self.beta * self.vel_rx_pos_out[l]
            self.vel_ry_pos_out[l] = -self.alpha * self.dc_dry_pos_out[l] + self.beta * self.vel_ry_pos_out[l]
            self.vel_rz_pos_out[l] = -self.alpha * self.dc_drz_pos_out[l] + self.beta * self.vel_rz_pos_out[l]
            self.vel_rx_neg_out[l] = -self.alpha * self.dc_drx_neg_out[l] + self.beta * self.vel_rx_neg_out[l]
            self.vel_ry_neg_out[l] = -self.alpha * self.dc_dry_neg_out[l] + self.beta * self.vel_ry_neg_out[l]
            self.vel_rz_neg_out[l] = -self.alpha * self.dc_drz_neg_out[l] + self.beta * self.vel_rz_neg_out[l]

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

    def weight_update_rmsprop(self, network):
        """
        Update weights and biases according to AdaGrad
        """
        epsilon = 10e-8
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma

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
            self.vel_b[l] = gamma * self.vel_b[l] + one_m_gamma * self.dc_db[l]**2
            self.vel_q[l] = gamma * self.vel_q[l] + one_m_gamma * self.dc_dq[l]**2

            self.vel_rx_pos_inp[l] = gamma * self.vel_rx_pos_inp[l] + one_m_gamma * self.dc_drx_pos_inp[l]**2
            self.vel_ry_pos_inp[l] = gamma * self.vel_ry_pos_inp[l] + one_m_gamma * self.dc_dry_pos_inp[l]**2
            self.vel_rz_pos_inp[l] = gamma * self.vel_rz_pos_inp[l] + one_m_gamma * self.dc_drz_pos_inp[l]**2
            self.vel_rx_neg_inp[l] = gamma * self.vel_rx_neg_inp[l] + one_m_gamma * self.dc_drx_neg_inp[l]**2
            self.vel_ry_neg_inp[l] = gamma * self.vel_ry_neg_inp[l] + one_m_gamma * self.dc_dry_neg_inp[l]**2
            self.vel_rz_neg_inp[l] = gamma * self.vel_rz_neg_inp[l] + one_m_gamma * self.dc_drz_neg_inp[l]**2

            self.vel_rx_pos_out[l] = gamma * self.vel_rx_pos_out[l] + one_m_gamma * self.dc_drx_pos_out[l]**2
            self.vel_ry_pos_out[l] = gamma * self.vel_ry_pos_out[l] + one_m_gamma * self.dc_dry_pos_out[l]**2
            self.vel_rz_pos_out[l] = gamma * self.vel_rz_pos_out[l] + one_m_gamma * self.dc_drz_pos_out[l]**2
            self.vel_rx_neg_out[l] = gamma * self.vel_rx_neg_out[l] + one_m_gamma * self.dc_drx_neg_out[l]**2
            self.vel_ry_neg_out[l] = gamma * self.vel_ry_neg_out[l] + one_m_gamma * self.dc_dry_neg_out[l]**2
            self.vel_rz_neg_out[l] = gamma * self.vel_rz_neg_out[l] + one_m_gamma * self.dc_drz_neg_out[l]**2

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

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to adagrad
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
            self.vel_b[l] += self.dc_db[l] ** 2
            self.vel_q[l] += self.dc_dq[l] ** 2

            self.vel_rx_pos_inp[l] += self.dc_drx_pos_inp[l] ** 2
            self.vel_ry_pos_inp[l] += self.dc_dry_pos_inp[l] ** 2
            self.vel_rz_pos_inp[l] += self.dc_drz_pos_inp[l] ** 2
            self.vel_rx_neg_inp[l] += self.dc_drx_neg_inp[l] ** 2
            self.vel_ry_neg_inp[l] += self.dc_dry_neg_inp[l] ** 2
            self.vel_rz_neg_inp[l] += self.dc_drz_neg_inp[l] ** 2

            self.vel_rx_pos_out[l] += self.dc_drx_pos_out[l] ** 2
            self.vel_ry_pos_out[l] += self.dc_dry_pos_out[l] ** 2
            self.vel_rz_pos_out[l] += self.dc_drz_pos_out[l] ** 2
            self.vel_rx_neg_out[l] += self.dc_drx_neg_out[l] ** 2
            self.vel_ry_neg_out[l] += self.dc_dry_neg_out[l] ** 2
            self.vel_rz_neg_out[l] += self.dc_drz_neg_out[l] ** 2

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

