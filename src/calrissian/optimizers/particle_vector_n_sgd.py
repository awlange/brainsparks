from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleVectorNSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Particle Vector networks
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=None, alpha_r=0.01, alpha_t=0.01, init_v=0.0,
                 n_threads=1, chunk_size=10, epsilon=10e-8, gamma2=0.1, alpha_decay=None, use_log=False):
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
        self.position_grad = position_grad  # Turn off position gradient?
        self.alpha_decay = alpha_decay

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "rmsprop":
            self.weight_update_func = self.weight_update_rmsprop
        if weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_momentum

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dr = None
        self.dc_dn = None

        # Mean squares
        self.ms_db = None
        self.ms_dr = None
        self.ms_dn = None

        self.vel_db = None
        self.vel_dr = None
        self.vel_dn = None

        self.t = 0
        self.gamma_t = 0.0
        self.gamma2_t = 0.0

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

                for t, result_t in enumerate(result):
                    tmp_dc_db = result_t[0]
                    tmp_dc_dr = result_t[1]
                    tmp_dc_dn = result_t[2]

                    if t == 0 and offset == 0:
                        self.dc_db = tmp_dc_db
                        self.dc_dr = tmp_dc_dr
                        self.dc_dn = tmp_dc_dn
                    else:
                        for l, tmp_b in enumerate(tmp_dc_db):
                            self.dc_db[l] += tmp_b
                        for l, tmp_r in enumerate(tmp_dc_dr):
                            for r in range(len(tmp_r)):
                                self.dc_dr[l][r] += tmp_r[r]
                        for l, tmp_n in enumerate(tmp_dc_dn):
                            for v in range(len(tmp_n)):
                                self.dc_dn[l][v] += tmp_n[v]

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
                    self.dc_db, self.dc_dr, self.dc_dn = network.cost_gradient(mini_X, mini_Y)

                # Update weights and biases
                self.weight_update_func(network)

                # Alpha decay
                if self.alpha_decay is not None:
                    self.alpha *= 1.0 - self.alpha_decay

                if self.verbosity > 1 and m % self.cost_freq == 0:
                    c = network.cost(data_X, data_Y)
                    print("Cost at epoch {} mini-batch {}: {:g}".format(epoch, m, c))
                    # TODO: could output projected time left based on mini-batch times

                    # Temporary
                    # print(np.sqrt(self.ms_dq[0]))

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
            for r in range(layer.nr):
                layer.positions[r] -= self.alpha * self.dc_dr[l+1][r]
            for v in range(layer.nv):
                layer.nvectors[v] -= self.alpha * self.dc_dn[l+1][v]

        layer = network.particle_input
        l = -1
        for r in range(layer.nr):
            layer.positions[r] -= self.alpha * self.dc_dr[l+1][r]
        for v in range(layer.nv):
            layer.nvectors[v] -= self.alpha * self.dc_dn[l+1][v]

    def weight_update_momentum(self, network):
        pass

    def weight_update_rmsprop(self, network):
        """
        Update weights and biases according to RMSProp
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        alpha = self.alpha
        epsilon = self.epsilon  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None:
            self.ms_db = []
            self.ms_dr = [[np.zeros(network.particle_input.output_size) for _ in range(network.particle_input.nr)]]
            self.ms_dn = [[np.zeros(network.particle_input.output_size) for _ in range(network.particle_input.nv)]]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dr.append([np.zeros(layer.output_size) for _ in range(layer.nr)])
                self.ms_dn.append([np.zeros(layer.output_size) for _ in range(layer.nv)])

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)

            for r in range(layer.nr):
                self.ms_dr[l + 1][r] = gamma * self.ms_dr[l + 1][r] + one_m_gamma * (self.dc_dr[l + 1][r] * self.dc_dr[l + 1][r])
                layer.positions[r] -= alpha * self.dc_dr[l + 1][r] / np.sqrt(self.ms_dr[l + 1][r] + epsilon)

            for v in range(layer.nv):
                self.ms_dn[l + 1][v] = gamma * self.ms_dn[l + 1][v] + one_m_gamma * (self.dc_dn[l + 1][v] * self.dc_dn[l + 1][v])
                layer.nvectors[v] -= alpha * self.dc_dn[l + 1][v] / np.sqrt(self.ms_dn[l + 1][v] + epsilon)

        # Input layer
        layer = network.particle_input
        l = -1
        for r in range(layer.nr):
            self.ms_dr[l + 1][r] = gamma * self.ms_dr[l + 1][r] + one_m_gamma * (
            self.dc_dr[l + 1][r] * self.dc_dr[l + 1][r])
            layer.positions[r] -= alpha * self.dc_dr[l + 1][r] / np.sqrt(self.ms_dr[l + 1][r] + epsilon)

        for v in range(layer.nv):
            self.ms_dn[l + 1][v] = gamma * self.ms_dn[l + 1][v] + one_m_gamma * (
            self.dc_dn[l + 1][v] * self.dc_dn[l + 1][v])
            layer.nvectors[v] -= alpha * self.dc_dn[l + 1][v] / np.sqrt(self.ms_dn[l + 1][v] + epsilon)

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to Adagrad
        """
        pass

