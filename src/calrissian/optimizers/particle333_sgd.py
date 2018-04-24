from .optimizer import Optimizer

import numpy as np
import time
import sys

from multiprocessing import Pool


class Particle333SGD(Optimizer):
    """
    Stochastic gradient descent optimization
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, epsilon=10e-8, gamma2=0.1, alpha_decay=None, fixed_input=False,
                 n_threads=1, chunk_size=1):
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
        self.fixed_input = fixed_input

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
        self.dc_dz = None
        self.dc_dr_inp = None
        self.dc_dr_out = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_z = None
        self.vel_r_inp = None
        self.vel_r_out = None

        # Mean squares
        self.ms_db = None
        self.ms_dq = None
        self.ms_dz = None
        self.ms_dr_inp = None
        self.ms_dr_out = None

        self.ms_b = None
        self.ms_q = None
        self.ms_z = None
        self.ms_r_inp = None
        self.ms_r_out = None

        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.pool = None

    def cost_gradient_parallel(self, network, data_X, data_Y):
        if self.pool is None:
            self.pool = Pool(processes=self.n_threads)

        data_XY_list = []
        offset = 0
        while offset < len(data_X):
            data_X_sub = data_X[offset:(offset+self.chunk_size), :]
            data_Y_sub = data_Y[offset:(offset+self.chunk_size), :]
            data_XY_list.append((data_X_sub, data_Y_sub, self.n_threads))
            offset += self.chunk_size

        result = self.pool.map(network.cost_gradient_thread, data_XY_list, chunksize=1)

        for t, result_t in enumerate(result):
            tmp_dc_db = result_t[0]
            tmp_dc_dq = result_t[1]
            tmp_dc_dz = result_t[2]
            tmp_dc_dr_inp = result_t[3]
            tmp_dc_dr_out = result_t[4]

            if t == 0:
                self.dc_db = tmp_dc_db
                self.dc_dq = tmp_dc_dq
                self.dc_dz = tmp_dc_dz
                self.dc_dr_inp = tmp_dc_dr_inp
                self.dc_dr_out = tmp_dc_dr_out

            else:
                for l, tmp_b in enumerate(tmp_dc_db):
                    self.dc_db[l] += tmp_b
                for l, tmp_q in enumerate(tmp_dc_dq):
                    self.dc_dq[l] += tmp_q
                for l, tmp_z in enumerate(tmp_dc_dz):
                    self.dc_dz[l] += tmp_z
                for l, tmp_r in enumerate(tmp_dc_dr_inp):
                    self.dc_dr_inp[l] += tmp_r
                for l, tmp_r in enumerate(tmp_dc_dr_out):
                    self.dc_dr_out[l] += tmp_r

    def optimize(self, network, data_X, data_Y):
        """
        :return: optimized network
        """
        optimize_start_time = time.time()

        indexes = np.arange(len(data_X))

        if self.verbosity > 1:
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
                if self.n_threads > 1:
                    self.cost_gradient_parallel(network, mini_X, mini_Y)
                else:
                    self.dc_db, self.dc_dq, self.dc_dz, self.dc_dr_inp, self.dc_dr_out = network.cost_gradient(mini_X, mini_Y)
                
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

            if self.verbosity > 1:
                c = network.cost(data_X, data_Y)
                print("Cost after epoch {}: {:g}".format(epoch, c))
                print("Epoch time: {:g} s".format(time.time() - epoch_start_time))

        if self.verbosity > 1:
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
            layer.zeta -= self.alpha * self.dc_dz[l]
            layer.r_out -= self.alpha * self.dc_dr_out[l]
            if not (self.fixed_input and l == 0):
                layer.r_inp -= self.alpha * self.dc_dr_inp[l]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent with simple momentum
        """
        # Initialize RMS to zero
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_z = []
            self.vel_r_inp = []
            self.vel_r_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_z.append(np.zeros(layer.zeta.shape))
                self.vel_r_inp.append(np.zeros(layer.r_inp.shape))
                self.vel_r_out.append(np.zeros(layer.r_out.shape))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_z[l] = -self.alpha * self.dc_dz[l] + self.beta * self.vel_z[l]
            self.vel_r_inp[l] = -self.alpha * self.dc_dr_inp[l] + self.beta * self.vel_r_inp[l]
            self.vel_r_out[l] = -self.alpha * self.dc_dr_out[l] + self.beta * self.vel_r_out[l]

            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.zeta += self.vel_z[l]
            layer.r_out += self.vel_r_out[l]
            if not (self.fixed_input and l == 0):
                layer.r_inp += self.vel_r_inp[l]

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
            self.ms_dz = []
            self.ms_dr_inp = []
            self.ms_dr_out = []
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_dz.append(np.zeros(layer.zeta.shape))
                self.ms_dr_inp.append(np.zeros(layer.r_inp.shape))
                self.ms_dr_out.append(np.zeros(layer.r_out.shape))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma * self.ms_dq[l] + one_m_gamma * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dz[l] = gamma * self.ms_dz[l] + one_m_gamma * (self.dc_dz[l] * self.dc_dz[l])
            self.ms_dr_inp[l] = gamma * self.ms_dr_inp[l] + one_m_gamma * (self.dc_dr_inp[l] * self.dc_dr_inp[l])
            self.ms_dr_out[l] = gamma * self.ms_dr_out[l] + one_m_gamma * (self.dc_dr_out[l] * self.dc_dr_out[l])

            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.q -= alpha * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon)
            layer.zeta -= alpha * self.dc_dz[l] / np.sqrt(self.ms_dz[l] + epsilon)
            layer.r_out -= alpha * self.dc_dr_out[l] / np.sqrt(self.ms_dr_out[l] + epsilon)
            if not (self.fixed_input and l == 0):
                layer.r_inp -= alpha * self.dc_dr_inp[l] / np.sqrt(self.ms_dr_inp[l] + epsilon)

