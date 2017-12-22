from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleVectorSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Particle Vector networks
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True,
                 n_threads=1, chunk_size=10, epsilon=10e-8, gamma2=0.1,
                 alpha_decay=None,
                 fixed_input=False):
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
        self.fixed_input = fixed_input

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
        self.dc_dzeta = None

        # Mean squares
        self.ms_db = None
        self.ms_drx = None
        self.ms_dry = None
        self.ms_drz = None
        self.ms_drw = None
        self.ms_dnx = None
        self.ms_dny = None
        self.ms_dnz = None
        self.ms_dnw = None
        self.ms_dzeta = None

        self.vel_db = None
        self.vel_drx = None
        self.vel_dry = None
        self.vel_drz = None
        self.vel_drw = None
        self.vel_dnx = None
        self.vel_dny = None
        self.vel_dnz = None
        self.vel_dnw = None
        self.vel_dzeta = None

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
                    tmp_dc_dzeta = result_t[3]

                    if t == 0 and offset == 0:
                        self.dc_db = tmp_dc_db
                        self.dc_dr = tmp_dc_dr
                        self.dc_dn = tmp_dc_dn
                        self.dc_dzeta = tmp_dc_dzeta
                    else:
                        for l, tmp_b in enumerate(tmp_dc_db):
                            self.dc_db[l] += tmp_b
                        for i in range(4):
                            for l, tmp_r in enumerate(tmp_dc_dr[i]):
                                self.dc_dr[i][l] += tmp_r
                            for l, tmp_n in enumerate(tmp_dc_dn[i]):
                                self.dc_dn[i][l] += tmp_n
                        for l, tmp_z in enumerate(tmp_dc_dzeta):
                            self.dc_dzeta[l] += tmp_z

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
            layer.rx -= self.alpha * self.dc_dr[0][l+1]
            layer.ry -= self.alpha * self.dc_dr[1][l+1]
            layer.rz -= self.alpha * self.dc_dr[2][l+1]
            layer.rw -= self.alpha * self.dc_dr[3][l+1]
            layer.nx -= self.alpha * self.dc_dn[0][l+1]
            layer.ny -= self.alpha * self.dc_dn[1][l+1]
            layer.nz -= self.alpha * self.dc_dn[2][l+1]
            layer.nw -= self.alpha * self.dc_dn[3][l+1]
            layer.zeta -= self.alpha * self.dc_dzeta[l+1]

        if not self.fixed_input:
            network.particle_input.rx -= self.alpha * self.dc_dr[0][0]
            network.particle_input.ry -= self.alpha * self.dc_dr[1][0]
            network.particle_input.rz -= self.alpha * self.dc_dr[2][0]
            network.particle_input.rw -= self.alpha * self.dc_dr[3][0]
        network.particle_input.nx -= self.alpha * self.dc_dn[0][0]
        network.particle_input.ny -= self.alpha * self.dc_dn[1][0]
        network.particle_input.nz -= self.alpha * self.dc_dn[2][0]
        network.particle_input.nw -= self.alpha * self.dc_dn[3][0]
        network.particle_input.zeta -= self.alpha * self.dc_dzeta[0]

    def weight_update_momentum(self, network):
        if self.vel_db is None:
            self.vel_db = []
            self.vel_drx = [np.zeros(network.particle_input.output_size)]
            self.vel_dry = [np.zeros(network.particle_input.output_size)]
            self.vel_drz = [np.zeros(network.particle_input.output_size)]
            self.vel_drw = [np.zeros(network.particle_input.output_size)]
            self.vel_dnx = [np.zeros(network.particle_input.output_size)]
            self.vel_dny = [np.zeros(network.particle_input.output_size)]
            self.vel_dnz = [np.zeros(network.particle_input.output_size)]
            self.vel_dnw = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_db.append(np.zeros(layer.b.shape))
                self.vel_drx.append(np.zeros(layer.output_size))
                self.vel_dry.append(np.zeros(layer.output_size))
                self.vel_drz.append(np.zeros(layer.output_size))
                self.vel_drw.append(np.zeros(layer.output_size))
                self.vel_dnx.append(np.zeros(layer.output_size))
                self.vel_dny.append(np.zeros(layer.output_size))
                self.vel_dnz.append(np.zeros(layer.output_size))
                self.vel_dnw.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_db[l] *= self.beta
            self.vel_drx[l+1] *= self.beta
            self.vel_dry[l+1] *= self.beta
            self.vel_drz[l+1] *= self.beta
            self.vel_drw[l+1] *= self.beta
            self.vel_dnx[l+1] *= self.beta
            self.vel_dny[l+1] *= self.beta
            self.vel_dnz[l+1] *= self.beta
            self.vel_dnw[l+1] *= self.beta

            self.vel_db[l] -= self.alpha * self.dc_db[l]
            self.vel_drx[l+1] -= self.alpha * self.dc_dr[0][l+1]
            self.vel_dry[l+1] -= self.alpha * self.dc_dr[1][l+1]
            self.vel_drz[l+1] -= self.alpha * self.dc_dr[2][l+1]
            self.vel_drw[l+1] -= self.alpha * self.dc_dr[3][l+1]
            self.vel_dnx[l+1] -= self.alpha * self.dc_dn[0][l+1]
            self.vel_dny[l+1] -= self.alpha * self.dc_dn[1][l+1]
            self.vel_dnz[l+1] -= self.alpha * self.dc_dn[2][l+1]
            self.vel_dnw[l+1] -= self.alpha * self.dc_dn[3][l+1]

            layer.b += self.vel_db[l]
            layer.rx += self.vel_drx[l+1]
            layer.ry += self.vel_dry[l+1]
            layer.rz += self.vel_drz[l+1]
            layer.rw += self.vel_drw[l+1]
            layer.nx += self.vel_dnx[l+1]
            layer.ny += self.vel_dny[l+1]
            layer.nz += self.vel_dnz[l+1]
            layer.nw += self.vel_dnw[l+1]

        self.vel_drx[0] *= self.beta
        self.vel_dry[0] *= self.beta
        self.vel_drz[0] *= self.beta
        self.vel_drw[0] *= self.beta
        self.vel_dnx[0] *= self.beta
        self.vel_dny[0] *= self.beta
        self.vel_dnz[0] *= self.beta
        self.vel_dnw[0] *= self.beta

        self.vel_drx[0] -= self.alpha * self.dc_dr[0][0]
        self.vel_dry[0] -= self.alpha * self.dc_dr[1][0]
        self.vel_drz[0] -= self.alpha * self.dc_dr[2][0]
        self.vel_drw[0] -= self.alpha * self.dc_dr[3][0]
        self.vel_dnx[0] -= self.alpha * self.dc_dn[0][0]
        self.vel_dny[0] -= self.alpha * self.dc_dn[1][0]
        self.vel_dnz[0] -= self.alpha * self.dc_dn[2][0]
        self.vel_dnw[0] -= self.alpha * self.dc_dn[3][0]

        if not self.fixed_input:
            network.particle_input.rx += self.vel_drx[0]
            network.particle_input.ry += self.vel_dry[0]
            network.particle_input.rz += self.vel_drz[0]
            network.particle_input.rw += self.vel_drw[0]
        network.particle_input.nx += self.vel_dnx[0]
        network.particle_input.ny += self.vel_dny[0]
        network.particle_input.nz += self.vel_dnz[0]
        network.particle_input.nw += self.vel_dnw[0]

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
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_drw = [np.zeros(network.particle_input.output_size)]
            self.ms_dnx = [np.zeros(network.particle_input.output_size)]
            self.ms_dny = [np.zeros(network.particle_input.output_size)]
            self.ms_dnz = [np.zeros(network.particle_input.output_size)]
            self.ms_dnw = [np.zeros(network.particle_input.output_size)]
            self.ms_dzeta = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_drw.append(np.zeros(layer.output_size))
                self.ms_dnx.append(np.zeros(layer.output_size))
                self.ms_dny.append(np.zeros(layer.output_size))
                self.ms_dnz.append(np.zeros(layer.output_size))
                self.ms_dnw.append(np.zeros(layer.output_size))
                self.ms_dzeta.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_drx[l + 1] = gamma * self.ms_drx[l + 1] + one_m_gamma * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] = gamma * self.ms_dry[l + 1] + one_m_gamma * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] = gamma * self.ms_drz[l + 1] + one_m_gamma * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])
            self.ms_drw[l + 1] = gamma * self.ms_drw[l + 1] + one_m_gamma * (self.dc_dr[3][l + 1] * self.dc_dr[3][l + 1])
            self.ms_dnx[l + 1] = gamma * self.ms_dnx[l + 1] + one_m_gamma * (self.dc_dn[0][l + 1] * self.dc_dn[0][l + 1])
            self.ms_dny[l + 1] = gamma * self.ms_dny[l + 1] + one_m_gamma * (self.dc_dn[1][l + 1] * self.dc_dn[1][l + 1])
            self.ms_dnz[l + 1] = gamma * self.ms_dnz[l + 1] + one_m_gamma * (self.dc_dn[2][l + 1] * self.dc_dn[2][l + 1])
            self.ms_dnw[l + 1] = gamma * self.ms_dnw[l + 1] + one_m_gamma * (self.dc_dn[3][l + 1] * self.dc_dn[3][l + 1])
            self.ms_dzeta[l + 1] = gamma * self.ms_dzeta[l + 1] + one_m_gamma * (self.dc_dzeta[l + 1] * self.dc_dzeta[l + 1])

            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.rx -= alpha * self.dc_dr[0][l+1] / np.sqrt(self.ms_drx[l + 1] + epsilon)
            layer.ry -= alpha * self.dc_dr[1][l+1] / np.sqrt(self.ms_dry[l + 1] + epsilon)
            # layer.rz -= alpha * self.dc_dr[2][l+1] / np.sqrt(self.ms_drz[l + 1] + epsilon)
            # layer.rw -= alpha * self.dc_dr[3][l+1] / np.sqrt(self.ms_drw[l + 1] + epsilon)
            layer.nx -= alpha * self.dc_dn[0][l + 1] / np.sqrt(self.ms_dnx[l + 1] + epsilon)
            layer.ny -= alpha * self.dc_dn[1][l + 1] / np.sqrt(self.ms_dny[l + 1] + epsilon)
            # layer.nz -= alpha * self.dc_dn[2][l + 1] / np.sqrt(self.ms_dnz[l + 1] + epsilon)
            # layer.nw -= alpha * self.dc_dn[3][l + 1] / np.sqrt(self.ms_dnw[l + 1] + epsilon)
            layer.zeta -= alpha * self.dc_dzeta[l + 1] / np.sqrt(self.ms_dzeta[l + 1] + epsilon)

        # Input layer
        self.ms_drx[0] = gamma * self.ms_drx[0] + one_m_gamma * (self.dc_dr[0][0] * self.dc_dr[0][0])
        self.ms_dry[0] = gamma * self.ms_dry[0] + one_m_gamma * (self.dc_dr[1][0] * self.dc_dr[1][0])
        self.ms_drz[0] = gamma * self.ms_drz[0] + one_m_gamma * (self.dc_dr[2][0] * self.dc_dr[2][0])
        self.ms_drw[0] = gamma * self.ms_drw[0] + one_m_gamma * (self.dc_dr[3][0] * self.dc_dr[3][0])
        self.ms_dnx[0] = gamma * self.ms_dnx[0] + one_m_gamma * (self.dc_dn[0][0] * self.dc_dn[0][0])
        self.ms_dny[0] = gamma * self.ms_dny[0] + one_m_gamma * (self.dc_dn[1][0] * self.dc_dn[1][0])
        self.ms_dnz[0] = gamma * self.ms_dnz[0] + one_m_gamma * (self.dc_dn[2][0] * self.dc_dn[2][0])
        self.ms_dnw[0] = gamma * self.ms_dnw[0] + one_m_gamma * (self.dc_dn[3][0] * self.dc_dn[3][0])
        self.ms_dzeta[0] = gamma * self.ms_dzeta[0] + one_m_gamma * (self.dc_dzeta[0] * self.dc_dzeta[0])
        if not self.fixed_input:
            network.particle_input.rx -= alpha * self.dc_dr[0][0] / np.sqrt(self.ms_drx[0] + epsilon)
            network.particle_input.ry -= alpha * self.dc_dr[1][0] / np.sqrt(self.ms_dry[0] + epsilon)
            # network.particle_input.rz -= alpha * self.dc_dr[2][0] / np.sqrt(self.ms_drz[0] + epsilon)
            # network.particle_input.rw -= alpha * self.dc_dr[3][0] / np.sqrt(self.ms_drw[0] + epsilon)
        network.particle_input.nx -= alpha * self.dc_dn[0][0] / np.sqrt(self.ms_dnx[0] + epsilon)
        network.particle_input.ny -= alpha * self.dc_dn[1][0] / np.sqrt(self.ms_dny[0] + epsilon)
        # network.particle_input.nz -= alpha * self.dc_dn[2][0] / np.sqrt(self.ms_dnz[0] + epsilon)
        # network.particle_input.nw -= alpha * self.dc_dn[3][0] / np.sqrt(self.ms_dnw[0] + epsilon)
        network.particle_input.zeta -= alpha * self.dc_dzeta[0] / np.sqrt(self.ms_dzeta[0] + epsilon)

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to Adagrad
        """
        alpha = self.alpha
        # epsilon = self.epsilon  # small number to avoid division by zero
        #
        # # Initialize RMS to zero
        # if self.ms_db is None or self.ms_dq is None:
        #     self.ms_db = []
        #     self.ms_dq = []
        #     self.ms_drx = [np.zeros(network.particle_input.output_size)]
        #     self.ms_dry = [np.zeros(network.particle_input.output_size)]
        #     self.ms_drz = [np.zeros(network.particle_input.output_size)]
        #     self.ms_dnx = [np.zeros(network.particle_input.output_size)]
        #     self.ms_dny = [np.zeros(network.particle_input.output_size)]
        #     self.ms_dnz = [np.zeros(network.particle_input.output_size)]
        #     for l, layer in enumerate(network.layers):
        #         self.ms_db.append(np.zeros(layer.b.shape))
        #         self.ms_dq.append(np.zeros(layer.q.shape))
        #         self.ms_drx.append(np.zeros(layer.output_size))
        #         self.ms_dry.append(np.zeros(layer.output_size))
        #         self.ms_drz.append(np.zeros(layer.output_size))
        #         self.ms_dnx.append(np.zeros(layer.output_size))
        #         self.ms_dny.append(np.zeros(layer.output_size))
        #         self.ms_dnz.append(np.zeros(layer.output_size))
        #
        # for l, layer in enumerate(network.layers):
        #     self.ms_db[l] += (self.dc_db[l] * self.dc_db[l])
        #     self.ms_dq[l] += (self.dc_dq[l] * self.dc_dq[l])
        #     self.ms_drx[l + 1] += (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
        #     self.ms_dry[l + 1] += (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
        #     self.ms_drz[l + 1] += (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])
        #     self.ms_dnx[l + 1] += (self.dc_dn[0][l + 1] * self.dc_dn[0][l + 1])
        #     self.ms_dny[l + 1] += (self.dc_dn[1][l + 1] * self.dc_dn[1][l + 1])
        #     self.ms_dnz[l + 1] += (self.dc_dn[2][l + 1] * self.dc_dn[2][l + 1])
        #
        #     layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
        #     # layer.q -= self.alpha_q * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon)
        #     layer.rx -= alpha * self.dc_dr[0][l + 1] / np.sqrt(self.ms_drx[l + 1] + epsilon)
        #     layer.ry -= alpha * self.dc_dr[1][l + 1] / np.sqrt(self.ms_dry[l + 1] + epsilon)
        #     layer.rz -= alpha * self.dc_dr[2][l + 1] / np.sqrt(self.ms_drz[l + 1] + epsilon)
        #     layer.nx -= alpha * self.dc_dn[0][l + 1] / np.sqrt(self.ms_dnx[l + 1] + epsilon)
        #     layer.ny -= alpha * self.dc_dn[1][l + 1] / np.sqrt(self.ms_dny[l + 1] + epsilon)
        #     layer.nz -= alpha * self.dc_dn[2][l + 1] / np.sqrt(self.ms_dnz[l + 1] + epsilon)
        #
        # # Input layer
        # self.ms_drx[0] += (self.dc_dr[0][0] * self.dc_dr[0][0])
        # self.ms_dry[0] += (self.dc_dr[1][0] * self.dc_dr[1][0])
        # self.ms_drz[0] += (self.dc_dr[2][0] * self.dc_dr[2][0])
        # self.ms_dnx[0] += (self.dc_dn[0][0] * self.dc_dn[0][0])
        # self.ms_dny[0] += (self.dc_dn[1][0] * self.dc_dn[1][0])
        # self.ms_dnz[0] += (self.dc_dn[2][0] * self.dc_dn[2][0])
        # network.particle_input.rx -= alpha * self.dc_dr[0][0] / np.sqrt(self.ms_drx[0] + epsilon)
        # network.particle_input.ry -= alpha * self.dc_dr[1][0] / np.sqrt(self.ms_dry[0] + epsilon)
        # network.particle_input.rz -= alpha * self.dc_dr[2][0] / np.sqrt(self.ms_drz[0] + epsilon)
        # network.particle_input.nx -= alpha * self.dc_dn[0][0] / np.sqrt(self.ms_dnx[0] + epsilon)
        # network.particle_input.ny -= alpha * self.dc_dn[1][0] / np.sqrt(self.ms_dny[0] + epsilon)
        # network.particle_input.nz -= alpha * self.dc_dn[2][0] / np.sqrt(self.ms_dnz[0] + epsilon)

