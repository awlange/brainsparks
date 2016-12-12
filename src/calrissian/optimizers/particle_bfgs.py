from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleBFGS(Optimizer):
    """
    BFGS
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=None, alpha_r=0.01, alpha_t=0.01, init_v=0.0,
                 n_threads=1, chunk_size=10, epsilon=10e-8, gamma2=0.1, alpha_decay=None, use_log=False):
        """
        :param alpha: learning rate
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

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_dt = None
        self.dc_dzeta = None

        # BFGS stuff
        self.s_b = None
        self.s_q = None
        self.s_rx = None
        self.s_ry = None
        self.s_rz = None
        self.s_t = None
        self.y_b = None
        self.y_q = None
        self.y_rx = None
        self.y_ry = None
        self.y_rz = None
        self.y_t = None
        self.p_b = None
        self.p_q = None
        self.p_rx = None
        self.p_ry = None
        self.p_rz = None
        self.p_t = None
        self.Bk_inv = None
        self.total = 0

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
                    tmp_dc_dq = result_t[1]
                    tmp_dc_dr = result_t[2]
                    tmp_dc_dt = result_t[3]
                    tmp_dc_dzeta = result_t[4]

                    if t == 0 and offset == 0:
                        self.dc_db = tmp_dc_db
                        self.dc_dq = tmp_dc_dq
                        self.dc_dr = tmp_dc_dr
                        self.dc_dt = tmp_dc_dt
                        self.dc_dzeta = tmp_dc_dzeta
                    else:
                        for l, tmp_b in enumerate(tmp_dc_db):
                            self.dc_db[l] += tmp_b
                        for l, tmp_q in enumerate(tmp_dc_dq):
                            self.dc_dq[l] += tmp_q
                        for l, tmp_t in enumerate(tmp_dc_dt):
                            self.dc_dt[l] += tmp_t
                        for l, tmp_z in enumerate(tmp_dc_dzeta):
                            self.dc_dzeta[l] += tmp_z
                        for i in range(3):
                            for l, tmp_r in enumerate(tmp_dc_dr[i]):
                                self.dc_dr[i][l] += tmp_r

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
                    self.dc_db, self.dc_dq, self.dc_dr, self.dc_dt = network.cost_gradient(mini_X, mini_Y)

                # Update weights and biases
                self.weight_update(network)

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

    def flatten(self, network, b, q, rx, ry, rz, t):
        flat = np.zeros(self.total)

        j = 0
        for i in range(network.particle_input.output_size):
            flat[j+i] = rx[0][i]
            j += 1
        for i in range(network.particle_input.output_size):
            flat[j+i] = ry[0][i]
            j += 1
        for i in range(network.particle_input.output_size):
            flat[j+i] = rz[0][i]
            j += 1
        for i in range(network.particle_input.output_size):
            flat[j+i] = t[0][i]
            j += 1
        
        for l, layer in enumerate(network.layers):
            for i in range(layer.output_size):
                flat[j+i] = rx[l+1][i]
                j += 1
            for i in range(layer.output_size):
                flat[j+i] = ry[l+1][i]
                j += 1
            for i in range(layer.output_size):
                flat[j+i] = rz[l+1][i]
                j += 1
            for i in range(layer.output_size):
                flat[j+i] = t[l+1][i]
                j += 1
            for i in range(layer.output_size):
                flat[j+i] = q[l][i]
                j += 1
            for i in range(layer.output_size):
                flat[j+i] = b[l][0][i]
                j += 1
                
        return flat

    def unflatten(self, network, flat):
        b = []
        q = []
        rx = [np.zeros(network.particle_input.output_size)]
        ry = [np.zeros(network.particle_input.output_size)]
        rz = [np.zeros(network.particle_input.output_size)]
        t = [np.zeros(network.particle_input.output_size)]
        for l, layer in enumerate(network.layers):
            b.append(np.zeros(layer.b.output_size))
            q.append(np.zeros(layer.q.output_size))
            rx.append(np.zeros(layer.output_size))
            ry.append(np.zeros(layer.output_size))
            rz.append(np.zeros(layer.output_size))
            t.append(np.zeros(layer.output_size))

        j = 0
        for i in range(network.particle_input.output_size):
            rx[0][i] = flat[j + i]
            j += 1
        for i in range(network.particle_input.output_size):
            ry[0][i] = flat[j + i]
            j += 1
        for i in range(network.particle_input.output_size):
            rz[0][i] = flat[j + i]
            j += 1
        for i in range(network.particle_input.output_size):
            t[0][i] = flat[j + i]
            j += 1

        for l, layer in enumerate(network.layers):
            for i in range(layer.output_size):
                rx[l + 1][i] = flat[j + i]
                j += 1
            for i in range(layer.output_size):
                ry[l + 1][i] = flat[j + i]
                j += 1
            for i in range(layer.output_size):
                rz[l + 1][i] = flat[j + i]
                j += 1
            for i in range(layer.output_size):
                t[l + 1][i] = flat[j + i]
                j += 1
            for i in range(layer.output_size):
                q[l][i] = flat[j + i]
                j += 1
            for i in range(layer.output_size):
                b[l][0][i] = flat[j + i]
                j += 1

        return b, q, rx, ry, rz, t

    def weight_update(self, network):
        self.s_b = None
        self.s_q = None
        self.s_rx = None
        self.s_ry = None
        self.s_rz = None
        self.s_t = None
        self.y_b = None
        self.y_q = None
        self.y_rx = None
        self.y_ry = None
        self.y_rz = None
        self.y_t = None
        self.p_b = None
        self.p_q = None
        self.p_rx = None
        self.p_ry = None
        self.p_rz = None
        self.p_t = None
        self.Bk_inv = None

        # Initialize
        if self.Bk_inv is None:

            self.s_b = []
            self.s_q = []
            self.s_rx = [np.zeros(network.particle_input.output_size)]
            self.s_ry = [np.zeros(network.particle_input.output_size)]
            self.s_rz = [np.zeros(network.particle_input.output_size)]
            self.s_t = [np.zeros(network.particle_input.output_size)]
            self.y_b = []
            self.y_q = []
            self.y_rx = [np.zeros(network.particle_input.output_size)]
            self.y_ry = [np.zeros(network.particle_input.output_size)]
            self.y_rz = [np.zeros(network.particle_input.output_size)]
            self.y_t = [np.zeros(network.particle_input.output_size)]
            self.p_b = []
            self.p_q = []
            self.p_rx = [np.zeros(network.particle_input.output_size)]
            self.p_ry = [np.zeros(network.particle_input.output_size)]
            self.p_rz = [np.zeros(network.particle_input.output_size)]
            self.p_t = [np.zeros(network.particle_input.output_size)]

            self.total = 4 * network.particle_input.output_size
            for l, layer in enumerate(network.layers):
                self.s_b.append(np.zeros(layer.b.output_size))
                self.s_q.append(np.zeros(layer.q.output_size))
                self.s_rx.append(np.zeros(layer.output_size))
                self.s_ry.append(np.zeros(layer.output_size))
                self.s_rz.append(np.zeros(layer.output_size))
                self.s_t.append(np.zeros(layer.output_size))
                self.y_b.append(np.zeros(layer.b.output_size))
                self.y_q.append(np.zeros(layer.q.output_size))
                self.y_rx.append(np.zeros(layer.output_size))
                self.y_ry.append(np.zeros(layer.output_size))
                self.y_rz.append(np.zeros(layer.output_size))
                self.y_t.append(np.zeros(layer.output_size))
                self.p_b.append(np.zeros(layer.b.output_size))
                self.p_q.append(np.zeros(layer.q.output_size))
                self.p_rx.append(np.zeros(layer.output_size))
                self.p_ry.append(np.zeros(layer.output_size))
                self.p_rz.append(np.zeros(layer.output_size))
                self.p_t.append(np.zeros(layer.output_size))
                self.total += 6 * layer.b.output_size

            self.Bk_inv = np.eye(self.total)

        # 1. Compute pk
        p = self.Bk_inv.dot(network, self.flatten(self.dc_db, self.dc_dq, self.dc_dr[0], self.dc_dr[1], self.dc_dr[2], self.dc_dt))
        self.p_b, self.p_q, self.p_rx, self.p_ry, self.p_rz, self.p_t = self.unflatten(network, p)

        # 2+3. Update and take step
        self.s_t[0] = -self.alpha * self.p_t[0]
        self.s_rx[0] = -self.alpha * self.p_rx[0]
        self.s_ry[0] = -self.alpha * self.p_ry[0]
        self.s_rz[0] = -self.alpha * self.p_rz[0]
        for l, layer in enumerate(network.layers):
            self.s_b[l] = -self.alpha * self.p_b[l]
            self.s_q[l] = -self.alpha * self.p_q[l]
            self.s_t[l+1] = -self.alpha * self.p_t[l+1]
            self.s_rx[l+1] = -self.alpha * self.p_rx[l+1]
            self.s_ry[l+1] = -self.alpha * self.p_ry[l+1]
            self.s_rz[l+1] = -self.alpha * self.p_rz[l+1]

        network.particle_input.theta += self.s_t[0]
        network.particle_input.rx += self.s_rx[0]
        network.particle_input.ry += self.s_ry[0]
        network.particle_input.rz += self.s_rz[0]
        for l, layer in enumerate(network.layers):
            layer.b += self.s_b[l]
            layer.q += self.s_q[l]
            layer.theta += self.s_t[l+1]
            layer.rx += self.s_rx[l+1]
            layer.ry += self.s_ry[l+1]
            layer.rz += self.s_rz[l+1]

        # 4.




