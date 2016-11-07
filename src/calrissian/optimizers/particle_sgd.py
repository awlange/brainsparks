from .optimizer import Optimizer

import numpy as np
import time

from multiprocessing import Pool


class ParticleSGD(Optimizer):
    """
    Stochastic gradient descent optimization for Atomic layers
    """

    def __init__(self, alpha=0.01, beta=0.0, gamma=0.9, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, position_grad=True, alpha_b=0.01, alpha_q=0.01, alpha_r=0.01, alpha_t=0.01, init_v=0.0,
                 n_threads=1, chunk_size=10, epsilon=10e-8, gamma2=0.1):
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

        # Could make individual learning rates if we want, but it doesn't seem to matter much
        self.alpha_b = alpha
        self.alpha_q = alpha
        self.alpha_r = alpha
        self.alpha_t = alpha

        self.init_v = init_v
        self.residual = 0.0

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        elif weight_update == "rmsprop_momentum":
            self.weight_update_func = self.weight_update_rmsprop_momentum
        elif weight_update == "rmsprop":
            self.weight_update_func = self.weight_update_rmsprop
        elif weight_update == "adadelta":
            self.weight_update_func = self.weight_update_adadelta
        elif weight_update == "adam":
            self.weight_update_func = self.weight_update_adam
        elif weight_update == "diag":
            self.weight_update_func = self.weight_update_diag
        elif weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr = None
        self.dc_dt = None
        self.dc_dzeta = None

        # Velocities
        self.vel_b = None
        self.vel_q = None
        self.vel_rx = None
        self.vel_ry = None
        self.vel_rz = None
        self.vel_t = None

        # Mean squares
        self.ms_db = None
        self.ms_dq = None
        self.ms_drx = None
        self.ms_dry = None
        self.ms_drz = None
        self.ms_dt = None
        self.ms_dzeta = None

        self.ms_b = None
        self.ms_q = None
        self.ms_rx = None
        self.ms_ry = None
        self.ms_rz = None
        self.ms_t = None

        # Deltas
        self.del_b = None
        self.del_q = None
        self.del_rx = None
        self.del_ry = None
        self.del_rz = None
        self.del_t = None

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
                mini_X = shuffle_X[m:(m+self.mini_batch_size)]
                mini_Y = shuffle_Y[m:(m+self.mini_batch_size)]

                # Compute gradient for mini-batch
                if self.n_threads > 1:
                    self.cost_gradient_parallel(network, mini_X, mini_Y)
                else:
                    self.dc_db, self.dc_dq, self.dc_dr, self.dc_dt = network.cost_gradient(mini_X, mini_Y)

                # Remove translational degrees of freedom gradients (later: include rotational)
                self.remove_translational_gradients()

                # Update weights and biases
                self.weight_update_func(network)

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

    def remove_translational_gradients(self):
        """
        Subtract zeroth moment from position and phase gradients
        """
        # Phase
        n_dt = 0
        mean_dt = 0.0
        for dt in self.dc_dt:
            n_dt += len(dt)
            mean_dt += np.sum(dt)
        mean_dt /= n_dt

        if self.verbosity > 2:
            print("Mean theta gradient: {}".format(mean_dt))

        for l, dt in enumerate(self.dc_dt):
            self.dc_dt[l] -= mean_dt

        # Position
        n_dr = 0
        mean_dx = 0.0
        mean_dy = 0.0
        mean_dz = 0.0
        for l in range(len(self.dc_dr)):
            n_dr += len(self.dc_dr[0][l])
            mean_dx += np.sum(self.dc_dr[0][l])
            mean_dy += np.sum(self.dc_dr[1][l])
            mean_dz += np.sum(self.dc_dr[2][l])
        mean_dx /= n_dr
        mean_dy /= n_dr
        mean_dz /= n_dr

        if self.verbosity > 2:
            print("Mean position gradient: {} {} {}".format(mean_dx, mean_dy, mean_dz))

        for l in range(len(self.dc_dr)):
            self.dc_dr[0][l] -= mean_dx
            self.dc_dr[1][l] -= mean_dy
            self.dc_dr[2][l] -= mean_dz

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
        Update weights and biases according to steepest descent with simple momentum
        """
        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None or self.vel_rx is None or self.vel_ry is None or self.vel_rz is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx[l+1] = -self.alpha * self.dc_dr[0][l+1] + self.beta * self.vel_rx[l+1]
            self.vel_ry[l+1] = -self.alpha * self.dc_dr[1][l+1] + self.beta * self.vel_ry[l+1]
            self.vel_rz[l+1] = -self.alpha * self.dc_dr[2][l+1] + self.beta * self.vel_rz[l+1]
            self.vel_t[l+1] = -self.alpha * self.dc_dt[l+1] + self.beta * self.vel_t[l+1]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.theta += self.vel_t[l+1]
            if self.position_grad:
                layer.rx += self.vel_rx[l+1]
                layer.ry += self.vel_ry[l+1]
                layer.rz += self.vel_rz[l+1]

        self.vel_t[0] = -self.alpha * self.dc_dt[0][0] + self.beta * self.vel_t[0]
        network.particle_input.theta += self.vel_t[0]
        if self.position_grad:
            self.vel_rx[0] = -self.alpha * self.dc_dr[0][0] + self.beta * self.vel_rx[0]
            self.vel_ry[0] = -self.alpha * self.dc_dr[1][0] + self.beta * self.vel_ry[0]
            self.vel_rz[0] = -self.alpha * self.dc_dr[2][0] + self.beta * self.vel_rz[0]
            network.particle_input.rx += self.vel_rx[0]
            network.particle_input.ry += self.vel_ry[0]
            network.particle_input.rz += self.vel_rz[0]

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to AdaGrad
        """
        alpha = self.alpha
        epsilon = self.epsilon  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_dt = [np.zeros(network.particle_input.output_size)]
            self.ms_dzeta = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_dt.append(np.zeros(layer.output_size))
                self.ms_dzeta.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] += (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] += (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dt[l + 1] += (self.dc_dt[l + 1] * self.dc_dt[l + 1])
            self.ms_dzeta[l + 1] += (self.dc_dzeta[l + 1] * self.dc_dzeta[l + 1])
            self.ms_drx[l + 1] += (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] += (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] += (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.q -= alpha * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon)
            layer.theta -= alpha * self.dc_dt[l+1] / np.sqrt(self.ms_dt[l + 1] + epsilon)
            layer.rx -= alpha * self.dc_dr[0][l+1] / np.sqrt(self.ms_drx[l + 1] + epsilon)
            layer.ry -= alpha * self.dc_dr[1][l+1] / np.sqrt(self.ms_dry[l + 1] + epsilon)
            layer.rz -= alpha * self.dc_dr[2][l+1] / np.sqrt(self.ms_drz[l + 1] + epsilon)

        # Input layer
        self.ms_dt[0] += (self.dc_dt[0] * self.dc_dt[0])
        self.ms_dzeta[0] += (self.dc_dzeta[0] * self.dc_dzeta[0])
        self.ms_drx[0] += (self.dc_dr[0][0] * self.dc_dr[0][0])
        self.ms_dry[0] += (self.dc_dr[1][0] * self.dc_dr[1][0])
        self.ms_drz[0] += (self.dc_dr[2][0] * self.dc_dr[2][0])
        network.particle_input.theta  -= alpha * self.dc_dt[0] / np.sqrt(self.ms_dt[0] + epsilon)
        network.particle_input.zeta  -= alpha * self.dc_dzeta[0] / np.sqrt(self.ms_dzeta[0] + epsilon)
        network.particle_input.rx -= alpha * self.dc_dr[0][0] / np.sqrt(self.ms_drx[0] + epsilon)
        network.particle_input.ry -= alpha * self.dc_dr[1][0] / np.sqrt(self.ms_dry[0] + epsilon)
        network.particle_input.rz -= alpha * self.dc_dr[2][0] / np.sqrt(self.ms_drz[0] + epsilon)

    def weight_update_rmsprop(self, network):
        """
        Update weights and biases according to RMSProp
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        # one_m_gamma = self.gamma2
        alpha = self.alpha
        # epsilon = 10e-8  # small number to avoid division by zero
        epsilon = self.epsilon  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_dt = [np.zeros(network.particle_input.output_size)]
            self.ms_dzeta = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_dt.append(np.zeros(layer.output_size))
                self.ms_dzeta.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma * self.ms_dq[l] + one_m_gamma * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dt[l + 1] = gamma * self.ms_dt[l + 1] + one_m_gamma * (self.dc_dt[l + 1] * self.dc_dt[l + 1])
            # self.ms_dzeta[l + 1] = gamma * self.ms_dzeta[l + 1] + one_m_gamma * (self.dc_dzeta[l + 1] * self.dc_dzeta[l + 1])
            self.ms_drx[l + 1] = gamma * self.ms_drx[l + 1] + one_m_gamma * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] = gamma * self.ms_dry[l + 1] + one_m_gamma * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] = gamma * self.ms_drz[l + 1] + one_m_gamma * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

            layer.b -= alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon)
            layer.q -= alpha * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon)
            layer.theta -= alpha * self.dc_dt[l+1] / np.sqrt(self.ms_dt[l + 1] + epsilon)
            layer.rx -= alpha * self.dc_dr[0][l+1] / np.sqrt(self.ms_drx[l + 1] + epsilon)
            layer.ry -= alpha * self.dc_dr[1][l+1] / np.sqrt(self.ms_dry[l + 1] + epsilon)
            layer.rz -= alpha * self.dc_dr[2][l+1] / np.sqrt(self.ms_drz[l + 1] + epsilon)

        # Input layer
        self.ms_dt[0] = gamma * self.ms_dt[0] + one_m_gamma * (self.dc_dt[0] * self.dc_dt[0])
        # self.ms_dzeta[0] = gamma * self.ms_dzeta[0] + one_m_gamma * (self.dc_dzeta[0] * self.dc_dzeta[0])
        self.ms_drx[0] = gamma * self.ms_drx[0] + one_m_gamma * (self.dc_dr[0][0] * self.dc_dr[0][0])
        self.ms_dry[0] = gamma * self.ms_dry[0] + one_m_gamma * (self.dc_dr[1][0] * self.dc_dr[1][0])
        self.ms_drz[0] = gamma * self.ms_drz[0] + one_m_gamma * (self.dc_dr[2][0] * self.dc_dr[2][0])
        network.particle_input.theta -= alpha * self.dc_dt[0] / np.sqrt(self.ms_dt[0] + epsilon)
        # network.particle_input.zeta -= alpha * self.dc_dzeta[0] / np.sqrt(self.ms_dzeta[0] + epsilon)
        network.particle_input.rx -= alpha * self.dc_dr[0][0] / np.sqrt(self.ms_drx[0] + epsilon)
        network.particle_input.ry -= alpha * self.dc_dr[1][0] / np.sqrt(self.ms_dry[0] + epsilon)
        network.particle_input.rz -= alpha * self.dc_dr[2][0] / np.sqrt(self.ms_drz[0] + epsilon)

    def weight_update_rmsprop_momentum(self, network):
        """
        Update weights and biases according to RMSProp, plus momentum
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        alpha = self.alpha
        epsilon = 10e-8  # small number to avoid division by zero

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_dt = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_dt.append(np.zeros(layer.output_size))

        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None or self.vel_rx is None or self.vel_ry is None or self.vel_rz is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma * self.ms_dq[l] + one_m_gamma * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dt[l + 1] = gamma * self.ms_dt[l + 1] + one_m_gamma * (self.dc_dt[l + 1] * self.dc_dt[l + 1])
            self.ms_drx[l + 1] = gamma * self.ms_drx[l + 1] + one_m_gamma * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] = gamma * self.ms_dry[l + 1] + one_m_gamma * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] = gamma * self.ms_drz[l + 1] + one_m_gamma * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

            self.vel_b[l] = -self.alpha * self.dc_db[l] / np.sqrt(self.ms_db[l] + epsilon) + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha * self.dc_dq[l] / np.sqrt(self.ms_dq[l] + epsilon) + self.beta * self.vel_q[l]
            self.vel_rx[l+1] = -self.alpha * self.dc_dr[0][l+1] / np.sqrt(self.ms_drx[l + 1] + epsilon) + self.beta * self.vel_rx[l+1]
            self.vel_ry[l+1] = -self.alpha * self.dc_dr[1][l+1] / np.sqrt(self.ms_dry[l + 1] + epsilon) + self.beta * self.vel_ry[l+1]
            self.vel_rz[l+1] = -self.alpha * self.dc_dr[2][l+1] / np.sqrt(self.ms_drz[l + 1] + epsilon) + self.beta * self.vel_rz[l+1]
            self.vel_t[l+1] = -self.alpha * self.dc_dt[l+1] / np.sqrt(self.ms_dt[l + 1] + epsilon) + self.beta * self.vel_t[l+1]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.theta += self.vel_t[l+1]
            if self.position_grad:
                layer.rx += self.vel_rx[l+1]
                layer.ry += self.vel_ry[l+1]
                layer.rz += self.vel_rz[l+1]

        # Input layer
        self.ms_dt[0] = gamma * self.ms_dt[0] + one_m_gamma * (self.dc_dt[0] * self.dc_dt[0])
        self.ms_drx[0] = gamma * self.ms_drx[0] + one_m_gamma * (self.dc_dr[0][0] * self.dc_dr[0][0])
        self.ms_dry[0] = gamma * self.ms_dry[0] + one_m_gamma * (self.dc_dr[1][0] * self.dc_dr[1][0])
        self.ms_drz[0] = gamma * self.ms_drz[0] + one_m_gamma * (self.dc_dr[2][0] * self.dc_dr[2][0])

        self.vel_t[0] = -self.alpha * self.dc_dt[0][0] / np.sqrt(self.ms_dt[0] + epsilon) + self.beta * self.vel_t[0]
        self.vel_rx[0] = -self.alpha * self.dc_dr[0][0] / np.sqrt(self.ms_drx[0] + epsilon) + self.beta * self.vel_rx[0]
        self.vel_ry[0] = -self.alpha * self.dc_dr[1][0] / np.sqrt(self.ms_dry[0] + epsilon) + self.beta * self.vel_ry[0]
        self.vel_rz[0] = -self.alpha * self.dc_dr[2][0] / np.sqrt(self.ms_drz[0] + epsilon) + self.beta * self.vel_rz[0]
        network.particle_input.theta += self.vel_t[0]
        network.particle_input.rx += self.vel_rx[0]
        network.particle_input.ry += self.vel_ry[0]
        network.particle_input.rz += self.vel_rz[0]

    def weight_update_adam(self, network):
        """
        Update weights and biases according to ADAM
        """
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        gamma2 = self.gamma2
        one_m_gamma2 = 1.0 - gamma2
        alpha = self.alpha
        epsilon = 10e-8  # small number to avoid division by zero

        gamma_t = gamma
        gamma2_t = gamma2

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_dt = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_dt.append(np.zeros(layer.output_size))

        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t.append(np.zeros(layer.output_size))

        gv = 1.0 - self.gamma_t
        gm = 1.0 - self.gamma2_t

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma2 * self.ms_db[l] + one_m_gamma2 * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma2 * self.ms_dq[l] + one_m_gamma2 * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dt[l + 1] = gamma2 * self.ms_dt[l + 1] + one_m_gamma2 * (self.dc_dt[l + 1] * self.dc_dt[l + 1])
            self.ms_drx[l + 1] = gamma2 * self.ms_drx[l + 1] + one_m_gamma2 * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] = gamma2 * self.ms_dry[l + 1] + one_m_gamma2 * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] = gamma2 * self.ms_drz[l + 1] + one_m_gamma2 * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

            self.vel_b[l] = gamma2 * self.vel_b[l] + one_m_gamma * self.dc_db[l]
            self.vel_q[l] = gamma2 * self.vel_q[l] + one_m_gamma * self.dc_dq[l]
            self.vel_t[l + 1] = gamma2 * self.vel_t[l + 1] + one_m_gamma * self.dc_dt[l + 1]
            self.vel_rx[l + 1] = gamma2 * self.vel_rx[l + 1] + one_m_gamma * self.dc_dr[0][l + 1]
            self.vel_ry[l + 1] = gamma2 * self.vel_ry[l + 1] + one_m_gamma * self.dc_dr[1][l + 1]
            self.vel_rz[l + 1] = gamma2 * self.vel_rz[l + 1] + one_m_gamma * self.dc_dr[2][l + 1]

            layer.b -= (alpha * self.vel_b[l]/gv) / np.sqrt(self.ms_db[l]/gm + epsilon)
            layer.q -= (alpha * self.vel_q[l]/gv) / np.sqrt(self.ms_dq[l]/gm + epsilon)
            layer.theta -= (alpha * self.vel_t[l+1]/gv) / (np.sqrt(self.ms_dt[l + 1]/gm) + epsilon)
            layer.rx -= (alpha * self.vel_rx[l+1]/gv) / (np.sqrt(self.ms_drx[l + 1]/gm) + epsilon)
            layer.ry -= (alpha * self.vel_ry[l+1]/gv) / (np.sqrt(self.ms_dry[l + 1]/gm) + epsilon)
            layer.rz -= (alpha * self.vel_rz[l+1]/gv) / (np.sqrt(self.ms_drz[l + 1]/gm) + epsilon)

        # Input layer
        self.ms_dt[0] = gamma2 * self.ms_dt[0] + one_m_gamma2 * (self.dc_dt[0] * self.dc_dt[0])
        self.ms_drx[0] = gamma2 * self.ms_drx[0] + one_m_gamma2 * (self.dc_dr[0][0] * self.dc_dr[0][0])
        self.ms_dry[0] = gamma2 * self.ms_dry[0] + one_m_gamma2 * (self.dc_dr[1][0] * self.dc_dr[1][0])
        self.ms_drz[0] = gamma2 * self.ms_drz[0] + one_m_gamma2 * (self.dc_dr[2][0] * self.dc_dr[2][0])

        self.vel_t[0] = gamma * self.vel_t[0] + one_m_gamma * self.dc_dt[0]
        self.vel_rx[0] = gamma * self.vel_rx[0] + one_m_gamma * self.dc_dr[0][0]
        self.vel_ry[0] = gamma * self.vel_ry[0] + one_m_gamma * self.dc_dr[1][0]
        self.vel_rz[0] = gamma * self.vel_rz[0] + one_m_gamma * self.dc_dr[2][0]

        network.particle_input.theta  -= (alpha * self.vel_t[0]/gv) / (np.sqrt(self.ms_dt[0]/gm) + epsilon)
        network.particle_input.rx -= (alpha * self.vel_rx[0]/gv) / (np.sqrt(self.ms_drx[0]/gm) + epsilon)
        network.particle_input.ry -= (alpha * self.vel_ry[0]/gv) / (np.sqrt(self.ms_dry[0]/gm) + epsilon)
        network.particle_input.rz -= (alpha * self.vel_rz[0]/gv) / (np.sqrt(self.ms_drz[0]/gm) + epsilon)

        # Gamma powers
        self.t += 1
        self.gamma_t *= self.gamma_t
        self.gamma2_t *= self.gamma2_t

    def weight_update_adadelta(self, network):
        """
        Update weights and biases according to AdaDelta
        """
        alpha = self.alpha
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
        epsilon = 10e-8  # small number to avoid division by zero, as from paper

        # Initialize RMS to zero
        if self.ms_db is None or self.ms_dq is None:
            self.ms_db = []
            self.ms_dq = []
            self.ms_drx = [np.zeros(network.particle_input.output_size)]
            self.ms_dry = [np.zeros(network.particle_input.output_size)]
            self.ms_drz = [np.zeros(network.particle_input.output_size)]
            self.ms_dt = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_db.append(np.zeros(layer.b.shape))
                self.ms_dq.append(np.zeros(layer.q.shape))
                self.ms_drx.append(np.zeros(layer.output_size))
                self.ms_dry.append(np.zeros(layer.output_size))
                self.ms_drz.append(np.zeros(layer.output_size))
                self.ms_dt.append(np.zeros(layer.output_size))

            self.ms_b = []
            self.ms_q = []
            self.ms_rx = [np.zeros(network.particle_input.output_size)]
            self.ms_ry = [np.zeros(network.particle_input.output_size)]
            self.ms_rz = [np.zeros(network.particle_input.output_size)]
            self.ms_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.ms_b.append(np.zeros(layer.b.shape))
                self.ms_q.append(np.zeros(layer.q.shape))
                self.ms_rx.append(np.zeros(layer.output_size))
                self.ms_ry.append(np.zeros(layer.output_size))
                self.ms_rz.append(np.zeros(layer.output_size))
                self.ms_t.append(np.zeros(layer.output_size))

            self.del_b = []
            self.del_q = []
            self.del_rx = [np.zeros(network.particle_input.output_size)]
            self.del_ry = [np.zeros(network.particle_input.output_size)]
            self.del_rz = [np.zeros(network.particle_input.output_size)]
            self.del_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.del_b.append(np.zeros(layer.b.shape))
                self.del_q.append(np.zeros(layer.q.shape))
                self.del_rx.append(np.zeros(layer.output_size))
                self.del_ry.append(np.zeros(layer.output_size))
                self.del_rz.append(np.zeros(layer.output_size))
                self.del_t.append(np.zeros(layer.output_size))

        for l, layer in enumerate(network.layers):
            self.ms_db[l] = gamma * self.ms_db[l] + one_m_gamma * (self.dc_db[l] * self.dc_db[l])
            self.ms_dq[l] = gamma * self.ms_dq[l] + one_m_gamma * (self.dc_dq[l] * self.dc_dq[l])
            self.ms_dt[l + 1] = gamma * self.ms_dt[l + 1] + one_m_gamma * (self.dc_dt[l + 1] * self.dc_dt[l + 1])
            self.ms_drx[l + 1] = gamma * self.ms_drx[l + 1] + one_m_gamma * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
            self.ms_dry[l + 1] = gamma * self.ms_dry[l + 1] + one_m_gamma * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
            self.ms_drz[l + 1] = gamma * self.ms_drz[l + 1] + one_m_gamma * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

            self.del_b[l] = -alpha * self.dc_db[l] * np.sqrt((self.ms_b[l] + epsilon) / (self.ms_db[l] + epsilon))
            self.del_q[l] = -alpha * self.dc_dq[l] * np.sqrt((self.ms_q[l] + epsilon) / (self.ms_dq[l] + epsilon))
            self.del_t[l+1] = -alpha * self.dc_dt[l+1] * np.sqrt((self.ms_t[l+1] + epsilon) / (self.ms_dt[l+1] + epsilon))
            self.del_rx[l+1] = -alpha * self.dc_dr[0][l+1] * np.sqrt((self.ms_rx[l+1] + epsilon) / (self.ms_drx[l+1] + epsilon))
            self.del_ry[l+1] = -alpha * self.dc_dr[1][l+1] * np.sqrt((self.ms_ry[l+1] + epsilon) / (self.ms_dry[l+1] + epsilon))
            self.del_rz[l+1] = -alpha * self.dc_dr[2][l+1] * np.sqrt((self.ms_rz[l+1] + epsilon) / (self.ms_drz[l+1] + epsilon))

            layer.b += self.del_b[l]
            layer.q += self.del_q[l]
            layer.theta += self.del_t[l+1]
            layer.rx += self.del_rx[l+1]
            layer.ry += self.del_ry[l+1]
            layer.rz += self.del_rz[l+1]

            self.ms_b[l] = gamma * self.ms_b[l] + one_m_gamma * (self.del_b[l] * self.del_b[l])
            self.ms_q[l] = gamma * self.ms_q[l] + one_m_gamma * (self.del_q[l] * self.del_q[l])
            self.ms_t[l + 1] = gamma * self.ms_t[l + 1] + one_m_gamma * (self.del_t[l + 1] * self.del_t[l + 1])
            self.ms_rx[l + 1] = gamma * self.ms_rx[l + 1] + one_m_gamma * (self.del_rx[l + 1] * self.del_rx[l + 1])
            self.ms_ry[l + 1] = gamma * self.ms_ry[l + 1] + one_m_gamma * (self.del_ry[l + 1] * self.del_ry[l + 1])
            self.ms_rz[l + 1] = gamma * self.ms_rz[l + 1] + one_m_gamma * (self.del_rz[l + 1] * self.del_rz[l + 1])

        # Input layer
        l = -1
        layer = network.particle_input

        self.ms_dt[l + 1] = gamma * self.ms_dt[l + 1] + one_m_gamma * (self.dc_dt[l + 1] * self.dc_dt[l + 1])
        self.ms_drx[l + 1] = gamma * self.ms_drx[l + 1] + one_m_gamma * (self.dc_dr[0][l + 1] * self.dc_dr[0][l + 1])
        self.ms_dry[l + 1] = gamma * self.ms_dry[l + 1] + one_m_gamma * (self.dc_dr[1][l + 1] * self.dc_dr[1][l + 1])
        self.ms_drz[l + 1] = gamma * self.ms_drz[l + 1] + one_m_gamma * (self.dc_dr[2][l + 1] * self.dc_dr[2][l + 1])

        self.del_t[l+1] = -alpha * self.dc_dt[l+1] * np.sqrt((self.ms_t[l+1] + epsilon) / (self.ms_dt[l+1] + epsilon))
        self.del_rx[l+1] = -alpha * self.dc_dr[0][l+1] * np.sqrt((self.ms_rx[l+1] + epsilon) / (self.ms_drx[l+1] + epsilon))
        self.del_ry[l+1] = -alpha * self.dc_dr[1][l+1] * np.sqrt((self.ms_ry[l+1] + epsilon) / (self.ms_dry[l+1] + epsilon))
        self.del_rz[l+1] = -alpha * self.dc_dr[2][l+1] * np.sqrt((self.ms_rz[l+1] + epsilon) / (self.ms_drz[l+1] + epsilon))

        layer.theta += self.del_t[l+1]
        layer.rx += self.del_rx[l+1]
        layer.ry += self.del_ry[l+1]
        layer.rz += self.del_rz[l+1]

        self.ms_t[l+1] = gamma * self.ms_t[l + 1] + one_m_gamma * (self.del_t[l + 1] * self.del_t[l + 1])
        self.ms_rx[l+1] = gamma * self.ms_rx[l + 1] + one_m_gamma * (self.del_rx[l + 1] * self.del_rx[l + 1])
        self.ms_ry[l+1] = gamma * self.ms_ry[l + 1] + one_m_gamma * (self.del_ry[l + 1] * self.del_ry[l + 1])
        self.ms_rz[l+1] = gamma * self.ms_rz[l + 1] + one_m_gamma * (self.del_rz[l + 1] * self.del_rz[l + 1])

    def weight_update_diag(self, network):
        """
        Update weights and biases according to a thing
        """

        residual = self.epsilon

        # Initialize velocities to zero for momentum
        if self.vel_b is None or self.vel_q is None or self.vel_rx is None or self.vel_ry is None or self.vel_rz is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.vel_t = [np.zeros(network.particle_input.output_size)]
            self.del_b = []
            self.del_q = []
            self.del_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.del_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.del_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.del_t = [np.zeros(network.particle_input.output_size)]
            self.ms_b = []
            self.ms_q = []
            self.ms_rx = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.ms_ry = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.ms_rz = [np.random.uniform(-self.init_v, self.init_v, network.particle_input.output_size)]
            self.ms_t = [np.zeros(network.particle_input.output_size)]
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.zeros(layer.b.shape))
                self.vel_q.append(np.zeros(layer.q.shape))
                self.vel_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t.append(np.zeros(layer.output_size))
                self.del_b.append(np.zeros(layer.b.shape))
                self.del_q.append(np.zeros(layer.q.shape))
                self.del_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.del_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.del_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.del_t.append(np.zeros(layer.output_size))
                self.ms_b.append(np.zeros(layer.b.shape))
                self.ms_q.append(np.zeros(layer.q.shape))
                self.ms_rx.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.ms_ry.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.ms_rz.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.ms_t.append(np.zeros(layer.output_size))

            # For first iteration, just take a steepest small descent step
            for l, layer in enumerate(network.layers):
                self.del_b[l] = -self.alpha * self.dc_db[l]
                self.del_q[l] = -self.alpha * self.dc_dq[l]
                self.del_rx[l+1] = -self.alpha * self.dc_dr[0][l+1]
                self.del_ry[l+1] = -self.alpha * self.dc_dr[1][l+1]
                self.del_rz[l+1] = -self.alpha * self.dc_dr[2][l+1]
                self.del_t[l+1] = -self.alpha * self.dc_dt[l+1]
                layer.b += self.del_b[l]
                layer.q += self.del_q[l]
                layer.theta += self.del_t[l+1]
                layer.rx += self.del_rx[l+1]
                layer.ry += self.del_ry[l+1]
                layer.rz += self.del_rz[l+1]

                residual += np.sum(np.abs(self.dc_db[l]))
                residual += np.sum(np.abs(self.dc_dq[l]))
                residual += np.sum(np.abs(self.dc_dr[0][l+1]))
                residual += np.sum(np.abs(self.dc_dr[1][l+1]))
                residual += np.sum(np.abs(self.dc_dr[2][l+1]))
                residual += np.sum(np.abs(self.dc_dt[l+1]))

            layer = network.particle_input
            self.del_t[0] = -self.alpha * self.dc_dt[0][0]
            self.del_rx[0] = -self.alpha * self.dc_dr[0][0]
            self.del_ry[0] = -self.alpha * self.dc_dr[1][0]
            self.del_rz[0] = -self.alpha * self.dc_dr[2][0]
            layer.theta += self.del_t[0]
            layer.rx += self.del_rx[0]
            layer.ry += self.del_ry[0]
            layer.rz += self.del_rz[0]

            residual += np.sum(np.abs(self.dc_dr[0][0]))
            residual += np.sum(np.abs(self.dc_dr[1][0]))
            residual += np.sum(np.abs(self.dc_dr[2][0]))
            residual += np.sum(np.abs(self.dc_dt[0]))
            # print(residual)

        else:
            # All iterations after first
            gamma = self.gamma
            one_m_gamma = 1.0 - gamma
            alpha = self.alpha
            g2 = self.gamma2

            beta = self.beta / self.residual

            for l, layer in enumerate(network.layers):
                self.ms_b[l] *= self.gamma
                self.ms_q[l] *= self.gamma
                self.ms_rx[l+1] *= self.gamma
                self.ms_ry[l+1] *= self.gamma
                self.ms_rz[l+1] *= self.gamma
                self.ms_t[l+1] *= self.gamma
                self.ms_b[l] += one_m_gamma * np.abs((self.del_b[l] + self.epsilon)/((self.dc_db[l]-self.vel_b[l]) + self.epsilon))
                self.ms_q[l] += one_m_gamma * np.abs((self.del_q[l] + self.epsilon)/((self.dc_dq[l]-self.vel_q[l]) + self.epsilon))
                self.ms_rx[l+1] += one_m_gamma * np.abs((self.del_rx[l+1] + self.epsilon)/((self.dc_dr[0][l+1]-self.vel_rx[l+1]) + self.epsilon))
                self.ms_ry[l+1] += one_m_gamma * np.abs((self.del_ry[l+1] + self.epsilon)/((self.dc_dr[1][l+1]-self.vel_ry[l+1]) + self.epsilon))
                self.ms_rz[l+1] += one_m_gamma * np.abs((self.del_rz[l+1] + self.epsilon)/((self.dc_dr[2][l+1]-self.vel_rz[l+1]) + self.epsilon))
                self.ms_t[l+1] += one_m_gamma * np.abs((self.del_t[l+1] + self.epsilon)/((self.dc_dt[l+1]-self.vel_t[l+1]) + self.epsilon))

                self.del_b[l] = -alpha * self.dc_db[l] * (np.minimum(self.ms_b[l], g2*np.ones_like(self.ms_b[l])))
                self.del_q[l] = -alpha * self.dc_dq[l] * (np.minimum(self.ms_q[l], g2*np.ones_like(self.ms_q[l])))
                self.del_rx[l+1] = -alpha * self.dc_dr[0][l+1] * (np.minimum(self.ms_rx[l+1], g2*np.ones_like(self.ms_rx[l+1])))
                self.del_ry[l+1] = -alpha * self.dc_dr[1][l+1] * (np.minimum(self.ms_ry[l+1], g2*np.ones_like(self.ms_ry[l+1])))
                self.del_rz[l+1] = -alpha * self.dc_dr[2][l+1] * (np.minimum(self.ms_rz[l+1], g2*np.ones_like(self.ms_rz[l+1])))
                self.del_t[l+1] = -alpha * self.dc_dt[l+1] * (np.minimum(self.ms_t[l+1], g2*np.ones_like(self.ms_t[l+1])))
                layer.b += self.del_b[l]
                layer.q += self.del_q[l]
                layer.rx += self.del_rx[l+1]
                layer.ry += self.del_ry[l+1]
                layer.rz += self.del_rz[l+1]
                layer.theta += self.del_t[l+1]

                residual += np.sum(np.abs(self.dc_db[l]))
                residual += np.sum(np.abs(self.dc_dq[l]))
                residual += np.sum(np.abs(self.dc_dr[0][l+1]))
                residual += np.sum(np.abs(self.dc_dr[1][l+1]))
                residual += np.sum(np.abs(self.dc_dr[2][l+1]))
                residual += np.sum(np.abs(self.dc_dt[l+1]))

            layer = network.particle_input

            self.ms_rx[0] *= self.gamma
            self.ms_ry[0] *= self.gamma
            self.ms_rz[0] *= self.gamma
            self.ms_t[0] *= self.gamma
            self.ms_rx[0] += one_m_gamma * np.abs((self.del_rx[0] + self.epsilon)/((self.dc_dr[0][0]-self.vel_rx[0]) + self.epsilon))
            self.ms_ry[0] += one_m_gamma * np.abs((self.del_ry[0] + self.epsilon)/((self.dc_dr[1][0]-self.vel_ry[0]) + self.epsilon))
            self.ms_rz[0] += one_m_gamma * np.abs((self.del_rz[0] + self.epsilon)/((self.dc_dr[2][0]-self.vel_rz[0]) + self.epsilon))
            self.ms_t[0] += one_m_gamma * np.abs((self.del_t[0] + self.epsilon)/((self.dc_dt[0]-self.vel_t[0]) + self.epsilon))

            self.del_rx[0] = -alpha * self.dc_dr[0][0] * (np.minimum(self.ms_rx[0], g2*np.ones_like(self.ms_rx[0])))
            self.del_ry[0] = -alpha * self.dc_dr[1][0] * (np.minimum(self.ms_ry[0], g2*np.ones_like(self.ms_ry[0])))
            self.del_rz[0] = -alpha * self.dc_dr[2][0] * (np.minimum(self.ms_rz[0], g2*np.ones_like(self.ms_rz[0])))
            self.del_t[0] = -alpha * self.dc_dt[0][0] * (np.minimum(self.ms_t[0], g2*np.ones_like(self.ms_t[0])))
            layer.rx += self.del_rx[0]
            layer.ry += self.del_ry[0]
            layer.rz += self.del_rz[0]
            layer.theta += self.del_t[0]

            residual += np.sum(np.abs(self.dc_dr[0][0]))
            residual += np.sum(np.abs(self.dc_dr[1][0]))
            residual += np.sum(np.abs(self.dc_dr[2][0]))
            residual += np.sum(np.abs(self.dc_dt[0]))
            # print(residual)

        # Keep around last iteration's gradient and step size, with decay
        gamma = 0.0
        one_m_gamma = 1.0 - gamma
        for l, layer in enumerate(network.layers):
            self.vel_b[l] = np.copy(one_m_gamma * self.dc_db[l] + gamma * self.vel_b[l])
            self.vel_q[l] = np.copy(one_m_gamma * self.dc_dq[l] + gamma * self.vel_q[l])
            self.vel_rx[l+1] = np.copy(one_m_gamma * self.dc_dr[0][l+1] + gamma * self.vel_rx[l+1])
            self.vel_ry[l+1] = np.copy(one_m_gamma * self.dc_dr[1][l+1] + gamma * self.vel_ry[l+1])
            self.vel_rz[l+1] = np.copy(one_m_gamma * self.dc_dr[2][l+1] + gamma * self.vel_rz[l+1])
            self.vel_t[l+1] = np.copy(one_m_gamma * self.dc_dt[l+1] + gamma * self.vel_t[l+1])
        self.vel_rx[0] = np.copy(one_m_gamma * self.dc_dr[0][0] + gamma * self.vel_rx[0])
        self.vel_ry[0] = np.copy(one_m_gamma * self.dc_dr[1][0] + gamma * self.vel_ry[0])
        self.vel_rz[0] = np.copy(one_m_gamma * self.dc_dr[2][0] + gamma * self.vel_rz[0])
        self.vel_t[0] = np.copy(one_m_gamma * self.dc_dt[0][0] + gamma * self.vel_t[0])

        self.residual = residual