from .optimizer import Optimizer

import numpy as np
import time


class Particle2SGD(Optimizer):
    """
    Stochastic gradient descent optimization for Atomic layers
    """

    def __init__(self, alpha=0.01, beta=0.0, n_epochs=1, mini_batch_size=1, verbosity=2, weight_update="sd",
                 cost_freq=2, alpha_b=0.01, alpha_q=0.01, alpha_r=0.01, alpha_t=0.01, init_v=0.0, gamma=0.9):
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
        self.alpha_t = alpha

        # Weight update function
        self.weight_update = weight_update
        self.weight_update_func = self.weight_update_steepest_descent
        if weight_update == "momentum":
            self.weight_update_func = self.weight_update_steepest_descent_with_momentum
        elif weight_update == "adagrad":
            self.weight_update_func = self.weight_update_adagrad

        # Weight gradients, to keep around for a step
        self.dc_db = None
        self.dc_dq = None
        self.dc_dr_inp = None
        self.dc_dr_out = None
        self.dc_dt_inp = None
        self.dc_dt_out = None

        # Velocities
        self.init_v = init_v
        self.vel_b = None
        self.vel_q = None
        self.vel_rx_inp = None
        self.vel_ry_inp = None
        self.vel_rz_inp = None
        self.vel_rx_out = None
        self.vel_ry_out = None
        self.vel_rz_out = None
        self.vel_t_inp = None
        self.vel_t_out = None

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
                self.dc_db, self.dc_dq, self.dc_dr_inp, self.dc_dr_out, self.dc_dt_inp, self.dc_dt_out = network.cost_gradient(mini_X, mini_Y)

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
            layer.theta_inp -= self.alpha * self.dc_dt_inp[l]
            layer.theta_out -= self.alpha * self.dc_dt_out[l]
            layer.rx_inp -= self.alpha * self.dc_dr_inp[0][l]
            layer.ry_inp -= self.alpha * self.dc_dr_inp[1][l]
            layer.rz_inp -= self.alpha * self.dc_dr_inp[2][l]
            layer.rx_out -= self.alpha * self.dc_dr_out[0][l]
            layer.ry_out -= self.alpha * self.dc_dr_out[1][l]
            layer.rz_out -= self.alpha * self.dc_dr_out[2][l]

    def weight_update_steepest_descent_with_momentum(self, network):
        """
        Update weights and biases according to steepest descent
        TODO
        """
        # Initialize velocities to zero/finite for momentum
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_inp = []
            self.vel_ry_inp = []
            self.vel_rz_inp = []
            self.vel_t_inp = []
            self.vel_rx_out = []
            self.vel_ry_out = []
            self.vel_rz_out = []
            self.vel_t_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.random.uniform(-self.init_v, self.init_v, layer.b.shape))
                self.vel_q.append(np.random.uniform(-self.init_v, self.init_v, layer.q.shape))
                self.vel_rx_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_ry_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rz_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rx_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_t_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] = -self.alpha_b * self.dc_db[l] + self.beta * self.vel_b[l]
            self.vel_q[l] = -self.alpha_q * self.dc_dq[l] + self.beta * self.vel_q[l]
            self.vel_rx_inp[l] = -self.alpha_r * self.dc_dr_inp[0][l] + self.beta * self.vel_rx_inp[l]
            self.vel_ry_inp[l] = -self.alpha_r * self.dc_dr_inp[1][l] + self.beta * self.vel_ry_inp[l]
            self.vel_rz_inp[l] = -self.alpha_r * self.dc_dr_inp[2][l] + self.beta * self.vel_rz_inp[l]
            self.vel_t_inp[l] = -self.alpha_t * self.dc_dt_inp[l] + self.beta * self.vel_t_inp[l]
            self.vel_rx_out[l] = -self.alpha_r * self.dc_dr_out[0][l] + self.beta * self.vel_rx_out[l]
            self.vel_ry_out[l] = -self.alpha_r * self.dc_dr_out[1][l] + self.beta * self.vel_ry_out[l]
            self.vel_rz_out[l] = -self.alpha_r * self.dc_dr_out[2][l] + self.beta * self.vel_rz_out[l]
            self.vel_t_out[l] = -self.alpha_t * self.dc_dt_out[l] + self.beta * self.vel_t_out[l]
            layer.b += self.vel_b[l]
            layer.q += self.vel_q[l]
            layer.theta_inp += self.vel_t_inp[l]
            layer.theta_out += self.vel_t_out[l]
            layer.rx_inp += self.vel_rx_inp[l]
            layer.ry_inp += self.vel_ry_inp[l]
            layer.rz_inp += self.vel_rz_inp[l]
            layer.rx_out += self.vel_rx_out[l]
            layer.ry_out += self.vel_ry_out[l]
            layer.rz_out += self.vel_rz_out[l]

    def weight_update_adagrad(self, network):
        """
        Update weights and biases according to AdaGrad
        """
        
        epsilon = 10e-8

        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_inp = []
            self.vel_ry_inp = []
            self.vel_rz_inp = []
            self.vel_t_inp = []
            self.vel_rx_out = []
            self.vel_ry_out = []
            self.vel_rz_out = []
            self.vel_t_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.random.uniform(-self.init_v, self.init_v, layer.b.shape))
                self.vel_q.append(np.random.uniform(-self.init_v, self.init_v, layer.q.shape))
                self.vel_rx_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_ry_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rz_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rx_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_t_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))

        for l, layer in enumerate(network.layers):
            self.vel_b[l] += self.dc_db[l]**2
            self.vel_q[l] += self.dc_dq[l]**2
            self.vel_rx_inp[l] += self.dc_dr_inp[0][l]**2
            self.vel_ry_inp[l] += self.dc_dr_inp[1][l]**2
            self.vel_rz_inp[l] += self.dc_dr_inp[2][l]**2
            self.vel_rx_out[l] += self.dc_dr_out[0][l]**2
            self.vel_ry_out[l] += self.dc_dr_out[1][l]**2
            self.vel_rz_out[l] += self.dc_dr_out[2][l]**2
            self.vel_t_inp[l] += self.dc_dt_inp[l]**2
            self.vel_t_out[l] += self.dc_dt_out[l]**2

            layer.b += -self.alpha * self.dc_db[l] / np.sqrt(self.vel_b[l] + epsilon)
            layer.q += -self.alpha * self.dc_dq[l] / np.sqrt(self.vel_q[l] + epsilon)
            layer.rx_inp += -self.alpha * self.dc_dr_inp[0][l] / np.sqrt(self.vel_rx_inp[l] + epsilon)
            layer.ry_inp += -self.alpha * self.dc_dr_inp[1][l] / np.sqrt(self.vel_ry_inp[l] + epsilon)
            layer.rz_inp += -self.alpha * self.dc_dr_inp[2][l] / np.sqrt(self.vel_rz_inp[l] + epsilon)
            layer.rx_out += -self.alpha * self.dc_dr_out[0][l] / np.sqrt(self.vel_rx_out[l] + epsilon)
            layer.ry_out += -self.alpha * self.dc_dr_out[1][l] / np.sqrt(self.vel_ry_out[l] + epsilon)
            layer.rz_out += -self.alpha * self.dc_dr_out[2][l] / np.sqrt(self.vel_rz_out[l] + epsilon)
            layer.theta_inp += -self.alpha * self.dc_dt_inp[l] / np.sqrt(self.vel_t_inp[l] + epsilon)
            layer.theta_out += -self.alpha * self.dc_dt_out[l] / np.sqrt(self.vel_t_out[l] + epsilon)


    def weight_update_rmsprop(self, network):
        """
        Update weights and biases according to rmsprop
        """
    
        epsilon = 10e-8
        gamma = self.gamma
        one_m_gamma = 1.0 - gamma
    
        if self.vel_b is None or self.vel_q is None:
            self.vel_b = []
            self.vel_q = []
            self.vel_rx_inp = []
            self.vel_ry_inp = []
            self.vel_rz_inp = []
            self.vel_t_inp = []
            self.vel_rx_out = []
            self.vel_ry_out = []
            self.vel_rz_out = []
            self.vel_t_out = []
            for l, layer in enumerate(network.layers):
                self.vel_b.append(np.random.uniform(-self.init_v, self.init_v, layer.b.shape))
                self.vel_q.append(np.random.uniform(-self.init_v, self.init_v, layer.q.shape))
                self.vel_rx_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_ry_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rz_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_rx_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_ry_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_rz_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
                self.vel_t_inp.append(np.random.uniform(-self.init_v, self.init_v, layer.input_size))
                self.vel_t_out.append(np.random.uniform(-self.init_v, self.init_v, layer.output_size))
    
        for l, layer in enumerate(network.layers):
            self.vel_b[l] = gamma * self.vel_b[l] + one_m_gamma * self.dc_db[l]**2
            self.vel_q[l] = gamma * self.vel_q[l] + one_m_gamma * self.dc_dq[l]**2

            self.vel_rx_inp[l] = gamma * self.vel_rx_inp[l] + one_m_gamma * self.dc_dr_inp[0][l]**2
            self.vel_ry_inp[l] = gamma * self.vel_ry_inp[l] + one_m_gamma * self.dc_dr_inp[1][l]**2
            self.vel_rz_inp[l] = gamma * self.vel_rz_inp[l] + one_m_gamma * self.dc_dr_inp[2][l]**2

            self.vel_rx_out[l] = gamma * self.vel_rx_inp[l] + one_m_gamma * self.dc_dr_out[0][l]**2
            self.vel_ry_out[l] = gamma * self.vel_ry_inp[l] + one_m_gamma * self.dc_dr_out[1][l]**2
            self.vel_rz_out[l] = gamma * self.vel_rz_inp[l] + one_m_gamma * self.dc_dr_out[2][l]**2

            self.vel_t_inp[l] = gamma * self.vel_t_inp[l] + one_m_gamma * self.dc_dt_inp[l]**2
            self.vel_t_out[l] = gamma * self.vel_t_out[l] + one_m_gamma * self.dc_dt_out[l]**2

            layer.b += -self.alpha * self.dc_db[l] / np.sqrt(self.vel_b[l] + epsilon)
            layer.q += -self.alpha * self.dc_dq[l] / np.sqrt(self.vel_q[l] + epsilon)

            layer.rx_inp += -self.alpha * self.dc_dr_inp[0][l] / np.sqrt(self.vel_rx_inp[l] + epsilon)
            layer.ry_inp += -self.alpha * self.dc_dr_inp[1][l] / np.sqrt(self.vel_ry_inp[l] + epsilon)
            layer.rz_inp += -self.alpha * self.dc_dr_inp[2][l] / np.sqrt(self.vel_rz_inp[l] + epsilon)

            layer.rx_out += -self.alpha * self.dc_dr_out[0][l] / np.sqrt(self.vel_rx_out[l] + epsilon)
            layer.ry_out += -self.alpha * self.dc_dr_out[1][l] / np.sqrt(self.vel_ry_out[l] + epsilon)
            layer.rz_out += -self.alpha * self.dc_dr_out[2][l] / np.sqrt(self.vel_rz_out[l] + epsilon)

            layer.theta_inp += -self.alpha * self.dc_dt_inp[l] / np.sqrt(self.vel_t_inp[l] + epsilon)
            layer.theta_out += -self.alpha * self.dc_dt_out[l] / np.sqrt(self.vel_t_out[l] + epsilon)


