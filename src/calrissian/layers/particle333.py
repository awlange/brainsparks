from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class Particle333(object):
    """
    Monopole input, multipole output
    """

    def __init__(self, input_size=1, output_size=1, activation="sigmoid", potential="gaussian",
                 nr=3, nc=1, rand="uniform",
                 s=1.0, q=1.0, b=1.0, z=1.0,
                 apply_convolution=False,
                 input_shape=(1, 1, 1),   # n_x, n_y, n_channel
                 output_shape=(1, 1, 1),  # n_x, n_y, n_channel
                 input_delta=(1.0, 1.0, 1.0),   # delta_x, delta_y, delta_channel
                 output_delta=(1.0, 1.0, 1.0),  # delta_x, delta_y, delta_channel
                 output_pool_shape=(1, 1, 1),  # max pooling
                 output_pool_delta=(1.0, 1.0, 1.0)
                 ):

        self.input_size = input_size  # with conv, this is the number of input channels
        self.output_size = output_size  # with conv, this is the number of output channels
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)
        self.dz_potential = Potential.get_dz(potential)
        self.nr = nr
        self.nc = nc

        self.apply_convolution = apply_convolution
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_delta = input_delta
        self.output_delta = output_delta
        self.output_pool_shape = output_pool_shape
        self.output_pool_delta = output_pool_delta

        if self.apply_convolution:
            self.input_size = self.input_shape[2]
            self.output_size = self.output_shape[2]

        # Weight initialization
        if rand == "uniform":
            self.b = np.random.uniform(-b, b, (1, self.output_size))
        else:
            self.b = np.random.normal(0.0, b, (1, self.output_size))

        # Coefficients
        self.q = None
        if rand == "uniform":
            self.q = np.random.uniform(-q, q, (self.output_size, nc))
        else:
            self.q = np.random.normal(0.0, q, (self.output_size, nc))

        # Widths
        self.zeta = np.ones((self.output_size, nc))  # in the potential, these are squared so that they are always positive
        if rand == "uniform":
            self.zeta = np.random.uniform(-z, z, (self.output_size, nc))
        else:
            self.zeta = np.random.normal(1.0, z, (self.output_size, nc))

        # Positions
        # With convolution, these are the positions of the anchors
        self.r_inp = None  # input positions
        self.r_out = None  # output positions
        if rand == "uniform":
            self.r_out = np.random.uniform(-s, s, (self.output_size, nc, nr))
            self.r_inp = np.random.uniform(-s, s, (self.input_size, nr))
        else:
            self.r_out = np.random.normal(0.0, s, (self.output_size, nc, nr))
            self.r_inp = np.random.normal(0.0, s, (self.input_size, nr))

        # pooling cache to help with gradient
        self.z_pool_max_cache = None

        self.w = None

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        if self.apply_convolution:
            return self.compute_z_convolution(a_in)

        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            potential = np.zeros((1, len(atrans)))
            for c in range(self.nc):
                delta_r = self.r_inp - self.r_out[j][c]
                r = np.sqrt(np.sum(delta_r * delta_r, axis=1))
                potential += self.q[j][c] * self.potential(r, zeta=self.zeta[j][c])
            z[j] = self.b[0][j] + potential.dot(atrans)
        return z.transpose()

    def compute_z_convolution(self, a_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_shape[2], self.output_shape[1], self.output_shape[0], len(a_in)))
        self.z_pool_max_cache = np.zeros((self.output_shape[2], self.output_shape[1], self.output_shape[0], len(a_in)), dtype=np.int32)

        for j in range(self.output_size):
            z[j] += self.b[0][j]

            # x fast index, y slow index
            for jy in range(self.output_shape[1]):
                for jx in range(self.output_shape[0]):

                    z_pool = np.zeros((self.output_pool_shape[1] * self.output_pool_shape[0], len(a_in)))

                    for pool_jy in range(self.output_pool_shape[1]):
                        for pool_jx in range(self.output_pool_shape[0]):

                            for c in range(self.nc):
                                rjc = self.r_out[j][c]
                                rjc_x = rjc[0] + jx * self.output_delta[0] + pool_jx * self.output_pool_delta[0]
                                rjc_y = rjc[1] + jy * self.output_delta[1] + pool_jy * self.output_pool_delta[1]

                                for i in range(self.input_size):
                                    for iy in range(self.input_shape[1]):
                                        ri_y = self.r_inp[i][1] + iy * self.input_delta[1]
                                        for ix in range(self.input_shape[0]):
                                            ri_x = self.r_inp[i][0] + ix * self.input_delta[0]

                                            ioff = i*self.input_shape[1]*self.input_shape[0] + iy*self.input_shape[0] + ix

                                            dx = ri_x - rjc_x
                                            dy = ri_y - rjc_y
                                            dz = self.r_inp[i][2] - rjc[2]
                                            r = np.sqrt(dx*dx + dy*dy + dz*dz)
                                            potential = self.potential(r, zeta=self.zeta[j][c])

                                            # Assumption: data is organized with x being fast (columns),
                                            # y being slow indexes (rows)
                                            z_pool[pool_jy * self.output_pool_shape[0] + pool_jx] += self.q[j][c] * potential * atrans[ioff]
                                            # z[j][jy][jx] += self.q[j][c] * potential * atrans[ioff]

                    # # Apply max pooling
                    pool_maxes = np.argmax(z_pool, axis=0)
                    for p, pval in enumerate(pool_maxes):
                        self.z_pool_max_cache[j][jy][jx][p] = pval
                        z[j][jy][jx][p] += z_pool[pval][p]

        z = z.reshape((self.output_shape[2] * self.output_shape[1] * self.output_shape[0], len(a_in)))

        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self):
        wt = np.zeros((self.output_size, self.input_size))

        for j in range(self.output_size):
            potential = np.zeros((1, self.input_size))
            for c in range(self.nc):
                delta_r = self.r_inp - self.r_out[j][c]
                r = np.sqrt(np.sum(delta_r * delta_r, axis=1))
                potential += self.q[j][c] * self.potential(r)
            wt[j] = potential

        self.w = wt.transpose()
        return self.w
