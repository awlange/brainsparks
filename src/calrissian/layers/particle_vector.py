from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleVectorInput(object):
    def __init__(self, output_size, s=1.0):
        self.output_size = output_size

        # Positions
        self.rx = np.random.normal(0.0, s, output_size)
        self.ry = np.random.normal(0.0, s, output_size)
        self.rz = np.random.normal(0.0, s, output_size)

        # Vectors
        self.nx = np.random.normal(0.0, s, output_size)
        self.ny = np.random.normal(0.0, s, output_size)
        self.nz = np.random.normal(0.0, s, output_size)

        # Normalize
        d = np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2)
        self.nx /= d
        self.ny /= d
        self.nz /= d

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.nx, self.ny, self.nz

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())


class ParticleVector(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0, q=None, b=None, boff=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))

        # Charges
        if q is None:
            q = g
        self.q = np.random.uniform(-q, q, output_size)

        # Positions
        self.rx = np.random.normal(0.0, s, output_size)
        self.ry = np.random.normal(0.0, s, output_size)
        self.rz = np.random.normal(0.0, s, output_size)

        # Vectors
        self.nx = np.random.normal(0.0, s, output_size)
        self.ny = np.random.normal(0.0, s, output_size)
        self.nz = np.random.normal(0.0, s, output_size)

        # Normalize
        d = np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2)
        self.nx /= d
        self.ny /= d
        self.nz /= d

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.nx, self.ny, self.nz

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def compute_z(self, a_in, r_in):
        """
        Vectorized v2.0

        :param a_in:
        :param r_in:
        :return:
        """
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        r_in_x = r_in[0]
        r_in_y = r_in[1]
        r_in_z = r_in[2]
        r_in_nx = r_in[3]
        r_in_ny = r_in[4]
        r_in_nz = r_in[5]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            w_ji = np.exp(-(dx**2 + dy**2 + dz**2))
            dot = r_in_nx * self.nx[j] + r_in_ny * self.ny[j] + r_in_nz * self.nz[j]
            w_ji *= dot
            z[j] = self.b[0][j] + self.q[j] * w_ji.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self, r_in):
        return None

    def compute_w_j(self, r_in, j):
        return None

    def compute_w_i(self, r_in, i):
        return None
