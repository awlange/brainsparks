from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleDipoleTreeInput(object):
    """
    Particle dipole approach with treecode for fast potential evaluation

    Particle positions are ordered such that even indexes are positive, odd are negative.
    Paired by (n, n+1)
    """

    def __init__(self, output_size, s=1.0, cut=10.0):
        self.output_size = output_size

        self.cut = cut
        self.cut2 = cut*cut

        # Particle positions
        self.rx = np.random.uniform(-s, s, 2*output_size)
        self.ry = np.random.uniform(-s, s, 2*output_size)
        self.rz = np.random.uniform(-s, s, 2*output_size)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())


class ParticleDipoleTree(object):
    """
    Particle dipole approach with treecode for fast potential evaluation

    Particle positions are ordered such that even indexes are positive, odd are negative.
    Paired by (n, n+1)
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0, cut=10.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.cut = cut
        self.cut2 = cut*cut

        # Weight initialization
        c = np.sqrt(1.0 / (input_size + output_size))
        self.b = np.random.uniform(-c, c, (1, output_size))

        # Charges
        c = 1.0
        self.q = np.random.uniform(-c, c, output_size)

        # Particle positions
        self.rx = np.random.uniform(-s, s, 2*output_size)
        self.ry = np.random.uniform(-s, s, 2*output_size)
        self.rz = np.random.uniform(-s, s, 2*output_size)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), self.get_rxyz()

    def compute_z(self, a_in, r_in):
        """
        Use treecode to compute
        """
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))

        r_in_x = r_in[0]
        r_in_y = r_in[1]
        r_in_z = r_in[2]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            potential = np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential = np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential -= np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            z[j] = self.b[0][j] + self.q[j] * potential.dot(atrans)

        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)
