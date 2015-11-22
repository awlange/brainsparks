from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleInput(object):
    def __init__(self, output_size):
        self.output_size = output_size

        # Positions
        s = 1.0
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())


class Particle(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", zeta=0.5):
        # super().__init__("Atomic", True)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.zeta = zeta

        # Weight initialization
        s = 0.1
        self.b = np.zeros((1, output_size))

        # Charges
        s = 1.0 / np.sqrt(self.input_size)
        self.q = np.random.uniform(-s, s, output_size)

        # Positions
        s = 1.0
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

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
        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            w_ji = np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
            z[j] = self.b[0][j] + self.q[j] * w_ji.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

