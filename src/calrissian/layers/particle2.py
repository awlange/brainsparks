from .layer import Layer
from ..activation import Activation

import numpy as np


class Particle2(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", zeta=1.0, s=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.zeta = zeta

        # Weight initialization
        c = np.sqrt(6.0 / (input_size + output_size))
        self.b = np.random.uniform(-c, c, (1, output_size))

        # Charges
        c = 1.0
        self.q = np.random.uniform(-c, c, output_size)

        # Positions
        self.rx_inp = np.random.uniform(-s, s, input_size)
        self.ry_inp = np.random.uniform(-s, s, input_size)
        self.rz_inp = np.random.uniform(-s, s, input_size)
        self.rx_out = np.random.uniform(-s, s, output_size)
        self.ry_out = np.random.uniform(-s, s, output_size)
        self.rz_out = np.random.uniform(-s, s, output_size)

        # Phase
        self.theta_inp = np.random.uniform(0, 2*np.pi, input_size)
        self.theta_out = np.random.uniform(0, 2*np.pi, output_size)

        # Matrix
        self.w = None

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            dx = self.rx_inp - self.rx_out[j]
            dy = self.ry_inp - self.ry_out[j]
            dz = self.rz_inp - self.rz_out[j]
            w_ji = np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
            dt = self.theta_inp - self.theta_out[j]
            w_ji *= np.cos(dt)
            z[j] = self.b[0][j] + self.q[j] * w_ji.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)
