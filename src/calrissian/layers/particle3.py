from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class Particle3(object):
    """
    2D monopole input, dipole output
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", potential="gaussian", s=1.0, q=None, b=None):

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)

        self.w = None

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(-b, b, (1, output_size))

        # Charges
        if q is None:
            q = g
        self.q = np.random.uniform(-q, q, output_size)

        self.rx_inp = np.random.uniform(-s, s, input_size)
        self.ry_inp = np.random.uniform(-s, s, input_size)

        self.rx_pos_out = np.random.uniform(-s, s, output_size)
        self.ry_pos_out = np.random.uniform(-s, s, output_size)
        self.rx_neg_out = np.random.uniform(-s, s, output_size)
        self.ry_neg_out = np.random.uniform(-s, s, output_size)

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            dx = self.rx_inp - self.rx_pos_out[j]
            dy = self.ry_inp - self.ry_pos_out[j]
            # potential = np.exp(-(dx**2 + dy**2))
            r = np.sqrt(dx ** 2 + dy ** 2)
            potential = self.potential(r)

            dx = self.rx_inp - self.rx_neg_out[j]
            dy = self.ry_inp - self.ry_neg_out[j]
            # potential -= np.exp(-(dx**2 + dy**2))
            r = np.sqrt(dx ** 2 + dy ** 2)
            potential -= self.potential(r)

            z[j] = self.b[0][j] + self.q[j] * potential.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self):
        w = np.zeros((self.input_size, self.output_size))
        for j in range(self.output_size):
            dx = self.rx_inp - self.rx_pos_out[j]
            dy = self.ry_inp - self.ry_pos_out[j]
            # potential = np.exp(-(dx**2 + dy**2))
            r = np.sqrt(dx ** 2 + dy ** 2)
            potential = self.potential(r)

            dx = self.rx_inp - self.rx_neg_out[j]
            dy = self.ry_inp - self.ry_neg_out[j]
            # potential -= np.exp(-(dx**2 + dy**2))
            r = np.sqrt(dx ** 2 + dy ** 2)
            potential -= self.potential(r)

            potential = self.q[j] * potential
            for i in range(self.input_size):
                w[i][j] = potential[i]

        self.w = w
        return w

    def compute_w_sum_square(self):
        total = 0.0
        for j in range(self.output_size):
            dx = self.rx_inp - self.rx_pos_out[j]
            dy = self.ry_inp - self.ry_pos_out[j]
            potential = np.exp(-(dx**2 + dy**2))

            dx = self.rx_inp - self.rx_neg_out[j]
            dy = self.ry_inp - self.ry_neg_out[j]
            potential -= np.exp(-(dx**2 + dy**2))

            potential = self.q[j] * potential
            total += np.sum(potential * potential)
        return total
