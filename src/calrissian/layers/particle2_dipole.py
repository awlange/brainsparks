from .layer import Layer
from ..activation import Activation

import numpy as np


class Particle2Dipole(object):
    """
    Dipole approximated as 2 coupled charges of equal magnitude, uncoupled
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", k_bond=1.0, k_eq=0.1, s=1.0, cut=10.0,
                 q=None, b=None):

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.w = None

        # Harmonic constraint coefficient and equilibrium
        self.k_bond = k_bond
        self.k_eq = k_eq
        self.cut = cut
        self.cut2 = cut*cut

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(-b, b, (1, output_size))

        # Charges
        if q is None:
            q = g
        # self.q = np.random.uniform(-q, q, output_size)
        self.q = np.random.choice([q, -q], size=output_size)

        self.rx_pos_inp = np.random.normal(0.0, s, input_size)
        self.ry_pos_inp = np.random.normal(0.0, s, input_size)
        self.rz_pos_inp = np.random.normal(0.0, s, input_size)
        self.rx_neg_inp = np.random.normal(0.0, s, input_size)
        self.ry_neg_inp = np.random.normal(0.0, s, input_size)
        self.rz_neg_inp = np.random.normal(0.0, s, input_size)

        self.rx_pos_out = np.random.normal(0.0, s, output_size)
        self.ry_pos_out = np.random.normal(0.0, s, output_size)
        self.rz_pos_out = np.random.normal(0.0, s, output_size)
        self.rx_neg_out = np.random.normal(0.0, s, output_size)
        self.ry_neg_out = np.random.normal(0.0, s, output_size)
        self.rz_neg_out = np.random.normal(0.0, s, output_size)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            dx = self.rx_pos_inp - self.rx_pos_out[j]
            dy = self.ry_pos_inp - self.ry_pos_out[j]
            dz = self.rz_pos_inp - self.rz_pos_out[j]
            potential = np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_pos_inp - self.rx_neg_out[j]
            dy = self.ry_pos_inp - self.ry_neg_out[j]
            dz = self.rz_pos_inp - self.rz_neg_out[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_neg_inp - self.rx_pos_out[j]
            dy = self.ry_neg_inp - self.ry_pos_out[j]
            dz = self.rz_neg_inp - self.rz_pos_out[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_neg_inp - self.rx_neg_out[j]
            dy = self.ry_neg_inp - self.ry_neg_out[j]
            dz = self.rz_neg_inp - self.rz_neg_out[j]
            potential += np.exp(-(dx**2 + dy**2 + dz**2))

            z[j] = self.b[0][j] + self.q[j] * potential.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self):
        w = np.zeros((self.input_size, self.output_size))
        for j in range(self.output_size):
            dx = self.rx_pos_inp - self.rx_pos_out[j]
            dy = self.ry_pos_inp - self.ry_pos_out[j]
            dz = self.rz_pos_inp - self.rz_pos_out[j]
            potential = np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_pos_inp - self.rx_neg_out[j]
            dy = self.ry_pos_inp - self.ry_neg_out[j]
            dz = self.rz_pos_inp - self.rz_neg_out[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_neg_inp - self.rx_pos_out[j]
            dy = self.ry_neg_inp - self.ry_pos_out[j]
            dz = self.rz_neg_inp - self.rz_pos_out[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = self.rx_neg_inp - self.rx_neg_out[j]
            dy = self.ry_neg_inp - self.ry_neg_out[j]
            dz = self.rz_neg_inp - self.rz_neg_out[j]
            potential += np.exp(-(dx**2 + dy**2 + dz**2))

            potential = self.q[j] * potential
            for i in range(self.input_size):
                w[i][j] = potential[i]

        self.w = w
        return w
