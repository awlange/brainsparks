from .layer import Layer
from ..activation import Activation

import numpy as np


class Particle4Input(object):
    def __init__(self, output_size, r_dim=3, s=1.0, t=None, phase_enabled=True):
        self.output_size = output_size
        self.phase_enabled = phase_enabled
        self.r_dim = r_dim

        # Positions
        self.r = np.random.normal(0.0, s, (output_size, r_dim))

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)

    def get_rxyz(self):
        return self.r, self.theta,

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())


class Particle4(object):

    def __init__(self, input_size=0, output_size=0, r_dim=3, activation="sigmoid", s=1.0, t=None, q=None, b=None, boff=0.0,
                 phase_enabled=True):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.phase_enabled = phase_enabled
        self.r_dim = r_dim

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
        self.r = np.random.normal(0.0, s, (output_size, r_dim))

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.r, self.theta

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
        r_in_r = r_in[0]
        r_in_theta = r_in[1]

        for j in range(self.output_size):
            dr = r_in_r - self.r[j]
            dd = np.sum(dr*dr, axis=1)
            dt = r_in_theta - self.theta[j]
            w_ji = np.exp(-dd) * np.cos(dt)
            z[j] = self.b[0][j] + self.q[j] * w_ji.dot(atrans)

        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self, r_in):
        w = np.zeros((self.input_size, self.output_size))
        r_in_r = r_in[0]
        r_in_theta = r_in[1]
        for j in range(self.output_size):
            dr = r_in_r - self.r[j]
            dd = np.sum(dr*dr, axis=1)
            dt = r_in_theta - self.theta[j]
            w_ji = np.exp(-dd) * np.cos(dt)
            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w
