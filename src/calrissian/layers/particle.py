from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleInput(object):
    def __init__(self, output_size, s=1.0, t=1.0, phase_enabled=False):
        self.output_size = output_size
        self.phase_enabled = phase_enabled

        # Positions
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)
        # self.rz = np.zeros(output_size)

        # Phase
        self.theta = np.random.uniform(0, 2*np.pi, output_size)
        # self.theta = np.random.uniform(-t, t, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.theta

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())


class Particle(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", zeta=1.0, s=1.0, t=1.0, q=None, b=None, boff=0.0,
                 phase_enabled=False, qw=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.phase_enabled = phase_enabled

        self.zeta = zeta

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))

        # Charges
        if q is None:
            q = g
        # self.q = np.ones(output_size)
        # self.q = np.random.uniform(-q, q, output_size)
        # self.q = np.random.normal(loc=0.0, scale=qw, size=output_size) + np.random.choice([q, -q], size=output_size)
        self.q = np.random.choice([q, -q], size=output_size)

        # Positions
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)
        # self.rz = np.zeros(output_size)

        # Phase
        self.theta = np.random.uniform(0, 2*np.pi, output_size)
        # self.theta = np.random.uniform(-t, t, output_size)

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.theta

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
        r_in_theta = r_in[3]

        if self.phase_enabled:
            for j in range(self.output_size):
                dx = r_in_x - self.rx[j]
                dy = r_in_y - self.ry[j]
                dz = r_in_z - self.rz[j]
                w_ji = np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
                dt = r_in_theta - self.theta[j]
                w_ji *= np.cos(dt)
                z[j] = self.b[0][j] + self.q[j] * w_ji.dot(atrans)
        else:
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

    def compute_w(self, r_in):

        w = np.zeros((self.input_size, self.output_size))

        r_in_x = r_in[0]
        r_in_y = r_in[1]
        r_in_z = r_in[2]
        r_in_theta = r_in[3]
        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            w_ji = self.q[j] * np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
            if self.phase_enabled:
                dt = r_in_theta - self.theta[j]
                w_ji *= np.cos(dt)
            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w

    def compute_w_j(self, r_in, j):
        """
        Only update the j-th column values
        """

        r_in_x = r_in[0]
        r_in_y = r_in[1]
        r_in_z = r_in[2]
        r_in_theta = r_in[3]
        dx = r_in_x - self.rx[j]
        dy = r_in_y - self.ry[j]
        dz = r_in_z - self.rz[j]
        w_ji = np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
        if self.phase_enabled:
            dt = r_in_theta - self.theta[j]
            w_ji *= self.q[j] * np.cos(dt)
        for i in range(self.input_size):
            self.w[i][j] = w_ji[i]

    def compute_w_i(self, r_in, i):
        """
        Only update the i-th row values
        """

        r_in_x = r_in[0][i]
        r_in_y = r_in[1][i]
        r_in_z = r_in[2][i]
        r_in_theta = r_in[3][i]
        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            w_ji = np.exp(-self.zeta * (dx**2 + dy**2 + dz**2))
            if self.phase_enabled:
                dt = r_in_theta - self.theta[j]
                w_ji *= self.q[j] * np.cos(dt)
            self.w[i][j] = w_ji[i]
