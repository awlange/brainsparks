from .layer import Layer
from ..activation import Activation

import numpy as np


class Particle5Input(object):
    def __init__(self, output_size, s=1.0, t=None, phase_enabled=True):
        self.output_size = output_size
        self.phase_enabled = phase_enabled

        # Positions
        self.rx = np.random.normal(0.0, s, output_size)
        self.ry = np.random.normal(0.0, s, output_size)
        self.rz = np.random.normal(0.0, s, output_size)
        self.rx2 = np.random.normal(0.0, s, output_size)
        self.ry2 = np.random.normal(0.0, s, output_size)
        self.rz2 = np.random.normal(0.0, s, output_size)

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
            self.theta2 = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)
            self.theta2 = np.random.uniform(0, 2*np.pi, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.rx2, self.ry2, self.rz2, self.theta, self.theta2

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())


class Particle5(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0, t=None, q=None, b=None, boff=0.0,
                 phase_enabled=True, qw=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.phase_enabled = phase_enabled

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))

        # Charges
        if q is None:
            q = g
        # self.q = np.ones(output_size)
        self.q = np.random.uniform(-q, q, output_size)
        # self.q = np.random.normal(loc=0.0, scale=qw, size=output_size) + np.random.choice([q, -q], size=output_size)
        # self.q = np.random.choice([q, -q], size=output_size)

        self.q2 = np.random.uniform(-q, q, output_size)

        # Positions
        # self.rx = np.random.uniform(-s, s, output_size)
        # self.ry = np.random.uniform(-s, s, output_size)
        # self.rz = np.random.uniform(-s, s, output_size)
        self.rx = np.random.normal(0.0, s, output_size)
        self.ry = np.random.normal(0.0, s, output_size)
        self.rz = np.random.normal(0.0, s, output_size)
        self.rx2 = np.random.normal(0.0, s, output_size)
        self.ry2 = np.random.normal(0.0, s, output_size)
        self.rz2 = np.random.normal(0.0, s, output_size)

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
            self.theta2 = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)
            self.theta2 = np.random.uniform(0, 2*np.pi, output_size)

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.rx2, self.ry2, self.rz2, self.theta, self.theta2

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
        r_in_x2 = r_in[3]
        r_in_y2 = r_in[4]
        r_in_z2 = r_in[5]
        r_in_theta = r_in[6]
        r_in_theta2 = r_in[7]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            dt = r_in_theta - self.theta[j]
            w_ji = self.q[j] * np.exp(-(dx**2 + dy**2 + dz**2)) * np.cos(dt)

            dx = r_in_x2 - self.rx2[j]
            dy = r_in_y2 - self.ry2[j]
            dz = r_in_z2 - self.rz2[j]
            dt = r_in_theta2 - self.theta2[j]
            w_ji += self.q2[j] * np.exp(-(dx**2 + dy**2 + dz**2)) * np.cos(dt)

            z[j] = self.b[0][j] + w_ji.dot(atrans)

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
        r_in_x2 = r_in[3]
        r_in_y2 = r_in[4]
        r_in_z2 = r_in[5]
        r_in_theta = r_in[6]
        r_in_theta2 = r_in[7]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            dt = r_in_theta - self.theta[j]
            w_ji = self.q[j] * np.exp(-(dx ** 2 + dy ** 2 + dz ** 2)) * np.cos(dt)

            dx = r_in_x2 - self.rx2[j]
            dy = r_in_y2 - self.ry2[j]
            dz = r_in_z2 - self.rz2[j]
            dt = r_in_theta2 - self.theta2[j]
            w_ji += self.q2[j] * np.exp(-(dx ** 2 + dy ** 2 + dz ** 2)) * np.cos(dt)

            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w
