from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class ParticleVectorInput(object):
    def __init__(self, output_size, rs=1.0, ns=1.0):
        self.output_size = output_size

        # Positions
        self.rx = np.random.normal(0.0, rs, output_size)
        self.ry = np.random.normal(0.0, rs, output_size)
        # self.rz = np.random.normal(0.0, s, output_size)
        # self.rw = np.random.normal(0.0, s, output_size)
        self.rz = np.zeros(output_size)
        self.rw = np.zeros(output_size)

        # Vectors
        self.nx = np.random.normal(0.0, ns, output_size)
        self.ny = np.random.normal(0.0, ns, output_size)
        # self.nz = np.random.normal(0.0, s, output_size)
        # self.nw = np.random.normal(0.0, s, output_size)
        self.nz = np.zeros(output_size)
        self.nw = np.zeros(output_size)
        # self.normalize(np.sqrt(output_size))

        # Widths
        self.zeta = np.abs(np.random.normal(1.0, 0.1, output_size))
        # self.zeta = np.ones(output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.rw, self.nx, self.ny, self.nz, self.nw, self.zeta

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())

    def normalize(self, s=1.0):
        """
        Ensure that vectors are normalized
        """
        d = s * np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2 + self.nw ** 2)
        self.nx /= d
        self.ny /= d
        self.nz /= d
        self.nw /= d


class ParticleVector(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", potential="gaussian",
                 s=1.0, q=None, b=None, boff=0.0, rs=1.0, ns=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))

        # Positions
        self.rx = np.random.normal(0.0, rs, output_size)
        self.ry = np.random.normal(0.0, rs, output_size)
        # self.rz = np.random.normal(0.0, s, output_size)
        # self.rw = np.random.normal(0.0, s, output_size)
        self.rz = np.zeros(output_size)
        self.rw = np.zeros(output_size)

        # Vectors
        self.nx = np.random.normal(0.0, ns, output_size)
        self.ny = np.random.normal(0.0, ns, output_size)
        # self.nz = np.random.normal(0.0, s, output_size)
        # self.nw = np.random.normal(0.0, s, output_size)
        self.nz = np.zeros(output_size)
        self.nw = np.zeros(output_size)
        # self.normalize(np.sqrt(output_size))

        # Widths
        self.zeta = np.abs(np.random.normal(1.0, 0.1, output_size))
        # self.zeta = np.ones(output_size)

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.rw, self.nx, self.ny, self.nz, self.nw, self.zeta

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def normalize(self, s=1.0):
        """
        Ensure that vectors are normalized
        """
        d = s * np.sqrt(self.nx ** 2 + self.ny ** 2 + self.nz ** 2 + self.nw ** 2)
        self.nx /= d
        self.ny /= d
        self.nz /= d
        self.nw /= d

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
        r_in_w = r_in[3]
        r_in_nx = r_in[4]
        r_in_ny = r_in[5]
        r_in_nz = r_in[6]
        r_in_nw = r_in[7]
        r_in_zeta = r_in[8]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            dw = r_in_w - self.rw[j]
            d = np.sqrt(dx**2 + dy**2 + dz**2 + dw**2)
            d *= np.sqrt(np.abs(self.zeta[j] * r_in_zeta))  # width
            w_ji = self.potential(d)
            dot = r_in_nx * self.nx[j] + r_in_ny * self.ny[j] + r_in_nz * self.nz[j] + r_in_nw * self.nw[j]
            w_ji *= dot
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
        r_in_w = r_in[3]
        r_in_nx = r_in[4]
        r_in_ny = r_in[5]
        r_in_nz = r_in[6]
        r_in_nw = r_in[7]
        r_in_zeta = r_in[8]

        for j in range(self.output_size):
            dx = r_in_x - self.rx[j]
            dy = r_in_y - self.ry[j]
            dz = r_in_z - self.rz[j]
            dw = r_in_w - self.rw[j]
            d = np.sqrt(dx**2 + dy**2 + dz**2 + dw**2)
            d *= np.sqrt(np.abs(self.zeta[j] * r_in_zeta))  # width
            w_ji = self.potential(d)
            dot = r_in_nx * self.nx[j] + r_in_ny * self.ny[j] + r_in_nz * self.nz[j] + r_in_nw * self.nw[j]
            w_ji *= dot
            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w

    def compute_w_j(self, r_in, j):
        return None

    def compute_w_i(self, r_in, i):
        return None
