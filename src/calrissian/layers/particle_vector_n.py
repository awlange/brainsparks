from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class ParticleVectorNInput(object):
    def __init__(self, output_size, nr=3, nv=3, sr=1.0, sv=1.0):
        self.output_size = output_size
        self.nv = nv
        self.nr = nr

        # Positions
        self.positions = []
        for i in range(nr):
            self.positions.append(np.random.normal(0.0, sr, output_size))

        # Vectors
        self.nvectors = []
        for i in range(nv):
            self.nvectors.append(np.random.normal(0.0, sv, output_size))
            # g = np.sqrt(1.0 / output_size)
            # self.nvectors.append(np.random.uniform(-g, g, output_size))
        # self.normalize()

    def get_rxyz(self):
        return self.positions, self.nvectors

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())

    def normalize(self):
        d = 0.0
        for v in range(self.nv):
            d += self.nvectors[v]**2
        d = np.sqrt(d)
        for v in range(self.nv):
            self.nvectors[v] /= d


class ParticleVectorN(object):

    def __init__(self, input_size=0, output_size=0, nr=3, nv=3, activation="sigmoid", potential="gaussian",
                 sr=1.0, sv=1.0, q=None, b=None, boff=0.0, uniform=False, p_dropout=-1.0, sigma_r=-1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.nr = nr
        self.nv = nv
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)
        self.p_dropout = p_dropout
        self.dropout_mask = None
        self.sigma_r = sigma_r

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))
        # self.b = np.random.normal(0.0, sv, (1, output_size))

        # Positions
        self.positions = []
        for i in range(nr):
            if uniform:
                self.positions.append(np.random.uniform(-sr, sr, output_size))
            else:
                self.positions.append(np.random.normal(0.0, sr, output_size))

        # Vectors
        self.nvectors = []
        for i in range(nv):
            self.nvectors.append(np.random.normal(0.0, sv, output_size))
            # self.nvectors.append(np.random.uniform(-g, g, output_size))
        # self.normalize()

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.positions, self.nvectors

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def normalize(self):
        d = 0.0
        for v in range(self.nv):
            d += self.nvectors[v]**2
        d = np.sqrt(d)
        for v in range(self.nv):
            self.nvectors[v] /= d

    def compute_z(self, a_in, r_in, apply_input_noise=False):
        """
        Vectorized v2.0

        :param a_in:
        :param r_in:
        :return:
        """
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        r_positions = r_in[0]
        r_nvectors = r_in[1]

        if apply_input_noise and self.sigma_r > 0.0:
            for r in range(self.nr):
                r_positions[r] += np.random.normal(0.0, self.sigma_r, r_positions[r].shape)

        for j in range(self.output_size):
            dd = 0.0
            for r in range(self.nr):
                dd += (r_positions[r] - self.positions[r][j])**2
            d = np.sqrt(dd)
            dot = 0.0
            for v in range(self.nv):
                dot += r_nvectors[v] * self.nvectors[v][j]
            w_ji = self.potential(d) * dot
            z[j] = self.b[0][j] + w_ji.dot(atrans)
        return z.transpose()

    def compute_a(self, z, apply_dropout=False):
        a = self.activation(z)
        if apply_dropout and self.p_dropout > 0.0:
            self.dropout_mask = np.random.binomial(1, self.p_dropout, a.shape)
            a *= self.dropout_mask
        return a

    def compute_da(self, z, apply_dropout=False):
        da = self.d_activation(z)
        if apply_dropout and self.p_dropout > 0.0:
            da *= self.dropout_mask
        return da

    def compute_w(self, r_in):
        w = np.zeros((self.input_size, self.output_size))
        r_positions = r_in[0]
        r_nvectors = r_in[1]

        for j in range(self.output_size):
            dd = 0.0
            for r in range(self.nr):
                dd += (r_positions[r] - self.positions[r][j])**2
            d = np.sqrt(dd)
            dot = 0.0
            for v in range(self.nv):
                dot += r_nvectors[v] * self.nvectors[v][j]
            w_ji = self.potential(d) * dot
            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w

    def compute_w_j(self, r_in, j):
        return None

    def compute_w_i(self, r_in, i):
        return None
