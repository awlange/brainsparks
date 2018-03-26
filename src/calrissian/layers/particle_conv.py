from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class ParticleConvInput(object):
    def __init__(self, output_size, s=1.0, nr=3, rand="uniform"):
        self.output_size = output_size
        self.nr = nr

        # Positions
        self.r = []
        for _ in range(output_size):
            if rand == "uniform":
                self.r.append(np.random.uniform(-s, s, nr))
            else:
                self.r.append(np.random.normal(0.0, s, nr))
        self.r = np.asarray(self.r)

    def get_rxyz(self):
        return self.r

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())


class ParticleConv(object):

    def __init__(self, output_size=0, rand="uniform",
                 activation="sigmoid", potential="gaussian",
                 nr=3, nc=1,
                 s=1.0, b=1.0):
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)
        self.nr = nr
        self.nc = nc

        # Weight initialization
        if rand == "uniform":
            self.b = np.random.uniform(-b, b, (1, output_size))
        else:
            self.b = np.random.normal(0.0, b, (1, output_size))

        # Coefficients
        self.q = None
        if rand == "uniform":
            self.q = np.random.uniform(-s, s, (output_size, nc))
        else:
            self.q = np.random.normal(0.0, s, (output_size, nc))

        # Positions
        self.r = None  # output positions
        self.rb = None  # basis positions
        if rand == "uniform":
            self.rb = np.random.uniform(-s, s, (output_size, nc, nr))
            self.r = np.random.uniform(-s, s, (output_size, nr))
        else:
            self.rb = np.random.normal(0.0, s, (output_size, nc, nr))
            self.r = np.random.normal(0.0, s, (output_size, nr))

        # Matrix
        self.w = None

    def get_rxyz(self):
        return self.r

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def compute_z(self, a_in, r_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))

        for j in range(self.output_size):
            w_ji = np.zeros((1, len(atrans)))
            for i in range(len(atrans)):
                for c in range(self.nc):
                    delta_r = r_in[i] - self.rb[j][c]
                    d = np.sqrt(np.sum(delta_r * delta_r))
                    w_ji[0][i] += self.q[j][c] * self.potential(d)

            z[j] = self.b[0][j] + w_ji.dot(atrans)

        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self, r_in):

        w = np.zeros((len(r_in), self.output_size))

        for j in range(self.output_size):
            for i in range(len(r_in)):
                for c in range(self.nc):
                    delta_r = r_in[i] - self.r[j][c]
                    d = np.sqrt(np.sum(delta_r * delta_r))
                    w[i][j] += self.q[j][c] * self.potential(d)

        self.w = w

        return w

