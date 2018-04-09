from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class Particle333(object):
    """
    Monopole input, multipole output
    """

    def __init__(self, input_size=1, output_size=1, activation="sigmoid", potential="gaussian",
                 nr=3, nc=1, rand="uniform",
                 s=1.0, q=1.0, b=1.0):

        self.input_size = input_size
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
            self.q = np.random.uniform(-q, q, (output_size, nc))
        else:
            self.q = np.random.normal(0.0, q, (output_size, nc))

        # Positions
        self.r_inp = None  # input positions
        self.r_out = None  # output positions
        if rand == "uniform":
            self.r_out = np.random.uniform(-s, s, (output_size, nc, nr))
            self.r_inp = np.random.uniform(-s, s, (input_size, nr))
        else:
            self.r_out = np.random.normal(0.0, s, (output_size, nc, nr))
            self.r_inp = np.random.normal(0.0, s, (input_size, nr))

        self.w = None

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            potential = np.zeros((1, len(atrans)))
            for c in range(self.nc):
                delta_r = self.r_inp - self.r_out[j][c]
                r = np.sqrt(np.sum(delta_r * delta_r, axis=1))
                potential += self.q[j][c] * self.potential(r)
            z[j] = self.b[0][j] + potential.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self):
        wt = np.zeros((self.output_size, self.input_size))

        for j in range(self.output_size):
            potential = np.zeros((1, self.input_size))
            for c in range(self.nc):
                delta_r = self.r_inp - self.r_out[j][c]
                r = np.sqrt(np.sum(delta_r * delta_r, axis=1))
                potential += self.q[j][c] * self.potential(r)
            wt[j] = potential

        self.w = wt.transpose()
        return self.w
