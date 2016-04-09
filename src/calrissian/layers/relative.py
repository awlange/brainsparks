from .layer import Layer
from ..activation import Activation

import numpy as np


class RelativeInput(object):
    def __init__(self, output_size, s=1.0):
        self.output_size = output_size

        # Relative variable
        self.x = np.random.uniform(-s, s, output_size)
        self.q = np.random.uniform(-s, s, output_size)

    def get_vars(self):
        return self.q, self.x

    def feed_forward(self, a_in):
        return a_in, self.get_vars()


class Relative(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Weight initialization
        c = np.sqrt(1.0 / (input_size + output_size))
        self.b = np.random.uniform(-c, c, (1, output_size))

        # Relative variable
        self.x = np.random.uniform(-s, s, output_size)
        self.q = np.random.uniform(-s, s, output_size)

    def get_vars(self):
        return self.q, self.x

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_vars())

    def compute_z(self, a_in, v_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))

        v_in_q = v_in[0]
        v_in_x = v_in[1]

        for j in range(self.output_size):
            w_ji = self.q[j] * (v_in_x - self.x[j])
            z[j] = self.b[0][j] + w_ji.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)
