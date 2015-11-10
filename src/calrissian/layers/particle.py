from .layer import Layer
from ..activation import Activation

import numpy as np
import math


class ParticleInput(object):
    def __init__(self, size):
        self.size = size

        # Positions
        s = 1.0
        self.r = np.random.uniform(-s, s, (size, 3))

        # Charges
        s = 1.0
        # self.q = np.random.uniform(-s, s, size)
        self.q = np.ones(size)

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in * self.q, self.r
        # return a_in, self.r


class Particle(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid"):
        # super().__init__("Atomic", True)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Weight initialization
        s = 1.0
        self.b = np.random.uniform(-s, s, (1, output_size))

        # Charges
        s = 0.1
        self.q = np.random.uniform(-s, s, output_size)
        # self.q = np.ones(output_size) + np.random.uniform(-s, s, output_size)
        # for i in range(output_size):
        #     if np.random.uniform(0, 1) > 0.5:
        #         self.q[i] *= -1.0

        # Positions
        s = 1.0
        self.r = np.random.uniform(-s, s, (output_size, 3))

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), self.r

    def compute_z(self, a_in, r_in):
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))
        for j in range(self.output_size):
            zj = z[j]
            q_j = self.q[j]
            r_jx = self.r[j][0]
            r_jy = self.r[j][1]
            r_jz = self.r[j][2]
            for i in range(len(r_in)):
                dx = r_in[i][0] - r_jx
                dy = r_in[i][1] - r_jy
                dz = r_in[i][2] - r_jz
                d2 = dx*dx + dy*dy + dz*dz
                w_ji = math.exp(-d2)  # gaussian pairwise kernel
                zj += w_ji * atrans[i]
            zj *= q_j
            bj = self.b[0][j]
            for k, a in enumerate(a_in):
                zj[k] += bj
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

