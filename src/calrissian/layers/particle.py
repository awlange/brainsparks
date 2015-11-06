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


class Particle(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid"):
        # super().__init__("Atomic", True)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Weight initialization
        s = 0.1
        self.b = np.random.uniform(-s, s, (1, output_size))

        # Charges
        s = 0.1
        self.q = np.random.uniform(-s, s, output_size)

        # Positions
        s = 1.0
        self.r = np.random.uniform(-s, s, (output_size, 3))

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), self.r

    def compute_z(self, a_in, r_in):
        z = np.zeros((len(a_in), self.output_size))
        for j in range(self.output_size):
            for k, a in enumerate(a_in):
                z[k][j] = self.b[0][j]
            q_j = self.q[j]
            r_j = self.r[j]
            for i in range(len(r_in)):
                dx = r_in[i][0] - r_j[0]
                dy = r_in[i][1] - r_j[1]
                dz = r_in[i][2] - r_j[2]
                d_ij = math.sqrt(dx*dx + dy*dy + dz*dz)
                w_ji = q_j * np.exp(-d_ij)  # exponential pairwise kernel
                for k, a in enumerate(a_in):
                    z[k][j] += w_ji * a[i]
        return z

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def build_weight_matrix(self, r_in):
        """
        For convenience in comparison to dense layer
        """
        w = np.zeros((self.output_size, self.input_size))
        for j in range(len(self.q)):
            q_j = self.q[j]
            r_j = self.r[j]
            for i in range(len(r_in)):
                dx = r_in[i][0] - r_j[0]
                dy = r_in[i][1] - r_j[1]
                dz = r_in[i][2] - r_j[2]
                d_ij = math.sqrt(dx*dx + dy*dy + dz*dz)
                w[j][i] = q_j * np.exp(-d_ij)  # exponential pairwise kernel
        return w

    def build_kernel_matrix(self, r_in):
        """
        For convenience in comparison to dense layer
        """
        K = np.zeros((self.output_size, self.input_size))
        for j in range(len(self.q)):
            r_j = self.r[j]
            for i in range(len(r_in)):
                dx = r_in[i][0] - r_j[0]
                dy = r_in[i][1] - r_j[1]
                dz = r_in[i][2] - r_j[2]
                d_ij = math.sqrt(dx*dx + dy*dy + dz*dz)
                K[j][i] = math.exp(-d_ij)  # exponential pairwise kernel
        return K

    def build_distance_matrix(self, r_in):
        """
        For convenience in comparison to dense layer
        """
        D = np.zeros((self.output_size, self.input_size))
        for j in range(len(self.q)):
            r_j = self.r[j]
            for i in range(len(r_in)):
                dx = r_in[i][0] - r_j[0]
                dy = r_in[i][1] - r_j[1]
                dz = r_in[i][2] - r_j[2]
                D[j][i] = math.sqrt(dx*dx + dy*dy + dz*dz)
        return D
