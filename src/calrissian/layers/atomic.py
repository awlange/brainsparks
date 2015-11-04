from .layer import Layer
from ..activation import Activation

import numpy as np


class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def distance(p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)


class AtomicInput(object):
    def __init__(self, size):
        self.output_size = size

        # Positions
        s = 3.0
        self.r = []
        for i in range(size):
            self.r.append(Point(np.random.uniform(-s, s), np.random.uniform(-s, s), np.random.uniform(-s, s)))


class Atomic(object):

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
        self.q = np.random.uniform(-s, s, output_size)

        # Positions
        s = 3.0
        self.r = []
        for i in range(output_size):
            self.r.append(Point(np.random.uniform(-s, s), np.random.uniform(-s, s), np.random.uniform(-s, s)))

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), self.r

    def compute_z(self, a_in, r_in):
        # More explicitly build the weight matrix in terms of charges and positions
        w = self.build_weight_matrix(r_in)
        return (w.dot(a_in) + self.b)[0]  # TODO... meh

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, A, r_in, sigma_Z=None):
        dc_db = None
        delta_q = None
        if sigma_Z is None:
            dc_db = prev_delta
            delta_q = dc_db  # not sure about this...
        else:
            w = self.build_weight_matrix(r_in)
            dc_db = prev_delta.dot(w) * sigma_Z

            # K = self.build_kernel_matrix(r_in)
            # dot_k = K.dot(A[1])
            # delta_q = K.sum(axis=0) * sigma_Z

        # TODO

        # dc_dq = np.outer(A, delta_q)
        dc_dw = np.outer(A, dc_db)

        return dc_db, dc_dw

    def build_weight_matrix(self, r_in):
        """
        For convenience in comparison to dense layer
        """
        w = np.zeros((self.output_size, self.input_size))
        for j in range(len(self.q)):
            q_j = self.q[j]
            r_j = self.r[j]
            for i in range(len(r_in)):
                d_ij = Point.distance(r_in[i], r_j)
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
                d_ij = Point.distance(r_in[i], r_j)
                K[j][i] = np.exp(-d_ij)  # exponential pairwise kernel
        return K

    def build_distance_matrix(self, r_in):
        """
        For convenience in comparison to dense layer
        """
        D = np.zeros((self.output_size, self.input_size))
        for j in range(len(self.q)):
            r_j = self.r[j]
            for i in range(len(r_in)):
                D[j][i] = Point.distance(r_in[i], r_j)
        return D
