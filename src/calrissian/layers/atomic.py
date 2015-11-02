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
        self.b = np.random.uniform(-s, s, output_size)

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
        z = np.copy(self.b)
        for j in range(len(self.q)):
            q_j = self.q[j]
            r_j = self.r[j]
            tmp = 0.0
            for i in range(len(a_in)):
                q_i = a_in[i]
                d_ij = Point.distance(r_in[i], r_j)
                tmp += q_i * np.exp(-d_ij)  # exponential pairwise kernel
            z[j] += q_j * tmp
        return np.asarray(z)

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, A, sigma_Z=None, dc_dw_l=None):
        dc_db = prev_delta if sigma_Z is None else self.w.dot(prev_delta) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw):
        return dc_db, dc_dw

    def forward_pass(self, a_in):
        return self.activation(a_in.dot(self.w) + self.b)

    def backward_pass(self):
        pass
