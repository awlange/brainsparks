from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleSparseInput(object):
    def __init__(self, output_size, s=1.0, t=None, phase_enabled=True, ktop=None):
        self.output_size = output_size
        self.phase_enabled = phase_enabled
        self.ktop = output_size if ktop is None else ktop

        # Positions
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.theta

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())


class ParticleSparse(object):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0, t=None, q=None, b=None, boff=0.0,
                 phase_enabled=True, ktop=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.phase_enabled = phase_enabled
        self.ktop = output_size if ktop is None else ktop

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(boff - b, boff + b, (1, output_size))

        # Charges
        if q is None:
            q = g
        self.q = np.random.uniform(-q, q, output_size)

        # Positions
        self.rx = np.random.uniform(-s, s, output_size)
        self.ry = np.random.uniform(-s, s, output_size)
        self.rz = np.random.uniform(-s, s, output_size)

        # Phase
        if t is not None:
            self.theta = np.random.uniform(-t, t, output_size)
        else:
            self.theta = np.random.uniform(0, 2*np.pi, output_size)

    def get_rxyz(self):
        return self.rx, self.ry, self.rz, self.theta

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def compute_z(self, a_in, r_in):
        """
        :param a_in: sparse map of index to value
        :param r_in:
        :return:
        """
        z = np.zeros((len(a_in), self.output_size))
        r_in_x = r_in[0]
        r_in_y = r_in[1]
        r_in_z = r_in[2]
        r_in_theta = r_in[3]

        for i, a_map in enumerate(a_in):
            for j in range(self.output_size):
                z[i][j] = self.b[0][j]
                for a_key, a_val in a_map.items():
                    dx = r_in_x[a_key] - self.rx[j]
                    dy = r_in_y[a_key] - self.ry[j]
                    dz = r_in_z[a_key] - self.rz[j]
                    w_ji = np.exp(-(dx**2 + dy**2 + dz**2))
                    dt = r_in_theta[a_key] - self.theta[j]
                    w_ji *= np.cos(dt)
                    z[i][j] += self.q[j] * w_ji * a_val
        return z

    def compute_a(self, z):
        """
        Linear activation, but only keep the top k activations
        """
        a = [{} for _ in range(len(z))]
        for i, zz in enumerate(z):
            for pair in sorted(zip(list(range(self.output_size)), list(zz)), key=(lambda x: x[1]))[:self.ktop]:
                a[i][pair[0]] = pair[1]
        return a

    def compute_da(self, z):
        a = self.compute_a(z)
        for i, amap in enumerate(a):
            for k in amap.keys():
                a[i][k] = 1.0
        return a
