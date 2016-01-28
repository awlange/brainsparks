from .layer import Layer
from ..activation import Activation
from .octree.octree import Octree

import numpy as np


class ParticleDipoleTreeInput(object):
    """
    Particle dipole approach with treecode for fast potential evaluation

    Particle positions are ordered such that first n indexes are positive, next n are negative.
    """

    def __init__(self, output_size, s=1.0, cut=1000.0, max_level=3, mac=0.0):
        self.output_size = output_size

        self.cut = cut
        self.cut2 = cut*cut

        # Particle positions
        self.rx = np.random.uniform(-s, s, 2*output_size)
        self.ry = np.random.uniform(-s, s, 2*output_size)
        self.rz = np.random.uniform(-s, s, 2*output_size)

        # Build Octree for this layer
        # TODO: for now this is keeping a duplicate copies of the data
        self.octree = Octree(max_level=max_level, p=1, n_particle_min=20, cut=self.cut, mac=mac)
        self.octree.build_tree(np.zeros(output_size), self.rx, self.ry, self.rz)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

    # def feed_forward(self, a_in):
    #     return a_in, (self.get_rxyz())

    def feed_forward(self, a_in):
        return a_in, self.octree

    def copy_pos_neg_positions(self, rx_pos, ry_pos, rz_pos, rx_neg, ry_neg, rz_neg):
        """
        For debugging purposes
        """
        for i in range(self.output_size):
            self.rx[i] = rx_pos[i]
            self.ry[i] = ry_pos[i]
            self.rz[i] = rz_pos[i]
            self.rx[self.output_size + i] = rx_neg[i]
            self.ry[self.output_size + i] = ry_neg[i]
            self.rz[self.output_size + i] = rz_neg[i]

        # rebuild octree
        self.octree.build_tree(np.zeros(self.output_size), self.rx, self.ry, self.rz)


class ParticleDipoleTree(object):
    """
    Particle dipole approach with treecode for fast potential evaluation

    Particle positions are ordered such that even indexes are positive, odd are negative.
    Paired by (n, n+1)
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", s=1.0, cut=1000.0, max_level=3, mac=0.0):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.cut = cut
        self.cut2 = cut*cut

        # Weight initialization
        c = np.sqrt(1.0 / (input_size + output_size))
        self.b = np.random.uniform(-c, c, (1, output_size))

        # Charges
        c = 1.0
        self.q = np.random.uniform(-c, c, output_size)

        # Particle positions
        self.rx = np.random.uniform(-s, s, 2*output_size)
        self.ry = np.random.uniform(-s, s, 2*output_size)
        self.rz = np.random.uniform(-s, s, 2*output_size)

        # Build Octree for this layer
        # TODO: for now this is keeping a duplicate copies of the data
        self.octree = Octree(max_level=max_level, p=1, n_particle_min=20, cut=self.cut, mac=mac)
        self.octree.build_tree(self.q, self.rx, self.ry, self.rz)

    def copy_pos_neg_positions(self, q, b, rx_pos, ry_pos, rz_pos, rx_neg, ry_neg, rz_neg):
        """
        For debugging purposes
        """
        for i in range(self.output_size):
            self.q[i] = q[i]
            self.b[0][i] = b[0][i]
            self.rx[i] = rx_pos[i]
            self.ry[i] = ry_pos[i]
            self.rz[i] = rz_pos[i]
            self.rx[self.output_size + i] = rx_neg[i]
            self.ry[self.output_size + i] = ry_neg[i]
            self.rz[self.output_size + i] = rz_neg[i]
        # rebuild octree
        self.octree.build_tree(self.q, self.rx, self.ry, self.rz)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def get_rxyz(self):
        return self.rx, self.ry, self.rz

    # def feed_forward(self, a_in, r_in):
    #     return self.compute_a(self.compute_z(a_in, r_in)), self.get_rxyz()

    def feed_forward(self, a_in, octree_in):
        return self.compute_a(self.compute_z(a_in, octree_in)), self.octree

    def compute_z(self, a_in, octree_in):
        """
        Use treecode to compute
        """
        z = np.zeros((self.output_size, len(a_in)))

        # potential = octree_in.compute_potential(self.rx, self.ry, self.rz, a_in)
        potential = octree_in.compute_potential2(self.rx, self.ry, self.rz, a_in)
        # potential = octree_in.compute_potential3(self.rx, self.ry, self.rz, a_in)

        # Coalesce particle dipole pairs
        for j in range(self.output_size):
            pot = (potential[j] - potential[self.output_size + j])
            z[j] = self.b[0][j] + self.q[j] * pot
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)
