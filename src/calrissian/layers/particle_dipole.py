from .layer import Layer
from ..activation import Activation

import numpy as np


class ParticleDipoleInput(object):
    """
    Dipole approximated as 2 coupled charges of equal magnitude
    """

    def __init__(self, output_size, k_bond=1.0, k_eq=0.1, s=1.0, cut=3.0):
        self.output_size = output_size

        # Harmonic constraint coefficient and equilibrium
        self.k_bond = k_bond
        self.k_eq = k_eq
        self.cut = cut
        self.cut2 = cut*cut

        # Positive charge positions
        self.rx_pos = np.random.uniform(-s, s, output_size)
        self.ry_pos = np.random.uniform(-s, s, output_size)
        self.rz_pos = np.random.uniform(-s, s, output_size)

        # Negative charge positions
        # Copy of positive charge position with small added noise
        # s = 1.1 * k_eq
        # self.rx_neg = np.copy(self.rx_pos) + np.random.uniform(-s, s, output_size)
        # self.ry_neg = np.copy(self.ry_pos) + np.random.uniform(-s, s, output_size)
        # self.rz_neg = np.copy(self.rz_pos) + np.random.uniform(-s, s, output_size)
        # Random
        self.rx_neg = np.random.uniform(-s, s, output_size)
        self.ry_neg = np.random.uniform(-s, s, output_size)
        self.rz_neg = np.random.uniform(-s, s, output_size)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def get_rxyz(self):
        return self.rx_pos, self.ry_pos, self.rz_pos, self.rx_neg, self.ry_neg, self.rz_neg

    def feed_forward(self, a_in):
        """
        Just scales the input by the charges
        Turned off for now
        """
        return a_in, (self.get_rxyz())

    def compute_bond_cost(self):
        dx = self.rx_pos - self.rx_neg
        dy = self.ry_pos - self.ry_neg
        dz = self.rz_pos - self.rz_neg
        dd = (np.sqrt(dx**2 + dy**2 + dz**2) - self.k_eq)**2
        return 0.5 * self.k_bond * np.sum(dd)

    def compute_bond_cost_gradient(self):
        dx = self.rx_pos - self.rx_neg
        dy = self.ry_pos - self.ry_neg
        dz = self.rz_pos - self.rz_neg
        dd = np.sqrt(dx**2 + dy**2 + dz**2)
        tmp = self.k_bond * (dd - self.k_eq) / dd
        tx = tmp * dx
        ty = tmp * dy
        tz = tmp * dz
        return tx, ty, tz


class ParticleDipole(object):
    """
    Dipole approximated as 2 coupled charges of equal magnitude
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", k_bond=1.0, k_eq=0.1, s=1.0, cut=3.0,
                 q=None, b=None):

        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.w = None

        # Harmonic constraint coefficient and equilibrium
        self.k_bond = k_bond
        self.k_eq = k_eq
        self.cut = cut
        self.cut2 = cut*cut

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        self.b = np.random.uniform(-b, b, (1, output_size))

        # Charges
        if q is None:
            q = g
        # self.q = np.random.uniform(-q, q, output_size)
        self.q = np.random.choice([q, -q], size=output_size)

        # Positive charge positions
        self.rx_pos = np.random.uniform(-s, s, output_size)
        self.ry_pos = np.random.uniform(-s, s, output_size)
        self.rz_pos = np.random.uniform(-s, s, output_size)

        # Negative charge positions
        # Copy of positive charge position with small added noise
        # s = 1.1 * k_eq
        # self.rx_neg = np.copy(self.rx_pos) + np.random.uniform(-s, s, output_size)
        # self.ry_neg = np.copy(self.ry_pos) + np.random.uniform(-s, s, output_size)
        # self.rz_neg = np.copy(self.rz_pos) + np.random.uniform(-s, s, output_size)
        # Random
        self.rx_neg = np.random.uniform(-s, s, output_size)
        self.ry_neg = np.random.uniform(-s, s, output_size)
        self.rz_neg = np.random.uniform(-s, s, output_size)

    def set_cut(self, cut):
        self.cut = cut
        self.cut2 = cut*cut

    def compute_bond_cost(self):
        dx = self.rx_pos - self.rx_neg
        dy = self.ry_pos - self.ry_neg
        dz = self.rz_pos - self.rz_neg
        dd = (np.sqrt(dx**2 + dy**2 + dz**2) - self.k_eq)**2
        return 0.5 * self.k_bond * np.sum(dd)

    def compute_bond_cost_gradient(self):
        dx = self.rx_pos - self.rx_neg
        dy = self.ry_pos - self.ry_neg
        dz = self.rz_pos - self.rz_neg
        dd = np.sqrt(dx**2 + dy**2 + dz**2)
        tmp = self.k_bond * (dd - self.k_eq) / dd
        tx = tmp * dx
        ty = tmp * dy
        tz = tmp * dz
        return tx, ty, tz

    def get_rxyz(self):
        return self.rx_pos, self.ry_pos, self.rz_pos, self.rx_neg, self.ry_neg, self.rz_neg

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), self.get_rxyz()

    def compute_z(self, a_in, r_in):
        """
        Vectorized v2.0

        :param a_in:
        :param r_in:
        :return:
        """
        atrans = a_in.transpose()
        z = np.zeros((self.output_size, len(a_in)))

        r_in_x_pos = r_in[0]
        r_in_y_pos = r_in[1]
        r_in_z_pos = r_in[2]
        r_in_x_neg = r_in[3]
        r_in_y_neg = r_in[4]
        r_in_z_neg = r_in[5]

        for j in range(self.output_size):
            dx = r_in_x_pos - self.rx_pos[j]
            dy = r_in_y_pos - self.ry_pos[j]
            dz = r_in_z_pos - self.rz_pos[j]
            # potential = 1.0 / np.sqrt(dx**2 + dy**2 + dz**2)
            potential = np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential = np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            dx = r_in_x_pos - self.rx_neg[j]
            dy = r_in_y_pos - self.ry_neg[j]
            dz = r_in_z_pos - self.rz_neg[j]
            # potential += -1.0 / np.sqrt(dx**2 + dy**2 + dz**2)
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential -= np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            dx = r_in_x_neg - self.rx_pos[j]
            dy = r_in_y_neg - self.ry_pos[j]
            dz = r_in_z_neg - self.rz_pos[j]
            # potential += -1.0 / np.sqrt(dx**2 + dy**2 + dz**2)
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential -= np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            dx = r_in_x_neg - self.rx_neg[j]
            dy = r_in_y_neg - self.ry_neg[j]
            dz = r_in_z_neg - self.rz_neg[j]
            # potential += 1.0 / np.sqrt(dx**2 + dy**2 + dz**2)
            potential += np.exp(-(dx**2 + dy**2 + dz**2))
            # d2 = dx**2 + dy**2 + dz**2
            # potential += np.piecewise(d2, [d2 <= self.cut2, d2 > self.cut2], [lambda x: np.exp(-x), 0.0])

            z[j] = self.b[0][j] + self.q[j] * potential.dot(atrans)
        return z.transpose()

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_w(self, r_in):
        w = np.zeros((self.input_size, self.output_size))
        r_in_x_pos = r_in[0]
        r_in_y_pos = r_in[1]
        r_in_z_pos = r_in[2]
        r_in_x_neg = r_in[3]
        r_in_y_neg = r_in[4]
        r_in_z_neg = r_in[5]

        for j in range(self.output_size):
            dx = r_in_x_pos - self.rx_pos[j]
            dy = r_in_y_pos - self.ry_pos[j]
            dz = r_in_z_pos - self.rz_pos[j]
            potential = np.exp(-(dx**2 + dy**2 + dz**2))

            dx = r_in_x_pos - self.rx_neg[j]
            dy = r_in_y_pos - self.ry_neg[j]
            dz = r_in_z_pos - self.rz_neg[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = r_in_x_neg - self.rx_pos[j]
            dy = r_in_y_neg - self.ry_pos[j]
            dz = r_in_z_neg - self.rz_pos[j]
            potential -= np.exp(-(dx**2 + dy**2 + dz**2))

            dx = r_in_x_neg - self.rx_neg[j]
            dy = r_in_y_neg - self.ry_neg[j]
            dz = r_in_z_neg - self.rz_neg[j]
            potential += np.exp(-(dx**2 + dy**2 + dz**2))

            potential = self.q[j] * potential
            for i in range(self.input_size):
                w[i][j] = potential[i]

        self.w = w
        return w
