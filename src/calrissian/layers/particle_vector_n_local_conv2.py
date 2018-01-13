from .layer import Layer
from ..activation import Activation
from ..potential import Potential

import numpy as np


class ParticleVectorNLocalConvolution2Input(object):
    def __init__(self, output_size, nr=3, nv=3, sr=1.0, sv=1.0, srl=None):
        self.output_size = output_size
        self.nv = nv
        self.nr = nr
        self.apply_convolution = False  # no conv here, just for consistency in gradient

        # Positions
        self.positions = []

        srl = srl if srl else [sr for _ in range(nr)]
        for i in range(nr):
            self.positions.append(np.random.normal(0.0, srl[i], output_size))

        # Vectors
        self.nvectors = []
        for i in range(nv):
            # self.nvectors.append(np.random.normal(0.0, sv, output_size))
            self.nvectors.append(np.random.uniform(-sv, sv, output_size))

    def get_rxyz(self):
        return self.positions, self.nvectors

    def feed_forward(self, a_in):
        return a_in, (self.get_rxyz())


class ParticleVectorNLocalConvolution2(object):
    """
    Local convolution with no pooling
    """

    def __init__(self, input_size=0, output_size=0, nr=3, nv=3, activation="sigmoid", potential="gaussian",
                 sr=1.0, sv=1.0, q=None, b=None, boff=0.0, uniform=False, p_dropout=-1.0, sigma_r=-1.0,
                 delta_r=0.0, apply_convolution=False, n_steps=None, srl=None):
        self.input_size = input_size
        self.output_size = output_size
        self.nr = nr
        self.nv = nv
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.potential = Potential.get(potential)
        self.d_potential = Potential.get_d(potential)
        self.p_dropout = p_dropout
        self.dropout_mask = None
        self.sigma_r = sigma_r
        self.delta_r = delta_r
        self.apply_convolution = apply_convolution

        # Convolution params
        # Just 2 dimensions for now
        self.n_steps = [-1, 0, 1] if n_steps is None else n_steps
        self.int_to_combo = []
        for ix, nx in enumerate(self.n_steps):
            for iy, ny in enumerate(self.n_steps):
                self.int_to_combo.append((ix, iy, 0))
                # for iz, nz in enumerate(n_steps):
                #     int_to_combo.append((ix, iy, iz))
        self.n_convolution = len(self.int_to_combo)

        # Weight initialization
        g = np.sqrt(2.0 / (input_size + output_size))
        if b is None:
            b = g
        # self.b = np.random.uniform(boff - b, boff + b, (1, output_size))
        self.b = np.random.uniform(-sv, sv, (1, output_size))
        # self.b = np.random.normal(0.0, sv, (1, output_size))

        # Positions
        srl = srl if srl else [sr for _ in range(nr)]
        self.positions = []
        for i in range(nr):
            if uniform:
                self.positions.append(np.random.uniform(-srl[i], srl[i], output_size))
            else:
                self.positions.append(np.random.normal(0.0, srl[i], output_size))

        # Vectors
        self.nvectors = []
        for i in range(nv):
            # self.nvectors.append(np.random.normal(0.0, sv, output_size))
            self.nvectors.append(np.random.uniform(-sv, sv, output_size))

        # Matrix
        self.w = None
        self.positions_cache = None
        self.activations_cache = None
        self.nvectors_cache = None

    def get_rxyz(self):
        if self.apply_convolution:
            self.update_caches()
            self.flatten_caches()
            return self.positions_cache, self.nvectors_cache
        else:
            return self.positions, self.nvectors

    def feed_forward(self, a_in, r_in):
        return self.compute_a(self.compute_z(a_in, r_in)), (self.get_rxyz())

    def update_caches(self):
        self.positions_cache = np.zeros((self.nr, self.output_size, self.n_convolution))
        self.nvectors_cache = np.zeros((self.nv, self.output_size, self.n_convolution))

        # Just 2 dimensions for now
        for j in range(self.output_size):
            ii = 0
            for ix, nx in enumerate(self.n_steps):
                pr0 = self.positions[0][j] + nx * self.delta_r
                for iy, ny in enumerate(self.n_steps):
                    pr1 = self.positions[1][j] + ny * self.delta_r

                    pr2 = self.positions[2][j]

                    self.positions_cache[0][j][ii] = pr0
                    self.positions_cache[1][j][ii] = pr1
                    self.positions_cache[2][j][ii] = pr2

                    for v in range(self.nv):
                        self.nvectors_cache[v][j][ii] = self.nvectors[v][j]

                    ii += 1

    def flatten_caches(self):
        # Flatten the convolution dimension
        self.positions_cache = self.positions_cache.reshape((self.nr, self.output_size * self.n_convolution))
        self.nvectors_cache = self.nvectors_cache.reshape((self.nv, self.output_size * self.n_convolution))

    def compute_z(self, a_in, r_in, apply_input_noise=False):
        """
        Vectorized v2.0

        :param a_in:
        :param r_in:
        :return:
        """
        atrans = a_in.transpose()
        z = None
        r_positions = r_in[0]
        r_nvectors = r_in[1]

        if apply_input_noise and self.sigma_r > 0.0:
            for r in range(self.nr):
                r_positions[r] += np.random.normal(0.0, self.sigma_r, r_positions[r].shape)

        if self.apply_convolution:
            z = np.zeros((self.output_size, self.n_convolution, len(a_in)))
            self.update_caches()

            # loop over nodes
            for j in range(self.output_size):
                # same for all translations
                dot = 0.0
                for v in range(self.nv):
                    dot += r_nvectors[v] * self.nvectors[v][j]
                dot_atrans = dot.reshape((self.input_size, 1)) * atrans  # cache outside the loops
                tmp_zj_xyz = np.ones((self.n_convolution, len(a_in))) * self.b[0][j]

                ii = 0
                for ix, nx in enumerate(self.n_steps):
                    pr0 = self.positions_cache[0][j][ii]
                    ddx = (r_positions[0] - pr0)**2
                    for iy, ny in enumerate(self.n_steps):
                        pr1 = self.positions_cache[1][j][ii]
                        ddy = (r_positions[1] - pr1)**2

                        pr2 = self.positions_cache[2][j][ii]
                        ddz = (r_positions[2] - pr2)**2

                        d = np.sqrt(ddx + ddy + ddz)
                        w_ji = self.potential(d)

                        tmp_zj_xyz[ii] += w_ji.dot(dot_atrans)
                        ii += 1

                        # for iz, nz in enumerate(n_steps):
                        #     pr2 = self.positions[2][j] + nz * self.delta_r
                        #     ddz = (r_positions[2] - pr2)**2
                        #
                        #     d = np.sqrt(ddx + ddy + ddz)
                        #     w_ji = self.potential(d)
                        #
                        #     tmp_zj_xyz[ii] = self.b[0][j] + w_ji.dot(dot_atrans)
                        #     ii += 1

                for ja, a in enumerate(tmp_zj_xyz):
                    z[j][ja] = a

            # Flatten the convolution dimension
            z = z.reshape((self.output_size * self.n_convolution, len(a_in)))
            self.flatten_caches()

        else:
            z = np.zeros((self.output_size, len(a_in)))

            for j in range(self.output_size):
                dd = 0.0
                for r in range(self.nr):
                    dd += (r_positions[r] - self.positions[r][j]) ** 2
                d = np.sqrt(dd)
                dot = 0.0
                for v in range(self.nv):
                    dot += r_nvectors[v] * self.nvectors[v][j]
                w_ji = self.potential(d) * dot
                z[j] = self.b[0][j] + w_ji.dot(atrans)

        return z.transpose()

    def compute_a(self, z, apply_dropout=False):
        a = self.activation(z)
        if apply_dropout and self.p_dropout > 0.0:
            self.dropout_mask = np.random.binomial(1, self.p_dropout, a.shape)
            a *= self.dropout_mask
        return a

    def compute_da(self, z, apply_dropout=False):
        da = self.d_activation(z)
        if apply_dropout and self.p_dropout > 0.0:
            da *= self.dropout_mask
        return da

    def compute_w(self, r_in):
        w = np.zeros((self.input_size, self.output_size))
        r_positions = r_in[0]
        r_nvectors = r_in[1]

        for j in range(self.output_size):
            dd = 0.0
            for r in range(self.nr):
                dd += (r_positions[r] - self.positions[r][j])**2
            d = np.sqrt(dd)
            dot = 0.0
            for v in range(self.nv):
                dot += r_nvectors[v] * self.nvectors[v][j]
            w_ji = self.potential(d) * dot
            for i in range(self.input_size):
                w[i][j] = w_ji[i]

        self.w = w

        return w

    def compute_w_j(self, r_in, j):
        return None

    def compute_w_i(self, r_in, i):
        return None
