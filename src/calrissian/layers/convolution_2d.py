from .layer import Layer
from ..activation import Activation

import numpy as np


class Convolution2D(Layer):

    def __init__(self, input_size=(1, 1), activation="sigmoid", n_filters=1, filter_size=(1, 1), stride=(1, 1),
                 weight_init="glorot"):
        """
        tuple indexing: 0 -> x, 1 -> y
        """

        super().__init__("Convolution2D", True)
        self.input_size = input_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Number of fields in convolution
        self.n_fields_x = 1 + (self.input_size[0] - self.filter_size[0]) // self.stride[0]
        self.n_fields_y = 1 + (self.input_size[1] - self.filter_size[1]) // self.stride[1]
        self.output_size = (self.n_filters * self.n_fields_x * self.n_fields_y)

        # Params
        s = np.sqrt(2.0 / (input_size[0] * input_size[1] + self.output_size))
        # self.b = np.random.uniform(-s, s, (1, self.n_filters))  # todo: check dimensions
        # self.w = np.random.uniform(-s, s, (self.n_filters, self.filter_size[0], self.filter_size[1]))

        # TEMPORARY for testing
        self.b = np.zeros((1, self.n_filters))  # todo: check dimensions
        self.w = np.zeros((self.n_filters, self.filter_size[0], self.filter_size[1]))

        for i in range(self.n_filters):
            self.b[0] = 0.01 * i
            for j in range(self.filter_size[0]):
                for k in range(self.filter_size[1]):
                    # self.w[i][j][k] = 0.001 * (i+2*j+3*k)
                    self.w[i][j][k] = 2.0

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        """
        a_in dimensions: (n data points), (input_size[0]), (input_size[1])
        :param a_in:
        :return:
        """

        out = np.zeros((len(a_in), self.n_filters, self.n_fields_x, self.n_fields_y))

        for n_data, a in enumerate(a_in):
            # Expand a
            expa = np.zeros((self.n_fields_x, self.n_fields_y, self.filter_size[0], self.filter_size[1]))
            for nfx in range(self.n_fields_x):
                for nfy in range(self.n_fields_y):
                    for fsx in range(self.filter_size[0]):
                        for fsy in range(self.filter_size[1]):
                            expa[nfx][nfy][fsx][fsy] = a[fsx + nfx*self.stride[0]][fsy + nfy*self.stride[1]]

            # TODO: dot with axis?
            for nf, w_filter in enumerate(self.w):
                out[n_data][nf] = np.sum(expa * w_filter, axis=(2, 3)) + self.b[0][nf]

        return out

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z, **kwargs):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, A, sigma_Z=None, dc_dw_l=None):
        dc_db = prev_delta if sigma_Z is None else self.w.dot(prev_delta) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw, **kwargs):
        return dc_db, dc_dw

    def forward_pass(self, a_in):
        return self.activation(a_in.dot(self.w) + self.b)

    def backward_pass(self):
        pass
