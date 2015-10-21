from .layer import Layer
from ..activation import Activation

import numpy as np


class Convolution1D(Layer):

    def __init__(self, input_size=0, n_filters=0, filter_size=0, activation="sigmoid", stride_length=1,
                 flatten_output=False):
        super().__init__("Convolution1D", True)
        self.input_size = input_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride_length = stride_length
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        self.flatten_output = flatten_output

        # Number of fields in convolution
        self.n_fields = 1 + (self.input_size - self.filter_size) // self.stride_length

        # Params
        # self.b = np.asarray([[1.05*(o+1) for o in range(n_filters)]])
        # self.w = np.transpose(np.asarray([[0.01*(i+o+1) for i in range(filter_size)] for o in range(n_filters)]))

        self.b = np.asarray([[0.5 for o in range(n_filters)]])
        self.w = np.transpose(np.asarray([[0.1*(i+1) for i in range(filter_size)] for o in range(n_filters)]))

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        # TODO: Seems like we should be able to optimize this better to take advantage of numpy better
        # Perhaps move the loop over filters to outer most loop
        z = []
        for i_data in range(len(a_in)):
            # For each filter, convolve the input left to right
            convolution = []
            i = 0
            for field in range(self.n_fields):
                convolution.append((a_in[i_data][i:(i+self.filter_size)].dot(self.w) + self.b)[0])
                i += self.stride_length
            z.append(np.transpose(convolution))

        z = np.asarray(z)
        if self.flatten_output:
            return Convolution1D.flatten(z)
        return z

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, A, sigma_Z=None):
        # For max pooling, we'll only need to compute the gradient w/r to the max field

        # Ensure shape
        prev_delta = prev_delta.reshape((self.n_filters, -1))

        dc_db = prev_delta if sigma_Z is None else self.w.dot(prev_delta) * sigma_Z

        # For each filter, convolve the input left to right
        convolution = []
        i = 0
        for field in range(self.n_fields):
            convolution.append(np.outer(A[i:(i+self.filter_size)], dc_db.transpose()[field]))
            i += self.stride_length
        dc_dw = np.transpose(convolution)

        return dc_db, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw, convolve=True):
        """
        Need to take sum across fields

        For max pooling, we'll probably need to do something else, like only use the max one

        :param dc_db:
        :param dc_dw:
        :return:
        """
        delta_b = np.sum(dc_db.reshape((self.n_filters, self.n_fields)), axis=1)
        delta_w = None
        if not convolve:
            delta_w = np.sum(dc_dw.reshape((self.n_filters, self.filter_size, self.n_fields)), axis=2)
        else:
            dc_dw_t = dc_dw.transpose()
            delta_w = np.zeros_like(self.w.transpose())
            for filter in range(self.n_filters):
                i = 0
                for field in range(self.n_fields):
                    delta_w[filter] += dc_dw_t[field + filter*self.n_fields][i:(i+self.filter_size)]
                    i += self.stride_length

        return delta_b, delta_w.transpose()

    def convolve_input(self, a_in):
        convolution = []
        i = 0
        for field in range(self.n_fields):
            convolution.append(a_in[i:(i+self.filter_size)])
            i += self.stride_length
        return np.transpose(convolution)

    @staticmethod
    def flatten(x):
        result = []
        for i, x in enumerate(x):
            result.append(x.flatten())
        return np.asarray(result)
