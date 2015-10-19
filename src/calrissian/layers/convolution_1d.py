from .layer import Layer
from ..activation import Activation

import numpy as np


class Convolution1D(Layer):

    def __init__(self, input_size=0, n_filters=0, filter_size=0, activation="sigmoid", stride_length=1):
        super().__init__("Convolution1D", True)
        self.input_size = input_size
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride_length = stride_length
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Number of fields in convolution
        self.n_fields = 1 + (self.input_size - self.filter_size) // self.stride_length

        # Params
        self.b = np.asarray([[1.05*o for o in range(n_filters)]])
        self.w = np.transpose(np.asarray([[0.01*(i+o) for i in range(filter_size)] for o in range(n_filters)]))

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
        return np.asarray(z)

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, sigma_Z, A):
        # TODO
        dc_db = self.w.dot(prev_delta) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw

    def compute_gradient_final_layer(self, prev_delta, A):
        dc_db = prev_delta  # correct

        z = []
        # For each filter, convolve the input left to right
        for filter in range(self.n_filters):
            convolution = []
            i = 0
            for field in range(self.n_fields):
                convolution.append(np.outer(A[i:(i+self.filter_size)], prev_delta[filter][field]))
                i += self.stride_length
            z.append(np.transpose(convolution))
        dc_dw = np.asarray(z)

        return dc_db, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw):
        """
        Need to take mean across filters

        For max pooling, we'll probably need to do something else

        :param dc_db:
        :param dc_dw:
        :return:
        """
        delta_b = np.mean(dc_db.reshape((self.n_filters, self.n_fields)), axis=1)
        delta_w = np.transpose(np.mean(dc_dw.reshape((self.filter_size, self.n_filters, self.n_fields)), axis=2))

        return delta_b, delta_w