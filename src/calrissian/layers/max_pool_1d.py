from .layer import Layer
from ..activation import Activation
from .convolution_1d import Convolution1D

import numpy as np


class MaxPool1D(Layer):
    """
    Convolve input but only feed forward the maximum activation
    """

    def __init__(self, input_size=0, n_filters=0, pool_size=1, stride_length=1, flatten_output=False):
        super().__init__("MaxPool1D", False)
        self.input_size = input_size  # TODO: make shape
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.stride_length = stride_length

        # Number of fields in convolution
        self.n_fields = 1 + (self.input_size - self.pool_size) // self.stride_length

        self.flatten_output = flatten_output

        # Not sure if needed
        self.activation = Activation.get("linear")
        self.d_activation = Activation.get_d("linear")

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        return a_in

    def compute_a(self, z):
        a = self.apply_max_pool(z)
        if self.flatten_output:
            return Convolution1D.flatten(a)
        return a

    def compute_da(self, z, a=None):

        return np.ones_like(self.apply_max_pool(z))

        # # Pluck out only the argmax gradients
        # result = np.zeros_like(z)
        # for i_data in range(len(z)):
        #     zi = z[i_data]
        #     for filter in range(self.n_filters):
        #         zf = zi[filter]
        #         i = 0
        #         for field in range(self.n_fields):
        #             field_list = zf[i:(i+self.pool_size)]
        #             max_index = np.argmax(field_list)
        #             result[i_data][filter][i + max_index] = 1.0
        #             i += self.stride_length
        #
        # return result

    def compute_gradient(self, prev_delta, A, sigma_Z=None):
        # return np.zeros_like(self.b), np.zeros_like(self.w)
        # # Pluck out only the argmax gradients

        result = np.zeros_like(A)
        for i_data in range(len(A)):
            zi = A[i_data]
            for filter in range(self.n_filters):
                zf = zi[filter]
                i = 0
                for field in range(self.n_fields):
                    field_list = zf[i:(i+self.pool_size)]
                    max_index = np.argmax(field_list)
                    result[i_data][filter][i + max_index] = 1.0
                    i += self.stride_length

        return result

    def compute_gradient_update(self, dc_db, dc_dw, A=None, convolve=True):
        return dc_db, dc_dw

    def apply_max_pool(self, a, argmax=False):
        max_func = np.argmax if argmax else np.max
        a_pool = []
        for i_data in range(len(a)):
            a_pool.append(self.convolve_max_pool(a[i_data], max_func))
        return np.asarray(a_pool)

    def convolve_max_pool(self, a, max_func=np.max):
        # Choose if we want the values or the indexes
        a_filter = []
        for filter in range(self.n_filters):
            convolution = []
            i = 0
            for field in range(self.n_fields):
                convolution.append(max_func(a[filter][i:(i+self.pool_size)]))
                i += self.stride_length
            a_filter.append(convolution)
        return a_filter


