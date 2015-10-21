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
        self.input_size = input_size
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
        z = []
        for i_data in range(len(a_in)):
            # For each filter, convolve the input left to right
            convolution = []
            i = 0
            for field in range(self.n_fields):
                x = a_in[i_data][i:(i+self.pool_size)]
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

