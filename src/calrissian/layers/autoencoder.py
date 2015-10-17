from .layer import Layer
from ..activation import Activation

import numpy as np


class Autoencoder(Layer):
    """
    TODO: a lot of things here... rethinking
    """

    def __init__(self, input_size=0, output_size=0, activation="sigmoid", symmetric_weights=False, freeze=False):
        super().__init__("Autoencoder")
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)
        self.symmetric_weights = symmetric_weights
        self.freeze = freeze  # In training, controls whether the weights should be allowed to change

        # Params
        self.b = np.asarray([[0.05*o for o in range(output_size)]])
        self.w = np.transpose(np.asarray([[0.01*(i+o) for i in range(input_size)] for o in range(output_size)]))
        self.b_decode = np.asarray([[0.05*o for o in range(output_size)]])
        if symmetric_weights:
            self.w_decode = None  # transpose of self.w
        else:
            self.w_decode = np.asarray([[0.01*(i+o) for i in range(input_size)] for o in range(output_size)])

    def feed_forward(self, a_in):
        return self.encode(a_in)

    def compute_z(self, a_in, decode=False):
        if decode:
            if self.symmetric_weights:
                return a_in.dot(self.w.transpose()) + self.b_decode
            return a_in.dot(self.w_decode) + self.b_decode
        return a_in.dot(self.w) + self.b

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def encode(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def decode(self, a_in):
        return self.compute_a(self.compute_z(a_in, decode=True))

    def compute_gradient(self, prev_delta, sigma_Z, A):
        """
        Computes this layer's weight and bias gradients for a single input data pass

        :return:
        """

        # TODO: figure out what to do with this

        dc_db = self.w.dot(prev_delta) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw
