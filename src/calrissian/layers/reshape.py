from .layer import Layer

import numpy as np


class Reshape(Layer):
    """
    A utility layer just to reshape each datum's ndarray
    """

    def __init__(self, reshape_from=(1, 1), reshape_to=(1, 1)):
        super().__init__("Reshape", False)
        self.reshape_from = reshape_from
        self.reshape_to = reshape_to

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        return a_in.reshape(self.reshape_to)

    def compute_a(self, z):
        return z

    def compute_da(self, z):
        return z

    def reverse_reshape(self, a):
        return a.reshape(self.reshape_from)
