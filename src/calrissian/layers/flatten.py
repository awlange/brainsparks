from .layer import Layer

import numpy as np


class Flatten(Layer):
    """
    A utility layer just to flatten each datum's ndarray
    """

    def __init__(self):
        super().__init__("Flatten", False)

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        result = []
        for i, a in enumerate(a_in):
            result.append(a.flatten())
        return np.asarray(result)

    def compute_a(self, z):
        return z

    def compute_da(self, z):
        return z
