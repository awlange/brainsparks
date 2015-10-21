from .layer import Layer
from ..activation import Activation

import numpy as np


class Flatten(Layer):
    """
    A utility layer just to flatten each datum's ndarray
    """

    def __init__(self):
        super().__init__("Flatten", False)
        self.activation = Activation.get("linear")
        self.d_activation = Activation.get_d("linear")

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        result = []
        for i, a in enumerate(a_in):
            result.append(a.flatten())
        return np.asarray(result)

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, A, sigma_Z=None, dc_dw_l=None):
        dc_db = prev_delta.reshape(sigma_Z.shape) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw
