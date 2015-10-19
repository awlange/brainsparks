from .layer import Layer
from ..activation import Activation

import numpy as np


class Dense(Layer):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid"):
        super().__init__("Dense", True)
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation.lower()
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Params
        self.b = np.asarray([[0.05*o for o in range(output_size)]])
        self.w = np.transpose(np.asarray([[0.01*(i+o) for i in range(input_size)] for o in range(output_size)]))

    def feed_forward(self, a_in):
        return self.compute_a(self.compute_z(a_in))

    def compute_z(self, a_in):
        return a_in.dot(self.w) + self.b

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)

    def compute_gradient(self, prev_delta, sigma_Z, A):
        dc_db = self.w.dot(prev_delta) * sigma_Z
        dc_dw = np.outer(A, dc_db)
        return dc_db, dc_dw

    def compute_gradient_final_layer(self, prev_delta, A):
        dc_db = prev_delta
        dc_dw = np.outer(A, prev_delta)
        return dc_db, dc_dw

    def compute_gradient_update(self, dc_db, dc_dw):
        return dc_db, dc_dw
