from src.sandbox.layer import Layer
from src.sandbox.activation import Activation

import numpy as np


class BatchDense(Layer):
    """
    NumPy-ified version of Dense. Takes batch inputs.
    """

    def __init__(self, input_size=0, output_size=0, activation="np_sigmoid"):
        super().__init__()
        self.type = "BatchDense"
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Params
        self.b = np.asarray([[0.05*o for o in range(output_size)]])
        self.w = np.transpose(np.asarray([[0.01*(i+o) for i in range(input_size)] for o in range(output_size)]))

    def feed_forward(self, a_in):
        return self.activation(self.compute_z(a_in))

    def compute_z(self, a_in):
        return a_in.dot(self.w) + self.b

    def compute_a(self, z):
        return self.activation(z)

    def compute_da(self, z):
        return self.d_activation(z)
