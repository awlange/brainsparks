
from src.sandbox.layer import Layer
from src.sandbox.activation import Activation

import src.sandbox.linalg as linalg


class Dense(Layer):

    def __init__(self, input_size=0, output_size=0, activation="sigmoid"):
        super().__init__()
        self.type = "Dense"
        self.input_size = input_size
        self.output_size = output_size
        self.activation = Activation.get(activation)
        self.d_activation = Activation.get_d(activation)

        # Params
        self.b = [0.05*o for o in range(output_size)]
        self.w = [[0.01*(i+o) for i in range(input_size)] for o in range(output_size)]

    def feed_forward(self, a_in):
        return [self.activation(zi) for zi in self.compute_z(a_in)]

    def compute_z(self, a_in):
        return linalg.mdotvpw(self.w, a_in, self.b)

    def compute_a(self, z):
        return [self.activation(zi) for zi in z]

    def compute_da(self, z):
        return [self.d_activation(zi) for zi in z]
