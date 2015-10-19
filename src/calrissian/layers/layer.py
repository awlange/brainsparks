import numpy as np


class Layer(object):

    def __init__(self, type, has_gradient):
        self.type = type
        self.has_gradient = has_gradient
        self.input_shape = (0, 0)
        self.output_shape = (0, 0)

        self.b = np.zeros((1, 1))
        self.w = np.zeros((1, 1))

    def feed_forward(self, a_in):
        return None

    def compute_z(self, a_in):
        return None

    def compute_a(self, z):
        return None

    def compute_da(self, z):
        return None

    def compute_gradient(self, prev_delta, sigma_Z, A):
        return None, None