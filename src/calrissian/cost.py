import numpy as np


class Cost(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        if name == "quadratic":
            return Cost.quadratic
        return None

    @staticmethod
    def get_d(name):
        if name == "quadratic":
            return Cost.d_quadratic
        return None

    @staticmethod
    def quadratic(y, a):
        return np.mean(np.square(np.linalg.norm(a-y, axis=1, ord=2))) * 0.5

    @staticmethod
    def d_quadratic(y, a):
        return (a - y) / len(y)
