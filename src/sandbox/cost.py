import src.sandbox.linalg as linalg
import numpy as np


class Cost(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        if name == "quadratic":
            return Cost.quadratic
        if name == "np_quadratic":
            return Cost.np_quadratic
        return None

    @staticmethod
    def get_d(name):
        if name == "quadratic":
            return Cost.d_quadratic
        if name == "np_quadratic":
            return Cost.np_d_quadratic
        return None

    @staticmethod
    def quadratic(y, a):
        """
        Cost for a single training data
        """
        return linalg.vsqdistw(a, y)

    @staticmethod
    def d_quadratic(y, a):
        """
        Cost derivative for a single training data
        """
        return linalg.vminw(a, y)

    @staticmethod
    def np_quadratic(y, a):
        return np.mean(np.square(np.linalg.norm(a-y, axis=1, ord=2))) * 0.5

    @staticmethod
    def np_d_quadratic(y, a):
        return (a - y) / len(y)
