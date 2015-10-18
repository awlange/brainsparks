import numpy as np


class Cost(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "quadratic" or nl == "l2" or nl == "mean_square_error" or nl == "mse":
            return Cost.quadratic
        if nl == "l1" or nl == "mean_absolute_error" or nl == "mae":
            return Cost.mae
        if nl == "categorical_cross_entropy":
            return Cost.categorical_cross_entropy
        if nl == "binary_cross_entropy":
            return Cost.binary_cross_entropy

        raise NameError("Cost function name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "quadratic" or nl == "l2" or nl == "mean_square_error" or nl == "mse":
            return Cost.d_quadratic
        if nl == "l1" or nl == "mean_absolute_error" or nl == "mae":
            return Cost.d_mae
        if nl == "categorical_cross_entropy":
            return Cost.d_categorical_cross_entropy
        if nl == "binary_cross_entropy":
            return Cost.d_binary_cross_entropy

        raise NameError("Cost function name {} is not implemented.".format(name))

    # Quadratic, L2, mean square error

    @staticmethod
    def quadratic(y, a):
        return 0.5 * np.mean(np.sum(np.square(a - y), axis=1))

    @staticmethod
    def d_quadratic(y, a, z):
        return ((a - y) * z) / len(y)

    # Mean absolute error, L1

    @staticmethod
    def mae(y, a):
        return np.mean(np.sum(np.abs(a - y), axis=1))

    @staticmethod
    def d_mae(y, a, z):
        diff = a - y
        return (np.piecewise(diff, [diff < 0, diff >= 0], [-1, 1]) * z) / len(y)

    @staticmethod
    def categorical_cross_entropy(y, a):
        return np.mean(-np.sum(y * np.log(a), axis=1))

    @staticmethod
    def d_categorical_cross_entropy(y, a, z):
        # z not used but here to maintain common interface
        # Note: only valid when used in combination with softmax activation in final layer
        return (a - y) / len(y)

    @staticmethod
    def binary_cross_entropy(y, a):
        return np.sum(-(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)))

    @staticmethod
    def d_binary_cross_entropy(y, a, z):
        # z not used but here to maintain common interface
        # Not to be used with softmax. This gradient is wrong for that... would probably need to compute Jacobian...
        return a - y
