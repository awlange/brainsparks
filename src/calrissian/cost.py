import numpy as np


class Cost(object):
    """
    Note: for multidimensional Y, we sum the cost over the dimensions in the outermost summation in the cost functions
    """

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "quadratic" or nl == "l2" or nl == "mean_square_error" or nl == "mse":
            return Cost().quadratic
        if nl == "mse_sparse":
            return Cost().quadratic_sparse
        if nl == "l1" or nl == "mean_absolute_error" or nl == "mae":
            return Cost().mae
        if nl == "categorical_cross_entropy":
            return Cost().categorical_cross_entropy
        if nl == "binary_cross_entropy":
            return Cost().binary_cross_entropy

        raise NameError("Cost function name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "quadratic" or nl == "l2" or nl == "mean_square_error" or nl == "mse":
            return Cost().d_quadratic
        if nl == "mse_sparse":
            return Cost().d_quadratic_sparse
        if nl == "l1" or nl == "mean_absolute_error" or nl == "mae":
            return Cost().d_mae
        if nl == "categorical_cross_entropy":
            return Cost().d_categorical_cross_entropy
        if nl == "binary_cross_entropy":
            return Cost().d_binary_cross_entropy

        raise NameError("Cost function name {} is not implemented.".format(name))

    # Quadratic, L2, mean square error

    def quadratic(self, y, a):
        """
        Supports multidimensional Y

        :param y:
        :param a:
        :return:
        """
        return 0.5 * np.sum(np.mean(np.sum(np.square(a - y), axis=1), axis=0))

    def d_quadratic(self, y, a, z):
        return ((a - y) * z) / len(y)

    # Mean absolute error, L1

    def mae(self, y, a):
        """
        Supports multidimensional Y

        :param y:
        :param a:
        :return:
        """
        return np.sum(np.mean(np.sum(np.abs(a - y), axis=1), axis=0))

    def d_mae(self, y, a, z):
        diff = a - y
        return (np.piecewise(diff, [diff < 0, diff >= 0], [-1, 1]) * z) / len(y)

    # Categorical cross entropy

    def categorical_cross_entropy(self, y, a):
        # In case of numerical instability (negative number or zero), set a element to epsilon
        a = np.maximum(a, np.ones_like(a) * 10e-10)
        return np.mean(-np.sum(y * np.log(a), axis=1), axis=0)

    def d_categorical_cross_entropy(self, y, a, z):
        # z not used but here to maintain common interface
        # Note: only valid when used in combination with softmax activation in final layer
        return (a - y) / len(y)

    # Binary cross entropy
    # TODO: this one doesn't take a mean so could be problematic for threaded gradient

    def binary_cross_entropy(self, y, a):
        # In case of numerical instability (negative number or zero), set a element to epsilon
        a = np.maximum(a, np.ones_like(a) * 10e-10)
        return np.sum(-(y * np.log(a) + (1.0 - y) * np.log(1.0 - a)))

    def d_binary_cross_entropy(self, y, a, z):
        # z not used but here to maintain common interface
        # Not to be used with softmax. This gradient is wrong for that... would probably need to compute Jacobian...
        return a - y

    def quadratic_sparse(self, y, a):
        # Is not taking mean per data point, but whatevs
        total = 0.0
        for i, amap in enumerate(a):
            for k, av in amap.items():
                total += (av - y[i].get(k, 0.0))**2
        total *= 0.5 / len(y)
        return total

    def d_quadratic_sparse(self, y, a, z):
        result = [{} for _ in range(len(z))]
        for i, amap in enumerate(a):
            for k, av in amap.items():
                result[i][k] = (av - y[i].get(k, 0.0)) * z[i][k] / len(y)
        return result
