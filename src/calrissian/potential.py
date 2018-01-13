import numpy as np

MAX_FLOAT = np.finfo(0.0).max


class Potential(object):

    def __init__(self):
        pass

    @staticmethod
    def get(name):
        nl = name.lower()
        if nl == "identity":
            return Potential().identity
        if nl == "linear":
            return Potential().linear
        if nl == "gaussian":
            return Potential().gaussian
        if nl == "exponential":
            return Potential().exponential
        if nl == "inverse":
            return Potential().inverse
        if nl == "log":
            return Potential().log
        if nl == "inv_multi":
            return Potential().inv_multi
        if nl == "lorentzian":
            return Potential().lorentzian
        if nl == "triangle":
            return Potential().triangle
        if nl == "lj":
            return Potential().lennard_jones
        if nl == "gwell":
            return Potential().gwell
        if nl == "gaussian_short":
            return Potential().gaussian_short

        raise NameError("Potential name {} is not implemented.".format(name))

    @staticmethod
    def get_d(name):
        nl = name.lower()
        if nl == "identity":
            return Potential().d_identity
        if nl == "linear":
            return Potential().d_linear
        if nl == "gaussian":
            return Potential().d_gaussian
        if nl == "exponential":
            return Potential().d_exponential
        if nl == "inverse":
            return Potential().d_inverse
        if nl == "log":
            return Potential().d_log
        if nl == "inv_multi":
            return Potential().d_inv_multi
        if nl == "lorentzian":
            return Potential().d_lorentzian
        if nl == "triangle":
            return Potential().d_triangle
        if nl == "lj":
            return Potential().d_lennard_jones
        if nl == "gwell":
            return Potential().d_gwell
        if nl == "gaussian_short":
            return Potential().d_gaussian_short

        raise NameError("Potential name {} is not implemented.".format(name))

    # Identity

    def identity(self, x):
        return np.ones(x.shape)

    def d_identity(self, x):
        return np.zeros(x.shape)

    # Linear

    def linear(self, x):
        return x

    def d_linear(self, x):
        return np.ones(x.shape)

    # Gaussian

    def gaussian(self, x):
        return np.exp(-x*x)

    def d_gaussian(self, x):
        return -2.0 * x * np.exp(-x*x)

    # Exponential

    def exponential(self, x):
        return np.exp(-x)

    def d_exponential(self, x):
        return -np.exp(-x)

    # Inverse
    def inverse(self, x):
        return 1.0/x

    def d_inverse(self, x):
        return -1.0/(x*x)

    # Log
    def log(self, x):
        return np.log(x)

    def d_log(self, x):
        return 1.0/x

    # Inverse multiquadratic
    def inv_multi(self, x):
        return 1.0/np.sqrt(1.0 + x*x)

    def d_inv_multi(self, x):
        return -x / np.power(1 + x*x, 1.5)

    # Lorentzian
    def lorentzian(self, x):
        return 1.0/(1.0 + x*x)

    def d_lorentzian(self, x):
        return -2.0 * x / (1.0 + x*x)**2

    # Triangle
    def triangle(self, x):
        return np.maximum(0, 1-x)

    def d_triangle(self, x):
        return np.piecewise(x, [x < 1.0, x >= 1.0], [-1.0, 0.0])

    # Lennard-Jones
    def lennard_jones(self, x):
        a = 1.0/x
        a2 = a*a
        a6 = a2*a2*a2
        a12 = a6
        return a12 - 2.0*a6

    def d_lennard_jones(self, x):
        a = 1.0/x
        a2 = a*a
        a6 = a2*a2*a2
        a12 = a6
        return 12*(a6 - a12) * a

    # Gaussian well

    def gwell(self, x):
        return np.exp(-x*x) - np.exp(-(x-2.0)**2)

    def d_gwell(self, x):
        return -2.0 * x * np.exp(-x*x) + 2.0 * (x - 2.0) * np.exp(-(x-2.0)**2)

    # Gaussian - short range

    def gaussian_short(self, x):
        a = 1.0/np.sqrt(0.1)
        return np.exp(-a * x*x)

    def d_gaussian_short(self, x):
        a = 1.0/np.sqrt(0.1)
        return -2.0 * a * x * np.exp(-a * x*x)
