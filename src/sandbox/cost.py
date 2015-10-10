import src.sandbox.linalg as linalg


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
