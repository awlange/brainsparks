import numpy as np


class BiasCoupleL2(object):
    """
    Experimental regularizer for coupling biases between layers
    """

    def __init__(self, coeff_lambda=0.25, couplings=None):
        """
        :param couplings: list of tuple couplings [((layer a, index a), (layer b, index b))]
        """
        self.coeff_lambda = coeff_lambda
        self.couplings = couplings if couplings is not None else []

    def cost(self, layers):
        c = 0.0
        for coupling in self.couplings:
            layer_a, index_a = coupling[0]
            layer_b, index_b = coupling[1]
            tmp = layers[layer_a].b[0][index_a] - layers[layer_b].b[0][index_b]
            c += tmp * tmp
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        """
        :param layers:
        :param dc_db:
        :param dc_dw: not changed
        :return:
        """
        two_lambda = 2.0 * self.coeff_lambda
        for coupling in self.couplings:
            layer_a, index_a = coupling[0]
            layer_b, index_b = coupling[1]
            tmp = two_lambda * (layers[layer_a].b[0][index_a] - layers[layer_b].b[0][index_b])
            dc_db[layer_a][0][index_a] += tmp
            dc_db[layer_b][0][index_b] -= tmp
        return dc_db, dc_dw

