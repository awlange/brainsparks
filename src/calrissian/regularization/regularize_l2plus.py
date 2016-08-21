import numpy as np


class RegularizeL2Plus(object):
    """
    L2 plus regularizer for weights, unit dot products
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        c = 0.0
        for layer in layers:
            w = layer.w
            for j in range(layer.output_size):
                for k in range(j, layer.output_size):
                    for i in range(layer.input_size):
                        c += w[i][j] * w[i][k]
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        for l, layer in enumerate(layers):
            w = layer.w
            tmp = np.zeros_like(w)

            for j in range(layer.output_size):
                for k in range(j, layer.output_size):
                    for i in range(layer.input_size):
                        tmp[i][j] += w[i][k]
                        tmp[i][k] += w[i][j]

            dc_dw[l] += self.coeff_lambda * tmp

        return dc_db, dc_dw
