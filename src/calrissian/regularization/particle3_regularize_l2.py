import numpy as np


class Particle3RegularizeL2(object):
    """
    L2 regularizer
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        # c = 0.0
        # for layer in layers:
        #     # c += np.sum(layer.q * layer.q)
        #     c += layer.compute_w_sum_square()
        # return self.coeff_lambda * c

        c = 0.0

        # L2 plus regularize
        # Compute the matrices
        for l, layer in enumerate(layers):
            w = layer.compute_w()
            wt = w.transpose()
            for j in range(layer.output_size):
                for k in range(layer.output_size):
                    if j == k:
                        continue
                    c += wt[j].dot(wt[k])**2
                    # c += np.abs(wt[j].dot(wt[k]))
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_dq):
        # two_lambda = 2.0 * self.coeff_lambda
        # for l, layer in enumerate(layers):
        #     dc_dq[l] += two_lambda * layer.q
        return dc_dq
