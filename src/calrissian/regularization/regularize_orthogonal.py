import numpy as np


class RegularizeOrthogonal(object):
    """
    Orthogonal
    """

    def __init__(self, coeff_lambda=0.0):
        self.coeff_lambda = coeff_lambda

    def cost(self, layers):
        c = 0.0
        for layer in layers:
            wt = layer.w.transpose()
            for j in range(layer.output_size):
                wtj = wt[j] / np.sqrt(wt[j].dot(wt[j]))
                for k in range(layer.output_size):
                    if j == k:
                        continue
                    wtk = wt[k] / np.sqrt(wt[k].dot(wt[k]))
                    c += np.abs(wtj.dot(wtk))

        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        for l, layer in enumerate(layers):
            wt = layer.w.transpose()
            tmp = np.zeros_like(wt)
            for j in range(layer.output_size):
                dj = np.sqrt(wt[j].dot(wt[j]))
                wtj = wt[j] / dj
                # TODO: simplify this
                s = 2 * (np.eye(len(wtj)) - np.outer(wtj, wtj)) / dj
                for k in range(layer.output_size):
                    if j == k:
                        continue
                    dk = np.sqrt(wt[k].dot(wt[k]))
                    wtk = wt[k] / dk
                    tmp[j] += wtk.dot(s) * np.sign(wtj.dot(wtk))

            dc_dw[l] += self.coeff_lambda * tmp.transpose()

        return dc_db, dc_dw
