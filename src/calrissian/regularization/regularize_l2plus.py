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
            # # Working version
            # w = layer.w
            # for j in range(layer.output_size):
            #     for k in range(j, layer.output_size):
            #         dot = 0.0
            #         for i in range(layer.input_size):
            #             dot += w[i][j] * w[i][k]
            #         c += dot*dot

            # # Working version 2
            # wt = layer.w.transpose()
            # for j in range(layer.output_size):
            #     for k in range(j, layer.output_size):
            #         dot = 0.0
            #         for i in range(layer.input_size):
            #             dot += wt[j][i] * wt[k][i]
            #         c += np.abs(dot)

            # # Vectorize
            # wt = layer.w.transpose()
            # for j in range(layer.output_size):
            #     wt_j = wt[j]
            #     for k in range(j+1, layer.output_size):
            #         c += wt_j.dot(wt[k])

            wt = layer.w.transpose()
            for j in range(layer.output_size):
                for k in range(j, layer.output_size):
                    c += np.abs(wt[j].dot(wt[k]))
                    # c += (wt[j].dot(wt[k]))**2
        return self.coeff_lambda * c

    def cost_gradient(self, layers, dc_db, dc_dw):
        for l, layer in enumerate(layers):
            # # Working version
            # w = layer.w
            # tmp = np.zeros_like(w)
            # for j in range(layer.output_size):
            #     for k in range(j, layer.output_size):
            #         dot = 0.0
            #         for i in range(layer.input_size):
            #             dot += w[i][j] * w[i][k]
            #
            #         for i in range(layer.input_size):
            #             tmp[i][j] += w[i][k] * dot
            #             tmp[i][k] += w[i][j] * dot
            # dc_dw[l] += 2.0 * self.coeff_lambda * tmp

            # # Working version 2
            # wt = layer.w.transpose()
            # tmp = np.zeros_like(wt)
            # for j in range(layer.output_size):
            #     for k in range(j, layer.output_size):
            #         dot = 0.0
            #         for i in range(layer.input_size):
            #             dot += wt[j][i] * wt[k][i]
            #
            #         s = np.sign(dot)
            #
            #         for i in range(layer.input_size):
            #             tmp[j][i] += wt[k][i] * s
            #             tmp[k][i] += wt[j][i] * s
            #
            # dc_dw[l] += self.coeff_lambda * tmp.transpose()

            # # Vectorize
            # wt = layer.w.transpose()
            # tmp = np.zeros_like(wt)
            # for j in range(layer.output_size):
            #     wt_j = wt[j]
            #     for k in range(j+1, layer.output_size):
            #         tmp[j] += wt[k]
            #         tmp[k] += wt_j
            # dc_dw[l] += self.coeff_lambda * tmp.transpose()

            wt = layer.w.transpose()
            tmp = np.zeros_like(wt)
            for j in range(layer.output_size):
                for k in range(j, layer.output_size):
                    s = np.sign(wt[j].dot(wt[k]))
                    # s = 2 * wt[j].dot(wt[k])
                    tmp[j] += wt[k] * s
                    tmp[k] += wt[j] * s
            dc_dw[l] += self.coeff_lambda * tmp.transpose()

        return dc_db, dc_dw
