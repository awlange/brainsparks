from src.sandbox.cost import Cost

import src.sandbox.linalg as linalg
import numpy as np


class Network(object):

    def __init__(self, cost="quadratic"):
        self.layers = []
        self.cost_function = Cost.get(cost)
        self.cost_d_function = Cost.get_d(cost)

    def add(self, layer):
        self.layers.append(layer)

    def forward_pass(self, input_vector):
        a = input_vector
        for layer in self.layers:
            a = layer.feed_forward(a)
        return a

    def full_cost(self, train_X, train_Y):
        c = 0.0
        for single_x, single_y in zip(train_X, train_Y):
            c += self.cost_function(single_y, self.forward_pass(single_x))
        c /= 2 * len(train_X)
        return c

    def batch_full_cost(self, train_X, train_Y):
        return self.cost_function(train_Y, self.forward_pass(train_X))

    def gradient_single(self, train_x, train_y):

        # For batch, want to compute A and Z for each input all at once
        sigma_Z = []
        A = [train_x]
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            a = layer.compute_a(z)
            A.append(a)
            sigma_Z.append(layer.compute_da(z))

        delta_L = linalg.vtimesw(self.cost_d_function(train_y, A[-1]), sigma_Z[-1])
        prev_delta = delta_L

        dCdb = [[] for _ in range(len(self.layers))]
        dCdw = [[] for _ in range(len(self.layers))]

        dCdb[-1] = delta_L
        dCdw[-1] = linalg.outer(delta_L, A[-2])

        # TODO: for batch, we should loop through each training example and add/reduce it to the gradient matrices

        for l_plus_1 in range(len(self.layers)-1, 0, -1):
            layer = self.layers[l_plus_1]
            delta = linalg.vtimesw(linalg.mdotv(linalg.transpose(layer.w), prev_delta), sigma_Z[l_plus_1-1])

            # Bias and weight dervis
            dCdb[l_plus_1-1] = delta
            dCdw[l_plus_1-1] = linalg.outer(delta, A[l_plus_1-1])

            prev_delta = delta

        # TODO: divide by N, were this not to be a "single" gradient

        return dCdb, dCdw

    def gradient_batch(self, train_X, train_Y):

        # Output gradients
        dCdb = []
        dCdw = []

        # For batch, we compute A and Z for each input all at once
        sigma_Z = []
        A = [train_X]
        for l, layer in enumerate(self.layers):
            z = layer.compute_z(A[l])
            A.append(layer.compute_a(z))
            sigma_Z.append(layer.compute_da(z))

            # Initialize
            dCdb.append(np.zeros(layer.b.shape))
            dCdw.append(np.zeros(layer.w.shape))

        delta_L = self.cost_d_function(train_Y, A[-1]) * sigma_Z[-1]  # Element-wise multiply

        # For each training case
        for i in range(len(train_X)):

            prev_delta = delta_L[i]
            dCdb[-1] += prev_delta
            dCdw[-1] += np.outer(A[-2][i], prev_delta)

            for l_plus_1 in range(len(self.layers)-1, 0, -1):
                layer = self.layers[l_plus_1]
                delta = layer.w.dot(prev_delta) * sigma_Z[l_plus_1-1][i]

                # Bias and weight derivatives
                dCdb[l_plus_1-1] += delta
                dCdw[l_plus_1-1] += np.outer(A[l_plus_1-1][i], delta)

                prev_delta = delta

        return dCdb, dCdw

