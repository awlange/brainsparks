from src.sandbox.cost import Cost

import src.sandbox.linalg as linalg


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

    def gradient_single(self, train_x, train_y):

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

        for l_plus_1 in range(len(self.layers)-1, 0, -1):
            layer = self.layers[l_plus_1]
            delta = linalg.vtimesw(linalg.mdotv(linalg.transpose(layer.w), prev_delta), sigma_Z[l_plus_1-1])

            # Bias and weight dervis
            dCdb[l_plus_1-1] = delta
            dCdw[l_plus_1-1] = linalg.outer(delta, A[l_plus_1-1])  # TODO: bug here, wtf is the operation

            prev_delta = delta

        return dCdb, dCdw

