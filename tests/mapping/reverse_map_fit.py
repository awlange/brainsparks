
import numpy as np
import pandas as pd

# BrainSparks imports
from src.calrissian.particle_network import ParticleNetwork
from src.calrissian.layers.particle import Particle
from src.calrissian.layers.particle import ParticleInput

from src.calrissian.network import Network
from src.calrissian.layers.dense import Dense

n_seed = 500
np.random.seed(n_seed)


def compute_matrices(input_net, i_layer=None):
    """
    Computes the weight matrix for each layer in the Particle Network
    With i_layer, only compute the matrix for the specified layer
    """
    r = input_net.particle_input.get_rxyz()
    for i, layer in enumerate(input_net.layers):
        if i_layer is None:
            layer.compute_w(r)
        elif i_layer == i:
            layer.compute_w(r)
        r = layer.get_rxyz()
    return input_net


def compute_error(net_mlp, net):
    """
    Computes the square error for all biases/weights for all layers
    """
    se = 0.0
    for l, layer in enumerate(net.layers):
        se += np.sum((net_mlp.layers[l].b - layer.b)**2)
        se += np.sum((net_mlp.layers[l].w - layer.w)**2)
    return se


def compute_grad_w(net_mlp, net):

    de_db = []
    for l, layer in enumerate(net.layers):
        de_db.append(2.0 * (layer.b - net_mlp.layers[l].b))

    print("analytical b")
    for b in de_db:
        print(b)

    # Common prefactor
    de_dw = []
    for l, layer in enumerate(net.layers):
        de_dw.append(2.0 * layer.w * (layer.w - net_mlp.layers[l].w))

    de_dq = []
    for l, layer in enumerate(net.layers):
        qq = np.zeros_like(layer.q)
        for j, qj in enumerate(layer.q):
            tmp = 0.0
            for i in range(layer.input_size):
                tmp += de_dw[l][i][j] / qj
            qq[j] = tmp
        de_dq.append(qq)
    print("analytical q")
    for q in de_dq:
        print(q)

    de_dt = []

    # input
    layer = net.particle_input
    next_layer = net.layers[0]
    tt = np.zeros_like(layer.theta)
    for j, tj in enumerate(layer.theta):
        tmp = 0.0
        for i in range(next_layer.output_size):
            dt = tj - next_layer.theta[i]
            tmp -= de_dw[0][j][i] * np.tan(dt)
        tt[j] = tmp
    de_dt.append(tt)

    # layers
    for l, layer in enumerate(net.layers):
        prev_layer = net.particle_input
        if l > 0:
            prev_layer = net.layers[l-1]

        # This layer matrix
        tt = np.zeros_like(layer.theta)
        for j, tj in enumerate(layer.theta):
            tmp = 0.0
            for i in range(prev_layer.output_size):
                dt = prev_layer.theta[i] - tj
                tmp += de_dw[l][i][j] * np.tan(dt)
            tt[j] = tmp

        # Next layer matrix
        if l+1 < len(net.layers):
            next_layer = net.layers[l+1]
            for j, tj in enumerate(layer.theta):
                tmp = 0.0
                for i in range(next_layer.output_size):
                    dt = tj - next_layer.theta[i]
                    tmp -= de_dw[l+1][j][i] * np.tan(dt)
                tt[j] += tmp

        de_dt.append(tt)

    print("analytical t")
    for t in de_dt:
        print(t)



def compute_fd_grad(net_mlp, net):

    h = 0.001

    # Bias
    fd_b = []
    for l, layer in enumerate(net.layers):
        lb = []
        for i, b in enumerate(layer.b[0]):
            hold = layer.b[0][i]
            layer.b[0][i] += h
            fp = compute_error(net_mlp, net)
            layer.b[0][i] -= 2*h
            fm = compute_error(net_mlp, net)
            lb.append((fp - fm) / (2*h))
            layer.b[0][i] = hold
        fd_b.append(np.asarray(lb))
    print("numerical b")
    for b in fd_b:
        print(b)

    # Charge
    fd_q = []
    for l, layer in enumerate(net.layers):
        lq = []
        for j, q in enumerate(layer.q):
            hold = layer.q[j]
            layer.q[j] += h
            compute_matrices(net)
            fp = compute_error(net_mlp, net)
            layer.q[j] -= 2*h
            compute_matrices(net)
            fm = compute_error(net_mlp, net)
            lq.append((fp - fm) / (2*h))
            layer.q[j] = hold
        fd_q.append(np.asarray(lq))
    print("numerical q")
    for q in fd_q:
        print(q)

    # Theta
    fd_t = []

    # input
    lt = []
    layer = net.particle_input
    for j, t in enumerate(layer.theta):
        hold = layer.theta[j]
        layer.theta[j] += h
        compute_matrices(net)
        fp = compute_error(net_mlp, net)
        layer.theta[j] -= 2*h
        compute_matrices(net)
        fm = compute_error(net_mlp, net)
        lt.append((fp - fm) / (2*h))
        layer.theta[j] = hold
    fd_t.append(np.asarray(lt))

    # layers
    for l, layer in enumerate(net.layers):
        lt = []
        for j, t in enumerate(layer.theta):
            hold = layer.theta[j]
            layer.theta[j] += h
            compute_matrices(net)
            fp = compute_error(net_mlp, net)
            layer.theta[j] -= 2*h
            compute_matrices(net)
            fm = compute_error(net_mlp, net)
            lt.append((fp - fm) / (2*h))
            layer.theta[j] = hold
        fd_t.append(np.asarray(lt))
    print("numerical theta")
    for t in fd_t:
        print(t)


def main():

    # MLP to be fit
    net_mlp = None
    with open("/Users/alange/programming/MNIST/store/classic_32.json", "r") as f:
        net_mlp = Network.read_from_json(f)

    # Initial particle network
    net = ParticleNetwork(cost="categorical_cross_entropy", particle_input=ParticleInput(784, s=2.0))
    net.append(Particle(784, 32, activation="tanh", zeta=1.0, s=2.0))
    net.append(Particle(32, 10, activation="softmax", zeta=1.0, s=2.0))
    compute_matrices(net)

    error = compute_error(net_mlp, net)

    compute_grad_w(net_mlp, net)
    compute_fd_grad(net_mlp, net)


if __name__ == "__main__":
    main()