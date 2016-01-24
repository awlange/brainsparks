
from src.calrissian.particle_dipole_network import ParticleDipoleNetwork
from src.calrissian.layers.particle_dipole import ParticleDipole
from src.calrissian.layers.particle_dipole import ParticleDipoleInput

from src.calrissian.particle_dipole_tree_network import ParticleDipoleTreeNetwork
from src.calrissian.layers.particle_dipole_tree import ParticleDipoleTree
from src.calrissian.layers.particle_dipole_tree import ParticleDipoleTreeInput

import numpy as np


def main():

    train_X = np.asarray([[0.2, -0.3], [0.1, -0.9], [0.3, 0.5]])
    train_Y = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

    # For comparison
    net = ParticleDipoleNetwork(cost="mse", particle_input=ParticleDipoleInput(2))
    net.append(ParticleDipole(2, 5, activation="sigmoid"))
    net.append(ParticleDipole(5, 3, activation="sigmoid"))

    print(net.predict(train_X))
    print(net.cost(train_X, train_Y))

    net2 = ParticleDipoleTreeNetwork(cost="mse", particle_input=ParticleDipoleTreeInput(2))
    net2.append(ParticleDipoleTree(2, 5, activation="sigmoid"))
    net2.append(ParticleDipoleTree(5, 3, activation="sigmoid"))
    # Make sure we have the same coordinates and charges
    net2.particle_input.copy_pos_neg_positions(net.particle_input.rx_pos, net.particle_input.ry_pos, net.particle_input.rz_pos,
                                               net.particle_input.rx_neg, net.particle_input.ry_neg, net.particle_input.rz_neg)
    for l in range(len(net.layers)):
        net2.layers[l].copy_pos_neg_positions(net.layers[l].q, net.layers[l].b,
                                              net.layers[l].rx_pos, net.layers[l].ry_pos, net.layers[l].rz_pos,
                                              net.layers[l].rx_neg, net.layers[l].ry_neg, net.layers[l].rz_neg)

    print(net2.predict(train_X))
    print(net2.cost(train_X, train_Y))


if __name__ == "__main__":

    # Ensure same seed
    n_seed = 777
    np.random.seed(n_seed)
    state = np.random.get_state()

    main()
