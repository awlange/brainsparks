from .box import Box

import numpy as np


class Octree(object):
    """
    Octree implementation to be used in particle dipole networks
    """

    def __init__(self, max_levels=3, p=0, n_particle_min=20, mac=0.0, cut=1000.0):
        self.max_level = max_levels
        self.p = p
        self.n_particle_min = n_particle_min
        self.mac = mac  # Multipole acceptance criterion
        self.cut = cut  # Distance cutoff

        # Root Box
        self.root_box = None

    def build_tree(self, q, rx, ry, rz):
        """
        Given positions, build the tree

        :param rx:
        :param ry:
        :param rz:
        :return:
        """

        # Duplicate charge data for ease of adding to boxes
        q2 = []
        for qi in q:
            q2.append(qi)  # positive
            q2.append(-qi)  # negative

        # Get min/max cartesian positions
        min_x = np.min(rx)
        max_x = np.max(rx)
        min_y = np.min(ry)
        max_y = np.max(ry)
        min_z = np.min(rz)
        max_z = np.max(rz)

        # Level 0 box
        buffer = 0.001
        self.root_box = Box((min_x-buffer, max_x+buffer), (min_y-buffer, max_y+buffer), (min_z-buffer, max_z+buffer), n_particle_min=self.n_particle_min, p=self.p)
        self.root_box.add(q2, rx, ry, rz, np.arange(len(rx)))
        if self.max_level == 0:
            self.root_box.leaf = True

        # Subsequent boxes
        level_boxes = self.root_box.children
        level = 1
        while len(level_boxes) > 0:
            next_level_boxes = []
            for level_box in level_boxes:
                # Divide parent box into 8 sub-boxes if it is not a leaf
                for child_box in level_box.children:
                    if not child_box.leaf:
                        child_box.divide()
                        # Max level check
                        if level >= self.max_level:
                            for box in child_box.children:
                                box.leaf = True
                        else:
                            next_level_boxes.append(child_box)

            level_boxes = next_level_boxes

    def compute_potential(self, rx, ry, rz, q_in):
        """
        Compute the potential at the given input coordinates by recursively traversing the octree
        :param rx: numpy array of x positions
        :param ry: numpy array of y positions
        :param rz: numpy array of z positions
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """
        potential = np.zeros((len(rx), len(q_in)))
        for i in range(len(rx)):
            potential[i] = self.compute_potential_recursive(rx[i], ry[i], rz[i], q_in, self.root_box)
        return potential

    def compute_potential_recursive(self, rxi, ryi, rzi, q_in, box):
        """
        Compute potential for single input coordinates (not a numpy array)
        :param rxi: x position
        :param ryi: y position
        :param rzi: z position
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        :return: the potential as approximated by the octree
        """
        potential = np.zeros(len(q_in))
        dx = box.center[0] - rxi
        dy = box.center[1] - ryi
        dz = box.center[2] - rzi
        R = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Beyond cutoff or MAC
        if R - box.radius > self.cut or box.radius / R <= self.mac:
            # Use the multipole expansion for potential

            # TODO: higher order multipoles
            # p = 0
            # potential = box.multipoles[0][0][0] * np.exp(-R*R)
            # TODO: do we recompute moments for each input? Or just ignore?
            # Do nothing, potential already zero above
            pass

        elif box.leaf:
            # Compute direct interaction
            dx = box.rx - rxi
            dy = box.ry - ryi
            dz = box.rz - rzi
            # box.set_box_q(q_in) # ???
            # tmp = box.q * np.exp(-(dx*dx + dy*dy + dz*dz))
            d2 = dx*dx + dy*dy + dz*dz
            exp = np.exp(-d2)

            # TODO: temporary kind of dumb way, but works
            # For each input data...
            for ip in range(len(potential)):
                # Get charges for this box
                box_q = np.zeros_like(dx)
                j = 0
                for ind in box.indexes:
                    i = ind // 2
                    box_q[j] = q_in[ip][i] if ind % 2 == 0 else -q_in[ip][i]  # sign of charge
                    # box_q[j] = 1 if ind % 2 == 0 else -1  # sign of charge
                    j += 1
                potential[ip] = np.sum(exp * box_q)  # check

        else:
            # Recurse
            for child_box in box.children:
                potential += self.compute_potential_recursive(rxi, ryi, rzi, q_in, child_box)

        return potential
