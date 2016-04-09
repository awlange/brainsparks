from .box import Box

import numpy as np
from numba import jit


class Octree(object):
    """
    Octree implementation to be used in particle dipole networks
    """

    def __init__(self, max_level=3, p=0, n_particle_min=20, mac=0.0, cut=1000.0):
        self.max_level = max_level
        self.p = p
        self.n_particle_min = n_particle_min
        self.mac = mac  # Multipole acceptance criterion
        self.cut = cut  # Distance cutoff

        # Root Box
        self.root_box = None
        self.levels = 0

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
        else:
            self.root_box.divide()

        # Subsequent boxes
        level_boxes = [self.root_box]
        level = 0
        while len(level_boxes) > 0:
            level += 1
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
        self.levels = level

    def set_dynamic_charges(self, q_in):
        """
        Set the given charges for each box in the tree

        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """
        # Make positive and negative copy of q_in charges
        q = np.concatenate((q_in, -q_in), axis=1)
        self.set_dynamic_charges_recursive(np.asarray(q), self.root_box)

    def set_dynamic_charges_recursive(self, q_in, box):
        """
        Recursive loop through the octree to set charges
        """
        box.set_dynamic_q(q_in)
        if not box.leaf:
            for child_box in box.children:
                self.set_dynamic_charges_recursive(q_in, child_box)

    def compute_potential(self, rx, ry, rz, q_in, set_dynamic_charges=True):
        """
        Compute the potential at the given input coordinates by recursively traversing the octree
        :param rx: numpy array of x positions
        :param ry: numpy array of y positions
        :param rz: numpy array of z positions
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """

        if set_dynamic_charges:
            self.set_dynamic_charges(q_in)

        potential = np.zeros((len(rx), len(q_in)))
        for i in range(len(rx)):
            self.compute_potential_recursive(potential[i], rx[i], ry[i], rz[i], self.root_box)
        return potential

    def compute_potential_recursive(self, potential, rxi, ryi, rzi, box):
        """
        Compute potential for single input coordinates (not a numpy array)
        :param rxi: x position
        :param ryi: y position
        :param rzi: z position
        """
        dx = box.center[0] - rxi
        dy = box.center[1] - ryi
        dz = box.center[2] - rzi
        R2 = dx*dx + dy*dy + dz*dz

        # Beyond cutoff or MAC
        tmp = self.cut + box.radius
        if R2 > tmp*tmp:
            # Do nothing!
            pass

        # elif box.radius / R <= self.mac:
        #     # Use the multipole expansion for potential
        #
        #     # TODO: higher order multipoles
        #     # p = 0
        #     # potential = box.multipoles[0][0][0] * np.exp(-R*R)
        #
        #     # total_charge_vector = np.sum(box.dynamic_q, axis=1)
        #     total_charge_vector = box.get_total_charge()
        #     potential += total_charge_vector * np.exp(-R2)

        elif box.leaf:
            # Compute direct interaction
            # Vectorized way with dynamic charges array
            dx = box.rx - rxi
            dy = box.ry - ryi
            dz = box.rz - rzi
            d2 = dx*dx + dy*dy + dz*dz
            potential += (np.exp(-d2)).dot(box.dynamic_q)

        else:
            # Recurse
            for child_box in box.children:
                self.compute_potential_recursive(potential, rxi, ryi, rzi, child_box)

    def compute_potential2(self, rx, ry, rz, q_in, set_dynamic_charges=True):
        """
        Compute the potential at the given input coordinates by recursively traversing the octree
        :param rx: numpy array of x positions
        :param ry: numpy array of y positions
        :param rz: numpy array of z positions
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """

        if set_dynamic_charges:
            self.set_dynamic_charges(q_in)
        potential = np.zeros((len(rx), len(q_in)))

        for i in range(len(rx)):
            potential_i = potential[i]
            rxi = rx[i]
            ryi = ry[i]
            rzi = rz[i]

            # Breadth-first search traversal
            box_queue = [self.root_box]
            while len(box_queue) > 0:

                # Get next box from queue
                box = box_queue.pop(0)

                dx = box.center[0] - rxi
                dy = box.center[1] - ryi
                dz = box.center[2] - rzi
                R2 = dx*dx + dy*dy + dz*dz

                # Beyond cutoff or MAC
                tmp = self.cut + box.radius
                if R2 > tmp*tmp:
                    # Do nothing!
                    pass

                # TODO: multipoles

                elif box.leaf:
                    # Compute direct interaction
                    # Vectorized way with dynamic charges array
                    bx = box.rx - rxi
                    by = box.ry - ryi
                    bz = box.rz - rzi
                    d2 = bx*bx + by*by + bz*bz
                    potential_i += (np.exp(-d2)).dot(box.dynamic_q)

                else:
                    # Recurse
                    for child_box in box.children:
                        box_queue.append(child_box)

        return potential

    def compute_potential3(self, rx, ry, rz, q_in, set_dynamic_charges=True):
        """
        Compute the potential at the given input coordinates by recursively traversing the octree
        :param rx: numpy array of x positions
        :param ry: numpy array of y positions
        :param rz: numpy array of z positions
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """

        if set_dynamic_charges:
            self.set_dynamic_charges(q_in)
        potential = np.zeros((len(rx), len(q_in)))

        # Breadth-first search traversal
        box_queue = [self.root_box]
        while len(box_queue) > 0:

            # Get next box from queue
            box = box_queue.pop(0)

            dx = box.center[0] - rx
            dy = box.center[1] - ry
            dz = box.center[2] - rz
            R2 = dx*dx + dy*dy + dz*dz

            # Beyond cutoff or MAC
            # Try piecewise?
            tmp = self.cut + box.radius
            tmp *= tmp

            for i in range(len(rx)):
                if R2[i] > tmp:
                    # Do nothing!
                    pass

                # TODO: multipoles

                elif box.leaf:
                    # Compute direct interaction
                    # Vectorized way with dynamic charges array
                    ddx = box.rx - rx[i]
                    ddy = box.ry - ry[i]
                    ddz = box.rz - rz[i]
                    d2 = ddx*ddx + ddy*ddy + ddz*ddz
                    potential[i] += (np.exp(-d2)).dot(box.dynamic_q)
                    # TODO: collect all points interacting with this box, then compute dot?

            for child_box in box.children:
                box_queue.append(child_box)

        return potential

    def compute_potential4(self, rx, ry, rz, q_in, set_dynamic_charges=True):
        """
        Compute the potential at the given input coordinates by recursively traversing the octree
        :param rx: numpy array of x positions
        :param ry: numpy array of y positions
        :param rz: numpy array of z positions
        :param q_in: input charges --> (n_input_data, n_input_nodes)
        """

        if set_dynamic_charges:
            self.set_dynamic_charges(q_in)
        potential = np.zeros((len(rx), len(q_in)))

        # Breadth-first search traversal
        box_queue = [self.root_box]
        while len(box_queue) > 0:

            # Get next box from queue
            box = box_queue.pop(0)

            dx = box.center[0] - rx
            dy = box.center[1] - ry
            dz = box.center[2] - rz
            R2 = dx*dx + dy*dy + dz*dz

            # Beyond cutoff or MAC
            # Try piecewise?
            tmp = self.cut + box.radius
            tmp *= tmp

            for i in range(len(rx)):
                if R2[i] > tmp:
                    # Do nothing!
                    pass

                # TODO: multipoles

                elif box.leaf:
                    # Compute direct interaction
                    # Vectorized way with dynamic charges array
                    ddx = box.rx - rx[i]
                    ddy = box.ry - ry[i]
                    ddz = box.rz - rz[i]
                    d2 = ddx*ddx + ddy*ddy + ddz*ddz
                    potential[i] += (np.exp(-d2)).dot(box.dynamic_q)
                    # TODO: collect all points interacting with this box, then compute dot?

            for child_box in box.children:
                box_queue.append(child_box)

        return potential
