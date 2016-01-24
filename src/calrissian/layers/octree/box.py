import numpy as np


class Box(object):
    """
    Box container class for Octree
    """

    def __init__(self, x_range, y_range, z_range, n_particle_min=20, p=0):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.center = (0.5*(x_range[1] + x_range[0]), 0.5*(y_range[1] + y_range[0]), 0.5*(z_range[1] + z_range[0]))

        self.n_particle_min = n_particle_min
        self.p = p

        # radius = distance from center to a corner of the box
        dx = self.center[0] - x_range[0]
        dy = self.center[1] - y_range[0]
        dz = self.center[2] - z_range[0]
        self.radius = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Leaf node
        self.leaf = False

        # Children boxes
        self.children = []

        # Particle positions in this box, to be added
        # TODO: may need to keep a map of these positions to their original index
        self.q = np.asarray([])
        self.rx = np.asarray([])
        self.ry = np.asarray([])
        self.rz = np.asarray([])
        self.indexes = np.asarray([])

        # Multipole moments
        # TODO: higher order
        self.moments = np.zeros((p+1, p+1, p+1))  # actually mor space than necessary, but easy for coding

    def length_x(self):
        return self.x_range[1] - self.x_range[0]

    def length_y(self):
        return self.y_range[1] - self.y_range[0]

    def length_z(self):
        return self.z_range[1] - self.z_range[0]

    def add(self, rq, rx, ry, rz, indexes):
        """
        Add particles to this box if they are within ranges. Compute multipole moments.

        :param q: charges for each particle
        :param rx: x positions
        :param ry: y positions
        :param rz: z positions
        :param indexes: indexes corresponding to original arrays
        :return:
        """
        q = []
        x = []
        y = []
        z = []
        ind = []
        for i, qi in enumerate(rq):
            if self.x_range[0] <= rx[i] < self.x_range[1]:
                if self.y_range[0] <= ry[i] < self.y_range[1]:
                    if self.z_range[0] <= rz[i] < self.z_range[1]:
                        q.append(qi)
                        x.append(rx[i])
                        y.append(ry[i])
                        z.append(rz[i])
                        ind.append(indexes[i])
        self.q = np.asarray(q)
        self.rx = np.asarray(x)
        self.ry = np.asarray(y)
        self.rz = np.asarray(z)
        self.indexes = np.asarray(ind)

        # Determine leaf by number of particles in this box
        if self.n_particles() <= self.n_particle_min:
            self.leaf = True

    def n_particles(self):
        return len(self.q)

    def is_empty(self):
        return self.n_particles() == 0

    def is_leaf(self):
        return self.leaf

    # def set_box_q(self, q_all):
    #     """
    #     To set dynamically changing charges
    #     :param q_all: (n_input_data, n_input_nodes)
    #     :return:
    #     """
    #     self.q = np.zeros((len(q_all), len(self.rx)))
    #     for k, qk in enumerate(q_all):
    #         i = 0
    #         for ind in self.indexes:
    #             self.q[k][i] = qk[ind]
    #             i += 1

    def compute_multipoles(self, q_in):
        # Compute multipole moments of this box
        # TODO: higher order. only doing monopole right now
        if self.n_particles() > 0:
            # p = 0
            q_sum = 0.0
            for i in self.indexes:
                q_sum += q_in[i]
            self.moments[0][0][0] = q_sum

    def divide(self):
        """
        Divide this box into 8 sub-boxes based on position ranges. Keep only those boxes that are not empty.
        """
        if self.is_leaf():
            return

        sub_boxes = []
        half_x = 0.5 * self.length_x()
        half_y = 0.5 * self.length_y()
        half_z = 0.5 * self.length_z()
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min, p=self.p))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min, p=self.p))

        # Add particles of this box to children boxes
        for sub_box in sub_boxes:
            sub_box.add(self.q, self.rx, self.ry, self.rz, self.indexes)

        # Only retain those sub_boxes that are not empty
        self.children = [box for box in sub_boxes if not box.is_empty()]
