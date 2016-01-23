import numpy as np


class Box(object):
    """
    Box container class for Octree
    """

    def __init__(self, x_range, y_range, z_range, n_particle_min=20):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.center = (0.5*(x_range[1] + x_range[0]), 0.5*(y_range[1] + y_range[0]), 0.5*(z_range[1] + z_range[0]))
        self.n_particle_min = n_particle_min

        # radius = distance from center to a corner of the box
        dx = self.center[0] - x_range[0]
        dy = self.center[1] - y_range[0]
        dz = self.center[2] - z_range[0]
        self.radius = np.sqrt(dx*dx + dy*dy + dz*dz)

        # Leaf node
        self.leaf = False

        # Particle positions in this box, to be added
        # TODO: may need to keep a map of these positions to their original index
        self.q = np.asarray([])
        self.rx = np.asarray([])
        self.ry = np.asarray([])
        self.rz = np.asarray([])

    def length_x(self):
        return self.x_range[1] - self.x_range[0]

    def length_y(self):
        return self.y_range[1] - self.y_range[0]

    def length_z(self):
        return self.z_range[1] - self.z_range[0]

    def add(self, q, rx, ry, rz):
        """
        Add particles to this box if they are within ranges

        :param q: charges for each particle
        :param rx: x positions
        :param ry: y positions
        :param rz: z positions
        :return:
        """
        q = []
        x = []
        y = []
        z = []
        for i, qi in enumerate(q):
            if self.x_range[0] <= rx[i] < self.x_range[1]:
                if self.y_range[0] <= ry[i] < self.y_range[1]:
                    if self.z_range[0] <= rz[i] < self.z_range[1]:
                        q.append(qi)
                        x.append(rx[i])
                        y.append(ry[i])
                        z.append(rz[i])
        self.q = np.asarray(q)
        self.rx = np.asarray(x)
        self.ry = np.asarray(y)
        self.rz = np.asarray(z)

        # Determine leaf by number of particles in this box
        if self.n_particles() <= self.n_particle_min:
            self.leaf = True

    def n_particles(self):
        return len(self.q)

    def is_empty(self):
        return self.n_particles() == 0

    def is_leaf(self):
        return self.leaf

    def divide(self):
        """
        Divide this box into 8 sub-boxes based on position ranges. Keep only those boxes that are not empty.
        :return: list of up to 8 non-empty Box objects
        """
        sub_boxes = []
        half_x = 0.5 * self.length_x()
        half_y = 0.5 * self.length_y()
        half_z = 0.5 * self.length_z()
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0], self.x_range[0]+half_x), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0], self.y_range[0]+half_y), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0], self.z_range[0]+half_z), n_particle_min=self.n_particle_min))
        sub_boxes.append(Box((self.x_range[0]+half_x, self.x_range[1]), (self.y_range[0]+half_y, self.y_range[1]), (self.z_range[0]+half_z, self.z_range[1]), n_particle_min=self.n_particle_min))

        # Add particles of this box to children boxes
        for sub_box in sub_boxes:
            sub_box.add(self.q, self.rx, self.ry, self.rz)

        # Only retain those sub_boxes that are not empty
        return [box for box in sub_boxes if not box.is_empty()]


class Octree(object):
    """
    Octree implementation to be used in particle dipole networks
    """

    def __init__(self, max_levels=3, p=1, n_particle_min=20):
        self.max_level = max_levels
        self.p = p
        self.n_particle_min = n_particle_min

        self.levels = []  # List of boxes on each level

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
        min_x = rx[0]
        max_x = rx[0]
        min_y = ry[0]
        max_y = ry[0]
        min_z = rz[0]
        max_z = rz[0]
        for i in range(len(rx)):
            min_x = min(min_x, rx[i])
            max_x = min(max_x, rx[i])
            min_y = min(min_y, ry[i])
            max_y = min(max_y, ry[i])
            min_z = min(min_z, rz[i])
            max_z = min(max_z, rz[i])

        # Level 0 box
        buffer = 0.001
        box = Box((min_x-buffer, max_x+buffer), (min_y-buffer, max_y+buffer), (min_z-buffer, max_z+buffer), n_particle_min=self.n_particle_min)
        box.add(q2, rx, ry, rz)
        if self.max_level == 0:
            box.leaf = True
        self.levels.append([box])

        # Subsequent boxes
        level = 1
        all_leaves = box.leaf
        while not all_leaves:
            level_boxes = []

            # Divide parent box into 8 sub-boxes if it is not a leaf
            for parent_box in self.levels[level-1]:
                if not parent_box.leaf:
                    sub_boxes = parent_box.divide()
                    if len(sub_boxes) > 0:
                        level_boxes.append(sub_boxes)

            # Max level check
            if level >= self.max_level:
                # Mark all level boxes as leaf
                all_leaves = True
                for level_box in level_boxes:
                    level_box.leaf = True
            else:
                # Check if all leaves
                all_leaves = True
                for level_box in level_boxes:
                    all_leaves = level_box.leaf and all_leaves

            # Add level boxes
            self.levels.append(level_boxes)
