"""
Spark (is not good for) linear algebra
"""

import numpy as np


class Vector(object):

    def __init__(self, sc, n_partitions=1, vec=list()):
        self.sc = sc
        self.rdd = sc.parallelize([(i, vi) for i, vi in enumerate(vec)], n_partitions)

    def v_add_w(self, w):
        return self.rdd.join(w.rdd).map(lambda x: (x[0], x[1][0] + x[1][1]))

    def v_min_w(self, w):
        return self.rdd.join(w.rdd).map(lambda x: (x[0], x[1][0] - x[1][1]))

    def v_times_w(self, w):
        return self.rdd.join(w.rdd).map(lambda x: (x[0], x[1][0] * x[1][1]))

    def v_div_w(self, w):
        return self.rdd.join(w.rdd).map(lambda x: (x[0], x[1][0] / x[1][1]))

    def v_dot_w(self, w):
        return self.rdd.join(w.rdd).map(lambda x: x[1][0] * x[1][1]).sum()

    def sqdist(self, w):
        return self.rdd.join(w.rdd).map(lambda x: (x[1][0] - x[1][1])**2).sum()


class Vector2(object):
    """
    Parallelized numpy array
    """

    def __init__(self, sc, vec=None, n_partitions=1, n_join_tasks=2):
        self.sc = sc
        self.n_partitions = n_partitions
        self.n_join_tasks = n_join_tasks
        self.rdd = None if vec is None \
            else sc.parallelize([(i, vi) for i, vi in enumerate(np.array_split(vec, n_partitions))], n_partitions)

    def v_add_w(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: (x[0], x[1][0] + x[1][1]))

    def v_min_w(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: (x[0], x[1][0] - x[1][1]))

    def v_times_w(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: (x[0], x[1][0] * x[1][1]))

    def v_div_w(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: (x[0], x[1][0] / x[1][1]))

    def v_dot_w(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: x[1][0].dot(x[1][1])).sum()

    def sqdist(self, w):
        return self.rdd.join(w.rdd, self.n_join_tasks).map(lambda x: np.sum((x[1][0] - x[1][1])**2)).sum()


def _dot(v, w):
    return sum(v[i] * w[i] for i in range(len(v)))


class Matrix(object):

    def __init__(self, sc, n_partitions=1, mat=list(list())):
        self.sc = sc
        self.n_partitions = n_partitions
        self.rdd = sc.parallelize([(i, mi) for i, mi in enumerate(mat)], n_partitions)

    def m_dot_v(self, v, broadcast=True):
        # may not be the best idea to collect... optional for already broadcast v
        w = self.sc.broadcast(v.rdd.sortByKey().map(lambda z: z[1]).collect()) if broadcast else v
        v = Vector(self.sc, self.n_partitions)
        v.rdd = self.rdd.map(lambda row: (row[0], sum(row[1][i] * w.value[i] for i in range(len(row[1])))))
        return v


class Matrix2(object):

    def __init__(self, sc, mat, n_partitions=1):
        self.sc = sc
        self.n_partitions = n_partitions
        self.rdd = sc.parallelize([(i, mi) for i, mi in enumerate(np.array_split(mat, n_partitions))], n_partitions)

    def m_dot_v(self, v, broadcast=True):
        tmp = v.rdd.sortByKey().map(lambda z: z[1]).collect()
        w = self.sc.broadcast(np.reshape(tmp, len(tmp) * len(tmp[0])))
        wa = np.asarray(w.value)
        v = Vector2(self.sc, n_partitions=self.n_partitions)
        v.rdd = self.rdd.map(lambda block: (block[0], block[1].dot(wa)))
        return v

    def m_dot_v_re(self, v):
        """
        redundant vector version
        """
        w = Vector2(self.sc, n_partitions=self.n_partitions)
        w.rdd = self.rdd.map(lambda block: (block[0], block[1].dot(v)))
        return w
