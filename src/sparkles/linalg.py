"""
Spark linear algebra
"""


class Vector(object):

    def __init__(self, sc, vec=list()):
        self.sc = sc
        self.rdd = sc.parallelize([(i, vi) for i, vi in enumerate(vec)])

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


def _dot(v, w):
    return sum(v[i] * w[i] for i in range(len(v)))


class Matrix(object):

    def __init__(self, sc, mat=list(list())):
        self.sc = sc
        self.rdd = sc.parallelize([(i, mi) for i, mi in enumerate(mat)])

    def m_dot_v(self, v, broadcast=True):
        # may not be the best idea to collect... optional for already broadcast v
        w = self.sc.broadcast(v.rdd.collect()) if broadcast else v
        v = Vector(self.sc)
        v.rdd = self.rdd.map(lambda row: (row[0], sum(row[1][i] * w.value[i][1] for i in range(len(row[1])))))
        return v
