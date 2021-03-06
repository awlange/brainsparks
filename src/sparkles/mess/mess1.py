from pyspark import SparkContext

from src.sparkles.linalg import *

import time
import sys


def main():
    sc = SparkContext(appName="matrix")

    n_partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    n = 4000

    v = Vector(sc, n_partitions, [x for x in range(n)])
    # w = Vector(sc, [2*x for x in range(n)])

    # vw = v.v_add_w(w)
    # print(vw.collect())

    m = Matrix(sc, n_partitions, [[x for x in range(n)] for _ in range(n)])

    time_start = time.time()

    for _ in range(5):
        mv = m.m_dot_v(v)
        # print(mv.rdd.take(10))
        mv.rdd.take(1000)

    print(time.time() - time_start)

    sc.stop()

if __name__ == "__main__":
    main()
