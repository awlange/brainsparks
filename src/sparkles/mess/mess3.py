from pyspark import SparkContext

from src.sparkles.linalg import *

import numpy as np

import time
import sys


def main():
    sc = SparkContext(appName="matrix")
    n_partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    n = 6000

    # v = Vector2(sc, np.asarray([x for x in range(n)]), n_partitions)
    v = np.asarray([x for x in range(n)])
    m = Matrix2(sc, np.asarray([[x for x in range(n)] for _ in range(n)]), n_partitions)

    time_start = time.time()

    for _ in range(5):
        mv = m.m_dot_v_re(v)
        mv.rdd.take(1000)

    print(time.time() - time_start)

    sc.stop()

if __name__ == "__main__":
    main()
