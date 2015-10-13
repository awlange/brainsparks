from pyspark import SparkContext

from src.sparkles.linalg import *

import numpy as np

import time
import sys


def main():
    sc = SparkContext(appName="matrix")

    n_partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    n = 10000000

    v = Vector2(sc, np.asarray([x for x in range(n)]), n_partitions)
    w = Vector2(sc, np.asarray([x for x in range(n)]), n_partitions)

    time_start = time.time()

    vw = v.sqdist(w)
    print(vw)

    print(time.time() - time_start)

    sc.stop()

if __name__ == "__main__":
    main()
