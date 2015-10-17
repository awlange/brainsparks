"""
Script entry point
"""

from src.sandbox.network import Network
from src.sandbox.dense import Dense

import src.sandbox.linalg as linalg
import numpy as np

import time


def main():
    n = 6000
    v = [x for x in range(n)]
    m = [[x for x in range(n)] for _ in range(n)]

    time_start = time.time()

    for _ in range(3):
        linalg.mdotv(m, v)

    print(time.time() - time_start)


def main2():
    n = 8000
    v = np.asarray([x for x in range(n)])
    m = np.asarray([[x for x in range(n)] for _ in range(n)])

    time_start = time.time()

    z = None
    for _ in range(3):
        z = m.dot(v)
        print(z.sum())


    print(time.time() - time_start)


if __name__ == "__main__":
    # main()
    main2()
