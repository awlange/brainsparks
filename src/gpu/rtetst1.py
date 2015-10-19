import numpy
from numpy.linalg import norm
import reikna.cluda as cluda
from reikna.linalg import MatrixMul

import reikna

import time


def main():
    api = cluda.ocl_api()
    # thr = api.Thread.create()
    thr = api.Thread.create({'exclude_devices': 'Iris Pro'})

    n = 6000
    m = 3000

    shape1 = (n, m)
    shape2 = (m, n)

    a = numpy.random.randn(*shape1).astype(numpy.float32)
    b = numpy.random.randn(*shape2).astype(numpy.float32)


    a_dev = thr.to_device(a)
    b_dev = thr.to_device(b)
    res_dev = thr.array((shape1[0], shape2[1]), dtype=numpy.float32)

    dot = MatrixMul(a_dev, b_dev, out_arr=res_dev)
    dotc = dot.compile(thr)

    gt = 0
    for i in range(10):
        thr.synchronize()
        gpu_start = time.time()
        dotc(res_dev, a_dev, b_dev)
        thr.synchronize()
        gt += time.time() - gpu_start
    print(gt)

    ct = 0
    res_reference = None
    for i in range(10):
        t = time.time()
        res_reference = numpy.dot(a, b)
        ct += time.time() - t
    print(ct)

    print(norm(res_dev.get() - res_reference) / norm(res_reference) < 1e-6)

if __name__ == "__main__":
    main()
