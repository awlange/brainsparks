from pyspark import SparkContext

from src.sparkles.linalg import *



def main():
    sc = SparkContext(appName="matrix")

    n = 16

    v = Vector(sc, [x for x in range(n)])
    # w = Vector(sc, [2*x for x in range(n)])
    # vw = v.v_add_w(w)
    # print(vw.collect())

    m = Matrix(sc, [[x for x in range(n)] for _ in range(n)])
    # print(sc.broadcast(v.rdd.collect()).value)
    mv = m.m_dot_v(v)

    print(mv.rdd.collect())


    sc.stop()

if __name__ == "__main__":
    main()
