from multiprocessing import Pool


def f(x):
    return x*x, 2, [x, x*x, x*x*x]

if __name__ == '__main__':
    with Pool(processes=5) as p:
        result = p.map_async(f, [1, 2, 3]).get()
        print(result)
