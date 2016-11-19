from pyspark import SparkContext, SparkConf


def main():
    sc = SparkContext(appName="test")

    data = [1, 2, 3, 4]
    lines = sc.parallelize(data)
    lineNum = lines.map(lambda s: int(s))
    total = lineNum.reduce(lambda a, b: a + b)

    print(total)

    sc.stop()

if __name__ == "__main__":
    main()
