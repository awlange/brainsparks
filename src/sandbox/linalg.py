"""
Basic dumb linear algebra
"""


def vaddw(v, w):
    """
    v + w
    """
    return [v[i] + w[i] for i in range(len(v))]


def vminw(v, w):
    """
    v - w
    """
    return [v[i] - w[i] for i in range(len(v))]


def vtimesw(v, w):
    """
    v * w (element-wise)
    """
    return [v[i] * w[i] for i in range(len(v))]


def vsqdistw(v, w):
    """
    |v - w|^2
    """
    return sum([(v[i] - w[i])**2 for i in range(len(v))])


def vdotw(v, w):
    """
    Vector-vector dot product
    """
    return sum([v[i] * w[i] for i in range(len(v))])


def mdotv(M, v):
    """
    M.v
    Matrix-vector product
    """
    return [vdotw(row, v) for row in M]


def mdotvpw(M, v, w):
    """
    M.v + w
    """
    return vaddw(mdotv(M, v), w)


def transpose(M):
    n_rows = len(M)
    n_cols = len(M[0])
    Mt = [[0.0 for _ in range(n_rows)] for __ in range(n_cols)]
    for i in range(n_rows):
        for j in range(n_cols):
            Mt[j][i] = M[i][j]
    return Mt


def outer(v, w):
    result = []
    for vi in v:
        x = []
        for wi in w:
            x.append(vi * wi)
        result.append(x)
    return result
