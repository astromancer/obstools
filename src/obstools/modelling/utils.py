import functools as ftl, operator as op


def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return ftl.reduce(op.mul, x)
