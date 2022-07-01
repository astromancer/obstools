import functools as ftl, operator as op


def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    return 1 if len(x) == 0 else ftl.reduce(op.mul, x)
