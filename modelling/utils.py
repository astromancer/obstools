import functools
import numbers
import operator

import logging
from pathlib import Path

import numpy as np

from recipes.string import get_module_name

# module level logger
logger = logging.getLogger(get_module_name(__file__, 2))


def prod(x):
    """Product of a list of numbers; ~40x faster vs np.prod for Python tuples"""
    if len(x) == 0:
        return 1
    return functools.reduce(operator.mul, x)


def assure_tuple(v):
    if isinstance(v, numbers.Integral):
        return v,
    if isinstance(v, tuple):
        return v
    else:
        raise ValueError('bad item %s of type %r' % (v, type(v)))


def make_shared_mem(loc, shape=None, dtype=None, fill=None, clobber=False):
    """
    Pre-allocate a writeable shared memory map as a container for the
    results of parallel computation. If file already exists and clobber is False
    open in update mode and fill will be ignored. Data persistence ftw.
    """

    # Note: Objects created by this function have no synchronization primitives
    # in place. Having concurrent workers write on overlapping shared memory
    # data segments, for instance by using inplace operators and assignments on
    # a numpy.memmap instance, can lead to data corruption as numpy does not
    # offer atomic operations. We do not risk that issue if each process is
    # updating an exclusive segment of the shared result array.

    # create folder if needed
    loc = Path(loc)
    filename = str(loc)
    folder = loc.parent
    if not folder.exists():
        logger.info('Creating folder: %s', str(folder))
        folder.mkdir(parents=True)

    new = not loc.exists()

    # update mode if existing file, else
    mode = 'w+' if new else 'r+'
    if dtype is None:
        dtype = 'f' if fill is None else type(fill)

    # create memmap
    if new:
        logger.info('Creating memmap of shape %s and dtype %s at %r',
                    shape, dtype, filename)
    else:
        logger.info('Loading memmap at %r', filename)

    mm = np.memmap(filename, dtype, mode, 0, shape)
    # mm = np.lib.format.open_memmap(str(loc), mode, dtype, shape)

    # overwrite data
    if (new or clobber) and fill:
        logger.info('Clobbering data with %g', fill)
        mm[:] = fill

    return mm


def make_shared_mem_nans(loc, shape=None, clobber=False):
    return make_shared_mem(loc, shape, fill=np.nan, clobber=clobber)


def cdist_tri(coo):
    """distance matrix with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    sdist = np.ma.masked_array(sdist)
    sdist[np.tril_indices(n)] = np.ma.masked
    return sdist
