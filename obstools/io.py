"""
Input / output helpers
"""

import logging
from pathlib import Path

# third-party libs
import numpy as np

# local libs
from recipes.logging import get_module_logger

from .utils import int2tup

# module level logger
logger = get_module_logger()


def load_memmap(loc, shape=None, dtype=None, fill=None, clobber=False):
    """
    Pre-allocate a writeable shared memory map as a container for the
    results of parallel computation. If file already exists and clobber is False
    open in update mode and fill will be ignored. Data persistence ftw.
    """

    # Note: Objects created by this function have no synchronization primitives
    #  in place. Having concurrent workers write on overlapping shared memory
    #  data segments, for instance by using inplace operators and assignments on
    #  a numpy.memmap instance, can lead to data corruption as numpy does not
    #  offer atomic operations. We do not risk that issue if each process is
    #  updating an exclusive segment of the shared result array.

    # create folder if needed
    loc = Path(loc)
    filename = str(loc)
    folder = loc.parent
    if not folder.exists():
        logger.info('Creating folder: %s', str(folder))
        folder.mkdir(parents=True)

    new = not loc.exists()

    # update mode if existing file, else
    mode = 'w+' if (new or clobber) else 'r+'
    if dtype is None:
        dtype = 'f' if fill is None else type(fill)

    # create memmap
    shape = int2tup(shape)
    if new:
        logger.info('Creating memmap of shape %s and dtype %r at %r',
                    shape, dtype, filename)
    else:
        logger.info('Loading memmap at %r', filename)

    # mm = np.memmap(filename, dtype, mode, 0, shape)
    # NOTE: using ` np.lib.format.open_memmap` here so that we get a small
    #  amount of header info for easily loading the array
    mm = np.lib.format.open_memmap(str(loc), mode, dtype, shape)

    if mm.shape != shape:
        logger.warning(f'Loaded memmap has shape {mm.shape}, which is '
                       f'different to that requested: {shape}. Overwrite: '
                       f'{clobber}')

    # overwrite data
    if (new or clobber) and (fill is not None):
        logger.debug('Over-writing data with %g', fill)
        mm[:] = fill

    return mm


def load_memmap_nans(loc, shape=None, dtype=None, clobber=False):
    return load_memmap(loc, shape, dtype, fill=np.nan, clobber=clobber)
