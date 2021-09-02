"""
Input / output helpers
"""

# std
import logging
import tempfile
from pathlib import Path

# third-party
import numpy as np

# local
from recipes.logging import logging, get_module_logger

# relative
from .utils import int2tup


from loguru import logger


def load_memmap(loc=None, shape=None, dtype=None, fill=None, clobber=False):
    """
    Pre-allocate a writeable shared memory map as a container for the
    results of parallel computation. If file already exists and clobber is False
    open in update mode and fill will be ignored. Data persistence ftw.
    """

    # NOTE: Objects created by this function have no synchronization primitives
    #  in place. Having concurrent workers write on overlapping shared memory
    #  data segments, for instance by using inplace operators and assignments on
    #  a numpy.memmap instance, can lead to data corruption as numpy does not
    #  offer atomic operations. We do not risk that issue if each process is
    #  updating an exclusive segment of the shared result array.

    if loc is None:
        fid, loc = tempfile.mkstemp('.npy')
        clobber = True  # fixme. messages below will be inaccurate

    loc = Path(loc)
    filename = str(loc)
    folder = loc.parent

    # create folder if needed
    if not folder.exists():
        logger.info('Creating folder: {!r:}', str(folder))
        folder.mkdir(parents=True)

    # update mode if existing file, else read
    new = not loc.exists()
    mode = 'w+' if (new or clobber) else 'r+'  # FIXME w+ r+ same??
    if dtype is None:
        dtype = 'f' if fill is None else type(fill)

    # create memmap
    shape = int2tup(shape)
    if new:
        logger.info('Creating memmap of shape {:s} and dtype {!r:} at {!r:}.',
                    shape, dtype, filename)
    else:
        logger.info('Loading memmap at {!r:}.', filename)

    # NOTE: using ` np.lib.format.open_memmap` here so that we get a small
    #  amount of header info for easily loading the array
    data = np.lib.format.open_memmap(filename, mode, dtype, shape)

    if data.shape != shape:
        logger.warning(f'Loaded memmap has shape {data.shape}, which is '
                       f'different to that requested: {shape}. Overwrite: '
                       f'{clobber}.')

    # overwrite data
    if (new or clobber) and (fill is not None):
        logger.debug('Overwriting data with %g.', fill)
        data[:] = fill

    return data


def load_memmap_nans(loc, shape=None, dtype=None, clobber=False):
    return load_memmap(loc, shape, dtype, fill=np.nan, clobber=clobber)
