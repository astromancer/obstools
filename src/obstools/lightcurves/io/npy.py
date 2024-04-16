import numpy as np
from loguru import logger


def write(filename, t, counts, std, mask=None):

    if np.ma.isMA(counts) or np.ma.isMA(std):
        mask = np.ma.getmaskarray(counts) | np.ma.getmaskarray(std)

    logger.info('Saving light curve data ({} rows, {} sources, {} masked '
                'points{}) to file: {}',
                len(t), len(counts), (0 if mask is None else mask.sum()),
                '', filename)

    # stack data
    data = stack_arrays(t, counts, std, mask)
    
    return np.save(filename, data)


def stack_arrays(t, flx, std, mask=None):
    """
    Stack light curve data into table for writing to file. Measurements for
    each star (Flux, σFlux, ...) columns are horizontally stacked.

    Parameters
    ----------
    t
    flx
    std
    mask

    Returns
    -------

    """
    nstars = len(flx)
    assert len(std) == nstars

    components = [flx, std]
    if mask is not None:
        assert len(mask) == nstars
        mask = mask.astype(int)
        components.append(mask)

    tbl = [t]
    for columns in zip(*components):
        tbl.extend(columns)

    # convert to array
    return np.array(tbl).T
