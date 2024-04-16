
# std
from pathlib import Path

# third-party
from loguru import logger

# local
from recipes import io

# relative
from . import npy, txt


def write(filename, t, counts, std, **kws):
    filename = Path(filename)
    ext = filename.suffix.lower().strip('.')
    if ext == 'txt':
        return txt.write(filename, t, counts, std, **kws)
    
    if ext in {'npy', 'dat'}:
        return npy.write(filename, t, counts, std, **kws)
    
    raise NotImplementedError()


def read(filename, hdu=None):

    filename = Path(filename)
    ext = filename.suffix.strip('.')
    if ext == 'txt':
        return txt.read(filename)

    if ext == 'npy':
        return load_memmap(hdu, filename)

    raise ValueError(f'Unsupported format: {ext!r}')


def load_memmap(hdu, filename, outfile=None, **kws):

    logger.info('Loading data for {}.', hdu.file.name)

    # CONFIG.pre_subtract
    # since the (gain) calibrated frames are being used below,
    # CCDNoiseModel(hdu.readout.noise)

    data = io.load_memmap(filename)
    flux = data['flux']
    return hdu.t.bjd, flux['value'].T, flux['sigma'].T
