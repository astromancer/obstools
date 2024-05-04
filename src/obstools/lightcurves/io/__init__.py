
# std
from enum import Enum
from pathlib import Path

# third-party
from loguru import logger

# local
from recipes import io

# relative
from . import npy, txt


# ---------------------------------------------------------------------------- #

class SupportedFileType(Enum):

    TXT = 'txt'
    NPY = 'npy'
    DAT = 'dat'
    # FITS = 'fits'
    # hd5

    @classmethod
    def _missing_(cls, ext):
        if isinstance(ext, Path):
            if member := getattr(cls, ext.suffix.strip('.').upper(), ()):
                return member

        raise ValueError(f'Unsupported format: {ext!r}')

    @classmethod
    def check(cls, file):
        if isinstance(file, str):
            return file.endswith(SUPPORTED)

        if isinstance(file, Path):
            return file.suffix.strip('.') in SUPPORTED

        raise TypeError(f'{type(file)}')


#
SUPPORTED = tuple(x.value for x in SupportedFileType)


# ---------------------------------------------------------------------------- #

def write(filename, t, counts, std, **kws):
    filename = Path(filename)
    writer = writers[SupportedFileType(filename).value]
    return writer(filename, t, counts, std, **kws)


def read(filename, hdu=None):

    filename = Path(filename)
    ext = SupportedFileType(filename).value

    if ext == 'txt':
        return txt.read(filename)

    if ext == 'npy':
        return load_memmap(filename, hdu)


def load_memmap(filename, hdu):
    logger.info('Loading data for {}.', hdu.file.name)

    # CONFIG.pre_subtract
    # since the (gain) calibrated frames are being used below,
    # CCDNoiseModel(hdu.readout.noise)

    flux = io.load_memmap(filename)['flux']
    return hdu.t.bjd, flux['value'].T, flux['sigma'].T


# IO workers
writers = {'txt': txt.write,
           'npy': npy.write,
           'dat': npy.write}

readers = {'txt': txt.read,
           'npy': load_memmap,
           'dat': load_memmap}
