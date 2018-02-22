import os
import re
import mmap
import warnings
import functools
import itertools as itt
import collections as coll
from io import BytesIO
# from pathlib import Path


# import scipy as sp
import numpy as np
from astropy.io import fits as pyfits
# import pylab as plt
# from matplotlib.transforms import  blended_transform_factory as btf

from recipes.io import parse
# from decor import path      #profile
from motley.progress import ProgressBar


# from IPython import embed


def fetchframe(filename, frame):
    return FitsCube(filename)[frame]


def fetch_first_frame(filename):
    """Quick load first data frame from fits file"""
    return fetchframe(filename, 0)


def fastheader(filename):
    """Get header from fits file.  Much quicker than pyfits.getheader for large files.
    Works with pathlib.Path objects."""
    with open(str(filename), 'rb') as fp:
        return pyfits.Header.fromfile(fp)


def fastheadkeys(filename, keys, defaults=()):
    header = quickheader(str(filename))
    if isinstance(keys, str):
        keys = keys,
    return [header.get(k, d) for (k, d) in itt.zip_longest(keys, defaults, fillvalue=None)]


quickheader = fastheader
quickheadkeys = fastheadkeys


class FitsCube(object):
    """
    A more efficient way of reading large fits files.  This class provides instant
    access to the frames without the need to load the entire data cube into RAM.
    Works well for multi-gigabyte files, that tend to take half the age of the
    universe to open with pyfits.open.

    The *data* attribute is a memory map to the fits image data which can
    be shared between multiple processes.  This class therefore also offers
    several advantaged over pyfits hdu objects i.t.o. parallelization.
    """
    # TODO: docstring example
    # Example
    # -------
    # import os, time
    # filename = 'myfile.fits'
    # sizeMb = os.stat(filename) / 2 ** 20  # size in Mb
    # print(sizeMb)  # a very large fits file
    # t0 = time.time()
    # ff = FitsCube(filename)
    # data = ff[0]
    # print(time.time() - t0)

    match_end = re.compile(b'END {77}\s*')

    def __init__(self, filename):
        """
        View image frames of 3D fits cube on demand by indexing.
        """
        filename = str(filename)  # for converting Path objects
        filesize = os.path.getsize(filename)

        # Have to scan for END of header - use filemap for efficiency
        with open(filename, 'rb') as fileobj:
            mm = mmap.mmap(fileobj.fileno(),
                           filesize,
                           access=mmap.ACCESS_READ)

        self.filename = filename

        # Find the index position of the first extension and data start
        mo = self.match_end.search(mm)
        if mo is None:
            raise TypeError('Could not find header END card. Is this a valid '
                            '.fits file?')
        self.data_start_bytes = dstart = mo.end()  # data starts here

        # header is before dstart
        header_str = mm[:dstart].decode()
        self.header = hdr = pyfits.Header.fromstring(header_str)

        # check if data is 3D
        nax = hdr['naxis']
        if nax not in (2, 3):
            raise TypeError('%s only works with 2D or 3D data!'
                            % self.__class__.__name__)

        # figure out the size of a data block
        shape = (hdr.get('naxis3', 1), hdr['naxis2'], hdr['naxis1'])
        self.ishape = nax1, nax2 = shape[1:]
        # NOTE: the order of axes on an numpy array are opposite of the order
        # specified in the FITS file.

        bitpix = hdr['bitpix']
        if bitpix == 16:
            dtype = '>i2'  # .format(bitpix//8)
        elif bitpix == -32:
            dtype = '>f4'
        else:
            dtype = '>f8'

        self.image_start_bytes = abs(bitpix) * nax1 * nax2 // 8
        self.bzero = hdr.get('bzero', 0)

        self.data = np.memmap(filename, dtype, 'r', dstart, shape)
        self.shape = shape
        # self.ndim = len(shape)

    def __getitem__(self, key):
        # NOTE: adding a float here converts from np.memmap to np.array
        return self.data[key] + self.bzero

    def __len__(self):
        return len(self.data)

    # @property
    # def shape(self):
    #     return self.data.shape

    # @property
    # def master_header(self):
    #     from recipes.io.tracewarn import warning_traceback_on, warning_traceback_off
    #     warning_traceback_on()
    #     warnings.warning('master_header deprecated')
    #     warning_traceback_off()
    #     return self.header

    # def __len__(self):
    #     return self.shape[-1]
    #
    # def __getitem__(self, i):
    #     """enable array-like indexing tricks"""
    #     # TODO: handle tuple
    #     if isinstance(i, (int, np.integer)):
    #         if i < 0:
    #             i += len(self)
    #         return self._get_from_slice(slice(i, i + 1)).squeeze()
    #
    #     elif isinstance(i, (list, np.ndarray)):
    #         return self._get_from_list(i)
    #
    #     elif isinstance(i, slice):
    #         return self._get_from_slice(i)
    #
    #     else:
    #         raise IndexError(
    #             'Only integers, and (continuous) slices are valid indices. '
    #             '(For now). Received: {}, {}'.format(type(i), i))
    #
    # def _get_from_slice(self, slice_):
    #     """Retrieve frames from slice of data along 3rd axis"""
    #     isize = self.image_start_bytes
    #     istart = slice_.start or 0
    #     istop = slice_.stop or len(self)
    #     istep = slice_.step or 1
    #     ispan = istop - istart
    #     shape = ((istop - istart) // istep,) + self.ishape
    #
    #     if istop > len(self):
    #         raise IndexError('index {} is out of bounds for axis 2 with size {}'
    #                          ''.format(slice_, len(self)))
    #
    #     if slice_.step:
    #         _buffer = BytesIO()
    #         iframes = range(len(self))[slice_]  # extract frame numbers as sequence of integers
    #         for j in iframes:
    #             _start = self.data_start_bytes + j * isize
    #             _buffer.write(self.filemap[_start:(_start + isize)])
    #         _buffer.seek(0)
    #         return self.bzero + np.ndarray(shape,
    #                                        dtype=self.dtype,
    #                                        buffer=_buffer.read()
    #                                        ).astype(float)
    #
    #     start = self.data_start_bytes + isize * istart
    #     end = start + isize * ispan
    #
    #     return self.bzero + np.ndarray(shape,
    #                                    dtype=self.dtype,
    #                                    buffer=self.filemap[start:end]
    #                                    ).astype(float)
    #
    # def _get_from_list(self, list_or_array):
    #     """Retrieve frames indicated by indices in list or array"""
    #     isize = self.image_start_bytes
    #     shape = (len(list_or_array),) + self.ishape
    #     _buffer = BytesIO()
    #     for j in list_or_array:
    #         _start = self.data_start_bytes + j * isize
    #         _buffer.write(self.filemap[_start:(_start + isize)])
    #     _buffer.seek(0)
    #     return self.bzero + np.ndarray(shape,
    #                                    dtype=self.dtype,
    #                                    buffer=_buffer.read()
    #                                    ).astype(float)
    #
    # def __iter__(self):
    #     for i in range(len(self)):
    #         yield self[i]

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()
        state.pop('filemap')
        return state

    def __setstate__(self, state):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(state)

        filename = str(self.filename)
        filesize = os.path.getsize(filename)

        with open(filename, 'rb') as fileobj:
            self.filemap = mmap.mmap(fileobj.fileno(),
                                     filesize,
                                     access=mmap.ACCESS_READ)

    # def display(self, *args, **kws):
    #     # NOTE: cannot pickle!!
    #     from grafico.imagine import VideoDisplay
    #     return VideoDisplay(self, *args, **kws)
    #


if __name__ == '__main__':
    import pickle

    pickle.loads()

    # from time import time
    #
    # saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
    # filelist = parse.to_list(saltpath + '/bxgp*.fits')
    #
    # t0 = time()
    # # print( len(filelist) )
    # keys = 'utc-obs', 'date-obs'
    # q = superheadhunter(filelist[:100], keys)
    # # q = headhunter( filelist[0], ('date-obs', 'utc-obs', 'deadtime') )
    #
    # print('Took', time() - t0, 's')
    # print()
    ##print( q )
    # for k,v in q.items():
    # print( k, len(v) )
    # ipshell()
