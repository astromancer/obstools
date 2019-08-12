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
    """
    Get header from fits file.  Much quicker than pyfits.getheader for large
    files. Works with pathlib.Path objects.
    """
    with open(str(filename), 'rb') as fp:
        return pyfits.Header.fromfile(fp)


def fastheadkeys(filename, keys, defaults=()):
    header = quickheader(str(filename))
    if isinstance(keys, str):
        keys = keys,
    return [header.get(k, d) for (k, d) in itt.zip_longest(keys, defaults,
                                                           fillvalue=None)]


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
    #     from graphical.imagine import VideoDisplay
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
