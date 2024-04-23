# std
import os
import re
import mmap
import itertools as itt

# third-party
import numpy as np
from astropy.io.fits import BITPIX2DTYPE, Header


# TODO: check if there is any performance diff in using the astropy fits regex
MATCH_END = re.compile(rb'END {77}\s*')


def fetch_frame(filename, frame):
    return FitsCube(filename)[frame]


def fetch_first_frame(filename):
    """Quick load first data frame from fits file"""
    return fetch_frame(filename, 0)


def fast_header(filename):
    """
    Get header from fits file.  Much quicker than pyfits.getheader for large
    files. Works with pathlib.Path objects.
    """
    with open(str(filename), 'rb') as fp:
        return Header.fromfile(fp)


def fast_header_keys(filename, keys, defaults=()):
    header = fast_header(str(filename))
    if isinstance(keys, str):
        keys = keys,
    return [header.get(k, d) for (k, d) in itt.zip_longest(keys, defaults,
                                                           fillvalue=None)]


def fast_sample(filename, subset, statistic, axis=0):
    ff = FitsPartial(filename, *subset)
    return statistic(ff[:], axis)


def parse_header(filename):
    """
    Extract primary header from fits file. return as str
    """

    # Have to scan for END of header - use memmap for efficiency
    with open(filename, 'rb') as fileobj:
        # create memmap
        mm = mmap.mmap(fileobj.fileno(),
                       os.path.getsize(filename),
                       access=mmap.ACCESS_READ)

    # Find the index position of the first extension and data start
    mo = MATCH_END.search(mm)
    if mo is None:
        raise TypeError('Could not find header END card. Is this a valid '
                        '.fits file? %r' % filename)

    # header is before data (no extensions!)
    data_start_bytes = mo.end()  # data starts here
    header = mm[:data_start_bytes].decode()

    return header, data_start_bytes, mm


# from motley.profiling import profile

class FitsPartial:
    """
    Fast reading subset of fits data.  Good for sampling, good for your
    mental health
    """

    def __init__(self, filename, start, stop):
        """
        View image frames of 3D fits cube on demand by indexing.
        """

        filename = str(filename)  # for converting Path objects
        hdr, data_start_bytes, mm = parse_header(filename)
        self.header = hdr = Header.fromstring(hdr)

        # check if data is 3D
        n_dim = hdr['NAXIS']
        if n_dim not in (2, 3):
            raise TypeError('%r only accepts 2D or 3D data!'
                            % self.__class__.__name__)

        # figure out the size of a data block
        bits_per_pixel = hdr['BITPIX']
        dtype = np.dtype(BITPIX2DTYPE[bits_per_pixel]).newbyteorder('>')
        n_frames = hdr.get('NAXIS3', 1)
        stop = min(stop, n_frames)
        assert 0 <= start < stop
        n = stop - start
        shape = (n, hdr['NAXIS2'], hdr['NAXIS1'])
        # NOTE: the order of axes on an numpy array are opposite of the order
        #  specified in the FITS file.
        image_size_bytes = abs(bits_per_pixel) * np.product(shape[1:]) // 8
        offset = data_start_bytes + image_size_bytes * start

        self.bzero = hdr.get('BZERO', 0)
        # TODO: bscale!!
        self.data = np.memmap(filename, dtype, 'r', offset, shape)

    def __getitem__(self, key):
        # NOTE: adding a float here converts from np.memmap to np.array
        return self.data[key] + self.bzero

    def __len__(self):
        return len(self.data)


class FitsCube:
    """
    A more efficient way of reading large fits files.  This class provides
    instant access to the frames without the need to load the entire data
    cube into RAM. Works well for multi-gigabyte files, that tend to take
    half the age of the universe to open with pyfits.open even when memmap=True.

    Write access currently not supported.

    The *data* attribute is a memory map to the fits image data which can
    be shared between multiple processes.  This class therefore also offers
    several advantaged over pyfits hdu objects i.t.o. parallelization.
    """

    # TODO: docstring example, show memory profile, speed test!!

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

    # @profile(report='bars')
    def __init__(self, filename):
        """
        View image frames of 3D fits cube on demand by indexing.
        """

        filename = str(filename)  # for converting Path objects
        hdr, data_start_bytes, mm = parse_header(filename)
        # can you make this even faster by skipping search for end card just
        # look for NAXIS\d, BITPIX, BZERO, BSCALE
        hdr = Header.fromstring(hdr)

        # check if data is 3D
        n_dim = hdr['NAXIS']
        if n_dim not in (2, 3):
            raise TypeError('%r only accepts 2D or 3D data!'
                            % self.__class__.__name__)

        # figure out the size of a data block
        bits_per_pixel = hdr['BITPIX']
        dtype = np.dtype(BITPIX2DTYPE[bits_per_pixel]).newbyteorder('>')
        shape = (hdr.get('NAXIS3', 1), hdr['NAXIS2'], hdr['NAXIS1'])
        # NOTE: the order of axes on an numpy array are opposite of the order
        #  specified in the FITS file.

        # self.image_start_bytes = abs(bits_per_pixel) * nax1 * nax2 // 8
        self.bzero = hdr.get('BZERO', 0)
        self.data = np.memmap(filename, dtype, 'r', data_start_bytes, shape)

        # self.shape = shape
        # self.ishape = nax1, nax2 = shape[1:]
        # self.ndim = len(shape)

    def __getitem__(self, key):
        # NOTE: adding a float here converts from np.memmap to np.array
        return self.data[key] + self.bzero

    def __len__(self):
        return len(self.data)

    # def __getstate__(self):
    #     # capture what is normally pickled
    #     state = self.__dict__.copy()
    #     state.pop('filemap')
    #     return state
    #
    # def __setstate__(self, state):
    #     # re-instate our __dict__ state from the pickled state
    #     self.__dict__.update(state)
    #
    #     filename = str(self.filename)
    #
    #     with open(filename, 'rb') as fileobj:
    #         self.filemap = mmap.mmap(fileobj.fileno(),
    #                                  os.path.getsize(filename),
    #                                  access=mmap.ACCESS_READ)

    # def display(self, *args, **kws):
    #     # NOTE: cannot pickle!!
    #     from scrawl.imagine import VideoDisplay
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
