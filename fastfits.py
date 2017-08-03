import os
import re
import mmap
import warnings
import functools
import itertools as itt
import collections as coll
from io import BytesIO
# from pathlib import Path
import multiprocessing as mp
import multiprocessing.pool

# import scipy as sp
import numpy as np
from astropy.io import fits as pyfits
# import pylab as plt
# from matplotlib.transforms import  blended_transform_factory as btf

from recipes.dict import DefaultOrderedDict
from recipes.io import parse
# from decor import path      #profile
from ansi.progress import ProgressBar


# from IPython import embed

# ====================================================================================================
def fastheader(filename):
    """Get header from fits file.  Much quicker than pyfits.getheader for large files.
    Works with pathlib.Path objects."""
    with open(str(filename), 'rb') as fp:
        return pyfits.Header.fromfile(fp)


quickheader = fastheader


def fastheadkeys(filename, keys, defaults=()):
    header = quickheader(str(filename))
    if isinstance(keys, str):
        keys = keys,
    return [header.get(k, d) for (k, d) in itt.zip_longest(keys, defaults, fillvalue=None)]


quickheadkeys = fastheadkeys


# ====================================================================================================
class FitsCube():
    """
    A more efficient way of reading large fits files.  This class provides instant
    access to the frames without the need to load the entire data cube into memory
    Works well for multi-gig files, that tend to take half the age of the universe
    to open with pyfits.open
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
        with open(filename, 'rb') as fileobj:
            self.filemap = filemap = mmap.mmap(fileobj.fileno(),
                                               filesize,
                                               access=mmap.ACCESS_READ)

        # Find the index position of the first extension and data start
        mo = self.match_end.search(filemap)
        self.data_start_bits = dstart = mo.end()  # data starts here

        # master header is before estart
        header_str = filemap[:dstart].decode()
        self.header = header = pyfits.Header.fromstring(header_str)

        # check if data is 3D
        nax = header['naxis']
        if nax not in (2, 3):
            raise TypeError('{} only works with 2D or 3D data!'.format(self.__class__.__name__))

        # figure out the size of a data block
        # NOTE: the order of axes on an Numpy array are opposite of the order specified in the FITS file.
        self.shape = header['naxis1'], header['naxis2'], header.get('naxis3', 1)
        self.ishape = nax1, nax2 = self.shape[1::-1]

        bitpix = header['bitpix']
        if bitpix == 16:
            self.dtype = '>i2'  # .format(bitpix//8)
        elif bitpix == -32:
            self.dtype = '>f4'
        else:
            self.dtype = '>f8'

        self.image_size_bits = abs(bitpix) * nax1 * nax2 // 8
        self.bzero = header.get('bzero', 0)

    @property
    def master_header(self):
        from recipes.io.tracewarn import warning_traceback_on, warning_traceback_off
        warning_traceback_on()
        warnings.warning('master_header depricated')
        warning_traceback_off()
        return self.header

    def __len__(self):
        return self.shape[-1]

    def __getitem__(self, i):
        """enable array-like indexing tricks"""
        # TODO: handle tuple
        if isinstance(i, (int, np.integer)):
            if i < 0:
                i += len(self)
            return self._get_from_slice(slice(i, i + 1)).squeeze()

        elif isinstance(i, (list, np.ndarray)):
            return self._get_from_list(i)

        elif isinstance(i, slice):
            return self._get_from_slice(i)

        else:
            raise IndexError(
                'Only integers, and (continuous) slices are valid indices. '
                '(For now). Received: {}, {}'.format(type(i), i))

    def _get_from_slice(self, slice_):
        """Retrieve frames from slice of data along 3rd axis"""
        isize = self.image_size_bits
        istart = slice_.start or 0
        istop = slice_.stop or len(self)
        istep = slice_.step or 1
        ispan = istop - istart
        shape = ((istop - istart) // istep,) + self.ishape

        if istop > len(self):
            raise IndexError('index {} is out of bounds for axis 2 with size {}'
                             ''.format(slice_, len(self)))

        if slice_.step:
            _buffer = BytesIO()
            iframes = range(len(self))[slice_]  # extract frame numbers as sequence of integers
            for j in iframes:
                _start = self.data_start_bits + j * isize
                _buffer.write(self.filemap[_start:(_start + isize)])
            _buffer.seek(0)
            return self.bzero + np.ndarray(shape,
                                           dtype=self.dtype,
                                           buffer=_buffer.read()
                                           ).astype(float)

        start = self.data_start_bits + isize * istart
        end = start + isize * ispan

        return self.bzero + np.ndarray(shape,
                                       dtype=self.dtype,
                                       buffer=self.filemap[start:end]
                                       ).astype(float)

    def _get_from_list(self, list_or_array):
        """Retrieve frames indicated by indices in list or array"""
        isize = self.image_size_bits
        shape = (len(list_or_array),) + self.ishape
        _buffer = BytesIO()
        for j in list_or_array:
            _start = self.data_start_bits + j * isize
            _buffer.write(self.filemap[_start:(_start + isize)])
        _buffer.seek(0)
        return self.bzero + np.ndarray(shape,
                                       dtype=self.dtype,
                                       buffer=_buffer.read()
                                       ).astype(float)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ====================================================================================================
def fetchframe(filename, frame):
    return FitsCube(filename)[frame]


def fetch_first_frame(filename):
    """Quick load first data frame from fits file"""
    return fetchframe(filename, 0)

    # fileobj = open(filename, 'rb')
    # filesize = os.path.getsize(filename)
    # filemap = mmap.mmap(
    # fileobj.fileno(), filesize, access=mmap.ACCESS_READ
    # )
    # fileobj.close()

    # mo = re.search( b'END {77}\s*', filemap )
    # dstart = mo.end()

    # header_str = filemap[:dstart].decode()
    # header = pyfits.Header.fromstring( header_str )
    # nax1, nax2 = header['naxis1'], header['naxis2']
    # bitpix = header['bitpix']

    # dsize = abs(bitpix) * nax1 * nax2 // 8
    # dend = dstart + dsize

    # shape = (nax2, nax1)
    # if bitpix==16:
    # dtype='>i2'#.format( bitpix//8 )
    # else:
    # dtype='>f4'
    # bzero = header.get('bzero', 0)

    # return bzero + np.ndarray( shape,
    # dtype=dtype,
    # buffer=filemap[dstart:dend],
    # ).astype(float)

    # return np.fromstring( filemap[dstart:dend], '>f32').reshape( nax1, nax2 )


# ****************************************************************************************************
class Extractor(object):
    def __init__(self, filelist, **kw):
        # WARNING:  Turns out this sequential extraction thing is SUPER slow!!
        # TODO: Checkout ipython-progress bar??????????
        """
        Extract frames from list of fits files.
        """
        self.start = start = kw.setdefault('start', 0)
        self.step = step = kw.setdefault('step', 1)
        self.clobber = kw.setdefault('clobber', True)

        self.keygrab = kw.setdefault('keygrab', None)
        self.headkeys = []

        self.filelist = parse.to_list(filelist)
        self.outfiles = []

        # map from integer extension number to (header, data) element
        self.data_element = lambda hdulist, i: (
            hdulist[i + 1].header, hdulist[i + 1].data)  # return header and data for extension i

        # read firts file
        first = pyfits.open(filelist[0], memmap=True)
        pheader = first[0].header

        # Assume all files have same number of extensions and calculate total number of frames.
        Next = pheader.get('nextend')
        self.Nfiles = Nfiles = len(filelist)
        self.Ntot = Ntot = (Nfiles * Next - start) // step
        self.stop = stop = kw.setdefault('stop', Ntot)

        self.padwidth = len(str(Ntot))  # for numerical file naming
        self.count = 0

        self.bar = kw.pop('progressbar', ProgressBar())  # initialise progress bar unless False

        # create master output file (HduList with header of first file)
        primary = pyfits.PrimaryHDU(header=pheader)
        self.master = pyfits.HDUList(primary)  # Use this as the master header for the output file
        self.mheader = self.master[0].header

        # TODO:update master header

    def loop(self, func):
        """
        loop through the files and extract selected frames.
        """
        start, stop, step = self.start, self.stop, self.step
        if self.bar:    self.bar.create(stop)
        for i, f in enumerate(self.filelist):
            start = start if i == 0 else start % step  # first file start at start, thereafter continue sequence of steps
            with pyfits.open(f, memmap=True) as hdulist:
                end = len(hdulist) - 1  # end point for each multi-extension cube

                datamap = map(functools.partial(self.data_element, hdulist),
                              range(start, end, step))

                for j, (header, data) in enumerate(datamap):
                    # exit clause
                    if self.count >= stop:
                        return

                    func(data, header)

                    # grab header keys
                    if not self.keygrab is None:
                        self.headkeys.append([header.get(k) for k in self.keygrab])

                    # show progress
                    if self.bar:
                        self.bar.progress(self.count)

                    self.count += 1

    def _burst(self, data, header):
        numstr = '{1:0>{0}}'.format(self.padwidth, self.count)
        fn = self.naming.format(numstr)
        self.outfiles.append(fn)
        pyfits.writeto(fn, data, header)

    def _multiext(self, data, header):
        self.master.append(pyfits.ImageHDU(data, header))

    def _cube(self, data, header):
        hdu = self.master[0]
        if hdu.data is None:
            hdu.data = data
        else:
            hdu.data = np.r_['0,3', hdu.data, data]  # stack data along 0th axis
            # pyfits.HeaderDiff

    def burst(self, naming='sci{}.fits'):
        """Save files individaully.  This is probably quite inefficient (?)"""
        self.naming = naming
        self.loop(self._burst)

        return self.outfiles

    def multiext(self):
        """Save file as one big multi-extension FITS file."""
        master = self.master
        header = self.mheader

        # work
        self.loop(self._multiext)

        # update header info
        header['nextend'] = header['NSCIEXT'] = len(master)
        # optional to update NCCDS, NSCIEXT here

        # verify FITS compliance
        master.verify()

        return master

    def cube(self):
        """Save file as one big 3D FITS data cube."""
        master = self.master
        header = self.mheader

        # work
        self.loop(self._cube)

        # update header info
        header.remove('nextend')
        header.remove('NSCIEXT')

        # verify FITS complience
        master.verify()

        return master


        # @path.to_string.auto
        # def write(self, outfilename):
        #     # optional to update NCCDS, NSCIEXT here
        #
        #     master.writeto(outfilename, clobber=self.clobber)
        #     master.close()


# ****************************************************************************************************
class SlotExtractor(object):  # TODO: find a new home for this class

    BLOCK = 2880
    match_end = re.compile(b'END {77}\s*')

    def __init__(self, filelist, **kw):
        # TODO: MULTIEXT, BURST METHODS
        """
        Extract frames from list of fits files.  Assuming all the files in the input list are
        fits complient and have the same basic structure, data extraction can be done ~10**6
        times faster than it can with pyfits (even without multiprocessing!!) by skipping unnecessary
        checks and other cuft.
        """
        self.start = kw.setdefault('start', 0)
        self.step = kw.setdefault('step', 1)
        self.stop = kw.setdefault('stop', None)

        keys = kw.setdefault('keygrab', None)
        self.headkeys = []

        self.filelist = parse.to_list(filelist)
        self.Nfiles = len(filelist)
        self.clobber = kw.setdefault('clobber', True)
        # self.outfiles = []

        self.data_buffer = BytesIO()
        self.text_buffer = []

        # keyword extraction setup
        if isinstance(keys, (str, type(None))):
            keys = keys,

        self.match_keys = matchmaker(*keys)
        self.keygrab = keys

        # initialise progress bar if needed
        self.bar = kw.pop('progressbar', ProgressBar())

        # memory buffer to reconstruct which frame came from which file
        self.filemem = []
        self.setup(self.filelist[0])

        # self.loop()

    def setup(self, filename):
        """extract necessary info from primary header and first extension header."""
        fileobj = open(filename, 'rb')
        filesize = os.path.getsize(filename)
        filemap = mmap.mmap(
            fileobj.fileno(), filesize, access=mmap.ACCESS_READ
        )
        fileobj.close()

        # Find the index position of the first extension and data start
        mo = self.match_end.search(filemap)
        estart = mo.end()  # extension 1 starts here
        # NOTE: There any not be extensions
        mo = self.match_end.search(filemap, estart)
        self.image_data_start = dstart = mo.end()  # data for extension 1 starts here

        # master header is before estart
        mheader_str = filemap[:estart].decode()
        self.header = mheader = pyfits.Header.fromstring(mheader_str)

        # header of first extension is between *estart* and *dstart*
        self.header_size = hsize = dstart - estart  # header size
        header_str = filemap[estart:dstart].decode()
        header = pyfits.Header.fromstring(header_str)

        # figure out the size of a data block
        self.ishape = nax2, nax1 = header['naxis2'], header['naxis1']
        # NOTE: the order of axes on an Numpy array are opposite of the order specified in the FITS file.
        bitpix = header['bitpix']  # NOTE: might be some fuckup value like -64
        if bitpix == 16:
            self.dtype = '>i2'  # .format(bitpix//8)
        if bitpix == 8:
            self.dtype = '>f8'
        else:
            self.dtype = '>f8'  # '>f4'  # data is big-endian??

        # NOTE: assume fits complience -- i.e. file is divided in chuncks of 2880 bytes
        BLOCK = self.BLOCK
        self.image_size_bits = dsize = abs(bitpix) * nax1 * nax2 // 8
        self.image_data_step = int(np.ceil((dsize + hsize) / BLOCK) * BLOCK)
        self.nextend = Next = mheader['nextend']  # number of fits extentions

        # expected number of frames to extract (assuming all have identical structure!)
        self.Ntot = (self.Nfiles * Next - self.start) // self.step  # FIXME too small by 1???

        # data value offset
        self.bzero = header.get('bzero', 0)

        self.h = []

    def loop(self):
        """Loop over the files and extract the data and optional header keywords"""
        start, stop, step = self.start, self.stop, self.step
        dstart, dstep, dsize = self.image_data_start, self.image_data_step, self.image_size_bits
        hsize = self.header_size
        self.count = 0

        if self.bar:
            self.bar.create(stop or len(self.filelist))  # FIXME: prints empty bar for size 1 loops

        # TODO: MULTIPROCESS!!
        for i, filename in enumerate(self.filelist):
            fileobj = open(filename, 'rb')
            filesize = os.path.getsize(filename)
            filemap = mmap.mmap(
                fileobj.fileno(), filesize, access=mmap.ACCESS_READ
            )
            start = start if i == 0 else start % step
            stop = filesize // dstep  # number of frames to extract from this file
            indices = dstart + dstep * np.arange(start, stop, step)
            for ix in indices:
                self.data_buffer.write(filemap[ix:ix + dsize])
                # grab header keys
                # FIXME: more efficient way of doing this
                head = filemap[ix - hsize:ix]
                # h = pyfits.header.Header.fromstring(head)
                self.h.append(head)
                raw_text = self.match_keys.findall(head)
                self.text_buffer.extend(raw_text)

            self.count += 1
            self.bar.progress(self.count)

            self.filemem.append((filename, i + 1))

            fileobj.close()
        self.bar.close()

    # TODO: MAKE PROPERTY??
    def get_data(self):
        """get the data from the buffer"""
        raw_data = np.frombuffer(self.data_buffer.getbuffer(), self.dtype)
        return raw_data.reshape(-1, *self.ishape)

    def get_keys(self, defaults=[], return_type='list'):
        return merger(self.text_buffer, self.keygrab, defaults, return_type)

    def multiext(self):
        raise NotImplementedError

    def cube(self):
        """Retrieve data as 3D FITS Cube"""
        # Run the main loop to write data into the buffer
        self.loop()

        # update header info
        header = self.header.copy()
        header.remove('nextend')
        header.remove('NSCIEXT')

        # NOTE: the order of axes on an Numpy array are opposite of that in the FITS file.
        # e.g. for a 2D image the "rows" (y-axis) are the first dimension, and the "columns"
        # (x-axis) are the second dimension.
        hdu = pyfits.PrimaryHDU(self.get_data(), header)
        return hdu

    def burst(self):
        raise NotImplementedError


# ****************************************************************************************************
class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False

    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# ****************************************************************************************************
# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class Pool(mp.pool.Pool):
    Process = NoDaemonProcess


# ====================================================================================================
CARD_LENGTH = 80
VALUE_INDICATOR = '= '  # The standard FITS value indicator
COMMENT_INDICATOR = '/'

end_str = ('END' + ' ' * 77).encode()
# FIXME:  Pattern below won't work when there are no comments: eg: NAXIS!!!!
# NOTE: USE PYFITS CARD PARSER??
card_parse_pattern = '(.*)%s(.*)%s(.*)' % (VALUE_INDICATOR, COMMENT_INDICATOR)
card_parser = re.compile(card_parse_pattern.encode())
cleaner = lambda b: b.strip(b"' ").decode()


# ====================================================================================================
def parse_and_clean(s):
    parts = card_parser.match(s).groups()
    key, val, comment = map(cleaner, parts)
    return key, val, comment


def parse_and_clean_no_comment(s):
    return parse_and_clean(s)[:2]


# ====================================================================================================
def init(filename, _matcher):
    """
    Function used to initialize the multiprocessing pool with shared filemap and
    regex matcher.
    """
    global filemap, matcher
    matcher = _matcher
    with open(filename, "rb") as fileobj:
        filemap = mmap.mmap(
            fileobj.fileno(), os.path.getsize(filename), access=mmap.ACCESS_READ
        )


# ====================================================================================================
def getchunks(filename, size=1024 * 1024):
    """Yield sequence of (start, size) chunk descriptors."""
    with open(filename, "rb") as f:
        while 1:
            start = f.tell()
            f.seek(size, 1)
            s = f.readline()  # skip forward to next line ending
            yield start, f.tell()
            if not s:
                break


# ====================================================================================================
def matchmaker(*keys):
    """Dynamically generate pre-compiled re matcher for finding keywords in FITS header.
    Parameters
    ----------
    keys:       sequence of keys
    """

    if keys == (None,):
        # return a null matcher that mimics a compiled regex, but always returns empty list
        # useful in that it avoids unnecessary if statements inside the main loop of extractors
        null_matcher = type('null_matcher', (),
                            dict(findall=lambda cls, x: []))
        return null_matcher()

        # if isinstance(keys, str):
        # keys = keys,

    # make sure the last keyword is the end card.  Need this to seperate the different headers post facto
    # FIXME!!  will not work if end is followed by more whitespace!!!!!!
    if keys[-1].lower() != 'end':
        keys += 'end',  # treat the end card as a key (with brackets for grouping

    # keywords can be followed by any character, end must be followed by whitespace only
    following = ['.'] * (len(keys) - 1) + [' ']
    # create the matching pattern for each keyword - consists of keyword followed by match for any
    # character following that up to total length of CARD_LENGTH (default 80) characters.
    # eg. FOO.{77}
    key_pattern_maker = lambda key, follow: '%s%s{%i}' % (key.upper(), follow,
                                                          CARD_LENGTH - len(key))
    # Join the individual patterns with the re OR flag, so we match for any of the keyword lines
    pattern = '|'.join(itt.starmap(key_pattern_maker, zip(keys, following)))

    return re.compile(pattern.encode())


# ====================================================================================================
def extractor(chunk):
    # results = []#defaultdict(list)
    # for j, mo in enumerate( matcher.finditer( filemap, *chunk )):
    # card_str = filemap[slice(*mo.span())]
    # results.append( card_str )
    return matcher.findall(filemap, *chunk)


# def merger(r):
# merged = defaultdict(list)
# for d in r:
# for k,v in d.items():
# merged[k] += v
# return merged

# ====================================================================================================
def merge2dict(raw, keys, defaults=[]):
    # NOTE: THIS MERGER WILL BE REDUNDANT IF YOU CAN MAKE THE REGEX CLEVER ENOUGH TO MATCH KEYWORD,VALUE,COMMENTS
    # TODO: BENCHMARK!
    # TODO: Optimize for speed!
    """
    Merge the results from the multiprocessing pool into a dictionary of lists.
    First identifying the END cards and split the results accordingly (by header).
    Loop through headers. Split each keyword into its parts (key, val, comment).
    Substitute missing values from given defaults (None) before aggregating the results.

    Parameters
    ----------
    raw:        List of matched lines from headers
    keys:       sequence of keywords by with to key resulting dictionary
    defaults:   Values to substitute (same order as keys) in case of missing keys.

    Returns
    -------
    D:          Dictionary of lists
    """
    keys = map(str.upper, keys)

    raw = np.asarray(raw)  # sometimes this is faster than np.array(_)
    w, = np.where(raw == end_str)  # location of END matches

    D = DefaultOrderedDict(list)
    # default key values (None if not given)
    defaults = coll.OrderedDict(itt.zip_longest(keys, defaults))  # preserve order of keys

    # iterate through headers. extract keyword values. if keyword missing, sub default. aggregate
    for seg in np.split(raw, w[:-1] + 1):  # split the raw data (last item in each segment is the end string )
        # write the extracted key-val pairs to buffer and sub defaults
        buffer = defaults.copy()
        buffer.update(map(parse_and_clean_no_comment, seg[:-1]))
        # append to the aggregator
        for key in defaults.keys():
            D[key].append(buffer[key])

    return D


# ====================================================================================================
def merger(raw, keys=None, defaults=[], return_type='raw'):
    if return_type == 'raw':
        return raw

    md = merge2dict(raw, keys, defaults)
    if return_type == 'dict':
        return md

    tmd = tuple(md.values())
    if return_type == 'list':
        return tmd

    if return_type == 'array':
        return np.asarray(tmd)


# @profile( follow=(merge2dict,) )
# ====================================================================================================
def headhunter(filename, keys, defaults=[], **kw):
    # TODO: BENCHMARK! Nchunks, filesize
    # TODO: OPTIMIZE?
    # TODO:  Read first N keys from multi-ext header???
    # WARNING:  No error handeling implemented!  Use with discretion!

    """Fast extraction of keyword-value from FITS header(s) using multiprocessing and memmory mapping.

    Parameters
    ----------
    filename:   str
        file to search
    keys:       sequence
        keywords to search for
    defaults:   sequence
        optional defaults to substitute in case of missing keywords

    Keywords
    --------
    Nchunks:            nt;    default 25
        Number of chunks to split the file into.  Tweaking this number can yield faster computation
        times depending on how many cores you have.
    with_parent:        bool;   default False
        whether the key values from the parent header should be kept in the results
    return_type:        str;   options: 'raw', 'dict', 'list', 'array'
        How the results should be merged:
        raw -->         raw matched strings are returned.
        dict -->        return dict of lists keyed on keywords
        list -->        return tuple of lists
        array -->       return 2D array of data values

    Returns
    -------
    dict of lists / list of str depending on the value of `merge`     """

    # print( filename )

    Nchunks = kw.get('Nchunks', 25)
    with_parent = kw.get('with_parent', False)
    return_type = kw.get('return_type', 'list')

    assert return_type in ('raw', 'dict', 'list', 'array')

    if isinstance(keys, re._pattern_type):
        matcher = keys
    else:
        if isinstance(keys, str):
            keys = keys,
        matcher = matchmaker(*keys)

    chunksize = max(1, os.path.getsize(filename) // Nchunks)

    pool = Pool(initializer=init, initargs=[filename, matcher])
    raw = pool.imap(extractor, getchunks(filename, chunksize))  # chunksize=10 (can this speed things up??)
    pool.close()
    pool.join()

    # concatenate the list of lists into single list (a for loop is *the* fastest way of doing this!!)
    results = []
    for r in raw:
        results.extend(r)

    if not with_parent:  # without parent header values (this is needed for the superheadhunter)
        ix = results.index(end_str)
        results = results[ix + 1:]

    return merger(results, keys, defaults, return_type)


# ====================================================================================================
# @profile()# follow=(headhunter,)dd
def superheadhunter(filelist, keys, defaults=[], **kw):
    # TODO: BENCHMARK! Nchunks, Nfiles
    # TODO: OPTIMIZE?
    """Headhunter looped over a list of files."""

    Nchunks = kw.get('Nchunks', 25)
    with_parent = kw.get('with_parent', False)
    return_type = kw.get('return_type', 'list')

    hunt = functools.partial(headhunter,
                             keys=keys,
                             Nchunks=Nchunks,
                             return_type='raw',
                             with_parent=False)

    pool = Pool()
    raw = pool.map(hunt, filelist)
    pool.close()
    # pool.join()

    # Flatten the twice nested list of string matches (this is the fastest way of doing this!!)
    results = []
    for r in raw:
        results.extend(r)

    return merger(results, keys, defaults, return_type)


# ====================================================================================================
if __name__ == '__main__':
    from time import time

    saltpath = '/media/Oceanus/UCT/Observing/SALT/V2400_Oph/20140921/product'
    filelist = parse.to_list(saltpath + '/bxgp*.fits')

    t0 = time()
    # print( len(filelist) )
    keys = 'utc-obs', 'date-obs'
    q = superheadhunter(filelist[:100], keys)
    # q = headhunter( filelist[0], ('date-obs', 'utc-obs', 'deadtime') )

    print('Took', time() - t0, 's')
    # print()
    ##print( q )
    # for k,v in q.items():
    # print( k, len(v) )
    # ipshell()
