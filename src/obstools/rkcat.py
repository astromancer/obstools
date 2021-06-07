"""
Read the Ritter-Kolb catalogue for Cataclysmic Variables and related objects.

The Final version (7.24) of 

The Ritter & Kolb Catalogue of Cataclysmic Binaries,
Low-Mass X-Ray Binaries and Related Objects

last updated (31 Dec 2015) 

available at: 
https://wwwmpa.mpa-garching.mpg.de/RKcat/cbcat

For the current and final edition (7.24), literature published before
31 Dec 2015 has, as far as possible, been taken into account. 
This edition has entries for 1429 CVs.


"""


# std libs
import re
import logging
import functools
import itertools as itt

# third-party libs
import numpy as np
import more_itertools as mit
from astropy.table import Table
from astropy.coordinates import SkyCoord

# local libs
from recipes import pprint
from recipes.logging import get_module_logger


# TODO: check coordinates are correct?  UPDATE WITH DATA FROM ALADIN?
# TODO: read companion catalogues
# TODO: read in meta data with column descriptions
# TODO: host as public catalogue that users can contribute to!?

# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)

# flags
# LIMIT_FLAGS = '<>'
# UNCERTAINTY_FLAGS = ':?'  # (categorical) uncertainties are flagged by ? or :
# MAG_FLAGS = 'UBVRIKpgr'
# PERIOD_FLAGS = '*'
# in case of object type SU or SH:  if followed by *, the
# orbital period has been estimated from the known superhump
# period using the empirical relation given by
# Stolz & Schoembs (1984, A&A 132, 187).

# indices for numeric data columns (magnitudes, temperatures, periods, masses)
NUMERIC_COLS_ASCII = np.r_[np.r_[5:10], np.r_[12:16]]
NUMERIC_COLS_ARRAY = np.sort(np.r_[NUMERIC_COLS_ASCII * 2,
                                   NUMERIC_COLS_ASCII * 2 + 1])


RGX_UNFLAG = re.compile(r'\s*([^\d]*)\s*([\d\.]*)([^d]*)')


FIELD_NAMES = ['name', 'alias',
               'flag1', 'flag2',
               'ra', 'dec',
               'type1', 'type2', 'type3', 'type4',
               'mag1', 'mag2', 'mag3', 'mag4',
               'T1', 'T2',
               'P0', 'P2', 'P3', 'P4',
               'EB', 'SB',
               'spectr2', 'spectr1',
               'q', 'q_E', 'Incl', 'Incl_E',
               'M1', 'M1_E', 'M2', 'M2_E']


def as_2d_array(tbl, col_names, dtype=None):
    """"""

    names = list(col_names)

    if dtype is None:
        # get the first dtype amongst unique set of dtypes in array...
        dtype = next(iter(set(next(zip(*tbl[names].dtype.fields.values())))))

    return tbl[names].as_array().view(dtype).reshape(-1, len(names))


def _get_col_indices(line, sep='|'):
    """
    Build regex pattern from a line. This can then be to read the rest of the
    data

    Parameters
    ----------


    """

    i, j = 0, 1
    ncols = line.count(sep)
    col_breaks = np.zeros(ncols + 1, int)

    while j <= ncols:
        i = line.index(sep, i + 1)
        col_breaks[j] = i + 1
        j += 1

    index_pairs = np.array([col_breaks[:-1], col_breaks[1:] - 1]).T
    col_widths = np.diff(index_pairs).squeeze()
    column_slices = list(map(slice, *index_pairs.T))
    return column_slices, col_widths


def _sanitize_cell(m, s):
    """
    Use the regex matcher to strip flags from cell data.

    Parameters
    ----------
    m
    s


    Returns
    -------
    (pre_flags, data, post_flags)

    """
    return m.match(s).groups()


def iter_lines(filename):
    """
    Iterate through lines containing actual data, i.e. skip the separator lines
    containing only dashes `-`

    Parameters
    ----------
    filename

    Returns
    -------

    """
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if not line.startswith(('\n', '-')):
                yield line.strip('\n')


def line_items(line, slices):
    """
    Split the row into the fixed width cells and strip whitespace from content.

    Parameters
    ----------
    line
    slices


    Returns
    -------

    """
    for _ in slices:
        yield line[_].strip()


def read_ascii(filename, mask_missing=False):
    """
    Read the data from Ritter-Kolb catalogue in `filename` into numpy string
    type array.

    Parameters
    ----------
    filename
    mask_missing


    Returns
    -------

    """
    # Read raw data
    lineGen = iter_lines(filename)
    h0, h1, *lines = lineGen  # unpack lines
    header = [h0, h1]

    # get column names
    column_slices, widths = _get_col_indices(lines[0])
    col_headers = list(mit.interleave(*(line_items(h, column_slices)
                                        for h in header)))

    # init data containers
    n = len(lines) // 2
    ncols = len(column_slices) * 2
    # dtypes = tuple('U%s' % w for w in widths)
    data = np.empty((n, ncols), 'U%s' % widths.max())  # somewhat inefficient

    ix_odd = np.arange(1, ncols, 2)
    ix_even = np.arange(0, ncols, 2)

    # populate data
    j = 0
    for i, line in enumerate(lines):
        # the next two lines interleaves the data from 2 rows which logically
        # belong to the same object.
        j, odd = divmod(i, 2)
        k = ix_odd if odd else ix_even
        #
        data[j, k] = list(line_items(line, column_slices))

    # mask missing data
    if mask_missing:
        data = np.ma.MaskedArray(data, (data == ''))

    logger.info('Data for %i objects successfully read.', (j + 1))
    return data, col_headers


def read_table(filename):
    """"""
    data, col_headers = read_ascii(filename)
    # data = np.ma.array(data, mask=(data == ''), fill_value='0')

    return Table(data, names=FIELD_NAMES,
                 masked=True,
                 meta=dict(header=col_headers))


def _convert_float(data, empty='', dtype=float):
    """Data type conversion hack for sh*tty database structure :("""

    new = np.ma.empty(data.shape, dtype)
    good = (data != empty)
    new[good] = data[good].astype(dtype)
    new.mask = ~good
    return new


def fmt_ra(x):
    return pprint.hms(x * 3600 / 15, unicode=True, precision=1)


def fmt_dec(x):
    return pprint.hms(x * 3600, sep='°’”', precision=1)


class RKCat(object):
    # TODO: inherit from Table??

    # @classmethod
    # def from_file(cls, filename):
    # ''

    def __init__(self, filename, mask_missing=True):
        # read the data
        data, header = read_ascii(filename, mask_missing)
        self.tbl = tbl = Table(data, names=FIELD_NAMES,
                               masked=True,
                               meta=dict(header=header))
        # Remove weird flag columns with `*` and `#`
        # TODO figure out wtf these actually mean
        _, flag_cols = self.col_group('flag')
        tbl.remove_columns(flag_cols)

        # clean numeric data
        self.pre_flags = None
        self.post_flags = None
        self.remove_flags()

        # units
        tbl['P0'].unit = tbl['T1'].unit = tbl['T2'].unit = 'day'
        tbl['P2'].unit = tbl['P3'].unit = tbl['P4'].unit = 's'
        # Masses
        tbl['M1'].unit = tbl['M1_E'].unit = 'Msun'
        tbl['M2'].unit = tbl['M2_E'].unit = 'Msun'
        # Inclination
        tbl['Incl'].unit = tbl['Incl_E'].unit = 'deg'

        # convert coordinates
        self.coords = coo = self.get_skycoord()

        tbl['ra'], tbl['dec'] = coo.ra, coo.dec
        tbl['ra'].format = fmt_ra
        tbl['dec'].format = fmt_dec

    def __repr__(self):
        return repr(self.tbl)

    def remove_flags(self):
        # regex matchers for numeric data columns (hard coded)
        # these extract the numbers from annoying entries like '<19.8p'
        # sre = _build_sre(pre_flags, post_flags)

        names = list(np.take(FIELD_NAMES, NUMERIC_COLS_ARRAY))
        sub = as_2d_array(self.tbl, names, 'U12').data
        pre, data, post = np.vectorize(_sanitize_cell)(RGX_UNFLAG, sub)
        data = _convert_float(data)

        for cname, new_data in zip(names, data.T):
            self.tbl[cname] = new_data

        # containers for flags
        n = len(self.tbl)
        flag_dtypes = np.dtype(list(zip(names, itt.repeat('U1'))))

        self.pre_flags = np.array(list(map(tuple, pre)),
                                  flag_dtypes).view(np.recarray)
        self.post_flags = np.array(list(map(tuple, post)),
                                   flag_dtypes).view(np.recarray)

        #
        if logger.getEffectiveLevel() >= logging.INFO:
            n_cleaned = np.sum(pre != '') + np.sum(post != '')
            logger.info('Flags stripped from %i data entries in %i columns',
                        n_cleaned, len(names))

    def restore_flags(self, names):
        """restore the stripped flags for the entire table"""
        raise NotImplementedError

        # names = list(np.setdiff1d(names, FIELD_NAMES))
        # numeric = np.take(FIELD_NAMES, NUMERIC_COLS_ARRAY)
        #
        # flagged = np.char.add(as_2d_array(self.tbl, cols),
        # )
        # for i, col in enumerate(cols):
        # self.tbl[col] = flagged[:, i]

    def col_group(self, name):
        ix, col_names = zip(*((i, c)
                              for i, c in enumerate(self.tbl.colnames)
                              if name in c))
        assert len(ix), 'Nope!'
        return ix, list(col_names)

    def select_by_type(self, *types, exclude=()):  # select_object_types

        # Type filter
        tbl = self.tbl
        # flg = self.flg

        l = np.zeros(len(tbl), bool)
        _, type_cols = self.col_group('type')

        a2D = as_2d_array(tbl, type_cols)
        # f2D = as_2d_array(flg, type_cols)

        # l = np.zeros(a2D.shape, bool)
        # f = np.zeros(len(tbl), 'U2')

        for i, t in enumerate(types):
            match = (a2D == t)
            l[match.any(1)] = False

        return self.tbl[l]  # , f2D
        # f = np.char.add(f, f2D[match])
        # return tbl[l], f

    def get_skycoord(self):
        """Convert string coordinates to SkyCoord"""
        # last column in the dec field is almost always 1 which I guess is
        # another mistake ...
        ra, dec = self.tbl['ra'], self.tbl['dec']
        dec = [dx.rsplit(' ', 1)[0] for dx in dec]
        return SkyCoord(ra, dec, unit=('h', 'deg'))


def to_XEphem(*args, **kw):
    if len(args) == 3:
        kw['name'], kw['ra'], kw['dec'] = args

# if __name__ == '__main__':
#     from obstools.airmass import altitude
#
#     # RKCat()
#     # '/media/Oceanus/UCT/Project/RKcat7.21_main.txt'
#     # '/media/Oceanus/UCT/Observing/RKCat7.23.main.txt'
#     filename = '/media/Oceanus/UCT/Observing/RKCat/7.23/RKCat7.23.main.txt'
#     fields, data = read_ascii(filename)
#
#     uncertain = np.vectorize(lambda s: s.endswith((':', '?')))(data)
#     cleaned = np.vectorize(lambda s: s.strip(':?'))(data)
#     empty = np.vectorize(lambda s: s == '')(data)
#     decmap = map(lambda dec: dec.rsplit(' ', 1)[0], dec)
#     coords = list(map(' '.join, zip(ra, decmap)))
#     # coords = Column( name='coords', data=SkyCoord( coords, unit=('h', 'deg') ) )
#     coords = SkyCoord(coords, unit=('h', 'deg'))
#     # TODO: Try use this as column??
#     ra_col = Column(coords.ra, 'ra', float)
#     dec_col = Column(coords.dec, 'dec', float)
#     i = table.colnames.index('ra')
#
#     table.add_columns([ra_col, dec_col], [i, i])
#
#     # Type filter
#     type_cols = [c for c in table.colnames if 'type' in c]
#     mtypes = ('AM', 'AS', 'LA', 'IP', 'LI')  # magnetic systems 'IP', 'LI'
#     ltype = np.array([any(t in mtypes for t in tt) for tt in table[type_cols]])
#
#     # Hour angle filter
#     # lra = (9 > coords.ra.hour) & (coords.ra.hour < 19)
#     # Magnitude filter
#     # mags = [c for c in table.colnames if 'mag' in c]
#     mag1 = np.vectorize(
#             lambda s: float(s.strip('>UBVRIKpgr') or 100))(tq['mag1'])
#     lmag = mag1 <= 18
#     tqq = tq[lmag]
#     lnight = (dusk < t.hours) | (t.hours < dawn)
#     t = t[lnight]
#
#     # update the IERS table and set the leap-second offset
#     iers_a = t.get_updated_iers_table(cache=True)
#     delta, status = t.get_delta_ut1_utc(iers_a, return_status=True)
#     t.delta_ut1_utc = delta
#
#     tqqq.sort('ra')
#
#     trep = tqqq.copy()
#     trep.remove_columns(('flag1', 'flag2',
#                          'T1', 'T2',
#                          'type4',
#                          'P2', 'P3', 'P4',
#                          'spectr1'))
#
#     #
#     coo_str = SkyCoord(ra=trep['ra'], dec=trep['dec'],
#                        unit='deg').to_string('hmsdms', sep=':')
#     ra_str, dec_str = zip(*map(str.split, coo_str))
#     ra_col = Column(ra_str, 'ra')
#     dec_col = Column(dec_str, 'dec')
