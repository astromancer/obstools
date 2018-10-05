"""
Read the Ritter-Kolb catalogue for Cataclysmic Variables and related objects.
"""

# builtin libs
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
from obstools.utils import fmt_hms
from recipes.string import get_module_name


# TODO: check coordinates are correct?  UPDATE WITH DATA FROM ALADIN?
# TODO: read companion catalogues
# TODO: read in meta data with column descriptions
# TODO: host as public catalogue that users can contribute to!?

# module level logger
log_name = get_module_name(__file__)
logger = logging.getLogger(log_name)
logger.setLevel(logging.INFO)

# flags
LIMIT_FLAGS = '<>'
UNCERTAINTY_FLAGS = ':?'  # (categorical) uncertainties are flagged by ? or :
MAG_FLAGS = 'UBVRIKpgr'
PERIOD_FLAGS = '*'
# in case of object type SU or SH:  if followed by *, the
# orbital period has been estimated from the known superhump
# period using the empirical relation given by
# Stolz & Schoembs (1984, A&A 132, 187).

# indices for numeric data columns (magnitudes, temperatures, periods, masses)
NUMERIC_COLS_ASCII = np.r_[np.r_[5:10], np.r_[12:16]]
NUMERIC_COLS_ARRAY = np.sort(np.r_[NUMERIC_COLS_ASCII * 2,
                                   NUMERIC_COLS_ASCII * 2 + 1])

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

    i = 0
    j = 1
    ncols = line.count(sep)
    cbreaks = np.zeros(ncols + 1, int)

    while j <= ncols:
        i = line.index(sep, i + 1)
        cbreaks[j] = i + 1
        j += 1

    index_pairs = np.array([cbreaks[:-1], cbreaks[1:] - 1]).T
    widths = np.diff(index_pairs).squeeze()
    column_slices = list(map(slice, *index_pairs.T))
    return column_slices, widths


def _build_sre(preflags='<>', postflags=':?'):
    """
    Build regex pattern for a column of width. Flag non-numeric characters
    will be removed.

    Parameters
    ----------
    width
    preflags
    postflags

    Returns
    -------


    """

    column_pattern = r'''
        \s*                             # match any leading whitespace
        ([%s]*)                         # match any pre-flags
        \s*                             # match any intervening whitespace
        ([\d\.]*)                       # match any number and decimal point
        ([%s]*)                         # match any post-flags
        ''' % (preflags, postflags)

    return re.compile(column_pattern, re.VERBOSE)


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
            if line.startswith(('\n', '-')):
                yield line.strip('\n')


def line_items(line, slices):
    """
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
    """Data type conversion hack for shitty database structure :("""

    new = np.ma.empty(data.shape, dtype)
    good = (data != empty)
    new[good] = data[good].astype(dtype)
    new.mask = ~good
    return new


class RKCat(object):  # TODO: inherit from Table??

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

        # containers for flags
        n = len(tbl)
        flag_dtypes = np.dtype(list(zip(tbl.colnames, itt.repeat('U1'))))
        self.pre_flags = np.recarray(n, flag_dtypes)
        self.post_flags = np.empty(n, flag_dtypes)

        # clean numeric data
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
        fmt_kws = dict(precision=1, short=False)
        tbl['ra'], tbl['dec'] = coo.ra, coo.dec
        tbl['ra'].format = functools.partial(fmt_hms, sep='ʰᵐˢ', **fmt_kws)
        tbl['dec'].format = functools.partial(fmt_hms, sep='°’”', **fmt_kws)

    def _unflags_columns(self, names, pre_flags, post_flags):
        # regex matchers for numeric data columns (hard coded)
        # these extract the numbers from annoying entries like '<19.8p'
        sre = _build_sre(pre_flags, post_flags)
        sub = as_2d_array(self.tbl, names, 'U12').data
        pre, data, post = np.vectorize(_sanitize_cell)(sre, sub)
        data = _convert_float(data)
        return pre, data, post

    def remove_flags(self):
        names = list(np.take(FIELD_NAMES, NUMERIC_COLS_ARRAY))
        pre, data, post = self._unflags_columns(names,
                                                LIMIT_FLAGS,
                                                MAG_FLAGS + UNCERTAINTY_FLAGS)

        for cname, new_data in zip(names, data.T):
            self.tbl[cname] = new_data

        pre_ = self.pre_flags[names]
        pre_[:] = pre.view(pre_.dtype).squeeze()

        post_ = self.pre_flags[names]
        post_[:] = pre.view(post_.dtype).squeeze()

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


# def clean_flags(col):

# # cln[0], P0sh = clean_flags(tbl['P0'], '*')
# logging.info('Cleaning column %s', col.name)
# matcher = re.compile(r'([^:?*]*)\s*([:?*]*)')
# cln, flg = zip(*(matcher.match(v).groups() for v in col))
# return np.array(cln), np.array(flg)

# def gen_widths(line):
# column_pattern = r'([^\|]+)\|'
# for i, m in enumerate(re.finditer(column_pattern, line)):
# s, e = m.span()
# w = (e - s)
# yield w

# def _build_column_sre(width, preflags='<>', postflags=':?'):
# """
# Build regex pattern for a column of width. Flag non-numeric characters
# will be removed.
#
#
# Parameters
# ----------
# width
# preflags
# postflags
#
# Returns
# -------
#
# """
#
# column_pattern = r'''
# \s*                             # match any leading whitespace
# ([%(preflags)s]*)               # match any pre-flags up to column width
# \s*                             # match any intervening whitespace
# ([\d\.]{0,%(width)i})           # match any number and decimal point
# ([%(postflags)s]*)              # match the flags
# ''' % dict(width=width, preflags=preflags, postflags=postflags)
#
# return re.compile(column_pattern, re.VERBOSE)

# def build_pattern(line):
# """
# Build regex pattern from a line. This can then be to read the rest of the
# data
#
# Parameters
# ----------
# line
#
# Returns
# -------
#
# """
# # , unflagIx=None
# # TODO: clean < >
# # if unflagIx is None:
#
# column_pattern = r'([^\|]+)\|'
# p = ''
# maxwidth = 0
# for i, m in enumerate(re.finditer(column_pattern, line)):
# s, e = m.span()
# w = (e - s)
#
# column_pattern = r'''
# \s*                             # match any leading whitespace
# ([^%(flags)s|]{0,%(width)i})    # match anything that's not considered a flag up to column width
# \s*                             # match any intervening whitespace
# ([%(flags)s]?)                  # match the flags
# \|?                             # column separator (sometimes missing due to typos in db)
# ''' % dict(width=w,
# flags=RKFLAGS)
# p += column_pattern
#
# # r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w
#
# # p += r'\s*([^\|]{0,%i})\s*\|?' % w
#
# # if unflagIx is None or i in unflagIx:
# #     print('strip', i)
# #     # pattern that captures data and flags seperately
# #     p += r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w
# #     # optional pipe here since database has errors where column separator is sometimes missing!
# # else:
# #     # pattern that captures data and empty 'flag'
# #     print('no strip', i)
# #     p += r'\s*([^\|]{0,%i})(\s*?)\|?' % w
# # p += r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w
#
# maxwidth = max(maxwidth, w)
#
# return re.compile(p, re.VERBOSE), maxwidth

#
# def split(matcher, line):
# dat = matcher.match(line).groups()
# return list(map(str.strip, dat))


# def read2(filename):
# fields = ['name', 'alias', 'flag1', 'flag2', 'ra', 'dec',
# 'type1', 'type2', 'type3', 'type4', 'mag1', 'mag2', 'mag3',
# 'mag4',
# 'T1', 'T2', 'P0', 'P2', 'P3', 'P4', 'EB', 'SB', 'spectr2',
# 'spectr1',
# 'q', 'q_E', 'Incl', 'Incl_E', 'M1', 'M1_E', 'M2', 'M2_E']
#
#
# lineGen = iter_lines(filename)
# h0, h1, *lines = lineGen
# header = [h0, h1]
# # data shape
# n = len(lines) // 2
# ncols = len(fields)
# nlinecols = (ncols // 2)
# # make re pattern
# matcher, w = build_split_pattern(lines[0])
#
# # init data
# data = np.empty((n, ncols), 'U%s' % w)
# # flags = np.empty((n, ncols), 'U1')
# # populate data
# for i, line in enumerate(lines):
# j, odd = divmod(i, 2)
# # s = nlinecols * odd
# sl = slice(odd, None, 2)
# dat = matcher.match(line).groups()
# data[j, sl] = np.char.strip(dat)
# # flags[j, sl] = flg
#
# logging.info('Data for %i objects successfully read.', (j  1))
# return data, np.array(fields), header

#
# def similar_str(fields, substr):
# ix, colnames = zip(*((i, c) for i, c in enumerate(fields) if substr in c))
# assert len(ix), 'Nope!'
# return ix, list(colnames)

# def split(matcher, line):
# dat, flg = zip(*grouper(matcher.match(line).groups(), 2))
# return dat, flg
#
#
# def build_split_pattern(line):
# column_pattern = r'([^\|]+)\|'
# p = ''
# maxwidth = 0
#
# for i, m in enumerate(re.finditer(column_pattern, line)):
# s, e = m.span()
# w = (e - s)
# p += r'([^\|]{0,%i})\|?' % w
# maxwidth = max(maxwidth, w)
#
# return re.compile(p), maxwidth

# try:
# # try convert to float
# values = values.astype(dtype)
# logging.info('Column %s converted to float', col.name)
# except ValueError as err:
# logging.info('Failed to convert column %s to float: %s', col.name,
# str(err))
#
# new = np.ma.empty(data.shape, values.dtype)
# new[~mask] = values
# new.mask = mask
#
# return new

# def clean_periods(tbl):
# pcols = [name for name in tbl.colnames if name[0] == 'P']
# sh = len(pcols), len(tbl)
# flags = np.empty(sh, 'U1')
#
# for i, p in enumerate(pcols):
# cln, flags[i] = clean_flags(tbl[p])
# tbl[p] = convert_ma(cln)
# # flags[i].mask = clean[i].mask        # all uncertainties for the empty ones masked
#
# return tbl, flags

# try:
# data[] = fill
# converted = data.astype(float)
# status = True
# except ValueError:
# converted = data
# status = False
#
# return status, converted


def to_XEphem(*args, **kw):
    if len(args) == 3:
        kw['name'], kw['ra'], kw['dec'] = args

# if __name__ == '__main__':
#     from pySHOC.airmass import altitude
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
