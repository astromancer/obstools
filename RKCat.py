import re
import logging
import functools
import itertools as itt
from collections import OrderedDict

# 1
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from recipes.iter import interleave, grouper

from IPython import embed


def gen_widths(line):
    column_pattern = r'([^\|]+)\|'
    for i, m in enumerate(re.finditer(column_pattern, line)):
        s, e = m.span()
        w = (e - s)
        yield w

def build_pattern(line):
    """
    Build regex pattern from a line. This can then be to read the rest of the data

    Parameters
    ----------
    line

    Returns
    -------

    """
    #, unflagIx=None
    # TODO: clean < >
    # if unflagIx is None:


    column_pattern = r'([^\|]+)\|'
    p = ''
    maxwidth = 0
    for i, m in enumerate(re.finditer(column_pattern, line)):
        s, e = m.span()
        w = (e - s)

        cpatr = r'''
        \s*                             # match any leading whitespace
        ([^%(flags)s|]{0,%(width)i})    # match anything that's not considered a flag up to column width
        \s*                             # match any intervening whitespace
        ([%(flags)s]?)                  # flags
        \|?                             # column seperator (sometimes missing due to errors in db)
        ''' % dict(width=w, flags=':?*<>')
        p += cpatr #r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w

        # p += r'\s*([^\|]{0,%i})\s*\|?' % w

        # if unflagIx is None or i in unflagIx:
        #     print('strip', i)
        #     # pattern that captures data and flags seperately
        #     p += r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w
        #     # optional pipe here since database has errors where column separator is sometimes missing!
        # else:
        #     # pattern that captures data and empty 'flag'
        #     print('no strip', i)
        #     p += r'\s*([^\|]{0,%i})(\s*?)\|?' % w
        #p += r'\s*([^:?*\|]{0,%i})\s*([:?*]?)\|?' % w

        maxwidth = max(maxwidth, w)

    return re.compile(p, re.VERBOSE), maxwidth

def split(matcher, line):
    dat, flg = zip(*grouper(matcher.match(line).groups(), 2))
    return dat, flg


def build_split_pattern(line):

    column_pattern = r'([^\|]+)\|'
    p = ''
    maxwidth = 0

    for i, m in enumerate(re.finditer(column_pattern, line)):
        s, e = m.span()
        w = (e - s)
        p += r'([^\|]{0,%i})\|?' % w
        maxwidth = max(maxwidth, w)

    return re.compile(p), maxwidth



def gen_datalines(filename):
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if line.startswith(('\n', '-')):
                continue
            yield line.strip('\n')



#
# def split(matcher, line):
#     dat = matcher.match(line).groups()
#     return list(map(str.strip, dat))

def read2(filename):
    fields = ['name', 'alias', 'flag1', 'flag2', 'ra', 'dec',
              'type1', 'type2', 'type3', 'type4', 'mag1', 'mag2', 'mag3', 'mag4',
              'T1', 'T2', 'P0', 'P2', 'P3', 'P4', 'EB', 'SB', 'spectr2', 'spectr1',
              'q', 'q_E', 'Incl', 'Incl_E', 'M1', 'M1_E', 'M2', 'M2_E']
    ncols = len(fields)

    lineGen = gen_datalines(filename)
    h0, h1, *lines = lineGen
    header = [h0, h1]
    # data shape
    n = len(lines) // 2
    ncols = len(fields)
    nlinecols = (ncols // 2)
    # make re pattern
    matcher, w = build_split_pattern(lines[0])

    # init data
    data = np.empty((n, ncols), 'U%s' % w)
    # flags = np.empty((n, ncols), 'U1')
    # populate data
    for i, line in enumerate(lines):
        j, odd = divmod(i, 2)
        #s = nlinecols * odd
        sl = slice(odd, None, 2)
        dat = matcher.match(line).groups()
        data[j, sl] = np.char.strip(dat)
        # flags[j, sl] = flg

    logging.info('Data for %i objects successfully read.', (j + 1))
    return data, np.array(fields), header

#
def similar_str(fields, substr):
    ix, colnames = zip(*((i, c) for i, c in enumerate(fields) if substr in c))
    assert len(ix), 'Nope!'
    return ix, list(colnames)


def read_data(filename):
    """"""
    fields = ['name', 'alias',
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
    ncols = len(fields)

    lineGen = gen_datalines(filename)
    h0, h1, *lines = lineGen
    header = [h0, h1]
    # data shape
    n = len(lines) // 2
    ncols = len(fields)
    nlinecols = (ncols // 2)
    # make re pattern
    # w = list(gen_widths(lines[0]))
    # starts = np.cumsum([0] + w)
    # slices = list(map(slice, *zip(*pairwise(starts))))

    # ix, pcols = similar_str(fields, 'P')
    matcher, w = build_pattern(lines[0])

    # init data
    data = np.empty((n, ncols), 'U%s' % w)
    flags = np.empty((n, ncols), 'U1')
    # populate data
    for i, l in enumerate(lines):
        j, odd = divmod(i, 2)
        #s = nlinecols * odd
        sl = slice(odd, None, 2)
        dat, flg = split(matcher, l)
        data[j, sl] = np.char.strip(dat)
        flags[j, sl] = flg

    iflag1 = fields.index('flag1')
    data[:, iflag1] = flags[:, iflag1]
    flags[:, iflag1] = ''

    logging.info('Data for %i objects successfully read.', (j + 1))
    return data, flags, np.array(fields), header

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_table(filename):
    """"""
    data, flags, fields, header = read_data(filename)
    # data = np.ma.array(data, mask=(data == ''), fill_value='0')
    table = Table(data, masked=True, names=fields, meta=dict(header=header))
    return table


# def clean_flags(col):
#     # (catagorical) uncertainties are flagged by ? or :
#     # in case of object type SU or SH:  if followed by *, the
#     # orbital period has been estimated from the known superhump
#     # period using the empirical relation given by
#     # Stolz & Schoembs (1984, A&A 132, 187).
#     # cln[0], P0sh = clean_flags(tbl['P0'], '*')
#     logging.info('Cleaning column %s', col.name)
#     matcher = re.compile(r'([^:?*]*)\s*([:?*]*)')
#     cln, flg = zip(*(matcher.match(v).groups() for v in col))
#     return np.array(cln), np.array(flg)


def convert_col(col, empty='', dtype=float):
    """Data type convertion hack"""
    data = col.data.data
    mask = (data == empty)
    values = data[~mask]
    try:
        # try convert to float
        values = values.astype(dtype)
        logging.info('Column %s converted to float', col.name)
    except ValueError as err:
        logging.info('Failed to convert column %s to float: %s', col.name, str(err))

    new = np.ma.empty(data.shape, values.dtype)
    new[~mask] = values
    new.mask = mask

    return new

# def clean_periods(tbl):
#     pcols = [name for name in tbl.colnames if name[0] == 'P']
#     sh = len(pcols), len(tbl)
#     flags = np.empty(sh, 'U1')
#
#     for i, p in enumerate(pcols):
#         cln, flags[i] = clean_flags(tbl[p])
#         tbl[p] = convert_ma(cln)
#         # flags[i].mask = clean[i].mask        # all uncertainties for the empty ones masked
#
#     return tbl, flags


    # try:
    #     data[] = fill
    #     converted = data.astype(float)
    #     status = True
    # except ValueError:
    #     converted = data
    #     status = False
    #
    # return status, converted


def as2D(tbl, colnames, dtype=None):
    if dtype is None:
        dtype = next(iter(set(next(zip(*tbl[colnames].dtype.fields.values())))))
    return tbl[colnames].as_array().view(dtype).reshape(-1, len(colnames))

# ****************************************************************************************************
class RKCat(object):

    def __init__(self, filename):

        # data, fields, header = read2(filename)
        # data = np.ma.array(data, mask=(data == ''), fill_value='0')
        data, flags, fields, header = read_data(filename)
        tbl = self.tbl = Table(data, masked=True, names=fields,
                               meta=dict(header=header))
        flg = self.flg = Table(flags, names=fields)

        # can't see wtf these actually mean
        _, flagCols = self.col_group('flag')
        tbl.remove_columns(flagCols)
        flg.remove_columns(flagCols)

        for name, col in tbl.columns.items():
            tbl[name] = convert_col(tbl[name])

        # units
        tbl['P0'].unit = tbl['T1'].unit = tbl['T2'].unit = 'day'
        tbl['P2'].unit = tbl['P3'].unit = tbl['P4'].unit = 's'
        # Masses
        tbl['M1'].unit = tbl['M1_E'].unit = tbl['M2'].unit = tbl['M2_E'].unit = 'Msun'
        # Inclination
        tbl['Incl'].unit = tbl['Incl_E'].unit = 'deg'

        # fix coordinates
        self.coords = coo = self.get_skycoord()
        tbl['ra'] = coo.ra.to_string('h', sep=':')
        tbl['dec'] = coo.dec.to_string(sep=':')


    def restore_flags(self, cols):
        """restore the stripped flags for the entire table"""
        # for i, col in enumerate(self.tbl[cols]):
        flagged = np.char.add(as2D(self.tbl, cols),
                              as2D(self.flg, cols))
        for i, col in enumerate(cols):
            self.tbl[col] = flagged[:, i]


    def col_group(self, name):
        ix, colnames = zip(*((i, c) for i, c in enumerate(self.tbl.colnames) if name in c))
        assert len(ix), 'Nope!'
        return ix, list(colnames)


    def select_by_type(self, *types, exclude=()):
        # Type filter
        tbl = self.tbl
        # flg = self.flg

        l = np.zeros(len(tbl), bool)
        _, typecols = self.col_group('type')

        a2D = as2D(tbl, typecols)
        # f2D = as2D(flg, typecols)

        #l = np.zeros(a2D.shape, bool)
        # f = np.zeros(len(tbl), 'U2')

        for i, t in enumerate(types):
            match = (a2D == t)
            l |= match.any(1)

        for i, t in enumerate(exclude):
            match = (a2D == t)
            l[match.any(1)] = False

        return self.tbl[l]#, f2D
            #f = np.char.add(f, f2D[match])


        return tbl[l], f

        # self.flags = flags
        # self.tbl = tbl
        #self.coords = self.get_skycoord(tbl)
        #self.tbl, self.Pflags = clean_periods(tbl)


    # def read_table(self, filename):
    #     """"""
    #     data, fields, header = read_data(filename)
    #     data = np.ma.array(data, mask=(data == ''))
    #     table = Table(data, masked=True, names=fields, meta=dict(header=header))
    #     return table



    # def clean(self, filename):
    #
    #     #self.header,
    #     tbl = read_table(filename)
    #     # (catagorical) uncertainties are flagged by ? or :
    #     #uncertain = np.vectorize(lambda s: s.endswith(('?', ':')))(tbl)
    #     self.coords = self.get_skycoord(tbl)
    #
    #     tbl, Pflags = clean_periods(tbl)
    #


    # def __getitem__(self, key):
    #     if isinstance(key, str):
    #         l = self.name == key
    #         if np.any(l):
    #             return OrderedDict(zip(self.Fields, self.Data.T[l][0]))
    #         else:
    #             raise KeyError('Object %s not found in catalogue...' % key)

    # @staticmethod
    def get_skycoord(self):
        """Convert string coordinates to SkyCoord"""
        # last column in the dec field is almost always 1 which I guess is another mistake
        ra, dec = self.tbl['ra'], self.tbl['dec']
        dec = [dx.rsplit(' ', 1)[0] for dx in dec]
        return SkyCoord(ra, dec, unit=('h', 'deg'))

    # def fix_coords(self):
    #     tbl = self.tbl
    #     coords =



# ****************************************************************************************************
def to_XEphem(*args, **kw):
    if len(args) == 3:
        kw['name'], kw['ra'], kw['dec'] = args
    elif len(args) == 2:
        kw['name'] = args[0]
        kw['ra'], kw['dec'] = args[1]

    kw.setdefault('mag', 15)
    kw.setdefault('epoch', 2000)

    return '{name},f|V,{ra},{dec},{mag},{epoch}'.format(**kw)


if __name__ == '__main__':
    from astropy.table import Table
    from astropy.table.column import Column

    from pySHOC.timing import Time
    from astropy.time import TimeDelta
    from pySHOC.airmass import altitude

    # RKCat()
    # fields, data = read_data( '/media/Oceanus/UCT/Project/RKcat7.21_main.txt' )
    # fields, data = read_data( '/media/Oceanus/UCT/Observing/RKCat7.23.main.txt' )
    # table = read_data( '/media/Oceanus/UCT/Project/RKcat7.21_main.txt' )
    fields, data = read_data('/media/Oceanus/UCT/Observing/RKCat/7.23/RKCat7.23.main.txt')
    uncertain = np.vectorize(lambda s: s.endswith((':', '?')))(data)
    cleaned = np.vectorize(lambda s: s.strip(':?'))(data)
    empty = np.vectorize(lambda s: s == '')(data)

    table = Table(cleaned, names=fields)

    # convert RA/DEC columns This is a massive HACK!!
    ra, dec = table['ra'], table['dec']
    decmap = map(lambda dec: dec.rsplit(' ', 1)[0], dec)
    coords = list(map(' '.join, zip(ra, decmap)))
    # coords = Column( name='coords', data=SkyCoord( coords, unit=('h', 'deg') ) )
    coords = SkyCoord(coords, unit=('h', 'deg'))  # TODO: Try use this as column??
    ra_col = Column(coords.ra, 'ra', float)
    dec_col = Column(coords.dec, 'dec', float)
    i = table.colnames.index('ra')
    table.remove_columns(('ra', 'dec'))
    table.add_columns([ra_col, dec_col], [i, i])

    # Type filter
    typecols = [c for c in table.colnames if 'type' in c]
    mtypes = ('AM', 'AS', 'LA', 'IP', 'LI')  # magnetic systems 'IP', 'LI'
    ltype = np.array([any(t in mtypes for t in tt) for tt in table[typecols]])

    # Hour angle filter
    # lra = (9 > coords.ra.hour) & (coords.ra.hour < 19)

    tq = table[ltype]  # &lra
    # tq.sort('ra')

    raise SystemExit

    # Magnitude filter
    # mags = [c for c in table.colnames if 'mag' in c]
    mag1 = np.vectorize(lambda s: float(s.strip('>UBVRIKpgr') or 100))(tq['mag1'])
    lmag = mag1 <= 18
    tqq = tq[lmag]

    # l = [ltype&lra][lmag]

    # Altitude filter
    t0 = Time('2015-12-03 00:00:00', scale='utc')
    interval = 300  # seconds
    td = TimeDelta(interval, format='sec')

    days = 3
    N = days * 24 * 60 * 60 / interval
    t = t0 + td * np.arange(N)

    dawn, dusk = 8, 18
    lnight = (dusk < t.hours) | (t.hours < dawn)
    t = t[lnight]

    iers_a = t.get_updated_iers_table(cache=True)  # update the IERS table and set the leap-second offset
    delta, status = t.get_delta_ut1_utc(iers_a, return_status=True)
    t.delta_ut1_utc = delta

    # lat, lon = 18.590549, 98.486546    #TNO
    # 28°45'38.3" N       17°52'53.9" W   +2332m
    lat, lon = 28.7606389, 17.88163888888889  # WHT
    lmst = t.sidereal_time('mean', longitude=lon)

    ra = np.radians(tqq['ra'].data[None].T)
    dec = np.radians(tqq['dec'].data[None].T)
    altmax = altitude(ra,
                      dec,
                      lmst.radian,
                      np.radians(lat)).max(1)
    lalt = np.degrees(altmax) >= 45

    # Orbital period filter
    pc = tqq['P0']
    pc[pc == ''] = '-1'
    Ph = pc.astype(float) * 24
    lP = (Ph < 5) & (Ph > 0)

    tqqq = tqq[lalt & lP]
    tqqq.sort('ra')

    trep = tqqq.copy()
    trep.remove_columns(('flag1', 'flag2', 'T1', 'T2', 'type4', 'P2', 'P3', 'P4', 'spectr1'))

    #
    coo_str = SkyCoord(ra=trep['ra'], dec=trep['dec'], unit='deg').to_string('hmsdms', sep=':')
    ra_str, dec_str = zip(*map(str.split, coo_str))
    ra_col = Column(ra_str, 'ra')
    dec_col = Column(dec_str, 'dec')

    trep.remove_columns(('ra', 'dec'))
    trep.add_columns([ra_col, dec_col], [2, 2])

    i = trep.colnames.index('P0')
    Pm = trep['P0'].astype(float) * 24 * 60
    col = Column(Pm, 'P Orb (min)', float)
    trep.remove_column('P0')
    trep.add_column(col, i)

    trep.show_in_browser()
