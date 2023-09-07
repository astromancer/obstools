"""
Write light curves to plain text in utf-8
"""

# std
import re
import textwrap
from pathlib import Path

# third-party
import numpy as np
import more_itertools as mit
from loguru import logger

# local
from recipes import op
from recipes.dicts import pformat
from recipes.io import read_lines
from recipes.config import ConfigNode


# ---------------------------------------------------------------------------- #
CONFIG = ConfigNode.load_module(__file__)

# write oflag data to file
FORMATSPEC_SRE = re.compile(r'%(\d{0,2})\.?(\d{0,2})([if])')

MULTILINE_CURLY_BRACKET = textwrap.dedent(
    '''
    ⎫
    ⎬ x%i stars
    ⎭
    '''
)
# U+23aa Sm CURLY BRACKET EXTENSION ⎪
# U+23ab Sm RIGHT CURLY BRACKET UPPER HOOK ⎫
# U+23ac Sm RIGHT CURLY BRACKET MIDDLE PIECE ⎬
# U+23ad Sm RIGHT CURLY BRACKET LOWER HOOK ⎭
# ---------------------------------------------------------------------------- #


def parse_format_spec(fmt):
    if mo := FORMATSPEC_SRE.match(fmt):
        return mo.groups()  # width, precision, dtype =
    else:
        raise ValueError('Nope!')


def format_list(data, fmt='%g', width=8, sep=','):
    lfmt = f'%-{width}s' * len(data)
    s = lfmt % tuple(np.char.mod(fmt + sep, data))
    return s[::-1].replace(',', ' ', 1)[::-1].join('[]')


def underline_ascii(text):
    return '\n'.join([text, '-' * len(text)])


def header_info_block(name, info):
    return '\n'.join(_header_info_block(name, info))


def _header_info_block(name, info):
    if name:
        yield underline_ascii(name)

    yield pformat(info, name='', lhs=str,  rhs=get_name, brackets='', sep='')
    yield ''


def get_name(o):
    return o.__name__ if callable(o) else str(o)


def check_column_widths(names, formats):
    widths, precisions, dtypes = [], [], []
    for name, fmt in zip(names, formats):
        width, precision, dtype = parse_format_spec(fmt)
        width = int(width or 1)
        width = max(width, len(name) + 1)

        widths.append(width)
        precisions.append(precision)
        dtypes.append(dtype)
    return widths, precisions, dtypes


# def make_format_spec(names, precisions, dtype='s'):
#     """
#     Adjust the column format specifiers to accommodate width of the column names
#     """
#
#     widths = list(map(len, names))
#     col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
#                                zip(widths, precisions, dtype)))
#     return widths, col_fmt_data

def write_header_aligned(a, out):
    # fn = '/home/hannes/work/pyshoc/pyshoc/data/SHOC1.txt'
    # a = np.genfromtxt(fn, dtype=None, names=True, encoding=None)
    a = a.astype([(_, t.replace('S', 'U')) for _, t in a.dtype.descr])

    widths = np.array(list(map(len, a.dtype.names)))
    fmt_head = ' '.join(map('%%-%is'.__mod__, widths))
    widths[0] += 2
    fmt = tuple(map('%%-%i%s'.__mod__, zip(widths, 's' * len(widths))))

    # out = '/home/hannes/work/pyshoc/pyshoc/data/SHOC1.txt'
    header = fmt_head % a.dtype.names
    np.savetxt(out, a, fmt, header=header)


def make_column_format(names, formats):
    """
    Adjust the column format specifiers to accommodate width of the column names
    """
    widths, precisions, dtypes = check_column_widths(names, formats)
    col_fmt_head = ''.join(map('%%-%is'.__mod__, [widths[0] - 2, *widths[1:]]))
    col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
                               zip(widths, precisions, dtypes)))
    return widths, col_fmt_head, col_fmt_data


def _make_name_format(col_widths, has_oflag):
    w0, *ww = col_widths
    w2 = map(sum, mit.grouper(ww, 2 + has_oflag, fillvalue=ww))  # 2-column widths
    return ''.join('%%-%is' % w for w in [w0 - 2, *w2])


def hstack_string(a, b, whitespace=1):
    a = a.rstrip('\n')
    b = b.rstrip('\n')
    assert a.count('\n') == b.count('\n')

    al = a.splitlines()
    bl = b.splitlines()
    w = max(map(len, al)) + whitespace

    return ''.join('{: <{}s}{}\n'.format(aa, w, bb) for aa, bb in zip(al, bl))


def get_column_info(nstars, has_oflag):

    _names, _units, descript = zip(*CONFIG.columns.values())
    col_info = dict(zip(_names, descript))

    names = [CONFIG.columns.time[0]]
    units = [CONFIG.columns.time[1]]
    formats = ['%18.9f']
    col_names_per_star = [CONFIG.columns.counts[0], CONFIG.columns.sigma[0]]
    col_units_per_star = [CONFIG.columns.counts[1], CONFIG.columns.sigma[1]]
    col_fmt_per_star = ['%12.3f', '%12.3f']

    # outlier detection parameters block
    if has_oflag:
        col_names_per_star.append(CONFIG.columns.outlier[0])
        col_units_per_star.append('')
        col_fmt_per_star.append('%i')
    else:
        # oflag_desc = ''
        col_info.pop(CONFIG.columns.outlier[0])
        col_info[''] = ' '  # place holder

    # column descriptions
    col_info_text = hstack_string('\n'.join(col_info.values()),
                                  MULTILINE_CURLY_BRACKET % nstars, 3)
    col_info.update(zip(col_info, col_info_text.splitlines()))

    # build column headers
    for _ in range(nstars):
        names.extend(col_names_per_star)
        units.extend(col_units_per_star)
        formats.extend(col_fmt_per_star)

    # prepend comment str in such a way as to not screw up alignment with data
    units = [f'[{u}]' for u in units]

    return names, units, formats, col_info


def make_header(title, obj_name, shape_info, has_oflag, meta=None):

    if meta is None:
        meta = {}
    # todo: delimiter ??

    # get column info
    nstars = shape_info['nstars']
    names, units, formats, col_info = get_column_info(nstars, has_oflag)
    # adjust the formatters
    col_widths, col_fmt_head, col_fmt_data = make_column_format(names, formats)

    info = {
        #  title, table shape info
        f'# {title}': shape_info,
        # column descriptions
        'Columns': col_info,
        **meta
    }

    # column headers block
    lines = _make_header(info, obj_name, nstars, has_oflag,
                         (names, units, col_widths, col_fmt_head))
    return '\n'.join(lines).replace('\n', '\n# ')[:-2], col_fmt_data


def _make_header(header_info, obj_name, nstars, has_oflag, col_spec):
    """

    Parameters
    ----------
    obj_name
    shape_info
    has_oflag
    meta:
        meta data that will be printed in the header

    Returns
    -------

    """
    *col_names_units, col_widths, col_fmt_head = col_spec

    # header blocks for additional meta data
    yield from map(header_info_block, *zip(*header_info.items()))

    # header as commented string
    # prepend comment character
    # header = '\n'.join(lines).replace('\n', '\n# ')

    yield (hline := '-' * (sum(col_widths) - 2))

    # object names
    obj_names = ('', obj_name, *(f'C{i}' for i in range(nstars - 1)))
    yield _make_name_format(col_widths, has_oflag) % obj_names

    # column titles
    for o in col_names_units:
        yield col_fmt_head % tuple(o)

    yield hline
    yield ''  # advance to new line


def make_table(t, flx, std, mask=None):
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


def write(filename, t, counts, std, mask=None,
          title=CONFIG.title, meta=None, obj_name='<unknown>'):
    """
    Write to text file

    Parameters
    ----------
    filename : str, Path
        Destination
    t : array
        Time stamps.
    counts : array 
        Source counts.
    std : array
        Uncertainty 
    mask : array, optional
        Masked values boolean array, by default None.
    title : str, optional
        Title for header, by default CONFIG.title
    meta : dict, optional
        Meta data for header, by default None
    obj_name : str, optional
        Name of the target, by default '<unknown>'

    """

    if meta is None:
        meta = {}

    if np.ma.isMA(counts) or np.ma.isMA(std):
        mask = np.ma.getmaskarray(counts) | np.ma.getmaskarray(std)

    logger.info('Saving light curve data ({} rows, {} masked points{}) to file: {}',
                len(t), filename, (0 if mask is None else mask.sum()),
                ', including meta data')

    # stack data
    tbl = make_table(t, counts, std, mask)

    nrows, ncols = tbl.shape
    nstars = len(counts)
    shape_info = dict(nrows=nrows,
                      ncols=ncols,
                      nstars=nstars)
    has_oflag = mask is not None
    header, col_fmt_data = make_header(title.format(obj_name),
                                       obj_name, shape_info, has_oflag, meta)

    # write to file
    with Path(filename).open('w') as fp:
        fp.write(header)
        np.savetxt(fp, tbl, col_fmt_data)


# alias
write_text = write


def read(filename):

    header = read_lines(filename, 25)
    data = np.loadtxt(filename)

    oflag = op.index(header, '# oflag', test=str.startswith, default=None)
    oflag = int(oflag is not None)
    step = 2 + oflag

    t = data[:, 0]
    flux, sigma, *oflag = (data[:, i::step] for i in range(1, 3 + oflag))

    if oflag:
        flux = np.ma.MaskedArray(flux, oflag[0])

    return t, flux.T, sigma.T


# alias
read_text = read
