"""
Write light curves to plain text in utf-8
"""
#TODO rename human ?

# std
import re
import textwrap
from pathlib import Path
from collections import OrderedDict as odict

# third-party
import numpy as np
import more_itertools as mit

# local
from recipes.dicts import pformat


# write oflag data to file
FORMATSPEC_SRE = re.compile(r'%(\d{0,2})\.?(\d{0,2})([if])')

MULTILINE_CURLY_BRACKET = textwrap.dedent(
    '''
    ⎫
    ⎬ x%i stars
    ⎭
    ''')


# U+23aa Sm CURLY BRACKET EXTENSION ⎪
# U+23ab Sm RIGHT CURLY BRACKET UPPER HOOK ⎫
# U+23ac Sm RIGHT CURLY BRACKET MIDDLE PIECE ⎬
# U+23ad Sm RIGHT CURLY BRACKET LOWER HOOK ⎭


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
    s = ''
    if name:
        s += underline_ascii(name)
        s += '\n'

    s += pformat(info, name='', rhs=get_name, brackets='', sep='')
    s += '\n'
    return s


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
    col_fmt_head = ''.join(map('%%-%is'.__mod__, widths))
    col_fmt_data = ''.join(map('%%-%i.%s%s'.__mod__,
                               zip(widths, precisions, dtypes)))
    return widths, col_fmt_head, col_fmt_data


def hstack_string(a, b, whitespace=1):
    a = a.rstrip('\n')
    b = b.rstrip('\n')
    assert a.count('\n') == b.count('\n')

    al = a.splitlines()
    bl = b.splitlines()
    w = max(map(len, al)) + whitespace

    return ''.join('{: <{}s}{}\n'.format(aa, w, bb) for aa, bb in zip(al, bl))


def get_column_info(nstars, has_oflag):
    # COL_NAME_TIME = 't'
    COL_NAME_TIME = 'bjd'
    COL_NAME_COUNTS = 'Flux'
    COL_NAME_SIGMA = 'σFlux'
    COL_NAME_OFLAG = 'oflag'

    # column descriptions
    COL_DESCRIPT = odict({
        # COL_NAME_TIME: 'Sidereal time in seconds since midnight',
        COL_NAME_TIME: 'Barycentric Julian Date',
        COL_NAME_COUNTS: 'Total integrated counts for star',
        COL_NAME_SIGMA: 'Standard deviation uncertainty on total counts',
        COL_NAME_OFLAG: 'Outlier flag'
    })

    names = [COL_NAME_TIME]
    units = ['day']
    formats = ['%18.9f']
    col_names_per_star = [COL_NAME_COUNTS, COL_NAME_SIGMA]
    col_units_per_star = ['adu', 'adu']
    col_fmt_per_star = ['%12.3f', '%12.3f']

    # outlier detection parameters block
    col_info = COL_DESCRIPT.copy()
    if has_oflag:
        col_names_per_star.append(COL_NAME_OFLAG)
        col_units_per_star.append('')
        col_fmt_per_star.append('%i')
    else:
        # oflag_desc = ''
        col_info.pop(COL_NAME_OFLAG)
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
    units[0] = f'# {units[0]}'
    names[0] = f'# {names[0]}'

    return names, units, formats, col_info


def make_header(obj_name, shape_info, has_oflag, meta=None):
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

    if meta is None:
        meta = {}
    # todo: delimiter ??

    nstars = shape_info['nstars']

    # get column info
    names, units, formats, col_info = get_column_info(nstars, has_oflag)
    # adjust the formatters
    col_widths, col_fmt_head, col_fmt_data = make_column_format(names, formats)

    # make header
    title = f'Light Curve for {obj_name}'

    # table shape info
    lines = [f'# {header_info_block(title, shape_info)}']

    # column descriptions
    lines.append(header_info_block('Columns', col_info))

    # header blocks for additional meta data
    lines.extend(header_info_block(sec_name, info)
                 for sec_name, info in meta.items())

    # header as commented string
    # prepend comment character
    header = '\n'.join(lines).replace('\n', '\n# ')

    # column headers block
    hline = '\n# ' + '-' * sum(col_widths)
    header += hline

    # object names
    obj_names = ['# ', obj_name] + ['C%i' % i for i in range(nstars - 1)]
    w0, *ww = col_widths
    w2 = map(sum, mit.grouper(2 + has_oflag, ww))  # 2-column widths
    col_fmt_names = ''.join('%%-%is' % w for w in [w0] + list(w2))
    header += '\n' + col_fmt_names % tuple(obj_names)

    # column titles
    for o in (names, units):
        header += '\n' + col_fmt_head % tuple(o)
    header += hline
    header += '\n'  # advance to new line

    return header, col_fmt_data


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

    return np.array(tbl).T


def write(filename, t, counts, std, mask=None, meta=None, obj_name='<unknown>'):
    if meta is None:
        meta = {}

    if np.ma.is_masked(counts):
        mask = counts.mask

    # stack data
    tbl = make_table(t, counts, std, mask)

    nrows, ncols = tbl.shape
    nstars = len(counts)
    shape_info = dict(nrows=nrows,
                      ncols=ncols,
                      nstars=nstars)
    has_oflag = mask is not None
    header, col_fmt_data = make_header(obj_name, shape_info, has_oflag, meta)

    # write to file
    with Path(filename).open('w') as fp:
        fp.write(header)
        np.savetxt(fp, tbl, col_fmt_data)
