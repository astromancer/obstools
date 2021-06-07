"""
Functions to read and parse the MASTER transient alert website:
    http://observ.pereplet.ru/MASTER_OT.html
"""


# std libs
from astropy.coordinates import jparser
from recipes.string import sub
import re
import itertools as itt
from datetime import datetime
from html.parser import HTMLParser
from collections import OrderedDict
from urllib.request import urlopen
import warnings

# third-party libs
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.table import Table

# local libs
from recipes.iter import first_true_idx

from .utils import fmt_ra, fmt_dec


URL = 'http://observ.pereplet.ru/MASTER_OT.html'

RGX_BAD_JCOO = re.compile(
    '(.*?J)'
    r'()([0-2]\d)([0-5]\d)([0-5]\d)\.?(\d{0,3})'
    r'([+-−])(\d{1,2})([0-5]\d)([0-6]\d)\.?(\d{0,3})'
)
RGX_NR = re.compile(r'[\d.]+')
RGX_BAD_DATE = re.compile(r'(\d{4}) ((?i:[a-z.]{3,4}))\s*(\d{2}) (\d+)')

COLUMN_PATTERNS = {
    'nr':      r'\s{0,3}(?:ART|-|[\d]{3,4})',  # ART? WTF!
    'name':    r'[\w.+\-−() ]{20,29}',
    'date':    r'20\d\d [\w.]{3,4} {1,2}[\d.X ]{2,6}',
    'type':    r'[a-zA-Z\|?/:]{1,5}',
    'mag':     r'[\d. ]{0,5}',
    'remark':  r'[\w/?|]{0,6}',
    # '[ATel GCN PZP CBET IAUC #\d?.,]+'),    #ATel|GCN|PZP|CBET|IAUC[ #\d?.,]+
    'ref':     r'[\w#\-.,? ]{0,13}',
    #     # FIXME: find a more general way of reading the last couple of columns
    'site':    'SA|Amu|Tun|Arg|Dom|Kis|Net|Ura|IAC|Iac|OAFA|Tav|NET|I,O',
    'pioneer': '[A-Z?,+]{0,6}',
    'comment': '.*'
}
# patterns = OrderedDict(patterns)
COLUMN_NAMES = list(COLUMN_PATTERNS.keys())


PARSER = re.compile(
    r'\s{0,5}'.join((f'(?P<{name}>{ptrn})'
                     for name, ptrn in COLUMN_PATTERNS.items()))
)


class MyHTMLParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        for attr in attrs:
            if (attr[0] == 'href') and attr[1]:
                self.links.append(attr[1])

    def handle_data(self, data):
        # if data not in ('\n',):
        self.data += data

    def reset(self):
        super().reset()
        self.data = ''
        self.links = []


def parse(report=True):
    """
    Attempt to parse the MASTER transient alert website
    """

    html = urlopen(URL).read()  # NOTE: THIS LINE IS SLOW

    lines = list(filter(None, html.decode().split('\n')))
    ix0 = first_true_idx(lines, is_numbered)
    ix1 = -first_true_idx(reversed(lines), is_numbered)
    raw_data = lines[ix0:ix1]

    data = np.empty((len(raw_data), len(COLUMN_NAMES)), 'O')
    links = []
    success = 0
    failures = {}

    html_parser = MyHTMLParser()   # strict=False
    for i, line in enumerate(raw_data):
        # strip html tags and extract links
        html_parser.reset()
        html_parser.feed(line)

        # get raw data
        clean = html_parser.data + html_parser.rawdata

        # parse the columns
        mo = PARSER.match(clean)
        if mo is None:
            # print(clean, '\n', err)
            failures[i] = clean
        else:
            data[i] = mo.groups()
            links.append(html_parser.links)
            success += 1

    if report:
        print(f'Successfully parsed {success} / {len(raw_data)} lines')
        print(f'{len(failures)} lines failed')

    # clean result
    data = np.char.array(data.astype('U')).strip()

    return data, links, failures


def load():
    data, links, failures = parse()

    # remove failures (all data 'None')
    data = np.delete(data, list(failures), 0)

    # create table
    dtype = np.dtype(list(zip(COLUMN_NAMES,
                              map('<U{}'.format, np.char.str_len(data).max(0))))
                     )
    tbl = Table(data, dtype=dtype, masked=True)

    # Extract coordinates RA, DEC from name and add as columns in the data table.
    ra, dec = np.vectorize(get_coords)(np.array(data[:, 1]))
    bad = (ra == -1) & (dec == -1)
    ra = np.ma.MaskedArray(ra, bad)
    dec = np.ma.MaskedArray(dec, bad)
    tbl.add_columns([ra, dec], [3, 3], ['ra', 'dec'])
    tbl['ra'].format = fmt_ra
    tbl['dec'].format = fmt_dec

    # convert mag to masked float
    mag = np.array(data[:, COLUMN_NAMES.index('mag')])
    mag = np.vectorize(get_mag)(mag)
    mag = np.ma.MaskedArray(mag, mag == -1)
    tbl.replace_column('mag', mag)

    # convert date to datetime
    tbl['date'] = np.vectorize(get_date)(tbl['date'])

    return tbl


def is_numbered(s):
    return s[0].isdigit()


def get_coords(name):
    try:
        return jparser.to_ra_dec_float(name)
    except ValueError as err:
        match = RGX_BAD_JCOO.search(name)
        if match:
            prefix, hms, dms = np.split(
                RGX_BAD_JCOO.search(name).groups(), [1, 6])
            warnings.warn(f'Bad value for arcseconds in name: {name!r}')
            return ((jparser._sexagesimal(hms) / jparser.SEXAG2DEG).sum() * 15,
                    (jparser._sexagesimal(dms) / jparser.SEXAG2DEG).sum())

        # placeholder for uninterprable values
        warnings.warn(f'Could not retreive coordinates from name: {name!r}')
        return -1, -1


def get_mag(m):
    mo = RGX_NR.match(m)
    if mo:
        return float(m)
    return -1


def write_ascii(filename, data):
    col_names = list(data.dtype.fields.keys())
    col_names[0] = '# ' + col_names[0]
    data = np.vstack([col_names, data]).astype('U')

    fmt = list(map('%-{}s'.format, np.char.str_len(data).max(0)))
    np.savetxt(filename, data[::-1], fmt)


def get_date(date):
    try:
        yr, month, day = date.split()
    except ValueError:
        # couple of these stupid typos:     2015 Mar  15 834
        yr, month, day, frac = date.split()
        day += f'.{frac}'

    # more stupid typos FFS
    month = sub(month.strip('.'), {'Spt': 'Sep',
                                   'Mrt': 'Mar',
                                   'Nar': 'Mar',
                                   'Avg': 'Aug'})
    mfmt = '%b'
    if len(month) > 3:
        mfmt = '%B'

    s = f'{yr} {month}'
    cur = day.strip('X')
    for mult in (24, 60, 60):
        i, frac = divmod(float(cur), 1)
        s += f' {int(i)!s:0>2s}'
        cur = frac * mult
    s += f' {int(cur)!s:0>2s}'
    return datetime.strptime(s, f'%Y {mfmt} %d %H %M %S')


def is_observable(coords, date=None, days=1, interval=300, altcut=45):
    from obstools.airmass import altitude

    # Altitude filter
    if date is None:
        t0 = Time(Time.now().iso.split()[0])
    else:
        t0 = Time(date, scale='utc')

    td = TimeDelta(interval, format='sec')
    N = days * 24 * 60 * 60 / interval
    td = td * np.arange(N)
    t = t0 + td

    dawn, dusk = 3, 19  # TODO: determine (fast)
    h = td.to('h').value % 24
    lnight = (dusk < h) | (h < dawn)
    t = t[lnight]

    lat, lon = -32.375823, 20.81
    lmst = t.sidereal_time('mean', longitude=20.81)

    ra, dec = np.radians(coords)[:, None, :].T
    alt = altitude(ra,
                   dec,
                   lmst.radian[:, None],
                   np.radians(lat))

    lalt = np.degrees(alt.max(0)) >= altcut

    return t, alt, lalt


if __name__ == '__main__':
    from obstools.jparser import Jparser

    # from ansi.table import Table
    data, links, failures = parse()
    # masterDb = np.rec.fromrecords(success, names=col_names)

    # coords = Jparser.many2deg(masterDb.name)
    # t, alt, lalt = is_observable(coords)

    # # Type filter
    # cvtypes = ('DN', 'NL', 'CV')
    # ltyp = np.any([masterDb.type == typ for typ in cvtypes], 0)

    # # Recentness filter
    # discovered = Time(np.vectorize(convert_date)(masterDb.date))
    # lrec = (Time.now() - discovered).to('d').value < 100

    # l = ltyp & lalt & lrec
