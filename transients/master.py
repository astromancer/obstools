"""
Functions to read and parse the MASTER transient alert website: http://observ.pereplet.ru/MASTER_OT.html
"""

import re
import urllib.request
import itertools as itt
from datetime import datetime
from html.parser import HTMLParser
from collections import OrderedDict

import numpy as np
from astropy.time import Time, TimeDelta

from recipes.iter import first_true_idx


# from obstools.jparser import Jparser


class MyHTMLParser(HTMLParser):
    def __init__(self, *args, **kwargs):
        HTMLParser.__init__(self, *args, **kwargs)
        self.data = ''
        self.links = []

    def handle_starttag(self, tag, attrs):
        # print("Start tag:", tag)
        for attr in attrs:
            if attr[0] == 'href':
                if attr[1]:
                    self.links.append(attr[1])
                    # print("     attr:", attr)

    # def handle_endtag(self, tag):
    #   print("End tag  :", tag)

    def handle_data(self, data):
        # if data not in ('\n',):
        self.data += data

# from decor.profiler import profiler
#
# @profiler.histogram
def parse_master():
    """
    Attempt to parse the MASTER transient alert website
    """
    master = 'http://observ.pereplet.ru/MASTER_OT.html'
    response = urllib.request.urlopen(master)
    html = response.read()                          #NOTE: THIS LINE IS SLOW
    htmllines = html.split(b'\n')

    isnumbered = lambda s: s[0].isdigit()
    lines = map(bytes.decode, htmllines)  # map to str
    lines = list(filter(None, lines))
    ix0 = first_true_idx(lines, isnumbered)
    ix1 = -first_true_idx(reversed(lines), isnumbered)
    data = lines[ix0:ix1]

    #meta = lines[ix1:]
    #pions = first_true_idx(meta, lambda l: l.startswith('Pioneer'))
    #pione = first_true_idx(meta, lambda l: 'Types of optical transients' in l)
    #pioneers = re.findall('([A-Z]{2}) -',  ' '.join(meta[pions+1:pione]))

    patterns = (('nr',      'ART|[ \d-]{4}'),               #ART? WTF!
                ('name',    '.+?'),
                ('date',    '20\d\d [\w.]{3,4} {1,3}[\d.]+'),
                ('type',    '[\w\|?/:]{1,5}'),
                ('mag',     '[\d.]{0,5}'),
                ('remark',  '[\w/]{0,6}'),
                ('ref',     '.+'), #'[ATel GCN PZP CBET IAUC #\d?.,]+'),    #ATel|GCN|PZP|CBET|IAUC[ #\d?.,]+
                # FIXME: find a more general way of reading the last couple of columns
                ('site',    'SA|Amu|Tun|Arg|Dom|Kis|Net|Ura|IAC|Iac|OAFA'),
                ('pioneer', '[A-Z?]*'),
                ('comment', '.*'))
    patterns = OrderedDict(patterns)
    colNames = list(patterns.keys())
    m = list(itt.starmap('(?P<{}>{})'.format, patterns.items()))
    master_pattern = '\s{0,5}'.join(m[:])
    parser = re.compile(master_pattern)

    success, links = [], []
    image_urls = []
    fail, flinks = [], []
    ixFail = []
    txt = []
    for i, l in enumerate(data):
        try:
            html_parser = MyHTMLParser()  # strict=False
            html_parser.feed(l)

            iurl = html_parser.links[0].strip() if len(html_parser.links) else None
            nolinks = html_parser.data + html_parser.rawdata
            mo = parser.match(nolinks)
            vals = list(map(str.strip, mo.groups()))
            #name = mo.groupdict()['name']

            success.append(vals)
            txt.append(html_parser.data)
            image_urls.append(iurl)
            links.append(html_parser.links)

        except Exception as err:
            # print(l, '\n', err)
            fail.append(l)
            ixFail.append(i)
            flinks.append(html_parser.links)

    print('%s lines successfully parsed' % len(success))
    print('%s lines failed' % len(fail))

    return success, colNames,  image_urls, links, fail, ixFail

def convert_date(datestr):
    y, mn, d = datestr.split()
    mn = mn.strip('.').replace('Spt', 'Sep').replace('Mrt', 'Mar')
    d = float(d)
    ds = str(int(d))
    #h = (float(d) - di) * 24
    fixed = ' '.join((y, mn, ds))
    try:
        return datetime.strptime(fixed, '%Y %b %d')
    except ValueError as err:
        pass
    return datetime.strptime(fixed, '%Y %B %d')


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
    success, colNames, image_urls, links, fail, ixFail = parse_master()
    masterDb = np.rec.fromrecords(success, names=colNames)

    coords = Jparser.many2deg(masterDb.name)
    t, alt, lalt = is_observable(coords)

    # Type filter
    cvtypes = ('DN', 'NL', 'CV')
    ltyp = np.any([masterDb.type == typ for typ in cvtypes], 0)

    # Recentness filter
    discovered = Time(np.vectorize(convert_date)(masterDb.date))
    lrec = (Time.now() - discovered).to('d').value < 100

    l = ltyp & lalt & lrec



