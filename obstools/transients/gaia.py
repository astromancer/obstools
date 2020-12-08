import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from astropy.table import Table

from . import fmt_ra, fmt_dec


def iter_data(csvfile):
    with Path(csvfile).open() as fp:
        for row in csv.reader(fp):
            if row:
                yield row


def load(csvfile):

    data = list(iter_data(csvfile))
    names = list(np.char.array(data[0]).strip('# '))
    data = np.array(data[1:])

    # use float dtype for numeric columns
    dtype = [f'<U{w}' for w in np.char.str_len(data).max(0)]
    for name in ['RaDeg', 'DecDeg', 'AlertMag']:
        dtype[names.index(name)] = float

    # create Table
    tbl = Table(np.array(data), names=names, dtype=dtype, masked=True)
    tbl['RaDeg'].format = fmt_ra
    tbl['DecDeg'].format = fmt_dec

    # masked float column
    hm = np.ma.array(tbl['HistoricMag'])
    l = (hm == '')
    hm[l] = -1
    hm = np.ma.MaskedArray(np.array(hm).astype(float), l)
    tbl.replace_column('HistoricMag', hm)

    # rename columns
    tbl.rename_columns(['Name', 'Date', 'RaDeg', 'DecDeg'],
                       ['name', 'date', 'ra', 'dec'])

    # dates
    tbl.replace_column('date', np.vectorize(
        datetime.fromisoformat)(tbl['date']))

    return tbl
