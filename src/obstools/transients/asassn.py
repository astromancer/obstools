# third-party
import numpy as np
from astropy.time import Time
from astropy.table import Table

# relative
from .utils import fmt_dec, fmt_ra


def load(filename):
    # filter asassn cvs
    data = np.genfromtxt(filename,
                         dtype=None, names=True,
                         delimiter=',', encoding='utf8')

    tbl = Table(data)
    tbl.rename_column('ASASSN_Name', 'name')
    tbl.rename_column('RAJ2000', 'ra')
    tbl.rename_column('DEJ2000', 'dec')
    tbl.add_column(Time(tbl['EpochHJD'], format='jd').datetime,
                   tbl.colnames.index('EpochHJD'),
                   'date')

    tbl['ra'].format = fmt_ra
    tbl['dec'].format = fmt_dec

    return tbl
