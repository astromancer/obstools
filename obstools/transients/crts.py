from pathlib import Path
from datetime import datetime

import numpy as np
from astropy.table import Table, vstack


def load(folder):
    """
    Read the html tables and merge

    http://crts.caltech.edu/
    """

    tables = []
    folder = Path(folder)
    for filename in folder.glob('*.html'):
        tbl = Table.read(filename, format='html')

        # make "del mag" column str type so it stacks without error
        tbl.replace_column('del mag', tbl['del mag'].astype(str))
        # rename XXX imgs column
        tel = filename.stem.split('_')[-1].strip('2')
        img = f'{tel} imgs'
        if img in tbl.colnames:
            tbl.rename_column(img, 'imgs')
        # ensure consistent column names
        if 'SSS ID' in tbl.colnames:
            tbl.rename_column('SSS ID', 'CRTS ID')

        tables.append(tbl)

    # aggregate
    crts = vstack(tables)

    # rename columns
    crts.rename_column('CRTS ID', 'name')
    crts.rename_column('RA (J2000)', 'ra')
    crts.rename_column('Dec (J2000)', 'dec')
    crts.rename_column('Date', 'date')

    # date
    crts['date'] = np.vectorize(get_date)(crts['date'].astype(str))

    return crts


def get_date(item):
    return datetime.strptime(item, '%Y%m%d')


if __name__ == '__main__':

    folder = '/home/hannes/Desktop/PhD/thesis/build/ch1/scripts/rate/data/'
    crts = load(folder)
