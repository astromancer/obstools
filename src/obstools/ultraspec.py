
# std libs
import re
import itertools as itt
from collections import defaultdict

# third-party libs
from astropy.table import vstack
from astropy.io.ascii import read
# import warnings
# from pathlib import Path


# from recipes.io import read_file_line

# from tsa.tfr import TimeFrequencyRepresentation as TFR

# from ansi.str import banner


# from decor import print_args


def read_data(filepath):
    table = read(str(filepath))
    set_column_names(table)

    return table


def set_column_names(table):
    namesline = next(
        filter(lambda l: l.startswith('name'), table.meta['comments']))
    namesline = namesline.strip('#').rstrip('*num_aper\n')
    cols1, _, rcols = namesline.partition('[')
    cols1 = cols1.replace('name/', '').split()
    rcols = rcols.strip(']').split()

    Ncols = len(table.columns)  # number of columns
    Nstars = (Ncols - len(cols1)) // len(rcols)
    colnames = cols1[:]
    for i in map(str, range(1, Nstars + 1)):
        colnames.extend(map(''.join, zip(rcols, itt.repeat(i))))

    for oldname, newname in zip(table.colnames, colnames):
        table[oldname].name = newname


def as2Dfloat(table, colid):
    """
    Extaract data for all columns starting with colid return as 2D array of floats shape (Naps, Ndata)
    """
    colnames = tuple(filter(lambda c: c.startswith(colid), table.colnames))
    # tuple(colid+str(i) for i in range(1,Naps+1))
    a = table[colnames].as_array()
    dtype = list(zip(a.dtype.names, (float,) * len(colnames)))
    return a.astype(dtype).view(float).reshape(len(table), -1)


get_multiple_columns = as2Dfloat


def get_target_name(header):
    matcher = re.compile(r'target\s+=\s+([\w\s]+)')
    hline = next(filter(matcher.match, header))
    return matcher.match(hline).groups()[0].strip()


def read_list(filelist, bad=(), stack=True, keyed_on='target'):
    # Load data
    # path = Path('/media/Oceanus/UCT/Observing/TNO/data_20151108')
    # bad = ()  # 'run023_RXJ0325.log', 'run019_RXJ0325.log'

    Tables = defaultdict(list)
    for filepath in filelist:
        if str(filepath).endswith(bad):
            continue

        tbl = read_data(filepath)
        if keyed_on == 'target':
            key = get_target_name(tbl.meta['comments'])
        elif keyed_on == 'filename':
            key = str(filepath.name)
        else:
            raise ValueError('keyed_on argument %r invalid' % keyed_on)

        Tables[key].append(tbl)

        # extract target name from filename
        # tffn = re.match('.*run[\d]{3}_(\w+)\.log', str(filepath)).groups()[0]
        # banner(target, tffn, swoosh='=', bg='cyan', width=80)

    # Concatenate and sort
    if stack:
        for key, tbls in Tables.items():
            tbl = Tables[key] = vstack(tbls)
            tbl.sort('mjd')

    return Tables
