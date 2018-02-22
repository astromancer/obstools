import motley
from motley.progress import ProgressBar
from motley.table import Table
from recipes.logging import LoggingMixin

import numpy as np


def mad(a):
    return np.median(np.abs(a - np.median(a)))


def null_func(*args):
    pass


class ProgressLogger(ProgressBar, LoggingMixin):
    # def __init__(self, **kws):
    #     ProgressBar.__init__(self, **kws)
    #     if not log_progress:
    #         self.progress = null_func

    def create(self, end):
        self.end = end
        self.every = np.ceil((10 ** -(self.sigfig + 2)) * self.end)  # only have to updat text every so often

    def progress(self, state, info=None):
        if self.needs_update(state):
            bar = self.get_bar(state)
            self.logger.info('Progress: %s' % bar)


# class ProgressPrinter(ProgressBar):
#     def __init__(self, **kws):
#         ProgressBar.__init__(self, **kws)
#         if not print_progress:
#             self.progress = self.create = null_func

def progressFactory(log=True, print_=True):
    if not log:
        global ProgressLogger   # not sure why this is neded
        class ProgressLogger(ProgressLogger):
            progress = null_func

    if not print_:
        class ProgressPrinter(ProgressBar):
            progress = create = null_func

    return ProgressLogger, ProgressBar



def table_coords(coo, ix_fit, ix_scale, ix_loc):

    # TODO: maybe add flux estimate

    # create table: coordinates
    ocoo = np.array(coo[:, ::-1], dtype='O')
    cootbl = Table(ocoo,
                col_headers=list('xy'),
                col_head_props=dict(bg='g'),
                row_headers=range(len(coo)),    #starts numbering from 0
                # number_rows=True,
                align='>', # easier to read when right aligned
                )

    # add colour indicators for tracking / fitting / scaling info
    ms = 2
    m = np.zeros((len(coo) + 1, 3), int)
    # m[0] = 1, 2, 3

    for i, ix in enumerate((ix_fit, ix_scale, ix_loc)):
        m[ix, i] = i + 1

    # flag stars
    cols = 'gbm'
    labels = ('fit|', 'scale|', 'loc|')
    tags = np.empty(m.shape, dtype='U1')
    tags[m != 0] = 'x'
    # tags[:] = 'x' * ms


    col_headers = motley.rainbow(labels, bg=cols)
    tt = Table(tags,  title='\n',  #title,   # title_props=None,
               col_headers=col_headers,
               frame=False, align='^',
               col_borders='', cell_whitespace=0)
    tt.colourise(m, fg=cols)
    # ts = tt.add_colourbar(str(tt), ('fit|', 'scale|', 'loc|'))

    # join tables
    tbl = Table([[str(cootbl), str(tt)]], frame=False, col_borders='')
    return tbl


def table_cdist(sdist, window, _print=False):
    # from scipy.spatial.distance import cdist
    n = len(sdist)
    # check for stars that are close together
    # sdist = cdist(coo, coo)  # pixel distance between stars
    #sdist[np.tril_indices(n)] = np.inf # since the distance matrix is symmetric, ignore lower half
    #mask = (sdist == np.inf)

    # create distance matrix as table, highlighting stars that are potentially too close together
    #  and may cause problems
    bg = 'light green'
    # tbldat = np.ma.array(sdist, mask=mask, copy=True)
    tbl = Table(sdist, #tbldat,
                title='Distance matrix',
                col_headers=range(n),
                row_headers=range(n),
                col_head_props=dict(bg=bg),
                row_head_props=dict(bg=bg),
                align='>')

    if sdist.size > 1:

        # FIXME: MaskError: Cannot convert masked element to a Python int.
        #/home/hannes/work/motley/table.py in colourise(self, states, *colours, **kws)
        #     651         #
        #     652
        # --> 653         propList = motley.get_state_dicts(states, *colours, **kws)
        #     654
        #     655         # propIter = motley._prop_dict_gen(*colours, **kws)
        #
        # /home/hannes/work/motley/ansi.py in get_state_dicts(states, *effects, **kws)
        #     132     nprops = len(propList)
        #     133     nstates = states.max() #ptp??
        # --> 134     istart = int(nstates - nprops + 1)
        #     135     return ([{}] * istart) + propList
        #     136
        #
        # /usr/local/lib/python3.5/dist-packages/numpy-1.14.0.dev0+39117c5-py3.5-linux-x86_64.egg/numpy/ma/core.py in __int__(self)
        #    4304                             "to Python scalars")
        #    4305         elif self._mask:
        # -> 4306             raise MaskError('Cannot convert masked element to a Python int.')
        #    4307         return int(self.item())
        #    4308
        #

        # Add colour as distance warnings
        c = np.zeros_like(sdist)
        c += (sdist < window / 2)
        c += (sdist < window)
        tbl.colourise(c, *' yr')
        tbl.show_colourbar = False
        tbl.flag_headers(c, bg=[bg]*3, fg='wyr')

    if _print and n > 1:
        print(tbl)

    return tbl#, c