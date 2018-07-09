import logging
import numbers
import motley
import numpy
from motley.progress import ProgressBar
from motley.table import Table
from recipes.logging import LoggingMixin
from recipes.string import get_module_name

import numpy as np

# module level logger
logger = logging.getLogger(get_module_name(__file__))


def null_func(*args):
    pass


class ProgressLogger(ProgressBar, LoggingMixin):
    # def __init__(self, **kws):
    #     ProgressBar.__init__(self, **kws)
    #     if not log_progress:
    #         self.progress = null_func

    def create(self, end):
        self.end = end
        self.every = np.ceil((10 ** -(self.sigfig + 2)) * self.end)
        # only have to update text every so often

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
        global ProgressLogger  # not sure why this is needed

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
                   row_headers=range(len(coo)),  # starts numbering from 0
                   # number_rows=True,
                   align='>',  # easier to read when right aligned
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
    tt = Table(tags, title='\n',  # title,   # title_props=None,
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
    # sdist[np.tril_indices(n)] = np.inf # since the distance matrix is symmetric, ignore lower half
    # mask = (sdist == np.inf)

    # create distance matrix as table, highlighting stars that are potentially too close together
    #  and may cause problems
    bg = 'light green'
    # tbldat = np.ma.array(sdist, mask=mask, copy=True)
    tbl = Table(sdist,  # tbldat,
                title='Distance matrix',
                col_headers=range(n),
                row_headers=range(n),
                col_head_props=dict(bg=bg),
                row_head_props=dict(bg=bg),
                align='>')

    if sdist.size > 1:
        # FIXME: MaskError: Cannot convert masked element to a Python int.
        # /home/hannes/work/motley/table.py in colourise(self, states, *colours, **kws)
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
        tbl.flag_headers(c, bg=[bg] * 3, fg='wyr')

    if _print and n > 1:
        print(tbl)

    return tbl  # , c


class ImageSampler(object):
    def __init__(self, data, sample_size=None):
        assert data.ndim == 3
        self.data = data
        self.default_sample_size = sample_size
        self._axis = 0

    def sample(self, n=None, subset=None):
        """
        Select a sample of `n` images randomly in the interval `subset`

        Parameters
        ----------
        n
        subset

        Returns
        -------

        """

        if n is None and self.sample_size is None:
            raise ValueError('Please give sample size (or initialize this '
                             'class with a sample size')

        if isinstance(subset, numbers.Integral):
            subset = (0, subset)  # treat like a slice

        #
        i0, i1 = subset

        # get frame indices
        # nfirst = min(n, i1 - i0)
        ix = np.random.randint(i0, i1, n)
        # create median image for init
        # logger.info('Selecting %i frames from amongst frames (%i->%i) for '
        #             'sample image.', n, i0, i1)
        return self.data[ix]

    def max(self, n=None, subset=None):
        return self.sample(n, subset).max(self._axis)

    def mean(self, n=None, subset=None):
        return self.sample(n, subset).mean(self._axis)

    def std(self, n=None, subset=None):
        return self.sample(n, subset).std(self._axis)

    def median(self, n=None, subset=None):
        return np.median(self.sample(n, subset), self._axis)


def rand_median(cube, ncomb, subset, nchoose=None):
    """
    median combine `ncomb`` frames randomly from amongst `nchoose` in the interval
    `subset`

    Parameters
    ----------
    cube
    ncomb
    subset
    nchoose

    Returns
    -------

    """
    if isinstance(subset, int):
        subset = (0, subset)  # treat like a slice

    i0, i1 = subset
    if nchoose is None:  # if not given, select from entire subset
        nchoose = i1 - i0

    # get frame indices
    nfirst = min(nchoose, i1 - i0)
    ix = np.random.randint(i0, i0 + nfirst, ncomb)
    # create median image for init
    logger.info('Combining %i frames from amongst frames (%i->%i) for '
                'reference image.', ncomb, i0, i0 + nfirst)
    return np.median(cube[ix], 0)


def duplicate_if_scalar(seq):
    """

    Parameters
    ----------
    seq : {number, array-like}

    Returns
    -------

    """
    # seq = np.atleast_1d(seq)
    if np.size(seq) == 1:
        seq = np.ravel([seq, seq])
    if np.size(seq) != 2:
        raise ValueError('Input should be of size 1 or 2')
    return seq
