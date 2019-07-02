"""
Miscellaneous utility functions
"""

# builtin libs
from IPython import embed
from recipes.dict import AttrReadItem, ListLike

import logging
import numbers
import itertools as itt

# third-party libs
import numpy as np
import more_itertools as mit
from scipy import ndimage
from scipy.stats import binned_statistic_2d

# local libs
import motley
from motley.table import Table
from motley.progress import ProgressBar
from recipes.logging import LoggingMixin
from recipes.introspection.utils import get_module_name

# module level logger
logger = logging.getLogger(get_module_name(__file__))


def null_func(*_):
    pass


def iter_repeat_last(it):
    """
    Yield items from the input iterable and repeat the last item indefinitely
    """
    it, it1 = itt.tee(mit.always_iterable(it))
    return mit.padded(it, next(mit.tail(1, it1)))


# class ContainerAttrGetterMixin(object):
#     """"""
#     # def __init__(self, data):
#     #     self.data = data
#     #     # types = set(map(type, data.flat))
#     #     # assert len(types) == 1
#
#     def __getattr__(self, key):
#         if hasattr(self, key):
#             return super().__getattr__(key)
#
#         # if hasattr(self.data[0, 0], key)
#         getter = operator.attrgetter(key)
#         return np.vectorize(getter, 'O')(self)


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
    # sdist[np.tril_indices(n)] = np.inf
    #  since the distance matrix is symmetric, ignore lower half
    # mask = (sdist == np.inf)

    # create distance matrix as table, highlighting stars that are potentially
    # too close together and may cause problems
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


def shift_combine(images, offsets, stat='mean', extend=False):
    """
    Statistics on shifted image stack

    Parameters
    ----------
    images
    offsets
    stat
    extend

    Returns
    -------

    """
    # convert to (masked) array
    images = np.asanyarray(images)
    offsets = np.asanyarray(offsets)

    # it can happen that `offsets` is masked (no stars in that frame)
    if np.ma.is_masked(offsets):
        # ignore images for which xy offsets are masked
        bad = offsets.mask.any(1)
        good = ~bad

        logger.info(f'Removing {bad.sum()} images from stack due to null '
                    f'detection')
        images = images[good]
        offsets = offsets[good]

    # get pixel grid ignoring masked elements
    shape = sy, sx = images.shape[1:]
    grid = np.indices(shape)
    gg = grid[:, None] - offsets[None, None].T
    if np.ma.is_masked(images):
        y, x = gg[:, ~images.mask]
        sample = images.compressed()
    else:
        y, x = gg.reshape(2, -1)
        sample = images.ravel()

    # use maximal area coverage. returned image may be larger than input images
    if extend:
        y0, x0 = np.floor(offsets.min(0))
        y1, x1 = np.ceil(offsets.max(0)) + shape + 1
    else:
        # returned image same size as original
        y0 = x0 = 0
        y1, x1 = np.add(shape, 1)

    # compute statistic
    yb, xb = np.ogrid[y0:y1, x0:x1]
    bin_edges = (yb.ravel() - 0.5, xb.ravel() - 0.5)
    results = binned_statistic_2d(y, x, sample, stat, bin_edges)
    image = results.statistic
    # mask nans (empty bins (pixels))

    # note: avoid downstream warnings by replacing np.nan with zeros and masking
    nans = np.isnan(image)
    image[nans] = 0
    return np.ma.MaskedArray(image, nans)


class ImageSampler(object):
    def __init__(self, data, sample_size=None):
        assert data.ndim == 3
        self.data = data
        self._axis = 0
        self.sample_size = sample_size
        # self.scaling_func = np.median
        # self.scaling = False

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
        logger.debug('Selecting %i frames from amongst frames (%i->%i) for '
                     'sample image.', n, i0, i1)
        return self.data[ix]

    def max(self, n=None, subset=None):
        return self.sample(n, subset).max(self._axis)

    def mean(self, n=None, subset=None):
        return self.sample(n, subset).mean(self._axis)

    def std(self, n=None, subset=None):
        return self.sample(n, subset).std(self._axis)

    def median(self, n=None, subset=None):
        return np.median(self.sample(n, subset), self._axis)

    # def create_sample_image(self, interval, sample_size, statistic=np.median):
    #     image = self.median(sample_size, interval)
    #     # todo can add scaling into sampler ?
    #     # scale = nd_sampler(image, np.median, 100)
    #     # image_NORM = image / scale
    #     mimage = np.ma.MaskedArray(image, BAD_PIXEL_MASK)
    #     return mimage  # , scale


class ImageSamplerHDUMixin(object):
    def get_sample_image(self, size, interval=(None,), func='median',
                         channel=...):
        sampler = ImageSampler(self.data[slice(*interval), channel])
        return getattr(sampler, func)(size)


class Record(AttrReadItem, ListLike):
    """
    Ordered dict with key access via attribute lookup. Also has some
    list-like functionality: indexing by int and appending new data.
    Best of both worlds.
    """
    pass


class LabelGroups(Record):
    """
    Makes sure values (labels) are always arrays.
    """
    _auto_name_fmt = 'group%i'

    def _allow_item(self, item):
        return bool(len(item))

    def _convert_item(self, item):
        return np.atleast_1d(item).astype(int)

    @property
    def sizes(self):
        return list(map(len, self.values()))
        # return [len(item) for item in self.values()]

    # def inverse(self):
    #     return {lbl: gid for gid, labels in self.items() for lbl in labels}


class LabelGroupsMixin(object):
    """Mixin class for grouping and labelling image segments"""

    def __init__(self, groups=None):
        self._groups = None
        self.set_groups(groups)

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, groups):
        self.set_groups(groups)

    def set_groups(self, groups):
        self._groups = LabelGroups(groups)

    # todo
    # def remove_group()
