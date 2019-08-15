"""
Extensions for segmentation images
"""

# std libs
import types
import inspect
import logging
import warnings
import functools
import itertools as itt
from operator import attrgetter

# third-party libs
import numpy as np
from scipy import ndimage
from astropy.utils import lazyproperty
from photutils.detection.core import detect_threshold
from photutils.segmentation.detect import detect_array
from photutils.segmentation import SegmentationImage, Segment

# local libs
from motley.table import Table
from recipes.logging import LoggingMixin
from recipes.pprint.misc import seq_repr_trunc
from recipes.introspection.utils import get_module_name
from obstools.modelling import UnconvergedOptimization
from obstools.phot.utils import (iter_repeat_last, duplicate_if_scalar,
                                 shift_combine,
                                 LabelGroupsMixin)

# from obstools.modelling.image import SegmentedImageModel
# from collections import namedtuple

# TODO: watershed segmentation on the negative image ?
# TODO: detect_gmm():


# module level logger


logger = logging.getLogger(get_module_name(__file__))


def methodize(func, instance):
    return types.MethodType(func, instance)


def is_lazy(_):
    return isinstance(_, lazyproperty)


def detect(image, mask=False, background=None, snr=3., npixels=7,
           edge_cutoff=None, deblend=False):
    """
    Image detection that returns a SegmentationHelper instance

    Parameters
    ----------
    image
    mask
    background
    snr
    npixels
    edge_cutoff:
        only used for threshold calculation
    deblend
    dilate

    Returns
    -------

    """

    logger.debug('Running detect(snr=%.1f, npixels=%i)', snr, npixels)

    if mask is None:
        mask = False  # need this for logical operators below to work

    # calculate threshold without edges so that std accurately measured for
    # region of interest
    if edge_cutoff:
        border = make_border_mask(image, edge_cutoff)
        mask = mask | border

    # separate pixel mask for threshold calculation (else the mask gets
    # duplicated to threshold array, which will skew the detection stats)
    if np.ma.isMA(image):
        mask = mask | image.mask
        image = image.data

    # # check mask reasonable
    # if mask.sum() == mask.size:

    # detection
    threshold = detect_threshold(image, snr, background, mask=mask)
    if mask is False:
        mask = None  # annoying photutils #HACK
    seg = detect_array(image, threshold, npixels, mask=mask)

    # check if anything detected
    no_sources = (np.sum(seg) == 0)
    if no_sources:
        logger.debug('No objects detected')

    if deblend and not no_sources:
        from photutils import deblend_sources
        seg = deblend_sources(image, seg, npixels).data

    # intentionally return an array
    return seg


def detect_measure(image, mask=False, background=None, snr=3., npixels=7,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)

    # for images with strong gradients, local median in annular region around
    # source is a better background estimator. Accurate count estimate here
    # is relatively important since we edit the background mask based on
    # source fluxes to account for photon bleed during frame transfer
    # bg = [np.median(image[region]) for region in
    #       seg.to_annuli(width=sky_width)]
    # bg = seg.median(image, 0)
    # sum = seg.sum(image) - bg * seg.areas
    return seg, seg.com_bg(image)  # , counts


class MultiThresholdBlobDetection(object):
    # algorithm defaults
    snr = (10, 7, 5, 3)
    npixels = (7, 5, 3)
    deblend = (True, False)
    dilate = (4, 2, 1)
    edge_cutoff = None
    max_iter = np.inf

    def __call__(self, *args, **kwargs):
        assert False, 'TODO'


def detect_loop(image, mask=None, snr=(10, 7, 5, 3), npixels=(7, 5, 3),
                deblend=(True, False), dilate=(4, 2, 1), edge_cutoff=None,
                max_iter=np.inf, bg_model=None, opt_kws=None, report=None):
    """
    Multi-threshold image blob detection, segmentation and grouping. This
    function runs multiple iterations of the blob detection algorithm on the
    same image, masking new sources after each round so that progressively
    fainter sources may be detected. By default the algorithm will continue
    looping until no new sources are detected. The number of iterations in
    the loop can also be controlled by specifying `max_iter`.  The arguments
    `snr`, `npixels`, `deblend`, `dilate` can be sequences, in which case each
    new detection round will use the next value in the sequence until the last
    value which will then be repeated for subsequent iterations. If scalar,
    this value will be used repeatedly for each round.

    A background model may also be provided.  This model will be fit to the
    image background region after each round of detection.  Control model
    optimization by passing `opt_kws` dict.

    Parameters
    ----------
    image
    mask
    snr
    npixels
    deblend
    dilate
    edge_cutoff
    max_iter
    min_iter
    bg_model
    opt_kws
    report

    Returns
    -------
    seg
    groups
    info
    result
    residual

    """
    # short circuit
    if max_iter == 0:
        # seg, groups, info, result, residual
        return np.zeros(image.shape, int), [], [], None, image

    # log
    logger.info('Running detection loop')
    lvl = logger.getEffectiveLevel()
    debug = logger.getEffectiveLevel() >= logging.DEBUG
    if report is None:
        report = lvl <= logging.INFO

    # make iterables
    var_names = ('snr', 'npixels', 'dilate', 'deblend')
    var_iters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
    var_gen = zip(*var_iters)

    # original_mask = mask
    if mask is None:
        mask = np.zeros(image.shape, bool)

    # first round detection without background model
    # opt_kws.setdefault('method', 'leastsq')
    opt_kws = opt_kws or {}
    residual = image
    result = None

    # get segmentation image
    seg = SegmentationHelper(np.zeros(image.shape, int))

    # label groups
    # keep track of group info + detection meta data
    groups = []
    info = []
    gof = []

    # detection loop
    counter = itt.count(0)
    while True:
        # keep track of iteration number
        count = next(counter)
        logger.debug('count %i', count)

        if count >= max_iter:
            logger.debug('break: max_iter reached')
            break

        # detect on residual image
        # noinspection PyTupleAssignmentBalance
        _snr, _npix, _dil, _debl = opts = next(var_gen)
        new_seg = seg.detect(residual, mask, None, _snr, _npix, edge_cutoff,
                             _debl, _dil)

        if not new_seg.data.any():
            logger.debug('break: no new detections')
            break

        # aggregate
        if count > 0:
            _, new_labels = seg.add_segments(new_seg)
        else:
            seg = new_seg
            new_labels = new_seg.labels

        # debug log!
        if debug:
            logger.debug('detect_loop: round %i: %i new detections: %s',
                         count, len(new_labels),
                         seq_repr_trunc(tuple(new_labels)))

        # update mask
        mask = mask | new_seg.to_bool()

        # group info
        groups.append(new_labels)  # groups[opts].extend(new_labels) ??
        info.append(dict(zip(var_names, opts)))

        if bg_model:
            # fit model, get residuals
            mimage = np.ma.MaskedArray(image, mask)
            try:
                result = bg_model.fit(mimage, **opt_kws)
                residual = bg_model.residuals(result, image)
                if report:
                    gof.append(bg_model.redchi(result, mimage))
            except UnconvergedOptimization as err:
                logger.info('Model optimization unsuccessful. Returning.')
                break
        # else:
        #     gof.append(None)

    # def report_detection(groups, info, bg_model, gof):
    if report:
        # log what you found
        from recipes import pprint

        seq_repr = functools.partial(seq_repr_trunc, max_items=3)

        # report detections here
        col_headers = ['snr', 'npix', 'dil', 'debl', 'n_obj', 'labels']
        info_list = list(map(list, map(dict.values, info)))
        tbl = np.column_stack([np.array(info_list, 'O'),
                               list(map(len, groups)),
                               list(map(seq_repr, groups))])
        if bg_model:
            col_headers.insert(-1, 'χ²ᵣ')
            tbl = np.insert(tbl, -1, list(map(pprint.numeric, gof)), 1)

        title = 'Object detections'
        if bg_model:
            title += f' with {bg_model.__class__.__name__} model'

        tbl_ = \
            Table(tbl,
                  title=title,
                  col_headers=col_headers,
                  total=(4,), minimalist=True)
        logger.info(f'\n{tbl_}')
        # print(logger.name)

    return seg, groups, info, result, residual


class SourceDetectionMixin(object):
    """
    Provides the `from_image` classmethod and the `detect` staticmethod that
    can be used to construct image models from images.
    """

    @classmethod
    def detect(cls, image, detect_sources, **kws):
        """

        Parameters
        ----------
        image
        detect_sources: bool
            Controls whether source detection algorithm is run. This argument
            provides a shortcut to the default source detection by setting
            `True`, or alternatively to skip source detection by setting
            `False`
        kws:
            Keywords for source detection algorithm

        Returns
        -------

        """
        # Detect objects & segment image
        if isinstance(detect_sources, dict):
            kws.update(detect_sources)

        if not ((detect_sources is True) or kws):
            kws['max_iter'] = 0  # short circuit

        return cls._detect(image, **kws)

    @classmethod
    def _detect(cls, image, *args, **kws):
        """
        Default blob detection algorithm.  Subclasses can override as needed
        """
        return detect_loop(image, *args, **kws)

    @classmethod
    def from_image(cls, image, detect_sources=True, **detect_opts):
        """
        Construct a instance of this class from an image.
        Sources in the image will be identified using `detect` method.
        Segments for detected sources will be added to the segmented image.
        Source groups will be added to `groups` attribute dict.


        Parameters
        ----------
        image
        detect_sources

        detect_opts

        Returns
        -------

        """

        # Basic constructor that initializes the model from an image. The
        # base version here runs a detection algorithm to separate foreground
        # objects and background, but doesn't actually include any physically
        # useful models. Subclasses can overwrite this method to add useful
        # models to the segments.

        # Detect objects & segment image
        seg, groups, info, result, residual = cls.detect(image, detect_sources,
                                                         **detect_opts)

        mdl = cls(seg)

        # add detected sources
        mdl.groups.update(groups)

        return mdl


def merge_segmentations(segmentations, xy_offsets, extend=True, f_accept=0.2,
                        post_merge_dilate=1):
    # merge detections masks by align, summation, threshold
    if isinstance(segmentations, (list, tuple)) and \
            isinstance(segmentations[0], SegmentationImage):
        segmentations = np.array([seg.data for seg in segmentations])
    else:
        segmentations = np.asarray(segmentations)

    n_images = len(segmentations)
    n_accept = max(f_accept * n_images, 1)

    eim = shift_combine(segmentations.astype(bool), xy_offsets, 'sum',
                        extend=extend)
    seg_image_extended, n_stars = ndimage.label(eim >= n_accept)

    seg_extended = SegmentationHelper(seg_image_extended)
    seg_extended.dilate(post_merge_dilate)
    return seg_extended


def select_rect_pad(segm, image, start, shape):
    """
    Get data from a sub-region of dimension `shape` from `image`
    beginning at index `start`. If the requested shape of the image
    is such that the image only partially overlaps with the data in the
    segmentation image, fill the non-overlapping parts with zeros (of
    the same dtype as the image)

    Parameters
    ----------
    image
    start
    shape

    Returns
    -------

    """
    if np.ma.is_masked(start):
        raise ValueError('Cannot select image subset for masked `start`')

    hi = np.array(shape)
    δtop = segm.shape - hi - start
    over_top = δtop < 0
    hi[over_top] += δtop[over_top]
    low = -np.min([start, (0, 0)], 0)
    oseg = tuple(map(slice, low, hi))

    # adjust if beyond limits of global segmentation
    start = np.max([start, (0, 0)], 0)
    end = start + (hi - low)
    iseg = tuple(map(slice, start, end))
    if image.ndim > 2:
        iseg = (...,) + iseg
        oseg = (...,) + oseg
        shape = (len(image),) + shape

    sub = np.zeros(shape, image.dtype)
    sub[oseg] = image[iseg]
    return sub


def inside_segment(coords, sub, grid):
    b = []
    ogrid = grid[0, :, 0], grid[1, 0, :]
    for j, (g, f) in enumerate(zip(ogrid, coords)):
        bi = np.digitize(f, g - 0.5)
        b.append(bi)

    mask = (sub == 0)
    if np.equal(grid.shape[1:], b).any() or np.equal(0, b).any():
        inside = False
    else:
        inside = not mask[b[0], b[1]]

    return inside


def _make_border_mask(data, xlow=0, xhi=None, ylow=0, yhi=None):
    """Edge mask"""
    mask = np.zeros(data.shape, bool)

    mask[:ylow] = True
    if yhi is not None:
        mask[yhi:] = True

    mask[:, :xlow] = True
    if xhi is not None:
        mask[:, xhi:] = True
    return mask


def make_border_mask(data, edge_cutoffs):
    if isinstance(edge_cutoffs, int):
        return _make_border_mask(data,
                                 edge_cutoffs, -edge_cutoffs,
                                 edge_cutoffs, -edge_cutoffs)
    edge_cutoffs = tuple(edge_cutoffs)
    if len(edge_cutoffs) == 4:
        return _make_border_mask(data, *edge_cutoffs)

    raise ValueError('Invalid edge_cutoffs %s' % edge_cutoffs)


def _2d_slicer(array, slice_, mask=None, compress=False):
    """
    Slice `np.ndarray` object along last 2 dimensions

    Parameters
    ----------
    array
    slice_
    mask
    extract

    Returns
    -------
    array or None
    """

    # print('!!!!!!!', array.shape, slice_, mask)

    if array is None:
        return None  # propagate `None`s

    # slice along last two dimensions
    slice_ = (Ellipsis,) + tuple(slice_)
    sub = array[slice_]

    if mask is None or mask is False:
        return sub

    if compress:
        return sub[..., ~mask]

    ma = np.ma.MaskedArray(sub, copy=True)
    ma[..., mask] = np.ma.masked
    return ma


class ModelledSegment(Segment):
    def __init__(self, segment_img, label, slices, area, model=None):
        super().__init__(segment_img, label, slices, area)
        self.model = model
        # self.grid =
        #


class SegmentedArray(np.ndarray):
    """
    Array subclass for keeping image segmentation data. Keeps a reference to
    the `SegmentationHelper` object that created it so that changing the
    segmentation array data triggers the lazyproperties to recompute the next
    time they are accessed.

    # note inplace operations on array will not trigger reset of parent
    # lazyproperties

    """

    def __new__(cls, input_array, parent):
        # Input data is array-like structure
        obj = np.array(input_array)

        # initialize with data
        super_ = super(SegmentedArray, cls)
        obj = super_.__new__(cls, obj.shape, int)
        super_.__setitem__(obj, ..., input_array)  # populate entire array

        # add SegmentationHelper instance as attribute to be updated upon
        # changes to segmentation data
        obj.parent = parent  # FIXME: this will be missed for new-from-template
        return obj

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        # set the data in the SegmentationHelper
        # print('Hitting up set data')
        self.parent.data = self

    def __reduce__(self):
        return SegmentedArray, (np.array(self), self.parent)

    #     constructor, init_args, *rest = np.ndarray.__reduce__(self)

    # def __getnewargs__(self):
    #     # These will be passed to the __new__() method upon unpickling.
    #     print('HI!!!!! ' * 10)
    #     return self, self.parent

    def __array_finalize__(self, obj):
        #
        if obj is None:
            # explicit constructor:  `SegmentedArray(data)`
            return

        # view casting or new-from-template constructor
        if hasattr(obj, 'parent'):
            self.parent = obj.parent

    def __array_wrap__(self, out_arr, context=None):
        # return a plain old array so we don't accidentally reset the parent
        # lazyproperties by edits on data of derived array
        return np.array(out_arr)

        # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        #     # return a plain old array so we don't accidentally reset the parent
        #     # lazyproperties by edits on data of derived array
        #     result = super(SegmentedArray, self).__array_ufunc__(ufunc, method,
        #                                                          *inputs, **kwargs)
        #     return np.array(result)

        # class Slices(np.recarray):
        #     # maps semantic corner positions to slice attributes
        #     _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}
        #
        #     def __new__(cls, parent):
        #         # parent in SegmentationHelper instance
        #         # get list of slices from super class SegmentationImage
        #         slices = SegmentationImage.slices.fget(parent)
        #         if parent.use_zero:
        #             slices = [(slice(None), slice(None))] + slices
        #
        #         # initialize np.ndarray with data
        #         super_ = super(np.recarray, cls)
        dtype = np.dtype(list(zip('yx', 'OO')))


#         obj = super_.__new__(cls, len(slices), dtype)
#         super_.__setitem__(obj, ..., slices)
#
#         # add SegmentationHelper instance as attribute
#         obj.parent = parent
#         return obj
#
#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         return NotImplemented

# simple container for 2-component objects
# yxTuple = namedtuple('yxTuple', ['y', 'x'])

import textwrap


# class Segments(list):
#     # maps semantic corner positions to slice attributes
#     _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}
#
#     def __repr__(self):
#         return ''.join((self.__class__.__name__, super().__repr__()))
#
#     def attrgetter(self, *attrs):
#         getter = operator.attrgetter(*attrs)
#         return list(map(getter, self))
#
#     # def of_labels(self, labels=None):
#     #     return self.seg.get_slices(labels)
#
#     def _get_corners(self, vh, slices):
#         # vh - vertical horizontal positions as two character string
#         yss, xss = (self._corner_slice_mapping[_] for _ in vh)
#         return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]
#
#     def lower_left_corners(self):
#         """lower left corners of segment slices"""
#         return self._get_corners('ll', self.seg.get_slices(labels))
#
#     def lower_right_corners(self, labels=None):
#         """lower right corners of segment slices"""
#         return self._get_corners('lr', self.seg.get_slices(labels))
#
#     def upper_right_corners(self, labels=None):
#         """upper right corners of segment slices"""
#         return self._get_corners('ur', self.seg.get_slices(labels))
#
#     def upper_left_corners(self, labels=None):
#         """upper left corners of segment slices"""
#         return self._get_corners('ul', self.seg.get_slices(labels))
#
#     llc = lower_left_corners
#     lrc = lower_right_corners
#     urc = upper_right_corners
#     ulc = upper_left_corners
#
#     def widths(self):
#         return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.x)))
#
#     def height(self):
#         return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.y)))
#
#     def extents(self, labels=None):
#         """xy sizes"""
#         slices = self.seg.get_slices(labels)
#         sizes = np.zeros((len(slices), 2))
#         for i, sl in enumerate(slices):
#             if sl is not None:
#                 sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
#                             for s, sz in zip(sl, self.seg.shape)]
#         return sizes
#
#     def grow(self, labels, inc=1):
#         """Increase the size of each slice in all directions by an increment"""
#         # z = np.array([slices.llc(labels), slices.urc(labels)])
#         # z + np.array([-1, 1], ndmin=3).T
#         urc = np.add(self.urc(labels), inc).clip(None, self.seg.shape)
#         llc = np.add(self.llc(labels), -inc).clip(0)
#         slices = [tuple(slice(*i) for i in yxix)
#                   for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
#         return slices
#
#     def around_centroids(self, image, size, labels=None):
#         com = self.seg.centroid(image, labels)
#         slices = self.around_points(com, size)
#         return com, slices
#
#     def around_points(self, points, size):
#
#         yxhw = duplicate_if_scalar(size) / 2
#         yxdelta = yxhw[:, None, None] * [-1, 1]
#         yxp = np.atleast_2d(points).T
#         yxss = np.round(yxp[..., None] + yxdelta).astype(int)
#         # clip negative slice indices since they yield empty slices
#         return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
#                           for sz, ss in zip(self.seg.shape, yxss))))
#
#     def plot(self, ax, **kws):
#         from matplotlib.patches import Rectangle
#         from matplotlib.collections import PatchCollection
#
#         kws.setdefault('facecolor', 'None')
#
#         rectangles = []
#         for (y, x) in self:
#             xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
#             w = x.stop - x.start
#             h = y.stop - y.start
#             r = Rectangle(xy, w, h)
#             rectangles.append(r)
#
#         windows = PatchCollection(rectangles, **kws)
#         ax.add_collection(windows)
#         return windows


class Slices(object):
    """
    Container emulation for tuples of slices.

    Aid selecting rectangular sub-regions of images.


    """
    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __init__(self, seg_or_slices):
        """
        Create container of slices from list of 2-tuple of slices, or from
        SegmentationImage.  If not

        Parameters
        ----------
        slices
        seg
        """

        # self awareness
        if isinstance(seg_or_slices, Slices):
            slices = seg_or_slices
            seg = slices.seg

        elif isinstance(seg_or_slices, SegmentationImage):
            # get slices from SegmentationImage
            seg = seg_or_slices
            slices = SegmentationImage.slices.fget(seg_or_slices)
        else:
            raise TypeError('%r should be initialized from '
                            '`SegmentationImage` or `Slices` object' %
                            self.__class__.__name__)

        # use object array as container so we can get items by indexing with
        # list or array which is really nice and convenient
        self.slices = np.empty(len(slices), 'O')
        self.slices[:] = slices

        # add SegmentationHelper instance as attribute
        self.seg = seg

    def __getitem__(self, key):
        # delegate item getting to `np.ndarray`
        return self.slices[key]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__,
                           np.array2string(self.slices))

    @property
    def x(self):
        _, x = zip(*self.slices)
        return x

    @property
    def y(self):
        y, _ = zip(*self.slices)
        return y

    # def of_labels(self, labels=None):
    #     return self.seg.get_slices(labels)

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]

    def lower_left_corners(self, labels=None):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self.seg.get_slices(labels))

    def lower_right_corners(self, labels=None):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self.seg.get_slices(labels))

    def upper_right_corners(self, labels=None):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self.seg.get_slices(labels))

    def upper_left_corners(self, labels=None):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self.seg.get_slices(labels))

    llc = lower_left_corners  # TODO: lazyproperties ???
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    def widths(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.x)))

    def height(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.y)))

    def extents(self, labels=None):
        """xy sizes"""
        slices = self.seg.get_slices(labels)
        sizes = np.zeros((len(slices), 2))
        for i, sl in enumerate(slices):
            if sl is not None:
                sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
                            for s, sz in zip(sl, self.seg.shape)]
        return sizes

    def grow(self, labels, inc=1):
        """Increase the size of each slice in all directions by an increment"""
        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T
        urc = np.add(self.urc(labels), inc).clip(None, self.seg.shape)
        llc = np.add(self.llc(labels), -inc).clip(0)
        slices = [tuple(slice(*i) for i in yxix)
                  for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
        return slices

    def around_centroids(self, image, size, labels=None):
        com = self.seg.centroid(image, labels)
        slices = self.around_points(com, size)
        return com, slices

    def around_points(self, points, size):

        yxhw = duplicate_if_scalar(size) / 2
        yxdelta = yxhw[:, None, None] * [-1, 1]
        yxp = np.atleast_2d(points).T
        yxss = np.round(yxp[..., None] + yxdelta).astype(int)
        # clip negative slice indices since they yield empty slices
        return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
                          for sz, ss in zip(self.seg.shape, yxss))))

    def plot(self, ax, **kws):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        from matplotlib.cm import get_cmap

        kws.setdefault('facecolor', 'None')

        rectangles = []
        slices = list(filter(None, self))
        n_slices = len(slices)

        ec = kws.get('edgecolor', None)
        ecs = kws.get('edgecolors', None)
        if ec is None and ecs is None:
            cmap = get_cmap(kws.get('cmap', 'gist_ncar'))
            ecs = cmap(np.linspace(0, 1, n_slices))
            kws['edgecolors'] = ecs

        # make the patches
        for y, x in slices:
            xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
            w = x.stop - x.start
            h = y.stop - y.start
            r = Rectangle(xy, w, h)
            rectangles.append(r)

        # collect
        windows = PatchCollection(rectangles, **kws)
        # plot
        ax.add_collection(windows)
        return windows


def format_doc(template):
    # Helper that attaches docstring to method given template
    def decorator(func):
        func.__doc__ = template % func.__name__
        return func

    return decorator


class SegmentationHelper(SegmentationImage, LoggingMixin):
    """
    Extends `photutils.segmentation.SegmentationImage` functionality

    Additions to the SegmentationImage class.
        * classmethod for construction from an image of stars
        * support for iterating over segments / slices
        * support for calculations on masked arrays
        * methods for basic statistics on segments (mean, median etc)
        * methods for calculating center of mass, counts, flux in segments
        * re-ordering (sorting) labels by counts in an image
        * preparing masks for photometry (2d, 3d)
        * dilating, eroding segments, transforming to annuli
        * selecting subsets of the segmentation image
        * adding segments from another instance / array
        * displaying as an image with segments labelled

    In terms of modelling, this class is a domain mapping layer that lives on
    top of images
    """

    # _allow_negative = False       # TODO: maybe

    # # TODO:
    # @classmethod
    # def gmm(cls, image, mask=False, background=None, snr=3., npixels=7,
    #            edge_cutoff=None, deblend=False, dilate=0):

    # Constructors
    # --------------------------------------------------------------------------
    @classmethod
    def from_image(cls, image, background=None, snr=3., npixels=7,
                   edge_cutoff=None, deblend=False, dilate=1, flux_sort=True):
        """
        Detect stars in an image and return a SegmentationHelper object

        Parameters
        ----------
        image
        snr
        npixels
        edge_cutoff
        deblend
        flux_sort
        dilate

        Returns
        -------

        """
        # detect
        obj = cls.detect(image, background, snr, npixels,
                         edge_cutoff, deblend, dilate)

        if flux_sort:
            obj.flux_sort(image)

        return obj

    @classmethod
    def detect(cls, image, mask=False, background=None, snr=3., npixels=7,
               edge_cutoff=None, deblend=False, dilate=0):
        """
        Image object detection that returns a SegmentationHelper instance

        Parameters
        ----------
        image
        mask
        background
        snr
        npixels
        edge_cutoff
        deblend
        dilate

        Returns
        -------

        """

        # segmentation image based on sigma-clipping
        seg = detect(image, mask, background, snr, npixels, edge_cutoff,
                     deblend)

        # Initialize
        obj = cls(seg)

        #
        if dilate:
            obj.dilate(iterations=dilate)

        if cls.logger.getEffectiveLevel() > logging.INFO:
            logger.debug('Detected %i objects across %i pixels.', obj.nlabels,
                         obj.to_bool().sum())

        return obj

    # --------------------------------------------------------------------------
    def _detect_refine(self, image, mask=False, background=None, snr=3.,
                       npixels=7, edge_cutoff=None, deblend=False,
                       dilate=0, ignore_labels=()):
        """
        Refine detection by running on image with current segments masked.

        note: This method will be run when calling `detect` from an instance
          of this class. It runs a detection on the background region of the
          image - i.e. all detected objects masked out except those in
          `ignore_labels`
        """
        #

        # update mask, get new labels
        mask = self.to_bool(None, ignore_labels) | mask

        cls = type(self)
        new = cls.detect(image, mask, background, snr, npixels, edge_cutoff,
                         deblend, dilate)

        # since we dilated the detection masks, we may now be overlapping
        # with previous detections. Remove overlapping pixels here
        if dilate:
            overlap = self.to_bool() & new.to_bool()
            new.data[overlap] = 0

        return new

    def __init__(self, data, use_zero=False):
        # self awareness
        if isinstance(data, SegmentationImage):
            data = data.data
            # loop the lazy properties and bind to the new instance to
            #  avoid recomputing unnecessarily
            for key, value in inspect.getmembers(data, is_lazy):
                setattr(self, key, value)

        # init parent
        super().__init__(data)  # lazyproperties will not be reset on init!
        self._use_zero = bool(use_zero)

        # hack so we can use `detect` from both class and instance
        self.detect = self._detect_refine

    # def __reduce__(self):
    #     # pickling helper
    #     return SegmentationHelper, (self.data, self._use_zero)

    def __array__(self, *args):
        """
        Array representation of the segmentation image (e.g., for
        matplotlib).
        """

        # add *args for case np.asanyarray(self, dtype=int)
        # this allows Segmentation class to be initialized from another
        # object of the same (or inherited) class more easily

        return self._data

    # Properties
    # --------------------------------------------------------------------------
    def _reset_lazy_properties(self):
        """Reset all lazy properties.  Will work for subclasses"""
        for key, value in inspect.getmembers(self.__class__, is_lazy):
            self.__dict__.pop(key, None)

    @property
    def data(self):
        """
        The 2D segmentation image.
        """
        return self._data

    @data.setter
    def data(self, data):
        # data = SegmentedArray(data, self)
        self.set_data(data)

    def set_data(self, value):
        # now subclasses can over-write this function to implement stuff
        # without touching the property definition above. This should be in the
        # base class!

        # Note however that the array which this property points to
        #  is mutable, so explicitly changing it's contents
        #  (via eg: ``seg.data[:] = 1``) will not delete the lazyproperties!
        #  There should be an array interface subclass for this

        # TODO: manage through class method allow_negative

        if np.any(~np.isfinite(value)):
            raise ValueError('data must not contain any non-finite values '
                             '(e.g. NaN, inf)')

        if '_data' in self.__dict__:
            # needed only when data is reassigned, not on init
            self._reset_lazy_properties()

        self._data = value

    @lazyproperty
    def has_zero(self):
        """Check if there are any zeros in the segmentation image"""
        return (self.data == 0).any()

    @lazyproperty
    def slices(self):
        """
        The minimal bounding box slices for each labeled region as a list.
        """

        # self.logger.debug('computing slices!')
        # get slices from parent (possibly compute via `find_objects`)
        return Slices(self)

    def get_slices(self, labels=None):
        if labels is None:
            return self.slices

        return self.slices[self.get_indices(labels)]

    #
    # @lazyproperty
    # def labels(self):
    #     # overwrite labels property to allow the use of zero label
    #     labels = np.unique(self.data)
    #     if (0 in labels) and not self.use_zero:
    #         return labels[1:]
    #     return labels
    #
    # @property
    # def labels_nonzero(self):
    #     """Positive segment labels"""
    #     if self.has_zero and self.use_zero:
    #         return self.labels[1:]
    #     return self.labels

    def resolve_labels(self, labels=None, ignore=(), allow_zero=False):
        """

        Get the list of labels that are in use.  Default is to return the
        full list of unique labels in the segmentation image.  If a sequence
        of labels is provided, this method will check whether they are valid
        - i.e. contained in the segmentation image

        Parameters
        ----------
        labels: sequence of int, optional
            labels to check
        ignore: sequence of int, optional
            labels to ignore
        allow_zero: bool
            Whether to raise exception on presence of `0` label



        Returns
        -------

        """
        if labels is None:
            labels = self.labels

        if len(ignore):
            labels = np.setdiff1d(labels, ignore)

        return self.has_labels(labels, allow_zero)

    def has_labels(self, labels, allow_zero=False):
        """
        Check that the sequence of labels are in the segmented image

        Parameters
        ----------
        labels
        allow_zero

        Returns
        -------

        """
        labels = np.atleast_1d(labels)
        valid = list(self.labels)

        if allow_zero:
            valid += [0]

        invalid = np.setdiff1d(labels, valid)
        if len(invalid):
            raise ValueError('Invalid label(s): %s' % str(tuple(invalid)))

        return labels

    # def index(self, labels):
    #     """Index the """
    #     labels = self.resolve_labels(labels)
    #     if self.is_consecutive:
    #         return labels - (not self.use_zero)
    #     else:
    #         return np.digitize(labels, self.labels) - (not self.use_zero)

    # --------------------------------------------------------------------------
    @lazyproperty  # TODO: Mixin!!!!
    def masks(self):  # todo: use attr_getter(self.segments, 'data_ma.mask')
        """
        For each (label, slice) pair: a boolean array of the cutout
        segmentation image containing False where
        any value in the segmented image is not equal to the value of that
        label.

        Returns
        -------
        dict of arrays (keyed on labels)
        """

        masks = {}
        if self.has_zero and self.use_zero:
            masks[0] = self.mask0

        for lbl, slice_ in self.enum_slices(self.labels_nonzero):
            seg = self.data[slice_]
            masks[lbl] = (seg != lbl)

        return masks

    @lazyproperty
    def mask0(self):  # todo use `masked_background` now in photutils
        return self.data == 0

    # @lazyproperty
    # def areas(self):
    #     areas = np.bincount(self.data.ravel())
    #     # since there may be no labelled segments in the image
    #     # `self.labels.min()` will raise ValueError
    #
    #     if self.has_zero and not self.use_zero:
    #         start = 1
    #     elif self.nlabels:
    #         start = 0
    #     else:
    #         start = self.labels.min()
    #     return areas[start:]

    # Boolean conversion / masking
    # --------------------------------------------------------------------------
    @classmethod
    def from_bool_3d(cls, masks):
        # todo check masks is 3d array
        data = np.zeros(masks.shape[1:], int)
        for i, mask in enumerate(masks):
            data[mask] = (i + 1)
        return cls(data)

    def to_bool_3d(self, labels=None, ignore_labels=()):
        """
        Expand segments into 3D sequence of masks, one segment label per image
        """
        labels = self.resolve_labels(labels)
        if len(ignore_labels):
            labels = np.setdiff1d(labels, ignore_labels)
        return self.data[None] == labels[:, None, None]

    def to_bool(self, labels=None, ignore_labels=()):
        """Mask where elements equal any of the `labels` """
        return self.to_bool_3d(labels, ignore_labels).any(0)

    # aliases
    from_boolean = from_bool = from_bool_3d
    to_boolean = to_bool
    to_boolean_3d = to_bool_3d

    def mask_segments(self, image, labels=None, ignore_labels=()):
        """
        Mask all segments having label in `labels`. default labels are all
        non-zero labels.
        """
        labels = self.resolve_labels(labels)
        if len(ignore_labels):
            labels = np.setdiff1d(labels, ignore_labels)

        return np.ma.array(image, mask=self.to_bool(labels))

    # alias
    mask_sources = mask_segments

    def mask_background(self, image):
        """Mask background (label <= 0)"""
        return np.ma.masked_where(self.data <= 0, image)

    def mask_3d(self, image, labels=None, mask0=True):
        """
        Expand image to 3D masked array, indexed on segment index with
        background (label<=0) masked.
        """
        return self._select_labels(image, labels, mask0, self.to_bool_3d)

    # Data extraction
    # --------------------------------------------------------------------------
    def select_subset(self, start, shape, type_=None):
        """
        FIXME: this description not entirely accurate - padding happens

        Create a smaller version of the segmentation image starting at
        yx-indices `start` and having shape `shape`

        Parameters
        ----------
        start
        shape
        type_:
            The desired type of output.  Can be any class (or callable) that
            accepts an array as initializer, but will typically be a subclass of
            `SegmentationImage`, or an `np.ndarray`.  If None (default),
            an instance of this class is returned.

        Returns
        -------

        """
        if type_ is None:
            type_ = self.__class__
        return type_(select_rect_pad(self, self.data, start, shape))

    @staticmethod
    def _select_labels(image, labels, mask0, masking_func):
        # method that does the work for isolating the labels
        mask = masking_func(labels)
        extracted = image * mask  # will upcast to 3d if mask is 3d
        if mask0:
            extracted = np.ma.MaskedArray(extracted, ~mask)
        return extracted

    def keep(self, labels, mask0=False):
        """
        Return segmentation data keeping only `labels` and zeroing / masking
        everything else. This is almost like `keep_labels`, except that here
        the internal data remain unchanged and instead a modified copy of the
        `SegmentationImage` returned.

        Parameters
        ----------
        labels
        mask0

        Returns
        -------

        """
        return self._select_labels(self.data, labels, mask0, self.to_bool)

    def select_3d(self, image, labels, mask0=False):
        """
        Get data of labels from image keeping only labeled regions and
        zeroing / masking everything else. Returns a 3d array with first
        dimension the size of labels
        """  # TODO: update for nd ?
        return self._select_labels(image, labels, mask0, self.to_bool_3d)

    def clone(self, keep_labels=all, remove=None):
        """Return a modified copy keeping only labels in `keep`"""
        if keep_labels is all:
            keep_labels = self.labels
        else:
            keep_labels = np.atleast_1d(keep_labels)

        if remove is not None:
            keep_labels = np.setdiff1d(keep_labels, remove)

        return self.__class__(self.keep(keep_labels))

    # segment / slice iteration
    # --------------------------------------------------------------------------
    def iter_slices(self, labels=None, filter_=False):
        """

        Parameters
        ----------
        labels: sequence of int
            Segment labels to be iterated over
        filter_: bool
            whether to filter empty labels (no pixels)

        Returns
        -------

        """

        slices = self.get_slices(labels)
        if filter_:
            slices = filter(None, slices)
        yield from slices

    def enum_slices(self, labels=None, filter_=True):
        """
        Yield (label, slice) pairs optionally filtering None slices
        """

        labels = self.resolve_labels(labels)
        pairs = zip(labels, self.slices[labels])
        if filter_:
            yield from ((i, s)
                        for i, s in pairs
                        if s is not None)
        else:
            yield from pairs

    def iter_segments(self, image, labels=None):
        """
        Yields labelled sub-regions of the image sequentially

        Parameters
        ----------
        image
        labels

        Returns
        -------

        """
        yield from self.coslice(image, labels=labels, flatten=True)

    def coslice(self, *arrays, labels=None, masked=False, flatten=False,
                enum=False, mask_these=(0,)):
        """
        Yields labelled sub-regions of multiple arrays sequentially as tuple of
        (optionally masked) arrays.

        Parameters
        ----------
        arrays: tuple of arrays
            May contain None, in which case None is returned in the same
            position instead of the sliced sub-array
        labels: array-like
            Sequence of integer labels
        masked: bool
            for each array and each segment cutout, mask all data
            "background" data (`label <= 0)`
        mask_these: sequence of int, optional
            index positions of the arrays for which the cutouts will be
            background-masked
        flatten: bool
            Return flattened arrays of data corresponding to label instead of a
            masked_array. This option is here since it is faster to check the
            sub-array labels than the entire frame if the slices have
            already been calculated.
            Note that both `mask` and `flatten` cannot be simultaneously True
        enum:
            enumerate with `label` value


        Yields
        ------
        (tuple of) (masked) array(s)
        """

        n = len(arrays)
        # checks
        if n == 0:
            raise ValueError('Need at least one array to slice')

        # check if arrays should be masked
        if not isinstance(masked, bool):
            raise TypeError('`masked` should be bool')
        # TODO: check if a boolean array was passed?

        if masked and flatten:
            raise ValueError("Use either mask or flatten. Can't do both.")

        # function that yields the result
        unpack = (next if n == 1 else tuple)

        # flag which arrays to mask
        mask_flags = np.zeros(len(arrays), bool)
        if masked:
            # mask_which_arrays = np.array(mask_which_arrays, ndmin=1)
            mask_flags[mask_these] = True
        if flatten:
            mask_flags[:] = True

        for lbl, slice_ in self.enum_slices(labels):
            subs = (_2d_slicer(a, slice_,
                               self.masks[lbl] if flg else False,
                               flatten)
                    for a, flg in zip(arrays, mask_flags))
            # note this propagates the `None`s in `arrays` tuple

            if enum:
                yield lbl, unpack(subs)
            else:
                yield unpack(subs)

    # Image methods
    # --------------------------------------------------------------------------
    _image_doc_template = textwrap.dedent(
            """
            %s pixel values in each segment ignoring any masked pixels.
    
            Parameters
            ----------
            image:  array-like, or masked array
                Image for which to calculate statistic
            labels: array-like
                labels
    
            Returns
            -------
            float or 1d array or masked array
            """)

    _support_stats = ['mean', 'median',
                      'minimum', 'minimum_position',
                      'maximum', 'maximum_position',
                      'extrema',
                      'variance', 'standard_deviation']
    _stats_aliases = {'minimum': 'min',
                      'maximum': 'max',
                      'minimum_position': 'argmin',
                      'maximum_position': 'argmax'}

    def _masked_stat(self, func, image, labels):

        self._check_image(image)
        labels = self.resolve_labels(labels, allow_zero=True)

        if np.ma.is_masked(image):
            # ignore masked pixels
            seg_data = self.data.copy()
            seg_data[image.mask] = self.max_label + 1
            # this label will not be used for statistic computation
            result = func(image, seg_data, labels)
            mask = (ndimage.sum(np.logical_not(image.mask),
                                self.data, labels) == 0)

            # get output mask
            return np.ma.MaskedArray(result, mask)
        else:
            return func(image, self.data, labels)

    for stat in _support_stats:
        func = getattr(ndimage, stat)
        ftl.partial(_masked_stat, func)
        format_doc(_image_doc_template)()

    @format_doc(_image_doc_template)
    def sum(self, image, labels=None):
        # TODO: look at ndimage._stats --> may be better to use here since
        #  it gets called internally anyway and you get the `counts` as a
        #  bonus...
        return self._masked_stat(image, labels, 'sum')

    for stat in

    @format_doc(_image_doc_template)
    def mean(self, image, labels=None):
        return self._masked_stat(image, labels, 'mean')

    @format_doc(_image_doc_template)
    def median(self, image, labels=None):
        return self._masked_stat(image, labels, 'median')

    # TODO:

    def _check_image(self, image):

        # convert list etc to array, but keep masked arrays
        image = np.asanyarray(image)

        # check shapes same
        if image.shape != self.shape:
            raise ValueError('Arrays does not have the same shape as '
                             'segmentation image')

        # TODO: maybe use numpy.broadcast_arrays if you want to do stats on
        #  higher dimensional data sets using this class.  You will have to
        #  think carefully on how to managed masked points, since replacing a
        #  label in the seg data will only work for 2d data.  May have to
        #  resort to `labelled_comprehension`

    def _relabel_masked(self, image, masked_pixels_label=None):
        # if image is masked, ignore masked pixels by using copy of segmentation
        # data with positions of masked pixels replaced by new label one
        # larger than current max label.
        # Note: Some statistical computations `mean`, `median`,
        #  `percentile` etc cannot be computed on the masked image by simply
        #  zero-filling the masked pixels since it will artificially skew the
        #  results. No significant performance difference between the 2 paths
        if masked_pixels_label is None:
            masked_pixels_label = self.max_label + 1
        masked_pixels_label = int(masked_pixels_label)

        if np.ma.is_masked(image):
            # ignore masked pixels
            seg_data = self.data.copy()
            seg_data[image.mask] = masked_pixels_label
            # this label will not be used for statistic computation
            return seg_data
        else:
            return self.data



    def thumbnails(self, image=None, labels=None):
        """
        Thumbnail cutout images based on segmentation

        Parameters
        ----------
        image
        labels

        Returns
        -------
        list of arrays
        """

        data = self.data if image is None else image
        self._check_image(data)
        return list(self.coslice(data, labels=labels))

    def argmax(self, image, labels=None):
        """Indices of maxima in each segment given an image"""

        # NOTE: ndimage.maximum_position exists - check if this is preferable
        #  pros: n-dimensional, performance??

        labels = self.resolve_labels(labels)
        self._check_image(image)

        n = len(labels)
        image.ndim()
        ix = np.empty((n, 2), int)

        for i, slices in enumerate(self.slices[labels]):
            sub = image[slices]
            llc = [s.start for s in slices]
            ix[i] = np.add(llc, np.divmod(np.ma.argmax(sub), sub.shape[1]))
        return ix

    def flux(self, image, labels=None, labels_bg=(0,),
             statistic_bg='median'):  #
        """
        An estimate of the net (background subtracted) source counts

        Parameters
        ----------
        image: ndarray
        labels: sequence of int
        labels_bg: sequence of int, np.ndarray, SegmentationImage
            if sequence, must be of length 1 or same length as `labels`
            if array of dimensionality 2, must be same shape as image

        statistic_bg

        Returns
        -------

        """
        self._check_image(image)
        labels = self.resolve_labels(labels)

        # for convenience, allow `labels_bg` to be a SegmentedImage!
        # todo: 3d array for overlapping bg regions?
        if isinstance(labels_bg, np.ndarray) and labels_bg.shape == image.shape:
            labels_bg = self.__class__(labels_bg)

        if isinstance(labels_bg, SegmentationImage):
            assert labels_bg.nlabels == len(labels)
            seg_bg = labels_bg.copy()
            # exclude sources from bg segmentation!
            seg_bg.data[self.to_bool(labels)] = 0
        else:
            labels_bg = self.resolve_labels(labels_bg, allow_zero=True)
            n_obj = len(labels)
            n_bg = len(labels_bg)
            if n_bg not in (n_obj, 1):
                raise ValueError(
                        'Unequal number of background / object labels (%i / %i)'
                        % (n_bg, n_obj))
            seg_bg = self

        # resolve background counts estimate function
        statistic_bg = statistic_bg.lower()
        known_estimators = ('mean', 'median')
        if isinstance(statistic_bg, str) and (statistic_bg in known_estimators):
            counts_bg_pp = getattr(seg_bg, statistic_bg)(image)
            counts_bg = counts_bg_pp * self.sum(np.ones_like(image), labels)
        else:  # TODO: allow callable?
            raise ValueError('Invalid value for parameter `statistic_bg`.')

        # mean counts per binned pixel (src + bg)
        counts = self.sum(image, labels)
        return counts - counts_bg

    def flux_sort(self, image, labels=None, labels_bg=(0,), bg_stat='median'):
        """
        Re-label segments for highest per-pixel counts in descending order
        """
        labels = self.resolve_labels(labels)
        unsorted_labels = np.setdiff1d(self.labels, labels)
        n_sort = len(labels)

        flx = self.flux(image, labels, labels_bg, bg_stat)
        order = np.argsort(flx)[::-1]

        # todo:
        # self.relabel_many(None, order + 1)

        # re-order segmented image labels
        forward_map = np.zeros(self.nlabels + 1, int)
        forward_map[1:n_sort + 1] = labels[order]
        forward_map[(n_sort + 1):] = unsorted_labels
        self.data = forward_map[self.data]
        return flx

    def count_sort(self, image, labels=None):
        # todo: sort_by_count / sort_by('sum')
        """
        Re-label segments such they are ordered counts (brightest source) in
        `image` descending order
        """
        labels = self.resolve_labels(labels)
        unsorted_labels = np.setdiff1d(self.labels, labels)

        counts = self.sum(image, labels)
        order = np.argsort(counts)[::-1]
        old_labels = labels[order]
        new_labels = np.arange(1, len(order) + 1)

        if len(np.intersect1d(unsorted_labels, new_labels)):
            raise NotImplementedError

        self.relabel_many(old_labels, new_labels)
        return counts[order]

    def snr(self, image, labels=None, labels_bg=(0,)):
        """
        A signal-to-noise ratio estimate.  Calculates the SNR by taking the
        ratio of total background subtracted flux to the total estimated
        uncertainty (including poisson, sky, and instrumental noise) in each
        segment as per Merlin & Howell '95
        """

        labels = self.resolve_labels(labels)
        counts = self.sum(image, labels)
        n_pix_src = self.sum(np.ones_like(image), labels)

        # estimate sky counts and noise from data
        flx_bg = self.median(image, labels_bg)  # contains sky, read, dark, etc
        n_pix_bg = self.sum(np.ones_like(image), labels_bg)

        # bg subtracted source counts
        signal = counts - flx_bg * n_pix_src
        # revised CCD equation from Merlin & Howell '95
        noise = np.sqrt(signal +  # ← poisson.      sky + instrument ↓
                        n_pix_src * (1 + n_pix_src / n_pix_bg) * flx_bg)
        return signal / noise

    # TODO: seg.centrality.com / seg.centrality.gmean /
    #  seg.centrality.mean / seg.centrality.argmax
    def com(self, image=None, labels=None):
        """
        Center of Mass for each labelled segment

        Parameters
        ----------
        image: 2d array or None
            if None, center of mass of segment for constant image is returned
        labels

        Returns
        -------

        """
        image = self.data.astype(bool) if image is None else image
        labels = self.resolve_labels(labels)
        return np.array(ndimage.center_of_mass(image, self.data, labels))

    def com_bg(self, image, labels=None, mask=None,
               background_estimator=np.ma.median, grid=None):
        # todo: better name ? com_deviation com_dev
        """
        Compute centre of mass of background subtracted (masked) image.
        Default is to use median statistic  as background estimator.
        Pre-subtraction improves accuracy of measurements in presence of noise.

        Parameters
        ----------
        image
        labels
        mask
        background_estimator
        grid

        Returns
        -------

        """

        if self.shape != image.shape:
            raise ValueError('Invalid image shape %s for segmentation of '
                             'shape %s' % (image.shape, self.shape))

        if background_estimator in (None, False):
            def bg_sub(image):
                return image
        else:
            def bg_sub(image):
                return image - background_estimator(image)

        if grid is None:
            grid = np.indices(self.shape)

        labels = self.resolve_labels(labels)

        # generalization for higher dimensional data
        axes_sum = tuple(range(1, image.ndim))

        counter = itt.count()
        com = np.empty((len(labels), 2))
        for lbl, (seg, sub, msk, grd) in self.coslice(self.data, image, mask,
                                                      grid, labels=labels,
                                                      enum=True):

            # ignore whatever is in this slice and is not same label as well
            # as any external mask
            use = (seg == lbl)
            if msk is not None:
                use &= ~msk

            #
            sub = bg_sub(sub)
            grd = grd[:, use]
            sub = sub[use]

            # compute sum
            sum_ = sub.sum()
            if sum_ == 0:
                warnings.warn(f'Function `com_bg` encountered zero-valued '
                              f'image segment at label {lbl}.')
                # can skip next

            # compute centre of mass
            com[next(counter)] = (sub * grd).sum(axes_sum) / sum_
            # may be nan / inf

        return com

    centroids = com_bg

    # --------------------------------------------------------------------------

    def relabel_many(self, old_labels, new_labels):
        """
        Reassign multiple labels

        Parameters
        ----------
        old_labels
        new_labels

        Returns
        -------

        """
        old_labels = self.resolve_labels(old_labels)

        if len(old_labels) != len(new_labels):
            raise ValueError('Unequal number of labels')

        # forward_map = np.hstack([0, self.labels])
        # there may be labels missing
        forward_map = np.arange(self.max_label + 1)
        forward_map[old_labels] = new_labels
        self.data = forward_map[self.data]

    def dilate(self, iterations=1, connectivity=4, labels=None, mask=None,
               copy=False):
        """
        Binary dilation on each labelled segment

        Parameters
        ----------
        iterations
        connectivity
        labels
        mask
        copy

        Returns
        -------

        """
        if not iterations:
            return self

        # expand masks to 3D sequence
        labels = self.resolve_labels(labels)
        masks = self.to_bool_3d(labels)

        if connectivity == 4:
            struct = ndimage.generate_binary_structure(2, 1)
        elif connectivity == 8:
            struct = ndimage.generate_binary_structure(2, 2)
        else:
            raise ValueError('Invalid connectivity={0}.  '
                             'Options are 4 or 8'.format(connectivity))

        # structure array needs to have same rank as masks
        struct = struct[None]
        masks = ndimage.binary_dilation(masks, struct, iterations, mask)
        data = np.zeros_like(self.data, subok=False)
        for lbl, mask in zip(labels, masks):
            data[mask] = lbl

        if labels is not None:
            no_dilate = self.keep(np.setdiff1d(self.labels, labels))
            mask = no_dilate.astype(bool)
            data[mask] = no_dilate[mask]

        if copy:
            return self.__class__(data)
        else:
            self.data = data
            return self

    def shift(self, offset):
        """
        Shift the data by (y, x) `offset` pixels

        Parameters
        ----------
        offset

        Returns
        -------

        """
        self.data = ndimage.shift(self.data, offset)
        # todo: optimization: update slices instead of re-running find_objects

    def to_annuli(self, buffer=0, width=5, labels=None):
        """
        Create 3D boolean array containing annular masks for sky regions.
        Regions containing labelled pixels from other stars within sky regions
        are masked out.

        Parameters
        ----------
        buffer
        width
        labels

        Returns
        -------

        """
        # sky annuli
        labels = self.resolve_labels(labels)
        masks = self.to_bool_3d(labels)
        # structure array needs to have same rank as masks
        struct = ndimage.generate_binary_structure(2, 1)[None]
        if buffer:
            m0 = ndimage.binary_dilation(masks, struct, iterations=buffer)
        else:
            m0 = masks
        m1 = ndimage.binary_dilation(m0, struct, iterations=width)
        msky = (m1 & ~m0)
        return msky

    def add_segments(self, data, label_insert=None, copy=False):
        """
        Add segments from another SegmentationImage.  New segments will
        over-write old segments in the pixel positions where they overlap.

        Parameters
        ----------
        data:
        label_insert:
            If None; new labels come after current ones.
            If int; value for starting label of new data. Current labels will be
              shifted upwards.

        Returns
        -------
        list of int (new labels)
"""

        assert self.shape == data.shape, \
            (f'Shape mismatch between {self.__class__.__name__} instance of '
             f'shape {self.shape} and data shape {data.shape}')

        # TODO: deal with overlapping segments

        if isinstance(data, SegmentationImage):
            input_labels = data.labels
            data = data.data.copy()  # copy to avoid altering original below
            # TODO: extend slices instead to avoid recompute + performance gain
        else:
            input_labels = np.unique(data)
            input_labels = input_labels[input_labels != 0]  # ignore background

        #
        n_input_labels = len(input_labels)
        if n_input_labels == 0:
            # nothing to add
            return self, input_labels

        # check if we have the same labels in both
        current_labels = self.labels
        duplicate_labels = np.intersect1d(input_labels, current_labels,
                                          assume_unique=True)
        has_dup_lbl = bool(len(duplicate_labels))

        # if replace:
        #     # new labels replace old ones (for duplicates)
        #     self.data[data.astype(bool)] = 0
        #     # zero all new labels at positions of old labels
        #     # note: this will not reset lazyproperties
        # else:
        #     # keep old labels over new ones (for duplicates)
        #     # zero new labels at positions of old labels (for duplicate labels)
        #     data[self.to_bool(duplicate_labels)] = 0

        new_labels = input_labels
        if label_insert is None:
            # new labels are numbered starting from one above current max label
            if has_dup_lbl:
                # relabel new labels
                new_labels = input_labels + self.max_label
                forward_map = np.zeros(input_labels.max() + 1)
                forward_map[input_labels] = new_labels
                data = forward_map[data]

        else:
            new_labels = np.arange(n_input_labels) + label_insert + 1
            current = self.data
            current[current >= label_insert] += new_labels.max()
            data = current + data

        #
        new_pix = data.astype(bool)

        if copy:
            seg_data_out = self.data.copy()
            # return seg_data_out, data, new_pix
            seg_data_out[new_pix] = data[new_pix]
            return self.__class__(seg_data_out), new_labels
        else:
            seg_data_out = self.data
            seg_data_out[new_pix] = data[new_pix]
            self._reset_lazy_properties()
            return self, new_labels

    def inside_segment(self, coords, labels=None):
        # filter COM positions that are outside of detection regions

        labels = self.resolve_labels(labels)
        assert len(coords) == len(labels)

        count = 0
        keep = np.zeros(len(coords), bool)

        # print('yo!', coords)

        grid = np.indices(self.shape)
        for i, (lbl, (sub, g)) in enumerate(
                self.coslice(self.data, grid, labels=labels, enum=True)):
            # print(i, lbl, )
            keep[i] = inside_segment(coords[i], sub, g)
            if not keep[i]:
                count += 1
                self.logger.debug(
                        'Discarding %i of %i at coords (%3.1f, %3.1f)',
                        count, len(coords), *coords[i])

        return keep

    def display(self, *args, **kws):
        """
        Plot the image using `ImageDisplay`

        Parameters
        ----------
        args
        kws

        Returns
        -------
        im: `ImageDisplay` instance

        """
        # TODO: disable sliders ....

        from scipy.cluster.vq import kmeans
        from graphical.imagine import ImageDisplay

        # draw segment labels
        draw_labels = kws.pop('draw_labels', True)
        draw_rect = kws.pop('draw_rect', False)

        # use categorical colormap (random, muted colours)
        cmap = 'gray' if (self.nlabels == 0) else self.make_cmap()
        # conditional here prevents bork on empty segmentation image
        kws.setdefault('cmap', cmap)

        # plot
        im = ImageDisplay(self.data, *args, **kws)

        if draw_rect:
            self.slices.plot(im.ax)

        if draw_labels:
            # add label text (number) on each segment
            yx = ndimage.center_of_mass(self.data, self.data, self.labels)

            for i, lbl in enumerate(self.labels):
                # compute minimal point distance from center-of-mass. Can
                # reveal disjointed segments, or segments with holes etc. For
                # these we add the label multiple times, one for each
                # separate part of the segment.
                # label positions computed as centers of kmean cluster
                # todo: use dbscan here for unknown number of clusters
                pos = np.array(yx[i])
                w = np.array(np.where(self.data == lbl))
                d = np.sqrt(np.square(pos[None].T - w).sum(0))
                dmin = d.min()
                if dmin > 1:
                    # center of mass not close to any pixels. disjointed segment
                    # try cluster
                    codebook, distortion = kmeans(w.T.astype(float), 2)
                else:
                    codebook = [pos]

                for pos in codebook:
                    im.ax.text(*pos[::-1], str(lbl),
                               color='w', fontdict=dict(weight='bold'),
                               va='center', ha='center')

        return im


class SegmentationGroups(SegmentationHelper, LabelGroupsMixin):

    def __init__(self, data, use_zero=False):
        super().__init__(data, use_zero)

    def add_segments(self, data, label_insert=None, group=None, copy=False):
        seg, new_labels = super().add_segments(data, label_insert, copy)
        self.groups[group] = new_labels
        return seg


class CachedPixelGrids():
    'things!'


class SegmentationGridHelper(SegmentationHelper):  # SegmentationModelHelper
    """Mixin class for image models with piecewise domains"""

    def __init__(self, data, grid=None, domains=None):
        # initialize parent
        super().__init__(data)

        if grid is None:
            grid = np.indices(data.shape)
        else:
            grid = np.asanyarray(grid)
            assert grid.ndim >= 2  # else the `coord_grids` property will fail

        self.grid = grid
        self._coord_grids = None

        # domains for each segment
        if domains is None:
            domains = {}
        else:
            assert len(domains) == self.nlabels
        # The coordinate domain for the models. Some models are more stable on
        # restricted intervals
        self.domains = domains

    @lazyproperty
    def coord_grids(self):
        return self.get_coord_grids()

    def get_coord_grids(self, labels):
        return dict(self.coslice(self.grid, labels=self.resolve_labels(labels),
                                 flatten=True, enum=True))

    # def get_coord_grids(self, labels=None):
    #     grids = {}
    #     labels = self.resolve_labels(labels)
    #     for lbl, sl in self.enum_slices(labels):
    #         grids[lbl] = self.get_grid(lbl, sl)
    #     return grids
    #
    # def get_grid(self, label, slice):
    #     sy, sx = slice
    #     dom = self.domains.get(label, None)
    #     if dom is None:
    #         ylo, yhi = sy.start, sy.stop
    #         xlo, xhi = sx.start, sx.stop
    #         yst, xst = 1, 1
    #     else:
    #         (ylo, yhi), (xlo, xhi) = self.domains[label]
    #         yst = (sy.stop - sy.start) * 1j
    #         xst = (sx.stop - sx.start) * 1j
    #     return np.mgrid[ylo:yhi:yst, xlo:xhi:xst]


if __name__ == '__main__':
    # pickling test
    import pickle

    z = np.zeros((25, 100), int)
    z[10:15, 30:40] = 1
    segm = SegmentationHelper(z)
    clone = pickle.loads(pickle.dumps(segm))
