"""
Extensions for segmentation images
"""
import inspect
import itertools as itt
import warnings

from operator import attrgetter

import numpy as np
from IPython import embed

from obstools.phot.utils import duplicate_if_scalar, iter_repeat_last
from recipes.logging import LoggingMixin
from scipy import ndimage

from photutils import SegmentationImage, Segment

# from collections import namedtuple

from astropy.stats.sigma_clipping import sigma_clip
from astropy.utils import lazyproperty

from photutils.detection.core import detect_threshold
from photutils.segmentation import detect_sources

import logging

from recipes.string import get_module_name

# TODO: watershed segmentation on the negative image ?

# module level logger
logger = logging.getLogger(get_module_name(__file__))


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
    edge_cutoff
    deblend
    dilate

    Returns
    -------

    """

    logger.info('Running detect(snr=%.1f, npixels=%i)', snr, npixels)

    if mask is None:
        mask = False

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

    # detection
    threshold = detect_threshold(image, snr, background, mask=mask)
    if mask is False:
        mask = None  # annoying photutils #HACK
    obj = detect_sources(image, threshold, npixels, mask=mask)

    #
    if np.all(obj.data == 0):
        logger.info('No objects detected')
        return obj

    # if border is not False:
    #     obj.remove_masked_labels(border, partial_overlap=False)

    if deblend:
        from photutils import deblend_sources
        obj = deblend_sources(image, obj, npixels)
        # returns altered copy of instance

    logger.info('Detected %i objects across %i pixels.', obj.nlabels,
                obj.data.astype(bool).sum())
    return obj


# TODO: detect_gmm():



def detect_loop(image, mask=None, bg_model=None, snr=(10, 7, 5, 3),
                npixels=(7, 5, 3), edge_cutoff=None, deblend=(True, False),
                dilate=(4, 1)):
    """
    Multi-threshold image blob detection, segmentation and grouping.

    `snr`, `npixels`, `dilate` can be sequences in which case the next value
    from each will be used in the next detection loop. If scalar, same value
    during each iteration while mask in updated

    Parameters
    ----------
    image
    mask
    bgmodel
    snr
    npixels
    dilate
    edge_cutoff
    deblend

    Returns
    -------

    """

    #
    logger.info('Running detection loop')

    # make iterables
    variters = tuple(map(iter_repeat_last, (snr, npixels, dilate, deblend)))
    vargen = zip(*variters)

    # segmentation data
    data = np.zeros(image.shape, int)
    original_mask = mask
    if mask is None:
        mask = np.zeros(image.shape, bool)

    # first round detection without background model
    residual = image
    results = None

    # keep track of group info + detection meta data
    if isinstance(bg_model, SegmentedImageModel):
        labels_bg = bg_model.segm.labels
    else:
        labels_bg = [0]

    # label groups for consecutive detections
    groups = LabelGroups(bg=labels_bg)
    groups.info = Record()
    groups._auto_name_fmt = groups.info._auto_name_fmt = 'stars%i'

    counter = itt.count(0)
    go = True
    j = max(labels_bg)
    while go:
        # keep track of iteration number
        count = next(counter)
        group_name = 'stars%i' % count

        # detect
        _snr, _npix, _dil, _debl = next(vargen)
        sh = SegmentationHelper.detect(residual, mask, None, _snr, _npix,
                                       edge_cutoff, _debl, _dil)

        # update mask, get new labels
        new_mask = sh.to_bool()
        new_data = new_mask & np.logical_not(mask)

        # since we dilated the detection masks, we may now be overlapping
        # with previous detections. Remove overlapping pixels here
        if dilate:
            overlap = data.astype(bool) & new_mask
            # print('overlap', overlap.sum())
            sh.data[overlap] = 0
            new_data[overlap] = False

        if not new_data.any():
            break

        # aggregate
        new_labelled = sh.data[new_data]
        new_labels = np.unique(new_labelled)
        data[new_data] += new_labelled + j
        mask = mask | new_mask
        # group info
        group = new_labels + j
        groups[group_name] = group
        groups.info[group_name] = \
            dict(snr=_snr, npixels=_npix, dilate=_dil, deblend=_debl)
        # update
        j += new_labels.max()

        #
        # logger.info('detect_loop: round nr %i: %i new detections: %s',
        #       count, len(group), tuple(group))
        cls.logger.info(
                'detect_loop: round nr %i: %i new detections: %s',
                count, len(group), seq_repr_trunc(tuple(group)))

        if count == 0:
            # initialise the background model
            # FIXME: better to do decide about ft streaks based on residuals
            # decide based on estimated star flux whether or not to add ft
            # streak model.
            threshold = 500
            ft, segm, strkLbl = FrameTransferBleed.from_image(
                    image, sh, threshold)
            # The frame transfer bleed model might be None if we only have
            # faint stars in the image
            if ft:
                groups['streaks'] = strkLbl

            # mdlr = cls(segm, (bg_model, ft), groups)
            # note both models might be None
        else:
            # add segments to ignore
            cls.logger.info('adding segments')
            _, labels = mdlr.segm.add_segments(sh, replace=True)

            mdlr.groups.append(labels)

        return None, results, residual, data, mask, groups

        # fit background
        mimage = np.ma.MaskedArray(image, original_mask)
        # return mdlr, mimage, segm
        results, residual = mdlr.minimized_residual(mimage)

        # print(results)

    # TODO: log what you found
    return mdlr, results, residual, data, mask, groups

    # background = results = None  # first round detection without bgmodel
    # groups = []
    # counter = itt.count(0)
    # j = 0
    # go = True
    # while go:
    #     # keep track of iteration number
    #     count = next(counter)
    #
    #     # detect
    #     sh = SegmentationHelper.detect(image, mask, background, next(_snr),
    #                                    next(_npix), edge_cutoff, deblend,
    #                                    next(_dil))
    #
    #     # update mask, get labels
    #     labelled = sh.data
    #     new_mask = labelled.astype(bool)
    #     new_data = new_mask & np.logical_not(mask)
    #     new_labels = np.unique(labelled[new_data])
    #
    #     # since we dilated the detection masks, we may now be overlapping with
    #     # previous detections. Remove overlapping pixels here
    #     # if dilate:
    #     #     overlap = mask & new_data
    #     #     new_data[overlap] = 0
    #
    #     if not new_data.any():
    #         # done
    #         # final group of detections are determined by sigma clipping
    #         if bgmodel:
    #             resi = bgmodel.residuals(results, image)
    #         else:
    #             resi = image
    #         resi_masked = np.ma.MaskedArray(resi, mask)
    #         resi_clip = sigma_clip(resi_masked)  # TODO: make this optional
    #         new_data = resi_clip.mask & ~resi_masked.mask
    #         new_data = ndimage.binary_dilation(new_data) & ~resi_masked.mask
    #         labelled, nlabels = ndimage.label(new_data)
    #         new_labels = np.unique(labelled[labelled != 0])
    #         go = False
    #
    #     # aggregate
    #     data[new_data] += labelled[new_data] + j
    #     mask = mask | new_mask
    #     groups.append(new_labels + j)
    #     j += new_labels.max()
    #
    #     logger.info('detect_loop: round nr %i: %i new detections: %s',
    #                 count, len(new_labels), tuple(new_labels))
    #     # print('detect_loop: round nr %i: %i new detections: %s' %
    #     #       (count, len(new_labels), tuple(new_labels)))
    #
    #     # slotmode HACK
    #     if count == 0:
    #         # BackgroundModel(sh, None, )
    #         # init bg model here
    #         segmbg = sh.copy()
    #
    #     if bgmodel:
    #         # fit background
    #         imm = np.ma.MaskedArray(image, mask)
    #         results, residu = bgmodel.minimized_residual(imm)
    #
    #         # background model
    #         background = bgmodel(results)
    #
    #     break
    #
    # # TODO: log what you found
    #
    # return data, groups, results


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

import operator


class Segments(list):
    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __repr__(self):
        return ''.join((self.__class__.__name__, super().__repr__()))

    def attrgetter(self, *attrs):
        getter = operator.attrgetter(*attrs)
        return list(map(getter, self))

    # def of_labels(self, labels=None):
    #     return self.segm.get_slices(labels)

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]

    def lower_left_corners(self):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self.segm.get_slices(labels))

    def lower_right_corners(self, labels=None):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self.segm.get_slices(labels))

    def upper_right_corners(self, labels=None):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self.segm.get_slices(labels))

    def upper_left_corners(self, labels=None):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self.segm.get_slices(labels))

    llc = lower_left_corners
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    def widths(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.x)))

    def height(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.y)))

    def extents(self, labels=None):
        """xy sizes"""
        slices = self.segm.get_slices(labels)
        sizes = np.zeros((len(slices), 2))
        for i, sl in enumerate(slices):
            if sl is not None:
                sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
                            for s, sz in zip(sl, self.segm.shape)]
        return sizes

    def grow(self, labels, inc=1):
        """Increase the size of each slice in all directions by an increment"""
        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T
        urc = np.add(self.urc(labels), inc).clip(None, self.segm.shape)
        llc = np.add(self.llc(labels), -inc).clip(0)
        slices = [tuple(slice(*i) for i in yxix)
                  for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
        return slices

    def around_centroids(self, image, size, labels=None):
        com = self.segm.centroid(image, labels)
        slices = self.around_points(com, size)
        return com, slices

    def around_points(self, points, size):

        yxhw = duplicate_if_scalar(size) / 2
        yxdelta = yxhw[:, None, None] * [-1, 1]
        yxp = np.atleast_2d(points).T
        yxss = np.round(yxp[..., None] + yxdelta).astype(int)
        # clip negative slice indices since they yield empty slices
        return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
                          for sz, ss in zip(self.segm.shape, yxss))))

    def plot(self, ax, **kws):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        kws.setdefault('facecolor', 'None')

        rectangles = []
        for (y, x) in self:
            xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
            w = x.stop - x.start
            h = y.stop - y.start
            r = Rectangle(xy, w, h)
            rectangles.append(r)

        windows = PatchCollection(rectangles, **kws)
        ax.add_collection(windows)
        return windows


class Slices(list):  # rename Segments, integrate with photutils
    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __init__(self, slices=(), segm=None):

        if len(slices) == 0:
            if segm is None:
                raise ValueError
            else:
                # get slices from parent
                slices = segm.__class__.slices.fget(segm)

        if isinstance(slices, Slices) and segm is None:
            segm = slices.segm

        # init parent class
        list.__init__(self, slices)

        # add SegmentationHelper instance as attribute
        self.segm = segm

    # def compute(self):
    #     self.segm.__class__.slices.fget(self.segm)

    def __repr__(self):
        return ''.join((self.__class__.__name__, super().__repr__()))

    # def __getitem__(self, key):

    @property
    def x(self):
        _, x = zip(*self)
        return x

    @property
    def y(self):
        y, _ = zip(*self)
        return y

    # def of_labels(self, labels=None):
    #     return self.segm.get_slices(labels)

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]

    def lower_left_corners(self, labels=None):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self.segm.get_slices(labels))

    def lower_right_corners(self, labels=None):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self.segm.get_slices(labels))

    def upper_right_corners(self, labels=None):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self.segm.get_slices(labels))

    def upper_left_corners(self, labels=None):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self.segm.get_slices(labels))

    llc = lower_left_corners
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    def widths(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.x)))

    def height(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.y)))

    def extents(self, labels=None):
        """xy sizes"""
        slices = self.segm.get_slices(labels)
        sizes = np.zeros((len(slices), 2))
        for i, sl in enumerate(slices):
            if sl is not None:
                sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
                            for s, sz in zip(sl, self.segm.shape)]
        return sizes

    def grow(self, labels, inc=1):
        """Increase the size of each slice in all directions by an increment"""
        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T
        urc = np.add(self.urc(labels), inc).clip(None, self.segm.shape)
        llc = np.add(self.llc(labels), -inc).clip(0)
        slices = [tuple(slice(*i) for i in yxix)
                  for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
        return slices

    def around_centroids(self, image, size, labels=None):
        com = self.segm.centroid(image, labels)
        slices = self.around_points(com, size)
        return com, slices

    def around_points(self, points, size):

        yxhw = duplicate_if_scalar(size) / 2
        yxdelta = yxhw[:, None, None] * [-1, 1]
        yxp = np.atleast_2d(points).T
        yxss = np.round(yxp[..., None] + yxdelta).astype(int)
        # clip negative slice indices since they yield empty slices
        return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
                          for sz, ss in zip(self.segm.shape, yxss))))

    def plot(self, ax, **kws):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        kws.setdefault('facecolor', 'None')

        rectangles = []
        for (y, x) in self:
            xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
            w = x.stop - x.start
            h = y.stop - y.start
            r = Rectangle(xy, w, h)
            rectangles.append(r)

        windows = PatchCollection(rectangles, **kws)
        ax.add_collection(windows)
        return windows


class SegmentationHelper(SegmentationImage, LoggingMixin):
    """
    Extends `photutils.SegmentationImage` functionality

    Additions to the SegmentationImage class.
        * classmethod for initializing from an image of stars
        * support for iterating over segments / slices
        * support for masked arrays
        * methods for calculating counts / flux in segments of image
        * preparing masks for photometry
        *
        *

    This class is a domain mapping layer that lives on top of images
    """

    # _allow_negative = False       # TODO:

    # def cmap(self):
    #   fixme: fails for all zero segments

    # Constructors
    # --------------------------------------------------------------------------
    @classmethod  # TODO: more general algorithms here - GMM etc
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

        # segmentation based on sigma-clipping
        obj = detect(image, mask, background, snr, npixels, edge_cutoff,
                     deblend)

        # Initialize
        new = cls(obj.data)

        #
        if dilate:
            new.dilate(iterations=dilate)
        #
        return new

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

    def refine(self, image, mask=False, background=None, snr=3., npixels=7,
               edge_cutoff=None, deblend=False, dilate=0, copy=False):
        """Refine detection by running on image with current segments masked"""

        image = self.mask_segments(image, self.labels_nonzero)
        new = self.__class__.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)
        segm, new_labels = self.add_segments(new, copy=copy)
        return segm

    # # TODO:
    # @classmethod
    # def gmm(cls, image, mask=False, background=None, snr=3., npixels=7,
    #            edge_cutoff=None, deblend=False, dilate=0):

    # def _from_self(self, data):
    # TODO: if lazyproperties already computed no need to recompute...

    # def refine(self, image, mask=False, bgmodel=None, snr=3., npixels=7,
    #            edge_cutoff=None, deblend=False, dilate=0):
    #     """Refine detection by running on image with segments masked"""
    #
    #     # make iterables
    #     _snr, _npix, _dil = (iter_repeat_last(o) for o in
    #                          (snr, npixels, dilate))
    #
    #     # mask known
    #     mask = mask | self.to_bool()
    #
    #     #
    #     go = True
    #     counter = itt.count()
    #     groups = [self.labels]
    #     background = results = None  # first round detection without bgmodel
    #     while go:
    #         count = next(counter)  # keep track of iteration number
    #         print('iter', count)
    #
    #         # re-run detection
    #         obj = self.detect(image, mask, background, next(_snr), next(_npix),
    #                           edge_cutoff, deblend, next(_dil))
    #
    #         # check if anything new was detected
    #         new_mask = obj.data.astype(bool)
    #         if new_mask.any():
    #             _, new_labels = self.add_segments(obj)
    #             print(new_labels, 'new_labels')
    #             groups.append(new_labels)
    #             mask = mask | new_mask
    #         else:
    #             go = False
    #             # finally sigma-clip on residuals  # TODO: make this optional
    #             if background is not None:
    #                 resi = image - background
    #                 resi_masked = np.ma.MaskedArray(resi, mask)
    #                 resi_clip = sigma_clip(resi_masked)
    #                 new = resi_clip.mask & ~resi_masked.mask
    #                 new = ndimage.binary_dilation(new) & ~resi_masked.mask
    #                 labelled, nlabels = ndimage.label(new)
    #                 #
    #                 _, new_labels = self.add_segments(labelled)
    #                 groups.append(new_labels)
    #
    #         # fit background
    #         if bgmodel:
    #             imm = np.ma.MaskedArray(image, mask)
    #             results = bgmodel.fit(imm)
    #             return results
    #             # background model
    #             background = bgmodel(results, bgmodel.grid)
    #         break
    #
    #     return groups, results, background

    # def __getitem__(self, labels):
    #     """Return instance of same class keeping `labels`"""
    #     new = self.copy()
    #     new.keep_labels(labels)
    #     return new

    def __init__(self, data, use_zero=False):
        super().__init__(data)
        self._use_zero = bool(use_zero)

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
        lazy = lambda _: isinstance(_, lazyproperty)
        for key, value in inspect.getmembers(self.__class__, lazy):
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

    def set_data(self, data):
        # now subclasses can over-write this function to implement stuff
        # without touching the property definition above. This should be in the
        # base class!

        # Note however that the array which this property points to
        # is mutable, so explicitly changing it's contents
        # (via eg: ``segm.data[:] = 1``) will not delete the lazyproperties!
        # There should be an array interface subclass for this

        # TODO: manage through class method allow_negative

        return SegmentationImage.data.fset(self, data)

    @lazyproperty
    def has_zero(self):
        """Check if there are any zeros in the segmentation image"""
        return (self.data == 0).any()

    @property
    def use_zero(self):
        """
        Allows the 0 label to be included in iteration across segments
        automatically. If `use_zero` is True, the first segment from
        iteration methods `iter_slices`, `enum_slices`, `iter_segments`,
        `coslice` will return the full frame.

        Returns
        -------

        """
        return self._use_zero

    @use_zero.setter
    def use_zero(self, b):
        b = bool(b)
        if b is self._use_zero:
            return

        if not self.has_zero:
            return

        # if we get here, we know there are 0s in the segmentation image,
        # and that the `use_zero` state has changed.
        # Avoid recomputing (slow) the `slices` and `labels` lazyproperties by
        # setting them explicitly below (via self.__class__.labels.fset etc)

        # change state.
        if self._use_zero:
            # ignore full frame
            slices = self.slices[1:]
            self.labels = self.labels[1:]
            self.masks.pop(0)
        else:
            # use full frame!
            slices = [tuple(map(slice, (0, 0), self.shape))]
            slices.extend(self.slices)
            self.labels = np.hstack([0, self.labels])
            self.masks.insert(0, self.mask0)

        # set
        self.slices = slices  # Slices(slices, self)
        self._use_zero = b

        del self.areas
        del self.segments

    # def segment0:

    @lazyproperty
    def segments(self):
        return Segments(super().segments)

    @lazyproperty
    def slices(self):
        """
        The minimal bounding box slices for each labeled region as a numpy
        object array.
        """

        # self.logger.debug('computing slices!')

        # get slices from parent (possibly compute via `find_objects`)
        slices = SegmentationImage.slices.fget(self)  # only non-zero labels

        if self.use_zero and self.has_zero:
            slices = [tuple(map(slice, (0, 0), self.shape))] + slices

        # return slices #np.array(slices, self._slice_dtype)
        return slices  # Slices(slices, self)

    def get_slices(self, labels=None):
        if labels is None:
            return self.slices
        labels = self.has_labels(labels) - (not self.use_zero)
        return [self.slices[_] for _ in labels]

    @lazyproperty
    def labels(self):
        # overwrite labels property to allow the use of zero label
        labels = np.unique(self.data)
        if (0 in labels) and not self.use_zero:
            return labels[1:]
        return labels

    @property
    def labels_nonzero(self):
        """Positive segment labels"""
        if self.has_zero and self.use_zero:
            return self.labels[1:]
        return self.labels

    def resolve_labels(self, labels=None):
        """

        Get the list of labels that are in use.  Default is to return the
        full list of unique labels in the segmentation image.  If a sequence
        of labels is provided, this method will check whether they are valid
        - i.e. contained in the segmentation image

        Parameters
        ----------
        labels: sequence of int, optional
            labels to check


        Returns
        -------

        """
        if labels is None:
            return self.labels
        return self.has_labels(labels)

    def has_labels(self, labels):
        """
        Check that the sequence of labels are in the segmented image

        Parameters
        ----------
        labels

        Returns
        -------

        """
        labels = np.atleast_1d(labels)
        valid = list(self.labels)
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
    #         return np.digitize(labels, self.labels)

    # --------------------------------------------------------------------------
    @lazyproperty
    def masks(self):
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
    def mask0(self):
        return self.data == 0

    @lazyproperty
    def areas(self):
        areas = np.bincount(self.data.ravel())
        if self.has_zero and not self.use_zero:
            return areas[1:]
        return areas

    # --------------------------------------------------------------------------

    # @lazyproperty
    # def windows(self):
    #     return RectangleSegments()

    # @lazyproperty
    # def all_labels(self):
    #     """The sorted array all the labels, not just the non-zero ones.     """
    #     # Useful to have this so we don't have to search for the index of the
    #     # passed labels against
    #     # the list of labels, we can just slice the all_labels array. numpy
    #     # will then do the check for valid labels and we don't have to
    #
    #     return np.unique(self.data)
    #

    # Boolean conversion / masking
    # --------------------------------------------------------------------------
    @classmethod
    def from_bool_3d(cls, masks):
        # todo check masks is 3d array
        data = np.zeros(masks.shape[1:], int)
        for i, mask in enumerate(masks):
            data[mask] = (i + 1)
        return cls(data)

    def to_bool_3d(self, labels=None):
        """
        Expand segments into 3D sequence of masks, one segment label per image
        """
        if labels is None:
            labels = self.labels_nonzero
        else:
            labels = self.resolve_labels(labels)
        return self.data[None] == labels[:, None, None]

    def to_bool(self, labels=None):
        """Mask where elements equal any of the `labels` """
        return self.to_bool_3d(labels).any(0)

    # aliases
    from_boolean = from_bool = from_bool_3d
    to_boolean = to_bool
    to_boolean_3d = to_bool_3d

    def mask_segments(self, image, labels=None):
        """
        Mask all segments having label in `labels`. default labels are all
        non-zero labels.
        """
        return np.ma.array(image, mask=self.to_bool(labels))

    # alias
    mask_foreground = mask_segments

    def mask_background(self, image):
        """Mask background (label <= 0)"""
        return np.ma.masked_where(self.data <= 0, image)

    def mask_3d(self, image, labels=None, mask0=True):
        """
        Expand image to 3D masked array, indexed on segment index with
        background (label<=0) masked.
        """
        return self._select_labels(image, labels, mask0, self.to_bool_3d)

    @staticmethod
    def _select_labels(image, labels, mask0, masking_func):
        # method that does the work for isolating the labels
        mask = masking_func(labels)
        extracted = image * mask  # will upcast to 3d if mask is 3d
        if mask0:
            extracted = np.ma.MaskedArray(extracted, ~mask)
        return extracted

    # Data extraction
    # --------------------------------------------------------------------------
    def select(self, labels, mask0=False):
        """
        Return segmentation data keeping only `labels` and zeroing / masking
        everything else. This is almost like `keep_labels`, except that here
        the internal data remain unchanged and instead a modified copy is
        returned.

        Parameters
        ----------
        labels
        mask0

        Returns
        -------

        """
        return self._select_labels(self.data, labels, mask0, self.to_bool)

    def select_3d(self, labels, mask0=False):
        """Return segmentation data keeping only requested labels"""
        return self._select_labels(self.data, labels, mask0, self.to_bool_3d)

    def clone(self, keep_labels=all, remove=None):
        """Return a modified copy keeping only labels in `keep`"""
        if keep_labels is all:
            keep_labels = self.labels
        else:
            keep_labels = np.atleast_1d(keep_labels)

        if remove is not None:
            keep_labels = np.setdiff1d(keep_labels, remove)

        return self.__class__(self.select(keep_labels))

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
        slices = self.get_slices(labels)
        pairs = zip(labels, slices)
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

    def coslice(self, *arrays, labels=None, masked_bg=False, flatten=False,
                enum=False, mask_which_arrays=(0,)):
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
        masked_bg: bool
            for each array and each segment cutout, mask all data
            "background" data (`label <= 0)`
        mask_which_arrays: sequence of int, optional
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

        # TODO: check if a boolean array was passed?
        if not isinstance(masked_bg, bool):
            raise TypeError('`masked_bg` should be bool')

        if masked_bg and flatten:
            raise ValueError("Use either mask or flatten. Can't do both.")

        # function that yields the result
        unpack = (next if n == 1 else tuple)

        # flag which arrays to mask
        mask_flags = np.zeros(len(arrays), bool)
        if masked_bg:
            # mask_which_arrays = np.array(mask_which_arrays, ndmin=1)
            mask_flags[mask_which_arrays] = True
        if flatten:
            mask_flags[:] = True

        for lbl, slice_ in self.enum_slices(labels):
            subs = (self._2d_slicer(a, slice_,
                                    self.masks[lbl] if flg else False,
                                    flatten)
                    for a, flg in zip(arrays, mask_flags))
            # note this propagates the `None`s in `arrays` tuple

            if enum:
                yield lbl, unpack(subs)
            else:
                yield unpack(subs)

    @staticmethod
    def _2d_slicer(array, slice_, mask=None, extract=False):
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

        if array is None:
            return None  # propagate `None`s

        # slice along last two dimensions
        slice_ = (Ellipsis,) + tuple(slice_)
        sub = array[slice_]

        if mask is None or mask is False:
            return sub

        if extract:
            return sub[..., ~mask]

        ma = np.ma.MaskedArray(sub, copy=True)
        ma[..., mask] = np.ma.masked
        return ma

    # Image methods
    # --------------------------------------------------------------------------
    def thumbnails(self, image=None, labels=None):
        """
        Thumbnail cutout images based on segmentation

        Parameters
        ----------
        image

        Returns
        -------
        list of arrays
        """
        data = self.data if image is None else image
        return list(self.coslice(data, labels=labels))

    def argmax(self, image, labels=None, mask=None):
        """Indices of maxima in each segment given an image"""

        if mask is not None:
            image = np.ma.array(image, mask=mask)

        labels = self.resolve_labels(labels)
        n = len(labels)
        ix = np.empty((n, 2), int)
        for i, (sy, sx) in self.enum_slices(labels):
            sub = image[sy, sx]
            llc = sy.start, sx.start
            ix[i] = np.add(llc, np.divmod(np.argmax(sub), sub.shape[1]))
        return ix

    def counts(self, image, labels=None):
        """
        Count raw pixel values in segments, ignoring any masked pixels

        Parameters
        ----------
        image
        labels

        Returns
        -------

        """
        labels = self.resolve_labels(labels)
        segm = self.data
        if np.ma.is_masked(image):
            segm = self.data.copy()
            segm[image.mask] = 0

        return ndimage.sum(image, segm, labels)

    def flux(self, image, labels=None):  # bgfunc=np.median
        """
        Mean per-pixel count in each segment
        """

        labels = self.resolve_labels(labels)
        counts = self.counts(image, labels)
        areas = self.areas[labels - 1]  # self.area(labels)

        # bg = bgfunc(image[self.data == 0])
        return (counts / areas)  # - bg

    def flux_sort(self, image):
        """
        Re-label segments for highest per-pixel counts in descending order
        """
        flx = self.flux(image)
        flx_srt, lbl_srt = zip(*sorted(zip(flx, self.labels), reverse=1))

        # TODO: do this like relabel_consecutive
        # re-order segmented image labels
        data = np.zeros_like(self.data, subok=False)
        for new, old in enumerate(lbl_srt, 1):
            # logging.debug('%i --> %i: across %i pixels', (old, new, m.sum()))
            data[self.data == old] = new
        self.data = data

    def snr(self, image, labels=None):
        """A crude snr estimate"""
        labels = self.resolve_labels(labels)

        flx = self.flux(image, labels)
        bgflx, bgstd = np.empty((2,) + flx.shape)
        for i, m in enumerate(self.to_annuli(1, 5, labels)):
            bgflx[i] = np.median(image[m])
            bgstd[i] = image[m].std()

        return (flx - bgflx) / bgstd  # Fixme

    def dilate(self, connectivity=4, iterations=1, mask=None):
        """
        Bianry dialtion on each labelled segment

        Parameters
        ----------
        connectivity
        iterations
        mask

        Returns
        -------

        """
        # expand masks to 3D sequence
        masks = self.to_bool_3d()

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
        for i, mask in enumerate(masks):
            data[mask] = (i + 1)

        self.data = data

    def com(self, image, labels=None):
        """
        Center of Mass coordinate for each label

        Parameters
        ----------
        image
        labels

        Returns
        -------

        """
        labels = self.resolve_labels(labels)
        return np.array(ndimage.center_of_mass(image, self.data, labels))

    def com_bg(self, image, labels=None, mask=None, bg_func=np.median):
        """
        Compute centre of mass with background pre-subtracted. Often more
        accurate than simple centroid.

        Parameters
        ----------
        image
        labels
        mask
        bg_func

        Returns
        -------

        """
        labels = self.resolve_labels(labels)

        counter = itt.count()
        com = np.empty((len(labels), 2))
        grid = np.indices(self.shape)  # fixme. don't recompute every time
        for lbl, (seg, sub, g, m) in self.coslice(self.data, image, grid, mask,
                                                  labels=labels, enum=1):
            #
            sub = sub - bg_func(sub)
            # ignore whatever is in this slice and is not same label
            sub[seg != lbl] = 0
            # ignore mask
            if m is not None:
                sub[m] = 0

            # compute sum
            sum_ = sub.sum()
            if sum_ == 0:
                warnings.warn('`com_bg` encountered zero-valued image '
                              'segment at label %i.' % lbl)  # can skip
            # compute centre of mass
            com[next(counter)] = (sub * g).sum((1, 2)) / sum_  # may be nan
        return com

    centroids = com_bg

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
        Create 3D boolean array containing annular masks for sky regions

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
        msky = (m1 & ~masks.any(0))
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

        NOTE: assumes segments do not physically overlap!

        """
        # TODO: deal with overlapping segements

        if isinstance(data, SegmentationImage):
            new_labels = data.labels
            data = data.data
            # TODO: extend slices instead to avoid recompute + performance gain
        else:
            new_labels = np.unique(data)
            new_labels = new_labels[new_labels != 0]  # ignore background

        #
        if len(new_labels) == 0:
            # nothing to add
            return self, new_labels

        # check if we have the same labels in both
        current_labels = self.labels
        duplicate_labels = np.intersect1d(new_labels, current_labels,
                                          assume_unique=True)
        has_dup_lbl = len(duplicate_labels)

        # if replace:
        #     # new labels replace old ones (for duplicates)
        #     self.data[data.astype(bool)] = 0
        #     # zero all new labels at positions of old labels
        #     # note: this will not reset lazyproperties
        # else:
        #     # keep old labels over new ones (for duplicates)
        #     # zero new labels at positions of old labels  (for duplicate labels)
        #     data[self.to_bool(duplicate_labels)] = 0


        if label_insert is None:
            # new labels are appended
            if has_dup_lbl:
                # relabel new labels
                relabel_new = np.arange(len(new_labels)) + self.max_label + 1
                for lbl, new_lbl in zip(new_labels, relabel_new):
                    data[data == lbl] = new_lbl
                # new updated labels
                new_labels = relabel_new
        else:
            new_labels = np.arange(len(new_labels)) + label_insert + 1
            current = self.data
            current[current >= label_insert] += new_labels.max()
            data = current + data

        new_pix = data.astype(bool)

        if copy:
            return self.__class__(data), new_labels
        else:
            self.data[new_pix] = data[new_pix]
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
        from graphical.imagine import ImageDisplay

        # label segments
        label = kws.pop('label', False)

        # use catagorical cmap (random, muted colours)
        cmap = 'gray' if (self.nlabels == 0) else self.cmap()
        # conditional prevents bork on empty segmentation image
        kws.setdefault('cmap', cmap)

        # plot
        im = ImageDisplay(self.data, *args, **kws)

        if label:
            # segment label (number) text central on each segment
            yx = ndimage.center_of_mass(self.data, self.data, self.labels)
            for i, lbl in enumerate(self.labels):
                im.ax.text(*yx[i][::-1], str(lbl),
                           color='w', fontdict=dict(weight='bold'),
                           va='center', ha='center')

        return im


class SegmentationGridHelper(SegmentationHelper):
    """Mixin class for image models with piecewise domains"""

    # TODO: make this a SelfAware class

    def __init__(self, data, grid=None, domains=None):
        # initialize parent
        super().__init__(self, data)

        if grid is None:
            grid = np.indices(data.shape)
        else:
            grid = np.asanyarray(grid)
            assert grid.ndim >= 2  # else the `grids` property will fail

        self.grid = grid
        self._coord_grids = None

        # domains for each segment
        if domains is None:
            domains = {}
        else:
            assert len(domains) == self.nlabels
        #
        self.domains = domains

    # @lazyproperty
    # def domains(self):
    #     """
    #     The coordinate domain for the models. Some models are more stable on
    #     restricted intervals
    #     """
    #     domains = {}
    #     for i, (sy, sx) in enumerate(self.iter_slices()):
    #         h = sy.stop - sy.start
    #         w = sx.stop - sx.start
    #         domains[i] = (0, h), (0, w)
    #         else:
    #             (ylo, yhi), (xlo, xhi) = self.domains[i]

    @lazyproperty
    def coord_grids(self):
        return self.get_coord_grids()

    def get_coord_grids(self, labels=None):
        grids = {}
        labels = self.resolve_labels(labels)
        for lbl, (sy, sx) in self.enum_slices(labels):
            dom = self.domains.get(lbl, None)
            h = sy.stop - sy.start
            w = sx.stop - sx.start
            if dom is None:
                (ylo, yhi), (xlo, xhi) = (0, h), (0, w)
            else:
                (ylo, yhi), (xlo, xhi) = self.domains[lbl]

            grids[lbl] = np.mgrid[ylo:yhi:h * 1j, xlo:xhi:w * 1j]
        return grids

    # def set_domains(self, domains):
    #
    #     assert

    def com_bg(self, image, labels=None, mask=None, bg_func=np.ma.median):
        """
        Compute centre of mass with background pre-subtracted. Often more
        accurate than simple centroid.

        Parameters
        ----------
        image
        labels
        mask
        bg_func

        Returns
        -------

        """
        labels = self.resolve_labels(labels)

        counter = itt.count()
        com = np.empty((len(labels), 2))
        for lbl, (seg, sub, m) in self.coslice(self.data, image, mask,
                                               labels=labels, enum=True):
            sub = sub - bg_func(sub)
            # ignore whatever is in this slice and is not same label
            sub[seg != lbl] = 0
            # ignore mask
            if m is not None:
                sub[m] = 0

            # compute sum
            sum_ = sub.sum()
            if sum_ == 0:
                warnings.warn('`com_bg` encountered zero-valued image '
                              'segment at label %i.' % lbl)  # can skip
            # compute centre of mass
            g = self.coord_grids[lbl - 1]  # NOTE: label 0 bad here
            # print(g.shape, sub.shape)
            com[next(counter)] = (sub * g).sum((1, 2)) / sum_  # may be nan
        return com

    # def radial_grids(self, labels=None, rmax=10):
