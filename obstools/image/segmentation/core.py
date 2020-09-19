"""
Extensions for segmentation images
"""

# std libs
import more_itertools as mit
import collections as col
import types
import inspect
import logging
import warnings
import itertools as itt
from operator import attrgetter
import numbers
from collections import namedtuple
# third-party libs

import numpy as np
from scipy import ndimage
from astropy.utils import lazyproperty
from obstools.image.segmentation.trace import trace_boundary

from .detect import detect
from photutils.segmentation import SegmentationImage  # , Segment

# local libs
from recipes.logging import LoggingMixin
from recipes.logging import get_module_logger
from obstools.phot.utils import LabelGroupsMixin
from obstools.image.utils import shift_combine
from recipes.containers.dicts import pformat

# from obstools.modelling.image import SegmentedImageModel
# from collections import namedtuple

# TODO: watershed segmentation on the negative image ?
# TODO: detect_gmm():


# module level logger
logger = get_module_logger()

#
# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])


def is_lazy(_):
    return isinstance(_, lazyproperty)


def _echo(_):
    return _


def image_sub(background_estimator):
    if background_estimator in (None, False):
        return _echo

    def sub(image):
        return image - background_estimator(image)

    return sub

# def autoreload_isinstance_hack(obj, kls):
#     for parent in obj.__class__.mro():
#         if parent.__name__ == kls.__name__:
#             return True


# def is_sequence(obj):
#     """Check if obj is non-string sequence"""
#     return isinstance(obj, (tuple, list, np.ndarray))


def merge_segmentations(segmentations, xy_offsets, extend=True, f_accept=0.2,
                        post_merge_dilate=1):
    """

    Parameters
    ----------
    segmentations
    xy_offsets
    extend
    f_accept
    post_merge_dilate

    Returns
    -------

    """
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
    seg_image_extended, n_sources = ndimage.label(eim >= n_accept,
                                                  structure=np.ones((3, 3)))

    # edge case: it may happen that when creating the boolean array above
    # with the threshold `n_accept`, that pixels on the edges of sources
    # become separated from the main source. eg:
    #                     ________________
    #                     |              |
    #                     |    ████      |
    #                     |  ████████    |
    #                     |    ██████    |
    #                     |  ██  ██      |
    #                     |              |
    #                     |              |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # These pixels will receive a different label if we do not provide a
    # block structure element.
    # Furthermore, it also sometimes happens for faint sources that the
    # thresholding splits the source in two, like this:
    #                     ________________
    #                     |              |
    #                     |    ████      |
    #                     |  ██          |
    #                     |      ████    |
    #                     |      ██      |
    #                     |              |
    #                     |              |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # After dilating post merger, we end up with two labels for a single source
    # We fix these by running the "blend" routine.  Note that this will
    #  actually blend blend sources that are touching but are actually
    #  distinct sources eg:
    #                     __________________
    #                     |    ██          |
    #                     |  ██████        |
    #                     |██████████      |
    #                     |  ██████        |
    #                     |    ████  ██    |
    #                     |      ████████  |
    #                     |        ████████|
    #                     |        ██████  |
    #                     |          ██    |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # If you have crowded fields you may need to run "deblend" again
    # afterwards to separate them again
    seg_extended = SegmentedImage(seg_image_extended)
    seg_extended.dilate(post_merge_dilate)
    seg_extended.blend()
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


def get_masking_flags(arrays, masked):
    # get flags determining which arrays in the sequence are to be mask
    mask_flags = np.zeros(len(arrays), bool)

    if masked in (None, False):
        return mask_flags

    if masked is True:
        # mask all arrays
        mask_flags[:] = True
        return mask_flags

    # must be a sequence of ints / possibly with ellipsis.
    # ints are indices telling us which arrays should be masked
    masked = set(masked)
    has_dots = Ellipsis in masked
    masked.discard(Ellipsis)
    for m in masked:
        if not isinstance(m, numbers.Integral):
            raise ValueError('`masked` should be bool or sequence of ints')
        mask_flags[m] = True

    if has_dots:
        # mask all remaining arrays
        mask_flags[max(masked):] = True

    return mask_flags


def _2d_slicer(array, slice_, mask=None, compress=False):
    """
    Slice `np.ndarray` object along last 2 dimensions

    Parameters
    ----------
    array:  np.ndarray
    slice_: tuple of slices
    mask:   bool, np.ndarray or None (default)
    compress: bool

    Returns
    -------
    array or None
    """

    # print('!!!!!!!', array.shape, slice_, mask)

    if array is None:
        return None  # propagate `None`s

    # slice along last two dimensions
    slice_ = (Ellipsis,) + tuple(slice_)
    cut = array[slice_]

    if mask is None or mask is False:
        return cut

    if compress:
        return cut[..., ~mask]

    ma = np.ma.MaskedArray(cut, copy=True)
    ma[..., mask] = np.ma.masked
    return ma


# class ModelledSegment(Segment):
#     def __init__(self, segment_img, label, slices, area, model=None):
#         super().__init__(segment_img, label, slices, area)
#         self.model = model
#         # self.grid =
#         #


class SegmentedArray(np.ndarray):
    """
    WORK IN PROGRESS

    Array subclass for keeping image segmentation data. Keeps a reference to
    the `SegmentedImage` object that created it so that changing the
    segmentation array data triggers the lazyproperties to recompute the next
    time they are accessed.

    # note inplace operations on array will not trigger reset of parent
    # lazyproperties, but setting data explicitly should

    """

    def __new__(cls, input_array, parent):
        # Input data is array-like structure
        obj = np.array(input_array)

        # initialize with data
        super_ = super(SegmentedArray, cls)
        obj = super_.__new__(cls, obj.shape, int)
        super_.__setitem__(obj, ..., input_array)  # populate entire array

        # add SegmentedImage instance as attribute to be updated upon
        # changes to segmentation data
        obj.parent = parent  # FIXME: this will be missed for new-from-template
        return obj

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        # set the data in the SegmentedImage
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
        #         # parent in SegmentedImage instance
        #         # get list of slices from super class SegmentationImage
        #         slices = SegmentationImage.slices.fget(parent)
        #         if parent.use_zero:
        #             slices = [(slice(None), slice(None))] + slices
        #
        #         # initialize np.ndarray with data
        #         super_ = super(np.recarray, cls)
        # dtype = np.dtype(list(zip('yx', 'OO')))


#         obj = super_.__new__(cls, len(slices), dtype)
#         super_.__setitem__(obj, ..., slices)
#
#         # add SegmentedImage instance as attribute
#         obj.parent = parent
#         return obj
#
#     def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
#         return NotImplemented


class vdict(dict):
    """
    Dictionary with vectorized item lookup
    """

    def __getitem__(self, key):
        # dispatch on np.ndarray for vectorized item getting with arbitrary
        # nesting
        if isinstance(key, (np.ndarray, list, tuple)):  # Container and not str??
            return [self[_] for _ in key]

        if key in (Ellipsis, None):
            return list(self.values())

        return super().__getitem__(key)


class Sliced(vdict):
    """
    Dict-like container for tuples of slices. Aids selecting rectangular
    sub-regions of images more easily.
    """

    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __init__(self, mapping, **kws):
        super().__init__()
        kws.update(mapping)
        for k, v in kws.items():
            self[k] = v  # ensure item conversion

    def __setitem__(self, key, value):
        # convert items to namedtuple so we can later do things like
        # >>> image[:, seg.slices[7].x]
        super().__setitem__(key, yxTuple(*value))

    @property
    def x(self):
        _, x = zip(*self.values())
        return vdict(zip(self.keys(), x))

    @property
    def y(self):
        y, _ = zip(*self.values())
        return vdict(zip(self.keys(), y))

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]

    def lower_left_corners(self, labels=...):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self[labels])

    def lower_right_corners(self, labels=...):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self[labels])

    def upper_right_corners(self, labels=...):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self[labels])

    def upper_left_corners(self, labels=...):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self[labels])

    llc = lower_left_corners
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    # def extents(self, labels=None):
    #     """xy sizes"""
    #     slices = self[labels]
    #     sizes = np.zeros((len(slices), 2))
    #     for i, sl in enumerate(slices):
    #         if sl is not None:
    #             sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
    #                         for s, sz in zip(sl, self.seg.shape)]
    #     return sizes

    def grow(self, labels, inc=1):
        """Increase the size of each slice in all directions by an increment"""
        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T
        urc = np.add(self.urc(labels), inc)  # .clip(None, self.seg.shape)
        llc = np.add(self.llc(labels), -inc).clip(0)
        slices = [tuple(slice(*i) for i in yxix)
                  for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
        return slices

    # def around_centroids(self, image, size, labels=None):
    #     com = self.seg.centroid(image, labels)
    #     slices = self.around_points(com, size)
    #     return com, slices
    #
    # def around_points(self, points, size):
    #
    #     yxhw = duplicate_if_scalar(size) / 2
    #     yxdelta = yxhw[:, None, None] * [-1, 1]
    #     yxp = np.atleast_2d(points).T
    #     yxss = np.round(yxp[..., None] + yxdelta).astype(int)
    #     # clip negative slice indices since they yield empty slices
    #     return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
    #                       for sz, ss in zip(self.seg.shape, yxss))))

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


# class Slices(object):
#     # FIXME: remove this now superceded by Sliced
#     """
#     Container emulation for tuples of slices. Aids selecting rectangular
#     sub-regions of images more easil
#     """
#     # maps semantic corner positions to slice attributes
#     _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}
#
#     def __init__(self, seg_or_slices):
#         """
#         Create container of slices from SegmentationImage, Slice instance, or
#         from list of 2-tuples of slices.
#
#         Slices are stored in a numpy object array.  The first item in the
#         array is a slice that will return the entire object to which it is
#         passed as item getter.  This represents the "background" slice.
#
#         Parameters
#         ----------
#         slices
#         seg
#         """
#
#         # self awareness
#         if isinstance(seg_or_slices, Slices):
#             slices = seg_or_slices
#             seg = slices.seg
#
#         elif isinstance(seg_or_slices, SegmentationImage):
#             # get slices from SegmentationImage
#             seg = seg_or_slices
#             slices = SegmentationImage.slices.fget(seg_or_slices)
#         else:
#             raise TypeError('%r should be initialized from '
#                             '`SegmentationImage` or `Slices` object' %
#                             self.__class__.__name__)
#
#         # use object array as container so we can get items by indexing with
#         # list or array which is really nice and convenient
#         # secondly, we include index 0 as the background ==> full array slice!
#         # this means we can index this object directly with an array of
#         # labels, or integer label, instead of needing the -1 every time you
#         # want a slice
#         self.slices = np.empty(len(slices) + 1, 'O')
#         self.slices[0] = (slice(None),) * seg.data.ndim
#         self.slices[1:] = slices
#
#         # add SegmentedImage instance as attribute
#         self.seg = seg


# import functools as ftl


# class Centrality():
#     'TODO: com / gmean / mean / argmax'  # maybe


class MaskedStatsMixin(object):
    """
    This class gives inheritors access to methods for doing statistics on
    segmented images (from `scipy.ndimage.measurements`)
    """

    # Each supported method is wrapped in the `MaskedStatistic` class upon
    # construction.  `MaskedStatistic` is a descriptor and will dynamically
    # attach to inheritors of this class that invoke the method via attribute
    # lookup eg: >>> obj.sum(image)
    #

    _supported = ['sum',
                  'mean', 'median',
                  'minimum', 'minimum_position',
                  'maximum', 'maximum_position',
                  #   'extrema', # return signature is different, so don't support
                  'variance', 'standard_deviation',
                  'center_of_mass']
    _aliases = {'minimum': 'min',
                'maximum': 'max',
                'minimum_position': 'argmin',  # minpos # minloc
                'maximum_position': 'argmax',  # maxpos # maxloc
                'standard_deviation': 'std',
                'center_of_mass': 'com'}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # add methods for statistics on masked image to inheritors
        for stat in cls._supported:
            method = MaskedStatistic(getattr(ndimage, stat))
            setattr(cls, stat, method)
            # also add aliases for convenience
            alias = cls._aliases.get(stat)
            if alias:
                setattr(cls, alias, method)


class MaskedStatistic(object):
    # Descriptor class that enables statistical computation on masked
    # input data with segmented images
    _doc_template = \
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
        """

    def __init__(self, func):
        self.func = func
        self.__name__ = name = func.__name__
        self.__doc__ = self._doc_template % name.title().replace('_', ' ')

    def __get__(self, seg, objtype=None):

        if seg is None:  # called from class
            return self

        # bind this class to the seg instance from whence the lookup came.
        # Essentially this binds the first argument `seg` in `__call__` below
        return types.MethodType(self, seg)

    def __call__(self, seg, image, labels=None):
        # handling of masked pixels for all statistical methods done here
        seg._check_input_data(image)
        labels = seg.resolve_labels(labels, allow_zero=True)

        worker = self.worker_ma if np.ma.is_masked(image) else self.worker
        if image.ndim == 2:
            result = worker(image, seg, labels)
        else:
            # TODO: may as well parallelize here!!
            result = [worker(im, seg, labels) for im in image]

        # ensure return array or masked array and not list.
        return np.asanyarray(result)

    def worker(self, image, seg, labels):
        return self.func(image, seg.data, labels)

    def worker_ma(self, image, seg, labels):
        # ignore masked pixels
        seg_data = seg.data
        original = seg_data[image.mask]
        seg_data[image.mask] = seg.max_label + 1
        # this label will not be used for statistic computation

        # wrap the compute in an exception clause since we would like to restore
        # the segmentation data if anything goes wrong during the compute
        try:
            # compute
            result = self.func(image, seg_data, labels)
        except:
            raise
        else:
            # now we have to check which labels may be completely masked in
            # image data, so we can mask those in output
            mask = (ndimage.sum(np.logical_not(image.mask),
                                seg_data, labels) == 0)
            # get output mask
            # for functions that return array-like results per segment (eg.
            # com), we have to up-cast the mask
            if mask.any():
                result, mask = np.broadcast_arrays(result, mask[np.newaxis].T)
            else:
                mask = False
            return np.ma.MaskedArray(result, mask)
        finally:
            # restore the original labels of the masked pixels
            seg.data[image.mask] = original


def radial_source_profile(image, seg, labels=None):
    com = seg.com(image, labels)
    grid = np.indices(image.shape)
    profiles = []
    for i, (sub, g) in enumerate(seg.coslice(image, grid, flatten=True,
                                             labels=labels)):
        r = np.sqrt(np.square(g - com[i, None].T).sum(0))
        profiles.append((r, sub))
    return profiles


class SegmentMasks(col.defaultdict):  # SegmentMasks
    """
    Container for segment masks
    """

    def __init__(self, seg):
        self.seg = seg
        col.defaultdict.__init__(self, None)

    def __missing__(self, label):
        # the mask is computed at lookup time here and inserted into the dict
        # automatically after this func executes

        # allow zero!
        if label != 0:
            self.seg.check_label(label)
        return self.seg.sliced(label) != label


class SegmentMasksMixin(object):
    @lazyproperty
    def masks(self):
        """
        A dictionary. For each label, a boolean array of the cutout segment
        with `False` wherever pixels have different labels. The dict will
        initially be empty - the masks are computed only when lookup happens.

        Returns
        -------
        dict of arrays (keyed on labels)
        """
        return SegmentMasks(self)


# SegmentImage
class SegmentedImage(SegmentationImage,  # base
                     MaskedStatsMixin,  # stats methods for masked images
                     SegmentMasksMixin,  # handles masks for foreground obj
                     LabelGroupsMixin,  # keeps track of label groups
                     LoggingMixin  # logger
                     ):
    """
    Extends `photutils.segmentation.SegmentationImage` functionality.

    Additions to the SegmentationImage class.
        * classmethod for construction from an image of sources

        * support for iterating over segments / slices

        * calculations on masked arrays
        * methods for statistics on segments (min, max, mean, median, ...)
        * methods for calculating center-of-mass, counts, flux in each segment
        * re-ordering (sorting) labels by any of the above statistics

        * preparing masks for photometry (2d, 3d)
        * dilating, eroding segments, transforming to annuli

        * selecting subsets of the segmentation image
        * adding segments from another instance / array

        * displaying as an image with segments labelled in matplotlib or on
          the console

        * keeps track of logical groupings of labels

        * allows all zero data - base class does not allow this which is
          perceived as unnecessarily restrictive
    """

    # In terms of modelling, this class is a also domain mapping layer that
    # lives on top of images.

    # _allow_negative = False       # TODO: maybe

    random_state = None

    # Constructors
    # --------------------------------------------------------------------------
    @classmethod
    def empty_like(cls, image):
        """
        Segmentation image with all zeros in the shape of `image`.
        Sources can be added later using the `add_segments` method
        """
        return cls(np.zeros(image.shape, int))

    @classmethod
    def gmm(cls, image, mask=False, n_components=5, plot=False, **kws):
        """
        Construct a SegmentedImage using Gaussian Mixture Model prediction
        for pixels.

        Parameters
        ----------
        image
        mask
        n_components
        kws

        Returns
        -------

        """

        from sklearn.mixture import GaussianMixture

        pixels = ...
        if (mask is not None) or (mask is not False):
            image = np.ma.MaskedArray(image, mask)
            pixels = ~image.mask

        # model
        gmm = GaussianMixture(n_components, **kws)
        y = np.ma.compressed(image).reshape(-1, 1)

        seg = np.zeros(image.shape, int)
        seg[pixels] = gmm.fit_predict(y)
        obj = cls(seg)

        if plot:
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap

            m = gmm.means_.T
            v = gmm.covariances_.T
            w = gmm.weights_ / np.sqrt(2 * np.pi * v)
            x = np.linspace(y.min(), y.max(), 250).reshape(-1, 1)
            components = w * np.exp(-0.5 * np.square((x - m)) / v).squeeze()

            fig, ax = plt.subplots()
            ax.hist(y.squeeze(), bins=100, density=True, log=True)
            for c in components.T:
                ax.plot(x, c, scaley=False)

            cmap = ListedColormap([l.get_color() for l in ax.lines])
            obj.display(cmap=cmap, draw_labels=False)

        return obj

    @classmethod
    def from_image(cls, image, background=None, snr=3., npixels=7,
                   edge_cutoff=None, deblend=False, dilate=1, flux_sort=True):
        """
        Detect sources in an image and return a SegmentedImage object

        Parameters
        ----------
        image
        background
        snr
        npixels
        edge_cutoff
        deblend
        flux_sort
        dilate

        # Returns
        -------

        """
        # detect
        # noinspection PyTypeChecker
        obj = cls.detect(image, background, snr, npixels,
                         edge_cutoff, deblend, dilate)

        if flux_sort:
            obj.flux_sort(image)

        return obj

    @classmethod
    def detect(cls, image, mask=False, background=None, snr=3., npixels=7,
               edge_cutoff=None, deblend=False, dilate=0, group_name=None):
        """
        Image object detection that returns a SegmentedImage instance

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
        group_name: hashable object
            Key representing the name of this group.  By default

        Returns
        -------

        """

        # segmentation image based on sigma-clipping
        arr = detect(image, mask, background, snr, npixels, edge_cutoff,
                     deblend)
        # here `seg` is an array

        # add group name
        if group_name is not None and not isinstance(group_name, col.Hashable):
            raise ValueError('Group name %r cannot be used since it is not a'
                             'hashable object.' % group_name)

        # Initialize
        seg = cls(arr)
        seg.groups[group_name] = seg.labels

        # dilate
        if dilate != 'auto':
            seg.dilate(iterations=dilate)

        if cls.logger.getEffectiveLevel() > logging.INFO:
            logger.debug('Detected %i objects across %i pixels.', seg.nlabels,
                         seg.to_binary().sum())

        return seg

    # --------------------------------------------------------------------------
    def _detect_refine(self, image, mask=False, background=None, snr=3.,
                       npixels=7, edge_cutoff=None, deblend=False,
                       dilate=0, group_name=None, ignore_labels=()):
        """
        Refine detection by running on image with current segments masked.

        note: This method will be run when calling `detect` from an instance
          of this class. It runs a detection on the background region of the
          image - i.e. all detected objects masked out except those in
          `ignore_labels`
        """
        #

        if self.nlabels and (dilate == 'auto'):
            self.auto_dilate(image, sigma=snr)
            dilate = 0

        # update mask, get new labels
        mask = self.to_binary(None, ignore_labels) | mask

        # run detect classmethod
        new = type(self).detect(image, mask, background, snr, npixels,
                                edge_cutoff, deblend, dilate, group_name)

        # update label groups
        # new.groups.update(self.groups)

        # since we dilated the detection masks, we may now be overlapping
        # with previous detections. Remove overlapping pixels here.
        if dilate:
            overlap = self.to_binary() & new.to_binary()
            new.data[overlap] = 0

        return new

    # def __new__(cls, data):
    #
    #     return super().__new__(cls)

    def __init__(self, data, label_groups=None):
        # self awareness
        if isinstance(data, SegmentationImage):
            # preserve label groups if no new mapping is given
            if label_groups is None:
                label_groups = data.groups

            data = data.data
            # loop the lazy properties and bind to the new instance to
            #  avoid recomputing unnecessarily
            for key, value in inspect.getmembers(data, is_lazy):
                setattr(self, key, value)

        # init parent
        super().__init__(data.astype(int))  # lazyproperties will not be reset
        # on init!
        # self._use_zero = bool(use_zero)

        # optional named groups
        LabelGroupsMixin.__init__(self, label_groups)

        # hack so we can use `detect` from both class and instance
        self.detect = self._detect_refine

    def __str__(self):
        params = ['shape', 'nlabels', 'groups']
        return pformat({p: getattr(self, p) for p in params},
                       self.__class__.__name__)

    def __reduce__(self):
        # for some reason default object.__reduce__ borks with TypeError when
        # unpickling
        return self.__class__, (self.data,)

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
        # TODO: base class method should suffice in recent versions - romove

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
        #  There could be an array interface subclass for this

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
        return 0 in self.data

    @lazyproperty
    def slices(self):
        """
        Segment bounding boxes as dict of tuple of slices.  The object
        returned by this method has builtin vectorized item getting, so you
        can get many slices at once (as a list) by indexing it with a tuple,
        list, or np.ndarray.  Eg:
        >>> seg.slices[0, 1, 2]

        You can also choose only row or column slices like this:
        >>> seg.slices.x[7, 8] # returns a list of slices
        which is very nice.
        """
        s = {0: (slice(None),) * self.data.ndim}
        s.update(zip(self.labels, SegmentationImage.slices.fget(self)))
        return Sliced(s)

    def sliced(self, label):
        """
        Shorthand for
        >>> seg.data[seg.slices[label]]

        Returns
        -------
        np.ndarray
        """
        self.check_label(label)
        return self.data[self.slices[label]]

    @lazyproperty
    def heights(self):
        return self.get_heights()

    def get_heights(self, labels=None):
        return self._get_hw(labels, 0)

    @lazyproperty
    def widths(self):
        return self.get_widths()

    def get_widths(self, labels=None):
        return self._get_hw(labels, 0)

    def _get_hw(self, labels, i):
        l = self.shape[::-1][i]
        labels = self.resolve_labels(labels)
        return np.subtract(*zip(*(s.indices(l)[1::-1] for s in
                                  getattr(self.slices, 'yx'[i])[labels])))

    @lazyproperty
    def heights(self):
        return self.slices.heights

    @lazyproperty
    def max_label(self):
        if len(self.labels):
            return super().max_label
        else:
            # otherwise `np.max` borks with empty sequence
            return 0

    def make_cmap(self, background_color='#000000', random_state=random_state):
        # this function fails for all zero data since `make_random_cmap`
        # squeezes the rgb values into an array with shape (3,).  The parent
        # tries to set the background colour (a tuple) in the 0th position of
        # this array and fails. This overwrite would not be necessary if
        # `make_random_cmap` did not squeeze

        from photutils.utils.colormaps import make_random_cmap
        from matplotlib import colors

        cmap = make_random_cmap(self.max_label + 1, random_state=random_state)
        cmap.colors = np.atleast_2d(cmap.colors)

        if background_color is not None:
            cmap.colors[0] = colors.hex2color(background_color)

        return cmap

    def resolve_labels(self, labels=None, ignore=None, allow_zero=False):
        """

        Get the list of labels from input.  Default is to return the
        full list of unique labels in the segmentation image.  If a sequence
        of labels is provided, this method will check whether they are valid
        - i.e. contained in the segmentation image.

        Parameters
        ----------
        labels: sequence of int, or label group key, optional
            Sequence of labels to check. Can also be a key (any hashable
            object except tuple), in which case the `groups` it will be
            interpreted as a key to the `groups` attribute dictionary.
        ignore: sequence of int, or label group key, optional
            labels to ignore
        allow_zero: bool
            Whether to raise exception on presence of `0` label

        Returns
        -------
        array of int (labels)
        """

        if labels is None:
            labels = self.labels

        if isinstance(labels, numbers.Integral):
            # allow passing integers for convenience
            labels = [labels]

        if isinstance(labels, tuple):
            # interpret tuples as sequences of labels, not as a group name
            labels = list(labels)

        if isinstance(labels, col.Hashable):
            # interpret as a group label
            if labels not in self.groups:
                raise ValueError('Could not interpret object %r as a set of '
                                 'labels, or as a key to any label group.' %
                                 labels)
            else:
                labels = self.groups[labels]

        # remove labels that are to be ignored
        if ignore is not None:
            labels = np.setdiff1d(labels, self.resolve_labels(ignore))

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
        array of int (labels)

        """
        labels = np.atleast_1d(labels)
        valid = list(self.labels)

        if allow_zero:
            valid += [0]

        invalid = np.setdiff1d(labels, valid)
        if len(invalid):
            raise ValueError('Invalid label(s): %s' % str(tuple(invalid)))

        return labels

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
    # @classmethod
    # def from_bool_3d(cls, masks):
    #     # todo check masks is 3d array
    #     data = np.zeros(masks.shape[1:], int)
    #     for i, mask in enumerate(masks):
    #         data[mask] = (i + 1)
    #     return cls(data)

    def to_binary(self, labels=None, ignore_labels=(), expand=False):
        """
        Create binary mask where all pixels having values in `labels`
        have been masked
        """
        m = self.to_binary_3d(labels, ignore_labels)
        if expand:
            return m

        return m.any(0)

    def to_binary_3d(self, labels=None, ignore_labels=()):
        """
        Expand segments into 3D sequence of masks, one segment label per image
        """
        labels = self.resolve_labels(labels)
        if len(ignore_labels):
            labels = np.setdiff1d(labels, ignore_labels)
        return self.data[None] == labels[:, None, None]

    # def to_bool(self, labels=None, ignore_labels=()):
    #
    #     return self.to_bool_3d(labels, ignore_labels).any(0)

    @classmethod
    def from_binary(cls, array):
        data, n_obj = ndimage.label(array)
        return cls(data)

    # aliases
    # from_boolean = from_bool = from_binary
    as_mask = to_binary  # to_mask = to_boolean = to_bool =

    def mask_image(self, image, labels=None, ignore_labels=()):
        """
        Mask all pixels in the image having labels in `labels`, while ignoring
        labels in `ignore_labels`. By default all pixels with non-zero
        labelled pixels will be masked.
        """
        labels = self.resolve_labels(labels)
        if len(ignore_labels):
            labels = np.setdiff1d(labels, self.resolve_labels(ignore_labels))

        return np.ma.array(image, mask=self.to_binary(labels))

    # alias
    mask_sources = mask_image

    def mask_background(self, image):
        """Mask background (label <= 0)"""
        return np.ma.masked_where(self.data <= 0, image)

    def mask_3d(self, image, labels=None, mask0=True):
        """
        Expand image to 3D masked array, 1st index on label, with background (
        label<=0) optionally masked.
        """
        return self._mask_pixels(image, labels, mask0, self.to_binary_3d)

    # Data extraction
    # --------------------------------------------------------------------------
    # TODO: cutout       better name?  
    def select_subset(self, start, shape, type_=None):
        """
        FIXME: this description not accurate - padding happens

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
    def _mask_pixels(image, labels, mask0, masking_func):
        # method that does the work for isolating data in image from set of
        # labels
        mask = masking_func(labels)
        extracted = image * mask  # will upcast to 3d if mask is 3d
        if mask0:
            extracted = np.ma.MaskedArray(extracted, ~mask)
        return extracted

    def keep(self, labels, mask0=False):
        """
        Return segmentation data as an array keeping only `labels` and
        zeroing / masking everything else. This is almost like `keep_labels`,
        except that here the internal data remain unchanged and instead a
        modified copy of the data array is returned.

        Parameters
        ----------
        labels
        mask0

        Returns
        -------

        """
        return self._mask_pixels(self.data, labels, mask0, self.to_binary)

    def select_3d(self, image, labels, mask0=False):
        """
        Get data from image keeping only labeled regions and
        zeroing / masking everything else.

        Parameters
        ----------
        image
        labels
        mask0

        Returns
        -------
        3d array with first         dimension the size of labels
        """
        # TODO: update for nd ?
        return self._mask_pixels(image, labels, mask0, self.to_binary_3d)

    def clone(self, keep_labels=..., remove=None):
        """
        Return a modified copy keeping only labels in `keep_labels` and
        removing those in `remove`
        """
        if keep_labels is ...:
            keep_labels = self.labels
        else:
            keep_labels = self.resolve_labels(keep_labels)

        if remove is not None:
            keep_labels = np.setdiff1d(keep_labels, remove)

        return self.__class__(self.keep(keep_labels))

    # segment / slice iteration
    # --------------------------------------------------------------------------
    def iter_slices(self, labels=None):
        """

        Parameters
        ----------
        labels: sequence of int
            Segment labels to be iterated over

        Returns
        -------

        """
        for label in self.resolve_labels(labels):
            yield self.slices[label]

    def enum_slices(self, labels=None):
        """
        Yield (label, slice) pairs optionally filtering None slices
        """

        for label in self.resolve_labels(labels):
            yield label, self.slices[label]

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
                enum=False):
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
        masked: bool or sequence of ints
            If True:
                For each cutout, mask all "background" pixels (i.e. those
                with values not equal to `label`.
            If sequence of ints:
                Represents index number in the sequence of arrays for which
                the returned cutout is required to be masked.
                If the sequence contains an ellipsis, this indicates that all
                 remaining arrays in the sequence will be returned masked:
                 eg: (4, ...) means mask all arrays beyond the fourth.
            If none of the above:
                no masking is done, cutouts are `np.ndarray`
        flatten: bool
            Yield flattened arrays of data corresponding to label instead of a
            masked_array. This option is here since it is faster to check the
            sub-array labels than the entire frame if the slices have
            already been calculated.
            Note that both `mask` and `flatten` cannot be simultaneously True
             and will raise a ValueError
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

        for i, a in enumerate(arrays):
            if (a is not None) and np.ndim(a) < 2:
                raise ValueError('All arguments should be (nd)images or `None`.'
                                 ' Array %i is %id.' % (i, np.ndim(a)))

        # flags which determine which arrays in the sequence are to be mask
        flags = get_masking_flags(arrays, masked)
        if flags.any() and flatten:
            raise ValueError(
                "Use either `masked` or `flatten`. Can't do both.")

        # function that yields the result. neat little trick avoid unit tuples
        unpack = (next if n == 1 else tuple)
        if enum:
            def yields(lbl, gen):
                return lbl, unpack(gen)
        else:
            def yields(lbl, gen):
                return unpack(gen)

        # flag which determined arrays in the sequence are to mask
        if flatten:
            flags[:] = True

        for lbl, slice_ in self.enum_slices(labels):
            # note this propagates the `None`s in `arrays` tuple
            cuts_gen = (_2d_slicer(a, slice_,
                                   self.masks[lbl] if flg else None,
                                   flatten)
                        for a, flg in zip(arrays, flags))
            yield yields(lbl, cuts_gen)

    # Image methods
    # --------------------------------------------------------------------------
    def _check_input_data(self, data):
        """
        Check that input data has correct dimensions for compute

        Parameters
        ----------
        data : [type]
            [description]

        Raises
        ------
        ValueError
            If the input data has incorrect shape for compute
        """

        # convert list etc to array, but keep masked arrays
        data = np.asanyarray(data)

        # check shapes same
        shape = self.shape
        # print(data.ndim, data.shape, self.shape, data.shape[-len(shape):])
        if (2 > data.ndim > 4) or (data.shape[-len(shape):] != shape):
            raise ValueError('Arrays does not have the correct shape for '
                             'computation with segmentation data. Input data '
                             f'shape: {data.shape}, segmentation data shape '
                             f'{shape}')

        # TODO: maybe use numpy.broadcast_arrays if you want to do stats on
        #  higher dimensional data sets using this class.  You will have to
        #  think carefully on how to managed masked points, since replacing a
        #  label in the seg data will only work for 2d data.  May have to
        #  resort to `labelled_comprehension`

    def _relabel_masked(self, image, masked_pixels_label=None):
        # if image is masked, ignore masked pixels by using copy of segmentation
        # data with positions of masked pixels replaced by new label one
        # larger than current max label.
        # Note: Some statistical computations eg `mean`, `median`,
        #  `percentile` etc cannot be computed on the masked image by simply
        #  zero-filling the masked pixels since it will artificially skew the
        #  results. There is no significant performance difference between
        #  the 2 code paths, I checked.
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

    def thumbnails(self, image=None, labels=None, masked=False):
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
        self._check_input_data(data)
        return list(self.coslice(data, labels=labels, masked=masked))

    def flux(self, image, labels=None, bg=(0,), statistic_bg='median'):  #
        """
        An estimate of the net (background subtracted) source counts, and its
        associated uncertainty

        Parameters
        ----------
        image: ndarray
        labels: sequence of int
        bg: sequence of int, np.ndarray, SegmentationImage
            if sequence, must be of length 1 or same length as `labels`
            if array of dimensionality 2, must be same shape as image

        statistic_bg

        Returns
        -------

        """
        self._check_input_data(image)
        labels = self.resolve_labels(labels)

        # estimate sky counts and noise from data
        counts, counts_bg_pp, n_pix_src, n_pix_bg = self._flux(image, labels, bg)
        
        # bg subtracted source counts
        signal = counts - counts_bg_pp * n_pix_src

        # revised CCD equation from Merlin & Howell '95
        noise = np.sqrt(signal +  # ← poisson.      sky + instrument ↓
                        n_pix_src * (1 + n_pix_src / n_pix_bg) * counts_bg_pp)

        return signal, noise

    def _flux(self, image, labels=None, bg=(0,), stat='median'):

        # TODO: 3d array for overlapping bg regions?
        if isinstance(bg, np.ndarray) and (bg.shape == image.shape):
            bg = self.__class__(bg)

        # for convenience, allow `bg` to be a SegmentedImage!
        if isinstance(bg, SegmentationImage):
            assert bg.nlabels == len(labels)
            seg_bg = bg.copy()
            # exclude sources from bg segmentation!
            seg_bg.data[self.to_binary(labels)] = 0
        else:
            # bg is a list of labels
            bg = self.resolve_labels(bg, allow_zero=True)
            n_obj = len(labels)
            n_bg = len(bg)
            if n_bg not in (n_obj, 1):
                raise ValueError('Unequal number of background / object labels '
                                 f'({n_bg} / {n_obj})')
            seg_bg = self

        # resolve background counts estimate function
        stat = stat.lower()
        known_estimators = ('mean', 'median')
        if stat not in known_estimators:
            # TODO: allow callable?
            raise ValueError('Invalid value for parameter background statistic '
                             f'`stat`: {stat}')

        # get background stat
        counts_bg_pp = getattr(seg_bg, stat)(image)
        ones = np.ones_like(image)
        n_pix_bg = seg_bg.sum(ones, bg)
        n_pix_src = self.sum(ones, labels)
        # counts_bg = counts_bg_pp * n_pix_src

        # mean counts per binned pixel (src + bg)
        counts = self.sum(image, labels)
        return counts, counts_bg_pp, n_pix_src, n_pix_bg

    def snr(self, image, labels=None, bg=(0,)):
        """
        A signal-to-noise ratio estimate.  Calculates the SNR by taking the
        ratio of total background subtracted flux to the total estimated
        uncertainty (including poisson, sky, and instrumental noise) in each
        segment as per Merlin & Howell '95
        """

        signal, noise = flux(self, image, labels, bg)
        return signal / noise

    # def noise(self, image)

    def flux_sort(self, image, labels=None, bg=(0,), bg_stat='median'):
        """
        Re-label segments for highest per-pixel counts in descending order
        """
        labels = self.resolve_labels(labels)
        unsorted_labels = np.setdiff1d(self.labels, labels)
        n_sort = len(labels)

        flx, _ = self.flux(image, labels, bg, bg_stat)
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

    # def com(self, image=None, labels=None):
    #     """
    #     Center of Mass for each labelled segment
    #
    #     Parameters
    #     ----------
    #     image: 2d array or None
    #         if None, center of mass of segment for constant image is returned
    #     labels
    #
    #     Returns
    #     -------
    #
    #     """
    #     image = self.data.astype(bool) if image is None else image
    #     labels = self.resolve_labels(labels)
    #     return np.array(ndimage.center_of_mass(image, self.data, labels))

    def com_bg(self, image, labels=None, mask=None,
               background_estimator=np.ma.median, grid=None):
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

        # get background subtraction function
        bg_sub = image_sub(background_estimator)

        if grid is None:
            grid = np.indices(self.shape)

        labels = self.resolve_labels(labels)

        # generalized for higher dimensional data
        axes_sum = tuple(range(1, image.ndim))

        counter = itt.count()
        com = np.empty((len(labels), image.ndim))
        # u  = np.empty((len(labels), image.ndim))  # uncertainty
        for lbl, (seg, sub, msk, grd) in self.coslice(
                self.data, image, mask, grid, labels=labels, enum=True):
            #
            i = next(counter)

            # ignore whatever is in this slice and is not same label as well
            # as any external mask
            use = (seg == lbl)
            if msk is not None:
                use &= ~msk

            # noise = np.sqrt(sub)
            grd = grd[:, use]
            sub = bg_sub(sub)[use]

            # compute sum
            sum_ = sub.sum()
            if sum_ == 0:
                warnings.warn(f'Function `com_bg` encountered zero-valued '
                              f'image segment at label {lbl}.')
                # can skip next
                com[i] = np.nan
            else:
                # compute centre of mass
                com[i] = (sub * grd).sum(axes_sum) / sum_

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

        # catch for empty label vectors
        if old_labels.size == 0:
            return

        # forward_map = np.hstack([0, self.labels])
        # there may be labels missing
        forward_map = np.arange(self.max_label + 1)
        forward_map[old_labels] = new_labels
        self.data = forward_map[self.data]

    def dilate(self, iterations=1, connectivity=4, labels=None, mask=None,
               copy=False, structure=None):
        """
        Binary dilation on each labelled segment. This makes most sense when
        sources are not physically touching - if this is not the case sources
        will blend into each other in unexpected ways.

        Parameters
        ----------
        iterations
        connectivity
        labels
        mask
        copy
        structure
            if given, connectivity is ignored

        Returns
        -------

        """
        if not iterations:
            return self

        if not isinstance(iterations, numbers.Integral):
            raise ValueError('`iterations` should be an integer')

        # expand masks to 3D sequence
        labels = self.resolve_labels(labels)
        masks = self.to_binary(labels, expand=True)

        if structure is None:
            d = {4: 1, 8: 2}.get(connectivity)
            if d:
                structure = ndimage.generate_binary_structure(2, d)
            else:
                raise ValueError('Invalid connectivity={0}.  '
                                 'Options are 4 or 8'.format(connectivity))

        # structure array needs to have same dimensionality as masks
        if structure.ndim == 2:
            structure = structure[None]

        masks = ndimage.binary_dilation(masks, structure, iterations, mask)
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

    def auto_dilate(self, image, labels=None, dmax=5, sigma=3):
        #
        from obstools.stats import mad

        labels = self.resolve_labels(labels)

        if len(labels) == 0:
            return self

        count = 0
        while count <= dmax:
            self.logger.debug('round %i', count)

            mim = self.mask_sources(image, labels)
            m = np.ma.median(mim)
            σ = mad(mim, m)

            # get annuli - test pixels
            b3 = self.to_annuli(0, 1, labels)
            slices = self.slices.grow(labels, 2)
            for label, b, s in zip(labels, b3, slices):
                bb = b[s]
                pixels = mim[s][bb]
                w = np.array(np.where(bb))
                dark = pixels - m < σ * sigma
                if dark.all():
                    labels = np.setdiff1d(labels, label)
                else:
                    logger.debug('label: %i: %i pixels added', label,
                                 sum(~dark))
                    bb[tuple(w[:, dark])] = False
                    self.data[s][bb] = label

            count += 1

    def blend(self):
        """
        Inverse operation of deblend.  Merge segments that are touching
        but have different labels.
        """

        # first identify which labels are touching
        # old, new = [], []
        # for lbl, a in zip(self.labels,
        #                   self.to_annuli(width=1, remove_sources=False)):
        #     u = np.unique(self.data[a])
        #     z = (u != 0)
        #     old.extend(u[z])
        #     new.extend([lbl] * z.sum())
        #
        # # now, relabel
        # self.relabel_many(old, new)

        data, nlabels = ndimage.label(self.to_binary())
        return self.__class__(data)

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

    def to_annuli(self, buffer=0, width=5, labels=None, remove_sources=True):
        """
        Create 3D boolean array containing annular masks for sky regions.
        Regions containing labelled pixels from other sources within sky regions
        are removed unless otherwise requested.

        Parameters
        ----------
        buffer
        width
        labels
        remove_sources

        Returns
        -------

        """
        # sky annuli
        labels = self.resolve_labels(labels)
        masks = self.to_binary(labels, expand=True)
        # structure array needs to have same rank as masks
        struct = ndimage.generate_binary_structure(2, 1)[None]
        if buffer:
            m0 = ndimage.binary_dilation(masks, struct, iterations=buffer)
        else:
            m0 = masks
        #
        m1 = ndimage.binary_dilation(m0, struct, iterations=width)
        msky = (m1 & ~m0)

        if remove_sources:
            msky &= np.logical_not(masks.any(0))

        return msky

    def add_segments(self, data, label_insert=None, copy=False,
                     group_name=None):
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
        copy
        group_name

        Returns
        -------
        list of int (new labels)
        """

        assert self.shape == data.shape, \
            (f'Shape mismatch between {self.__class__.__name__} instance of '
             f'shape {self.shape} and data shape {data.shape}')

        # TODO: option to deal with overlapping segments

        if isinstance(data, SegmentationImage):
            input_labels = data.labels
            groups = data.groups
            data = data.data.copy()  # copy to avoid altering original below
            # TODO: extend slices instead to avoid recompute + performance
            #  gain. Is this reasonable if there is overlap?
        else:
            input_labels = np.unique(data)
            input_labels = input_labels[input_labels != 0]  # ignore background
            groups = {}

        #
        n_input_labels = len(input_labels)
        if n_input_labels == 0:
            # nothing to add
            return self, input_labels

        # check if we have the same labels in both
        duplicate_labels = np.intersect1d(input_labels, self.labels,
                                          assume_unique=True)
        has_dup_lbl = bool(len(duplicate_labels))

        # remap input data so we have non-overlapping labels
        new_labels = input_labels
        forward_map = np.arange(input_labels.max() + 1)
        if label_insert is None:
            # new labels are numbered starting from one above current max label
            if has_dup_lbl:
                # remap new labels
                new_labels = input_labels + self.max_label
                forward_map = np.zeros(input_labels.max() + 1)
                forward_map[input_labels] = new_labels
                data = forward_map[data]
        else:
            # shift labels that are larger than `label_insert` upwards so
            # they follow numerically from largest label in `self`
            new_labels = np.arange(n_input_labels) + label_insert + 1
            delta = new_labels[-1]
            forward_map[forward_map > label_insert] += delta
            data = forward_map[data]

            # current = self.data
            # current[current >= label_insert] += delta
            # data = current + data

        # update groups
        for gid, nrs in groups.items():
            groups[gid] = forward_map[nrs]

        if group_name is not None:
            groups[group_name] = new_labels

        # mask for new label regions
        new_pix = data.astype(bool)
        if copy:
            seg_data_out = self.data.copy()
            seg_data_out[new_pix] = data[new_pix]
            groups.update(self.groups)
            return self.__class__(seg_data_out, groups), new_labels
        else:
            self.data[new_pix] = data[new_pix]
            self._reset_lazy_properties()
            self.groups.update(groups)
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

    # def _dist_com_to_nearest(self):

    def _label_positions(self):
        # compute distance from centroid to nearest pixel of same label.

        # If this distance is large, it means the segment potentially has a
        # hole or is arc shaped, or is disjointed. For these we add the label
        # multiple times, one label for each separate part of the segment.
        # Decide on where to put the labels based on kmeans with 2 clusters.
        # note. this will give bad positions for some shapes, but is good
        #  enough for jazz

        if self.nlabels == 0:
            return {}

        # noinspection PyUnresolvedReferences
        yx = self.com(self.data)
        # yx may contain nans if labels not sequential....
        yx = yx[~np.isnan(yx).any(1)]
        w = np.array(np.where(self.to_binary(expand=True)))
        splix, = np.where(np.diff(w[0]) >= 1)
        s = np.split(w[1:], splix + 1, 1)
        gy, gx = (np.array(g, 'O')
                  for g in zip(*s))  # each of these is a list of arrays
        magic = [None, ...][bool(len(set(map(len, gy))) - 1)]
        # explicit object arrays avoids DeprecationWarning
        d = ((gy - yx[:, 0, magic]) ** 2 +
             (gx - yx[:, 1, magic]) ** 2) ** 0.5
        # d is an array of arrays!

        pos = {}
        for i, (label, dmin) in enumerate(zip(self.labels, map(min, d))):
            if dmin > 1:
                from scipy.cluster.vq import kmeans
                # center of mass not close to any pixels. disjointed segment
                # try cluster
                # todo: could try dbscan here for unknown number of clusters
                pos[label], _ = kmeans(np.array((gy[i], gx[i]), float).T, 2)
            else:
                pos[label] = yx[i][None]
        return pos

    def display(self, cmap=None, contours=False, bbox=False, label=True,
                **kws):
        # TODO: DisplayMixin: seg.show(), seg.show.term(), seg.show.overlay
        # seg.show.labels()
        """
        Plot the image using the `ImageDisplay` class


        Parameters
        ----------
        cmap:
        contours: bool
            Should contours be drawn around object perimeter
        bbox: bool
            Should rectangles be drawn representing the segment bounding boxes
        label: bool
            Should the segments be labelled with numbers on the image
        kws:
            passed to `ImageDisplay`

        Returns
        -------
        im: `ImageDisplay` instance
        """
        from graphing.imagine import ImageDisplay

        # draw segment labels
        # draw_labels = kws.pop('draw_labels', True)
        draw_rect = kws.pop('draw_rect', False)

        # use categorical colormap (random, muted colours)
        cmap = 'gray' if (self.nlabels == 0) else self.make_cmap()
        # conditional here prevents bork on empty segmentation image

        kws.setdefault('sliders', False)

        # plot
        im = ImageDisplay(self.data, cmap=cmap, **kws)

        if contours:
            lines = self.get_contours()
            im.ax.add_collection(lines)

        if bbox:
            self.slices.plot(im.ax)

        if label:
            # add label text (number) on each segment
            self.draw_labels(im.ax, color='w', fontdict=dict(weight='bold'))

        return im

    def draw_labels(self, ax, **kws):

        kws_ = dict(va='center', ha='center')
        kws_.update(**kws)
        for lbl, pos in self._label_positions().items():
            for x, y in pos[:, ::-1]:
                ax.text(x, y, str(lbl), **kws_)

    def display_term(self, show_labels=True, frame=True, origin=0):
        """
        A lightweight visualization of the segmented image for display
        on console.  This creates a string with colourised "pixels" using ANSI
        escape sequences. Useful for visualising source cutouts or
        small segmented images.  The string representation is printed
        to stdout and returned by this function.


        The string returned by this function, when printed, might look
        something like this, but with the different segments each coloured
        according to its label:

           _________________________________________________
           |                                               |
           |                          ██                   |
           |         ██             ██████        ██       |
           |       ██████             ██        ████       |
           |     ██████████                         ██     |
           |       ██████                                  |
           |         ██                                    |
           |                                ████           |
           |               ██             ████████         |
           |             ██████             ██████         |
           |             ████████             ██           |
           |               ████                            |
           |                 ██                            |
           |                                               |
           |                                               |
           |                                               |
           |             ██                        ██      |
           |           ██████                    ██████    |
           |             ██                    ██████████  |
           |                                     ██████    |
           |                                       ██      |
           |_______________________________________________|


        Parameters
        ----------
        show_labels: bool
            Whether to add region labels (numbers)
        frame: bool
            should a frame be drawn around the image area

        Returns
        -------
        str
        """

        im = self.format_term(show_labels, frame, origin)
        print(im)
        return im

    def format_term(self, show_labels=True, frame=True, origin=0, cmap=None):
        import motley
        from motley import codes

        origin = int(origin)
        assert origin in (0, 1)

        # re-orient data
        o = 1 if origin else -1
        data = self.data[::o].ravel()

        BORDER = '⎪'  # U+23aa Sm CURLY BRACKET EXTENSION ⎪  # '|'
        nm = 2  # number of characters that represent a pixel
        n = self.nlabels
        nr, nc = self.shape

        # mapping from indices to labels
        forward_map = np.zeros(self.max_label + 1, int)
        forward_map[self.labels] = np.arange(1, n + 1)

        # colour map
        if cmap is None:
            cmap = self.make_cmap()
            colours = [0] + list(self.labels)
        else:
            from matplotlib.pyplot import get_cmap
            cmap = get_cmap(cmap)
            colours = np.linspace(0, 1, self.nlabels + 1)

        # get the 24 bit colours
        # noinspection PyTypeChecker
        colours = cmap(colours, bytes=True)[:, :3]

        # create markers (using ANSI bg codes)
        # markers = np.array(codes.from_list(bg=colours))
        markers = np.char.add(codes.from_list(bg=colours), '  ')
        # `markers` are just the ansi colour codes for each label with 2
        # whitespace added to represent a pixel.

        # row end markers.
        if frame:
            ul = codes.get('underline')
            endings = ((f'{codes.END}{BORDER}\n{BORDER}',) * (nr - 2) +
                       (f'{codes.END}{BORDER}\n{ul}{BORDER}',
                        f'{codes.END}{ul}{BORDER}{codes.END}\n'))
        else:
            endings = (f'{codes.END}\n',) * nr

        # newline positions / strings
        marks = col.defaultdict(str)

        # make line end markers.  New line begins with a coloured pixel.
        nli = np.arange(nc, nr * nc + 1, nc)
        marks.update(zip(nli, endings))

        # positions where switch to different label / ANSI strings at said pos
        swi, = np.where(np.diff(data, prepend=0).astype(bool))
        # each new row should also start with new colour
        swi = np.union1d(swi, np.arange(0, nr * nc, nc))
        for i in swi:
            marks[i] += markers[forward_map[data[i]]]

        # get positions / str for labels
        if show_labels:
            for lbl, pos in self._label_positions().items():
                # TODO: label text colour
                label = '%-2i' % lbl
                if origin == 0:
                    pos[:, 0] = nr - 1 - pos[:, 0]

                for i in np.ravel_multi_index(
                        np.round(pos).astype(int).T, self.shape):
                    if i in marks:
                        marks[i] = marks[i].replace('  ', label)
                    else:
                        marks[i] = markers[lbl].replace('  ', label)

        # create string
        i0 = 0
        im = ''
        if frame:
            im = motley.underline(' ' * ((nc + 1) * nm)) + '\n' + BORDER

        for i, mrk in sorted(marks.items(), key=lambda _: _[0]):
            im += ' ' * ((i - i0 - 1) * nm) + mrk
            i0 = i

        im += codes.END
        return im

    def get_boundary(self, label, offset=-0.5):
        """

        Fast boundary trace for segment `label`.  This algorithm is designed
        to be general enough to handle disjoint segments with a single label
        and therefore always returns a list even if the segment is monolithic.


        Parameters
        ----------
        label: int
            The segment label for which to get the boundary
        offset: float
            The offset that will be added to the points, by default -0.5, so
            that the contours line up with imaged displayed with bottom left pixel center

        Returns
        -------
        segments: list of arrays
            line segments tracing each part of the segment

        perimeter: array
            length of the circumference of each part of the segment
        """
        self.check_label(label)
        s = self.slices[label]
        b = (self.data[s] == label)
        origin = [_.start for _ in s]
        count = 0
        segments = []
        perimeter = []
        # noinspection PyUnresolvedReferences
        while b.any():
            pixels, boundary, p = trace_boundary(b)
            # get outline image, fill it, difference original, repeat.
            # This will trace outer as well as inner boundaries for
            # segments with holes.
            tmp = np.zeros_like(b)
            tmp[tuple(pixels.T)] = True
            b[ndimage.binary_fill_holes(tmp)] = False
            #
            segments.append((boundary + origin + offset)[:, ::-1])
            perimeter.append(p)

            if count > 10:
                break
            count += 1

        # correction factor for perimeter length (from
        return segments, np.multiply(0.95, perimeter)

    def get_boundaries(self, labels=None):
        """Get traced outlines for many labels"""
        return {label: self.get_boundary(label)
                for label in self.resolve_labels(labels)}

    @lazyproperty
    def traced(self):
        return self.get_boundaries()

    # @lazyproperty
    def get_contours(self, labels=None, **kws):
        """
        Get the collection of lines that trace the circumference of the
        segments.

        Parameters
        ----------
        labels
        kws

        Returns
        -------
        matplotlib.collections.LineCollection

        """
        from matplotlib.collections import LineCollection

        # if not 'colors' in kws:
        cmap = self.make_cmap()

        # note: use PathPatch if you want to be able to hatch the regions.
        #  at the moment you cannot hatch individual paths in PathCollection
        boundaries = self.get_boundaries(labels)
        contours, _ = zip(*boundaries.values())

        # noinspection PyTypeChecker
        kws.setdefault('colors', cmap(list(boundaries.keys())))
        return LineCollection(list(mit.flatten(contours)), **kws)

    def show_overlay(self, ax, labels=None, **kws):
        lines = self.get_contours(labels, **kws)
        ax.add_collection(lines)
        return lines

    @lazyproperty
    def perimeter(self):
        """
        Circumference of each segment. For segments with multiple separate

        Definition taken from
        "Digital Image Processing: An Algorithmic Introduction Using Java"
            by Wilhelm Burger, Mark J. Burge
            section 11.4.2

        link:
            https://books.google.co.za/books?id=jCEi9MVfxD8C&pg=PA224&lpg=PA224&dq=roundness+binary+image&source=bl&ots=vEXADHjhGT&sig=ACfU3U0TdWBHIzvd7Hto2qTDBA9dPVtnNw&hl=en&sa=X&ved=2ahUKEwjL4s6EqZDoAhWksXEKHYmBD8sQ6AEwD3oECAkQAQ#v=onepage&q=roundness%20binary%20image&f=false

        Returns
        -------

        """
        segments, perimeter = zip(*self.traced.values())
        return list(perimeter)

    @lazyproperty
    def roundness(self):
        """Circularity"""
        return (4 * np.pi * self.areas /
                np.square(list(map(sum, self.perimeter))))

    def circularize(self):
        x = (..., None, None)
        r = np.sqrt(self.areas / np.pi)[x]
        d = np.square(np.indices(self.shape)[:, None] -
                      self.com(self.data).T[x]).sum(0)
        return self.__class__(
            np.sum((np.sqrt(d) < r) * np.arange(1, self.nlabels + 1)[x], 0))


# class SegmentationGroups(SegmentedImage, LabelGroupsMixin):
#
#     def __init__(self, data, use_zero=False):
#         super().__init__(data, use_zero)
#
#     def add_segments(self, data, label_insert=None, group=None, copy=False):
#         seg, new_labels = super().add_segments(data, label_insert, copy)
#         self.groups[group] = new_labels
#         return seg


class CachedPixelGrids():
    'todo'  # TODO


class SegmentsModelHelper(SegmentedImage):  # SegmentationModelHelper
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
