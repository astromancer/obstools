# import inspect
import functools
import itertools as itt
import warnings
from operator import attrgetter

import more_itertools as mit
import numpy as np
from astropy.stats import median_absolute_deviation as mad
from astropy.stats.sigma_clipping import sigma_clip
from astropy.utils import lazyproperty
from obstools.phot.utils import duplicate_if_scalar
from photutils.detection.core import detect_threshold  # slow import!!
from photutils.segmentation import detect_sources, SegmentationImage
from recipes.dict import ListLike, AttrReadItem
from recipes.logging import LoggingMixin
from recipes.string import get_module_name
from scipy import ndimage
from scipy.spatial.distance import cdist

import logging

# from IPython import embed

# module level logger
logger = logging.getLogger(get_module_name(__file__))


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


def check_image_drift(cube, nframes, mask=None, snr=5, npixels=10):
    """Estimate the maximal positional drift for stars"""

    #
    logger.info('Estimating maximal image drift for %i frames.', nframes)

    # take `nframes` frames evenly spaced across data set
    step = len(cube) // nframes
    maxImage = cube[::step].max(0)  #

    segImx = SegmentationHelper.detect(maxImage, mask, snr=snr, npixels=npixels,
                                       dilate=3)

    mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                      for (xs, ys) in segImx.slices], 0)
    return mxshift, maxImage, segImx


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


def iter_repeat_last(it):
    it, it1 = itt.tee(mit.always_iterable(it))
    return mit.padded(it, next(mit.tail(1, it1)))


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

    # calculate threshold without edges so that std accurately measured for
    # region of interest
    if edge_cutoff:
        border = make_border_mask(image, edge_cutoff)
        mask = mask | border

    # separate mask for threshold calculation (else mask gets duplicated
    # to threshold array)
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


def detect_loop(image, mask=None, bgmodel=None, snr=3., npixels=7,
                edge_cutoff=None, deblend=False, dilate=1):
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

    # TODO: can you do better with GMM ???

    # make iterables
    _snr, _npix, _dil = (iter_repeat_last(o) for o in (snr, npixels, dilate))

    data = np.zeros(image.shape, int)
    if mask is None:
        mask = np.zeros(image.shape, bool)

    background = results = None  # first round detection without bgmodel
    groups = []
    counter = itt.count(0)
    j = 0
    go = True
    while go:
        # keep track of iteration number
        count = next(counter)

        # detect
        sh = SegmentationHelper.detect(image, mask, background, next(_snr),
                                       next(_npix), edge_cutoff, deblend,
                                       next(_dil))

        # update mask, get labels
        labelled = sh.data
        new_mask = labelled.astype(bool)
        new_data = new_mask & np.logical_not(mask)
        new_labels = np.unique(labelled[new_data])

        # since we dilated the detection masks, we may now be overlapping with
        # previous detections. Remove overlapping pixels here
        # if dilate:
        #     overlap = mask & new_data
        #     new_data[overlap] = 0

        if not new_data.any():
            # done
            # final group of detections are determined by sigma clipping
            if bgmodel:
                resi = bgmodel.residuals(results, image)
            else:
                resi = image
            resi_masked = np.ma.MaskedArray(resi, mask)
            resi_clip = sigma_clip(resi_masked)  # TODO: make this optional
            new_data = resi_clip.mask & ~resi_masked.mask
            new_data = ndimage.binary_dilation(new_data) & ~resi_masked.mask
            labelled, nlabels = ndimage.label(new_data)
            new_labels = np.unique(labelled[labelled != 0])
            go = False

        # aggregate
        data[new_data] += labelled[new_data] + j
        mask = mask | new_mask
        groups.append(new_labels + j)
        j += new_labels.max()

        logger.info('detect_loop: round nr %i: %i new detections: %s',
                    count, len(new_labels), tuple(new_labels))
        # print('detect_loop: round nr %i: %i new detections: %s' %
        #       (count, len(new_labels), tuple(new_labels)))

        # slotmode HACK
        if count == 0:
            # BackgroundModel(sh, None, )
            # init bg model here
            segmbg = sh.copy()

        if bgmodel:
            # fit background
            imm = np.ma.MaskedArray(image, mask)
            results, residu = bgmodel._fit_reduce(imm)

            # background model
            background = bgmodel(results)

        break

    # TODO: log what you found

    return data, groups, results


# class NullSlice():
#     """Null object pattern for getitem"""
#
#     def __getitem__(self, item):
#         return None

# constructor used for pickling
# def _construct_SegmentedArray(*args):
#     return SegmentedArray(*args)


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
        # Input array is an already formed ndarray instance
        obj = np.array(input_array)

        # initialize with data
        super_ = super(SegmentedArray, cls)
        obj = super_.__new__(cls, obj.shape, int)
        super_.__setitem__(obj, ..., input_array)

        # add SegmentationHelper instance as attribute to be updated upon
        # changes to segmentation data
        obj.parent = parent
        return obj

    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        # set the data in the SegmentationHelper
        print('Hitting up set data')
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
            return

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


class Slices(np.recarray):
    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __new__(cls, parent):
        # parent in SegmentationHelper instance
        # get list of slices from super class SegmentationImage
        slices = SegmentationImage.slices.fget(parent)
        if parent.allow_zero:
            slices = [(slice(None), slice(None))] + slices

        # initialize np.ndarray with data
        super_ = super(np.recarray, cls)
        dtype = np.dtype(list(zip('yx', 'OO')))
        obj = super_.__new__(cls, len(slices), dtype)
        super_.__setitem__(obj, ..., slices)

        # add SegmentationHelper instance as attribute
        obj.parent = parent
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return NotImplemented

    def from_labels(self, labels=None):
        return self.parent.get_slices(labels)

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        return [(getattr(y, yss), getattr(x, xss)) for (y, x) in slices]

    def lower_left_corners(self, labels=None):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self.parent.get_slices(labels))

    def lower_right_corners(self, labels=None):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self.parent.get_slices(labels))

    def upper_right_corners(self, labels=None):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self.parent.get_slices(labels))

    def upper_left_corners(self, labels=None):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self.parent.get_slices(labels))

    llc = lower_left_corners
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    def widths(self):
        return np.subtract(*zip(*map(attrgetter('stop', 'start'), self.x)))


    def extents(self, labels=None):

        slices = self.parent.get_slices(labels)
        sizes = np.zeros((len(slices), 2))
        for i, sl in enumerate(slices):
            if sl is not None:
                sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
                            for s, sz in zip(sl, self.parent.shape)]
        return sizes

    def enlarge(self, labels, inc=1):
        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T

        urc = np.add(self.urc(labels), 1).clip(None, self.parent.shape)
        llc = np.add(self.llc(labels), -1).clip(0)
        slices = [tuple(slice(*i) for i in yxix)
                  for yxix in zip(*np.swapaxes([llc, urc], -1, 0))]
        return slices

    def around_centroids(self, image, size, labels=None):
        com = self.parent.centroid(image, labels)
        slices = self.around_points(com, size)
        return com, slices

    def around_points(self, points, size):

        yxhw = duplicate_if_scalar(size) / 2
        yxdelta = yxhw[:, None, None] * [-1, 1]
        yxp = np.atleast_2d(points).T
        yxss = np.round(yxp[..., None] + yxdelta).astype(int)
        # clip negative slice indices since they yield empty slices
        return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
                          for sz, ss in zip(self.parent.shape, yxss))))

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
    Additions to the SegmentationImage class.
        * classmethod for initializing from an image
        * support for iterating over segments / slices
        * support for masked arrays
        * methods for calculating counts / flux in segments of image
        * preparing masks for photometry
        *
        *
    """
    # This class is essentially a mapping layer that lives on top of images

    _allow_zero = False  # TODO: maybe use_zero more appropriate
    _slice_dtype = np.dtype(list(zip('yx', 'OO')))

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

    # Properties
    # --------------------------------------------------------------------------
    @property
    def data(self):
        """
        The 2D segmentation image.
        """
        return self._data

    @data.setter
    def data(self, data):
        self.set_data(SegmentedArray(data, self))

    def set_data(self, data):
        # now subclasses can over-write this function to implement stuff
        # without touching the property definition above. This should be in the
        # base class!

        # Note however that the array which this property points to
        # is mutable, so explicitly changing it's contents
        # (via eg: `segm.data[:] = np.pi`) will not delete the lazyproperties!
        # This class should become an array subclass to handle those kind of
        # things
        # SegmentationImage.data.fset(self, data)
        #
        # del self.masks

        # TODO: manage through class method use_negative / allow_negative
        # if np.min(data) < 0:
        #     raise ValueError('The segmentation image cannot contain '
        #                      'negative integers.')
        self._data = data
        # be sure to delete any lazy properties to reset their values.
        del (self.data_masked, self.shape, self.labels, self.nlabels,
             self.max, self.slices, self.areas, self.is_sequential, self.masks)

    def __array__(self, *args):
        """
        Array representation of the segmentation image (e.g., for
        matplotlib).
        """

        # add *args for case np.asanyarray(self, dtype=int)
        # this allows Segmentation class to be initialized from another
        # object of the same (or inherited) class

        return self._data

    @property
    def allow_zero(self):
        return self._allow_zero

    @allow_zero.setter
    def allow_zero(self, b):
        b = bool(b)
        if b is self._allow_zero:
            return

        if self._allow_zero:
            self.slices = self.slices[1:]
            self.labels = self.labels[1:]
        else:
            slices = [(slice(None), slice(None))]
            slices.extend(self.slices)
            self.slices = np.array(slices, self._slice_dtype)
            self.labels = np.hstack([0, self.labels])
        self._allow_zero = b

    @lazyproperty
    def slices(self):
        """
        The minimal bounding box slices for each labeled region as a numpy
        object array.
        """

        self.logger.info('computing slices!!!')

        slices = SegmentationImage.slices.fget(self)
        if self.allow_zero:
            slices = [(slice(None), slice(None))] + slices

        return np.array(slices, self._slice_dtype)  # array of slices :))
        # return Slices(self)

    def get_slices(self, labels=None):
        if labels is None:
            return self.slices
        labels = self.check_labels(labels) - (not self.allow_zero)
        return self.slices[labels]
        # since self.slices is np.ndarray we can get items with array of labels

    @lazyproperty
    def labels(self):
        labels = np.array(np.unique(self.data))
        if not self.allow_zero:
            return labels[1:]
        return labels

    @property
    def labels_nonzero(self):
        labels = self.labels
        return labels[labels > 0]

    def resolve_labels(self, labels=None):
        if labels is None:
            return self.labels
        return self.check_labels(labels)

    def check_labels(self, labels):
        """Make sure we have valid labels"""
        labels = np.atleast_1d(labels)
        valid = list(self.labels)
        invalid = np.setdiff1d(labels, valid)
        if len(invalid):
            raise ValueError('Invalid label(s): %s' % str(tuple(invalid)))

        return labels

    def index(self, labels):
        labels = self.resolve_labels(labels)
        if self.is_sequential:
            return labels - (not self.allow_zero)
        else:
            return np.digitize(labels, self.labels)

    # --------------------------------------------------------------------------
    @lazyproperty
    def masks(self):
        """
        For each (label, slice) pair: a boolean array containing False where
        any value in the segmented image is not equal to the value of that
        label.

        Returns
        -------
        list of arrays
        """

        if not self.allow_zero:
            masks = {0: (self.data != 0)}
        else:
            masks = {}  # mask for label 0 will be computed below

        for lbl, slice_ in zip(self.labels, self.slices):
            seg = self.data[tuple(slice_)]  # tuple avoids FutureWarning
            masks[lbl] = (seg != lbl)

        return masks

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
        """Mask background"""
        return np.ma.masked_where(self.data <= 0, image)

    # Data extraction
    # --------------------------------------------------------------------------
    def extract(self, image, labels=None, mask0=True):
        """
        Extract data corresponding to labelled segments from image,
        optionally masking background. A flattened array is returned
        """
        return self._extract(image, labels, mask0, self.to_bool)

    def extract_3d(self, image, labels=None, mask0=True):
        """
        Expand image to 3D masked array, indexed on segment index with
        background (label 0) masked.
        """
        return self._extract(image, labels, mask0, self.to_bool_3d)

    @staticmethod
    def _extract(image, labels, mask0, to_mask):
        #  method that does the work for select / extract
        mask = to_mask(labels)
        extracted = image * mask
        if mask0:
            extracted = np.ma.MaskedArray(extracted, ~mask)
        return extracted

    # --------------------------------------------------------------------------
    def select(self, labels, mask0=False):
        """
        Return segmentation data keeping only `labels` and zeroing / masking
        everything else.

        Parameters
        ----------
        labels
        mask0

        Returns
        -------

        """
        return self._extract(self.data, labels, mask0, self.to_bool)

    def select_3d(self, labels, mask0=False):
        """Return segmentation data keeping only requested labels"""
        return self._extract(self.data, labels, mask0, self.to_bool_3d)

    def clone(self, keep=None, remove=None):
        """Return a modified copy keeping only labels in `keep`"""
        if keep is None:
            keep = self.labels
        else:
            keep = np.atleast_1d(keep)

        if remove is not None:
            keep = np.setdiff1d(keep, remove)

        return self.__class__(self.select(keep))

    # segment / slice iteration
    # --------------------------------------------------------------------------
    def iter_slices(self, labels=None, filter_=False):
        """

        Parameters
        ----------
        labels

        Returns
        -------

        """

        slices = self.get_slices(labels)
        if filter_:
            slices = filter(None, slices)
        yield from slices

    def enum_slices(self, labels=None, filter_=True):
        """Yield (label, slice) pairs optionally filtering None slices"""

        labels = self.resolve_labels(labels)
        slices = self.get_slices(labels)
        pairs = zip(labels, map(tuple, slices))
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

    def coslice(self, *arrays, labels=None, mask_bg=False, flatten=False,
                enum=False, mask_which_arrays=(0,)):
        """
        Yields labelled sub-regions of multiple arrays sequentially as tuple of
        (optionally masked) arrays.

        Parameters
        ----------
        arrays: tuple of arrays
            May contain None, in which case None is returned instead of the
            sliced sub-array
        labels: array-like
            Sequence of integer labels
        mask_bg: bool          # FIXME: mask_bg or mask_background better!!
            if True:
                for each array; mask all data in each slice not equal to 0 or
                `label`
        mask_which_arrays:
            which arrays in the sequence will have masked segments output
        flatten: bool
            Return flattened arrays of data corresponding to label. This
            option is here since it is faster to check the sub-array labels
            than the entire frame if the slices have already been calculated.
            Note that you cannot have both mask and extract being True
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
        if not isinstance(mask_bg, bool):
            raise ValueError('mask_bg should be bool')

        if mask_bg and flatten:
            raise ValueError("Use either mask or flatten. Can't do both.")

        # function that yields the result
        unpack = (next if n == 1 else tuple)

        # print('flatten', flatten)
        # print('*', labels, '*'*88)

        mask_flags = np.zeros(len(arrays), bool)
        if mask_bg:
            # mask_which_arrays = np.array(mask_which_arrays, ndmin=1)
            mask_flags[mask_which_arrays] = True
        if flatten:
            mask_flags[:] = True

        for lbl, slice_ in self.enum_slices(labels):
            # print(lbl, slice_)
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

        # snip snip
        slice_ = (Ellipsis,) + tuple(slice_)
        sub = array[slice_]

        if mask is None or mask is False:
            return sub

        if extract:
            return sub[..., ~mask]

        ma = np.ma.MaskedArray(sub)
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

    def imax(self, image, labels=None, mask=None):
        """Indices of maxima in each segment"""

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

    def counts(self, image, labels=None):  # todo: support masked arrays?
        """Count raw pixel values in segments"""
        labels = self.resolve_labels(labels)
        return ndimage.sum(image, self.data, labels)

    def flux(self, image, labels=None):  # bgfunc=np.median
        """Average per-pixel count in each segment"""

        labels = self.resolve_labels(labels)
        counts = self.counts(image, labels)
        areas = self.areas[labels-1] # self.area(labels)

        # bg = bgfunc(image[self.data == 0])
        return (counts / areas)  # - bg

    def flux_sort(self, image):
        """Re-label segments for highest per-pixel counts in descending order"""
        flx = self.flux(image)
        flx_srt, lbl_srt = zip(*sorted(zip(flx, self.labels), reverse=1))

        # TODO: do this like relabel_sequential
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

    # @property #lazy
    # def subgrids(self, labels=None):
    #     return list(self.gen_grids(labels))
    #
    # def gen_grids(self, labels):
    #
    #     ix = self.indices(labels)
    #     slices = np.take(self.slices, ix, 0)
    #     grid = np.indices(self.shape)  # TODO: move to init??
    #
    #     for s in slices:
    #         if s is None:
    #             yield None
    #         else:
    #             sly, slx = s
    #             yield grid[:, sly, slx]

    def com(self, image, labels=None):
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

    centroid = centroids = com_bg

    def shift(self, offset):
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
        # sky annulli
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

    def add_segments(self, data, label_insert=None, replace=False, copy=False):
        """

        Parameters
        ----------
        data
        label_insert:
            If None; new labels come after current ones.
            If int; value for starting label of new data. Current labels will be
              shifted upwards.

        Returns
        -------
        list of int (new labels)

        NOTE: Assumes non-overlapping segments
        """
        if isinstance(data, SegmentationImage):
            new_labels = data.labels
            data = data.data
            # TODO: extend slices instead to avoid recompute + performance gain
        else:
            new_labels = np.unique(data)[1:]

        if replace:
            # new labels replace old ones
            self.data[data.astype(bool)] = 0
        else:
            # keep old labels over new ones
            data[self.data.astype(bool)] = 0

        if label_insert is None:
            # new labels are appended
            off = self.labels.max()
            new_labels += off
            data[data != 0] += off
            new = self.data + data
        else:
            current = self.data
            current[current >= label_insert] += new_labels.max()
            new = current + data

        if copy:
            return self.__class__(new), new_labels
        else:
            self.data = new  # this will reset the `lazyproperty`s
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
        from graphical.imagine import ImageDisplay

        kws.setdefault('cmap', self.cmap())
        return ImageDisplay(self.data, *args, **kws)


class LabelUser(object):
    """
    Mixin class for inherited classes that use the `SegmentationHelper`.
    Adds the `use_label` and `ignore_label` properties.  Whenever a function
    that takes the `label` parameter is called with the default value `None`,
    the labels in `use_labels` will be used instead. Helps with dynamically
    including / excluding labels and with grouping labelled segments together

    If initialized without any arguments, all labels in the segmentation image
    will be in use.

    """

    def __init__(self, use_labels=None, ignore_labels=None):
        if use_labels is None:
            use_labels = self.segm.labels
        if ignore_labels is None:
            ignore_labels = []

        self._use_labels = np.setdiff1d(use_labels, ignore_labels)
        self._ignore_labels = np.asarray(ignore_labels)

    @property
    def ignore_labels(self):  # ignore_labels / use_labels
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, labels):
        self._ignore_labels = np.asarray(labels, int)
        self._use_labels = np.setdiff1d(self.segm.labels, labels)

    @property
    def use_labels(self):  # ignore_labels / use_labels
        return self._use_labels

    @use_labels.setter
    def use_labels(self, labels):
        self._use_labels = np.asarray(labels, int)
        self._ignore_labels = np.setdiff1d(self.segm.labels, labels)

    def resolve_labels(self, labels=None):
        if labels is None:
            return self.use_labels
        return self.segm.check_labels(labels)

    @property
    def nlabels(self):
        return len(self.use_labels)

    @property
    def sizes(self):
        return [len(labels) for labels in self.values()]


class Record(AttrReadItem, ListLike):
    pass


class LabelGroups(Record):
    """
    Ordered dict with key access via attribute lookup. Also has some
    list-like functionality: indexing by int and appending new data.
    Best of both worlds.  Also makes sure labels are always arrays.
    """
    _auto_name_fmt = 'group%i'

    def _allow_item(self, item):
        return bool(len(item))

    def _convert_item(self, item):
        return np.array(item, int)

    @property
    def sizes(self):
        return list(map(len, self.values()))
        # return [len(item) for item in self.values()]

    # def inverse(self):
    #     return {lbl: gid for gid, labels in self.items() for lbl in labels}


class LabelGroupsMixin(object):
    """Mixin class for grouping and labelling image segments"""

    def __init__(self, groups=None):
        if groups is None:
            groups = zip(('bg', 'group0'),
                         ([0], [self.segm.labels_nonzero]))

        self._groups = LabelGroups(groups)


    @property
    def groups(self):
        return self._groups

    # todo
    # def remove_group()


class GriddedSegments(SegmentationHelper):  # SegmentedGrid
    """Mixin class for gridded models"""

    def __init__(self, data, grid=None):
        SegmentationHelper.__init__(self, data)

        if grid is None:
            grid = np.indices(data.shape)
        else:
            grid = np.asanyarray(grid)
            assert grid.ndim >= 2  # else the `grids` property will fail

        self.grid = grid

    @lazyproperty
    def subgrids(self):
        grids = []
        for sy, sx in self.iter_slices(self.labels):
            grids.append(self.grid[..., sy, sx])
        return grids

    def set_data(self, data):
        SegmentationHelper.set_data(self, data)
        del self.subgrids

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
        for lbl, (seg, sub, m) in self.coslice(self.data, image, mask,
                                               labels=labels, enum=1):
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
            g = self.subgrids[lbl - 1]  # NOTE: label 0 bad here
            # print(g.shape, sub.shape)
            com[next(counter)] = (sub * g).sum((1, 2)) / sum_  # may be nan
        return com

    # def radial_grids(self, labels=None, rmax=10):


# TODO:
# optionally keep track of distribution of CoM for each star
# keep track of changes in relative positions, and switch off the update step when below threshold
# once stability is reached, start photometry


class StarTracker(LabelUser, LabelGroupsMixin, LoggingMixin):
    """
    A class to track stars in a CCD image frame to aid in doing time series photometry.

    Stellar positions in each new frame are calculated as the background subtracted centroids
    of the chosen image segments. Centroid positions of chosen stars are filtered for bad
    values (or outliers) and then combined by a weighted average. The weights are taken
    as the SNR of the star brightness. Also handles arbitrary pixel masks.

    Constellation of stars in image assumed unchanging. The coordinate positions of the stars
    with respect to the brightest star are updated upon each iteration.


    """

    # TODO: groups.bad_pixels

    # TODO: bayesian version
    # TODO: make multiprocessing safe
    # TODO: pickle!

    segmClass = SegmentationHelper
    snr_weighting = True  # TODO: manage as properties ?
    snr_cut = 3
    _distance_cut = 10
    _saturation_cut = 95  # %

    # this allows tracking stars with low snr, but not computing their
    # centroids which will have larger scatter

    # TODO:
    # def from_fits(cls, filename, snr=3., npixels=7, edge_cutoff=3, deblend=False,
    #                flux_sort=True, dilate=1, bad_pixel_mask=None, edge_mask=None)

    @classmethod
    def from_image(cls, image, mask=None, bgmodel=None, snr=3., npixels=7,
                   edge_cutoffs=None, deblend=False, dilate=1, flux_sort=True):
        """Initialize the tracker from an image"""

        # detect and group
        segdata, groups, p0bg = detect_loop(image, mask, bgmodel,
                                            snr, npixels, edge_cutoffs,
                                            deblend, dilate)
        # init helper
        sh = SegmentationHelper(segdata)
        # last group of detections are those that have been sigma clipped.
        # Don't include these in found stars

        use_labels = np.hstack(groups[:-1])
        # ignore_labels = groups[-1]

        # use_labels = groups[0]
        # ignore_labels = None

        # Center of mass
        found = sh.com_bg(image, use_labels, mask)

        # cls.best_for_tracking(image)
        obj = cls(found, sh, groups, use_labels, mask, edge_cutoffs)
        # log nice table with what's been found.
        obj.logger.info('Found the following stars:\n%s\n', obj.pprint())

        return obj, p0bg

    @classmethod
    def from_cube_segment(cls, cube, ncomb, subset, nrand=None, mask=None,
                          bgmodel=None, snr=(10., 3.), npixels=7,
                          edge_cutoffs=None, deblend=False, dilate=1,
                          flux_sort=True):

        # select `ncomb`` frames randomly from `nrand` in the interval `subset`
        if isinstance(subset, int):
            subset = (0, subset)  # treat like a slice

        i0, i1 = subset
        if nrand is None:  # if not given, select from entire subset
            nrand = i1 - i0
        nfirst = min(nrand, i1 - i0)
        ix = np.random.randint(i0, i0 + nfirst, ncomb)
        # create ref image for init
        image = np.median(cube[ix], 0)
        cls.logger.info('Combined %i frames from amongst frames (%i->%i) for '
                        'reference image.', ncomb, i0, i0 + nfirst)

        # init the tracker
        tracker, p0bg = SlotModeTracker.from_image(
                image, mask, bgmodel, snr, npixels, edge_cutoffs, deblend,
                dilate, flux_sort)
        return tracker, image, p0bg

    def __init__(self, coords, segm, label_groups=None, use_labels=None,
                 bad_pixel_mask=None, edge_cutoffs=None):
        """
        Use to track the location of stars on CCD frame

        Parameters
        ----------
        coords : array-like
            reference coordinates of the stars
        segm : SegmentationHelper instance
            (or inherited from)


        ignore_labels : array-like
            indices of stars to ignore when calculating frame shift. These stars
            will still be tracked by their relative position wrt the others.
            Their relative positions will also be updated upon each call.
        """
        # counter for incremental update
        self.counter = itt.count(1)  # TODO: multiprocess!!!
        # store zero and relative positions separately so we can update them
        # independently
        self.ir = 0  # TODO: optionally pass in

        # pixel coordinates of reference position from which the shift will
        # be measured
        self.yx0 = coords[self.ir]
        self.rvec = coords - coords[self.ir]
        self.segm = GriddedSegments(segm)
        self._original_data = self.segm.data

        # init groups
        LabelGroupsMixin.__init__(self, label_groups)

        # TODO: once you figure out how to use negative labels, these things can
        # live inside the segmentation image.  also adapt_segments??
        self.bad_pixel_mask = bad_pixel_mask
        if edge_cutoffs is not None:
            self.edge_mask = make_border_mask(self.segm.data, edge_cutoffs)
        else:
            self.edge_mask = None
        #
        self.offset = [0, 0]

        # label logic
        LabelUser.__init__(self, use_labels)

    def __call__(self, image, mask=None):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """

        # more robust that individual CoM track.
        # eg: cosmic rays:
        # '/media/Oceanus/UCT/Observing/data/Feb_2015/MASTER_J0614-2725/20150225.001.fits' : 236

        if not ((mask is None) or (self.bad_pixel_mask is None)):
            mask |= self.bad_pixel_mask
        else:
            mask = self.bad_pixel_mask  # might be None

        # calculate CoM
        com = self.segm.com_bg(image, self.use_labels, mask)
        # weights
        weights = self.get_weights(com)

        # update relative positions from CoM measures
        self.update_rvec(com)

        # calculate median com shift using current best relative positions
        # optionally using snr weighting scheme
        if self.snr_weighting or self.snr_cut:
            weights = self.set_snr_weights(weights, image)

        off = self.update_offset(com, weights)

        # finally return the new coordinates
        return self.rcoo + off

    @property
    def ngroups(self):
        return len(self.groups)

    @property
    def nsegs(self):
        return self.segm.nlabels

    @property
    def rcoo(self):
        """
        Reference coordinates (yx). Computed as initial coordinates + relative coordinates to
        allow updating the relative positions upon call
        """
        return self.yx0 + self.rvec

    @property
    def rcoo_xy(self):
        """Reference coordinates (xy)"""
        return self.rcoo[:, ::-1]

    def pprint(self):
        from motley.table import Table

        # TODO: say what is being ignored

        # coo = [:, ::-1]  # , dtype='O')
        tbl = Table(self.rcoo_xy,
                    col_headers=list('xy'),
                    col_head_props=dict(bg='g'),
                    row_headers=self.use_labels,
                    # number_rows=True,
                    align='>',  # easier to read when right aligned
                    )

        return tbl

    def sdist(self):
        coo = self.rcoo
        return cdist(coo, coo)

    def mask_segments(self, image, mask=None):
        """
        Prepare a background image by masking stars, bad pixels and whatever
        else
        """
        # mask stars
        imbg = self.segm.mask_segments(image)
        if mask is not None:
            imbg.mask |= mask

        if self.bad_pixel_mask is not None:
            imbg.mask |= self.bad_pixel_mask

        return imbg

    def prep_masks_phot(self, labels=None):
        # NOTE: this is the only method that really uses the edge_mask...

        segm = self.segm
        m3d = segm.to_bool_3d()
        sky_mask = m3d.any(0)
        if not self.bad_pixel_mask is None:
            sky_mask |= self.bad_pixel_mask
        if self.edge_mask is not None:
            sky_mask |= self.edge_mask

        labels = self.resolve_labels(labels)
        indices = self.segm.index(labels)

        phot_masks = sky_mask & ~m3d[indices]
        return phot_masks, sky_mask

    def get_weights(self, com):
        # use = self.use_labels - 1
        # ign = self.ignore_labels - 1
        weights = np.ones(len(self.use_labels))

        # flag outliers
        bad = self.is_bad(com)
        self.logger.debug('bad: %s', np.where(bad)[0])
        weights[bad] = 0
        # weights[use[bad]] = 0
        return weights

    def set_snr_weights(self, weights, image):
        # snr weighting scheme (for robustness)

        # if weights is None:
        # use = self.use_labels - 1
        ix = np.where(weights != 0)[0]
        lbl = ix + 1

        weights[ix] = snr = self.segm.snr(image, lbl)
        # self.logger.debug('snr: %s', snr)
        # self.logger.debug('weights: %s', weights)

        # ignore stars with low snr (their positions will still be tracked,
        # but not used to compute new positions)
        low_snr = snr < self.snr_cut
        if low_snr.sum() == len(lbl):
            self.logger.warning('SNR for all stars below cut: %s < %.1f',
                                snr, self.snr_cut)
            low_snr = snr < snr.max()  # else we end up with nans

        weights[ix[low_snr]] = 0

        self.logger.debug('weights: %s', weights)

        if np.all(weights == 0):
            self.logger.warning('All weights are 0!')
            weights += 1

        return weights

    def is_bad(self, coo):
        """
        improve the robustness of the algorithm by removing centroids that are
        bad.  Bad is inf / nan / outside segment.
        """

        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter COM positions that are outside of detection regions
        lbad2 = ~self.segm.inside_segment(coo, self.use_labels)

        return lbad | lbad2

    def is_outlier(self, coo, mad_thresh=5, jump_thresh=5):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """
        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter centroid measurements that are obvious errors (large jumps)
        r = np.sqrt(np.square(self.rcoo - coo).sum(1))
        lj = r > jump_thresh

        if len(coo) < 5:  # scatter large for small sample sizes
            lm = np.zeros(len(coo), bool)
        else:
            lm = r - np.median(r) > mad_thresh * mad(r)

        return lj | lm | lbad

    def update_offset(self, coo, weights):
        """
        Calculate the offset from the previous frame

        Parameters
        ----------
        coo
        weights

        Returns
        -------

        """
        # shift calculated as snr weighted mean of individual CoM shifts

        offset = np.ma.average(coo - self.rcoo, 0, weights)
        self.logger.debug('offset: (%.2f, %.2f)', *offset)

        # shift from previous frame
        shift = offset - self.offset
        ishift = np.round(shift).astype(int).data  # array not ma

        # if frame has shifted by more than one pixel, update the segmentationImage
        if np.abs(ishift).sum():  # any shift greater or equal one pixel
            # shift the segmentation data
            ioff = np.round(offset).astype(int).data
            self.logger.debug('shifting to offset: (%i, %i)', *ioff)
            # TODO: lock here
            self.segm.data = ndimage.shift(self._original_data, ioff)

            # update pixel offset from original
            # self.offset = np.round(offset).astype(int).data # array not ma
            self.offset = offset.data

        return offset

    def update_rvec(self, coo, weights=None):
        """"
        Incremental average of relative positions of stars (since they are
        considered static)
        """
        # see: https://math.stackexchange.com/questions/106700/incremental-averageing
        vec = coo - coo[self.ir]
        n = next(self.counter)
        if weights is not None:
            weights = (weights / weights.max() / n)[:, None]
        else:
            weights = 1. / n

        inc = (vec - self.rvec) * weights
        self.logger.debug('rvec increment:\n%s', str(inc))
        self.rvec += inc

    # def get_shift(self, image):
    #     com = self.segm.com_bg(image)
    #     l = ~self.is_outlier(com)
    #     shift = np.median(self.rcoo[l] - com[l], 0)
    #     return shift

    def best_for_tracking(self, image, close_cut=None, snr_cut=snr_cut,
                          saturation=None):
        """
        Find stars that are best suited for centroid tracking based on the
        following criteria:
        """
        too_bright, too_close, too_faint = [], [], []
        msg = 'Stars: %s too %s for tracking'
        if saturation:
            too_bright = self.too_bright(image, saturation)
            if len(too_bright):
                self.logger.debug(msg, str(too_bright), 'bright')
        if close_cut:
            too_close = self.too_close(close_cut)
            if len(too_close):
                self.logger.debug(msg, str(too_close), 'close')
        if snr_cut:
            too_faint = self.too_faint(snr_cut)
            if len(too_faint):
                self.logger.debug(msg, str(too_faint), 'faint')

        ignore = functools.reduce(np.union1d,
                                  (too_bright, too_close, too_faint))
        ix = np.setdiff1d(np.arange(len(self.coords)), ignore)
        if len(ix) == 0:
            self.logger.warning('No suitable stars found for tracking!')
        return ix

    # def auto_window(self):
    #     sdist_b = self.sdist[snr > self._snr_thresh]

    def too_faint(self, image, threshold=snr_cut):
        crude_snr = self.segm.snr(image)
        return np.where(crude_snr < threshold)[0]

    def too_close(self, threshold=_distance_cut):
        # Check for potential interference problems from stars that are close together
        return np.unique(np.ma.where(self.sdist < threshold))

    def too_bright(self, data, saturation, threshold=_saturation_cut):
        # Check for saturated stars by flagging pixels withing 1% of saturation level
        # TODO: make exact
        lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        # TODO check if you can improve speed here - dont have to check entire array?

        satpix = np.where((lower < data) & (data < upper))
        b = np.any(np.abs(np.array(satpix)[:, None].T - self.coords) < 3, 0)
        w, = np.where(np.all(b, 1))
        return w


# def get_streakmasks(self, image, flx_thresh=2.5e3, width=6):
#     # Mask streaks
#     flx = self.segm.flux(image)
#     strkCoo = self.rcoo[flx > flx_thresh]
#     w = np.multiply(width / 2, [-1, 1])
#     strkRng = strkCoo[:, None, 1] + w
#     strkSlc = map(slice, *strkRng.round(0).astype(int).T)
#
#     strkMask = np.zeros(image.shape, bool)
#     for sl in strkSlc:
#         strkMask[:, sl] = True
#
#     return strkMask


# ModelContainer = namedtuple('ModelContainer', ('psf', 'bg'))

# class BetterTracker(StarTracker):
#
#     @classmethod
#     def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
#                    flux_sort=True, dilate=True, bad_pixel_mask=None):
#         # create segmentationHelper
#         sh = cls.segmClass.from_image(
#             image, snr, npixels, edge_cutoff, deblend, flux_sort, dilate)
#
#         # Center of mass
#         found = sh.com_bg(image, None, bad_pixel_mask)
#
#         return cls(found, sh, None, bad_pixel_mask)


# def __call__(self, image, mask=None):
#
#     if not ((mask is None) or (self.bad_pixel_mask is None)):
#         mask |= self.bad_pixel_mask
#     else:
#         mask = self.bad_pixel_mask  # might be None
#
#     #image, results = self.background_subtract(image, mask)
#     # track stars (on bg subtracted image)
#     # print('returning')
#     return StarTracker.__call__(self, image)


class SlotModeTracker(StarTracker):
    snr_cut = 1.5

    @classmethod
    def from_image(cls, image, bgmodel=None, snr=3., npixels=7,
                   edge_cutoffs=None,
                   deblend=False, dilate=1, flux_sort=True, bad_pixel_mask=None,
                   streak_threshold=1e3):
        #
        import slotmode

        # get the bad pixel mask
        if bad_pixel_mask is None:
            bad_pixel_mask = slotmode.get_bad_pixel_mask(image)

        # create edge mask for sky photometry
        if edge_cutoffs is None:
            edge_cutoffs = slotmode.get_edge_cutoffs(image)

        # init parent class
        tracker, p0bg = super(SlotModeTracker, cls).from_image(
                image, bgmodel, snr, npixels, edge_cutoffs, deblend, dilate,
                flux_sort, bad_pixel_mask)

        # mask streaks
        # tracker.bright = tracker.bright_star_labels(image)
        # tracker.streaks = False
        # tracker.streaks = tracker.get_streakmasks(
        #         tracker.rcoo[tracker.bright - 1])
        return tracker, p0bg

    # def __call__(self, image, mask=None):
    #
    #     # mask = self.bad_pixel_mask | self.streaks
    #     com = StarTracker.__call__(self, image, mask)
    #     # return com
    #
    #     # update streak mask
    #     # self.streaks = self.get_streakmasks(com[:len(self.bright)])
    #     return com

    def bright_star_labels(self, image, flx_thresh=1e3):
        flx = self.segm.flux(image)
        w, = np.where(flx > flx_thresh)
        w = np.setdiff1d(w, self.ignore_labels)
        coo = self.rcoo[w]
        bsl = self.segm.data[tuple(coo.round(0).astype(int).T)]
        return bsl

    # def mask_segments(self, image, mask=None):
    #     imbg = StarTracker.mask_segments(self, image, mask)
    #     # imbg.mask |= self.streaks
    #     return imbg

    # def get_streakmasks(self, coo, width=6):
    #     # Mask streaks
    #
    #     w = np.multiply(width / 2, [-1, 1])
    #     strkRng = coo[:, None, 1] + w
    #     strkSlc = map(slice, *strkRng.round(0).astype(int).T)
    #
    #     strkMask = np.zeros(self.segm.data.shape, bool)
    #     for sl in strkSlc:
    #         strkMask[:, sl] = True
    #
    #     return strkMask

    # return grid, data arrays
    # return [(np.where(m), image[m]) for m in reg]

    # def get_edgemask(self, xlow=0, xhi=None, ylow=0, yhi=None):
    #     """Edge mask"""
    #     # can create by checking derivative of vignette model
    #     # return

    #     def add_streakmasks(self, image, flx_thresh=2.5e3, width=6):

    #         strkMask = self.get_streakmasks(image, flx_thresh, width)
    #         labelled, nobj = ndimage.label(strkMask)
    #         novr = strkMask & (self.segm.data != 0) # non-overlapping streaks
    #         labelled[novr] = 0

    #         streak_labels = self.segm.add_segments(labelled)
    #         self.ignore_labels = np.union1d(self.ignore_labels, streak_labels)
    #         return streak_labels

    # def refine(self, image, bgmodel=None, use_edge_mask=True, snr=3., npixels=6,
    #            edge_cutoff=0, deblend=False, flux_sort=False, dilate=1):
    #     # mask bad pixels / stars for bg fit
    #     # TODO: combine use_edge_mask with edge_cutoff
    #     # TODO: incorporate in from_image method??
    #
    #     # mask stars, streaks, bad pix
    #     imm = self.segm.mask_segments(image)  # stars masked
    #     # starmask = imm.mask.copy()
    #     use_edge_mask = use_edge_mask and self.edge_mask is not None
    #     if self.bad_pixel_mask is not None:
    #         imm.mask |= self.bad_pixel_mask
    #     if use_edge_mask:
    #         imm.mask |= self.edge_mask
    #     # if self.streaks is not None:
    #     #     imm.mask |= self.streaks
    #
    #     if bgmodel:
    #         # fit background
    #         results = bgmodel.fit(np.ma.array(imm, mask=self.streaks))  # = (ry, rx)
    #         # input image with background model subtracted
    #         im_bgs = bgmodel.residuals(results, image)
    #
    #         # do detection run without streak masks since there may be faint but
    #         # detectable stars partly in the streaks
    #         resi_sm = np.ma.array(im_bgs, mask=imm.mask)
    #         # prepare frame: fill bright stars
    #         resif = np.ma.filled(resi_sm, np.ma.median(resi_sm))
    #     else:
    #         resif = np.ma.filled(imm, np.ma.median(imm))
    #         im_bgs = image
    #
    #     # detect faint stars
    #     new_segm = SegmentationHelper.from_image(resif, snr, npixels,
    #                                              edge_cutoff, deblend,
    #                                              flux_sort, dilate)
    #     # since we dilated the detection masks, we may now be overlapping with
    #     # previous detections. Remove overlapping pixels here
    #     if dilate:
    #         overlap = new_segm.data.astype(bool) & self.segm.data.astype(bool)
    #         new_segm.data[overlap] = 0
    #
    #     # remove detections with small areas:
    #     # l = tracker.segm.areas[1:] < npixels
    #
    #     # add new detections
    #     faint_labels = self.segm.add_segments(new_segm)
    #     # calculate centroids of new detections
    #     if len(faint_labels):
    #         coords = self.segm.com_bg(im_bgs, mask=self.bad_pixel_mask)
    #         # ignore the faint stars for subsequent frame shift calculations
    #         self.ignore_labels = np.union1d(self.ignore_labels, faint_labels)
    #
    #         # update coords
    #         self.yx0 = coords[self.ir]
    #         self.rvec = coords - self.yx0
    #
    #         # set bg model
    #         self.bgmodel = bgmodel
    #
    #         return faint_labels, results, im_bgs
    #     else:
    #         return faint_labels, None, image

# from graphical.imagine import FitsCubeDisplay
# from matplotlib.widgets import CheckButtons
#
#
# class GraphicalStarTracker(
#         FitsCubeDisplay):  # TODO: Inherit from mpl Animation??
#
#     trackerClass = StarTracker
#     marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)
#
#     def __init__(self, filename, **kws):
#         FitsCubeDisplay.__init__(self, filename, ap_prop_dict={},
#                                  ap_updater=None, **kws)
#
#         self.outlines = None
#
#         # toDO
#         self.playing = False
#
#         # bbox = self.ax.get_position()
#         # rect = bbox.x0, bbox.y1, 0.2, 0.2
#         rect = 0.05, 0.825, 0.2, 0.2
#         axb = self.figure.add_axes(rect, xticks=[], yticks=[])
#         self.image_buttons = CheckButtons(axb, ('Tracking Regions', 'Data'),
#                                           (False, True))
#         self.image_buttons.on_clicked(self.image_button_action)
#
#         # window slices
#         # rect = 0.25, 0.825, 0.2, 0.2
#         # axb = self.figure.add_axes(rect, xticks=[], yticks=[])
#         # self.window_buttons = CheckButtons(axb, ('Windows', ), (False, ))
#         # self.window_buttons.on_clicked(self.toggle_windows)
#
#         self.tracker = None
#         self._xmarks = None
#         # self._windows = None
#
#     # def toggle_windows(self, label):
#
#     def image_button_action(self, label):
#         image = self.get_image_data(self.frame)
#         self.imagePlot.set_data(image)
#         self.figure.canvas.draw()
#
#     def init_tracker(self, first, **findkws):
#         if isinstance(first, int):
#             first = slice(first)
#         img = np.median(self.data[first], 0)
#         self.tracker = self.trackerClass.from_image(img, **findkws)
#
#         return self.tracker, img
#
#     @property
#     def xmarks(self):
#         # dynamically create track marks when first getting this property
#         if self._xmarks is None:
#             self._xmarks, = self.ax.plot(*self.tracker.rcoo[:, ::-1].T,
#                                          **self.marker_properties)
#         return self._xmarks
#
#     def get_image_data(self, i):
#         tr, d = self.image_buttons.get_status()
#         image = self.data[i]
#         trk = self.tracker
#         if (tr and d):
#             mask = trk.segm.to_bool(trk.use_labels)
#             data = np.ma.masked_array(image, mask=mask)
#             return data
#         elif tr:  # and not d
#             return trk.segm.data
#         else:
#             return image
#
#     def set_frame(self, i, draw=True):
#         self.logger.debug('set_frame: %s', i)
#
#         i = FitsCubeDisplay.set_frame(self, i, False)
#         # needs_drawing = self._needs_drawing()
#
#         # data = self.get_image_data(i)
#
#         centroids = self.tracker(self.data[i])  # unmask and track
#         self.xmarks.set_data(centroids[:, ::-1].T)
#
#         if self.outlines is not None:
#             segments = []
#             off = self.tracker.offset[::-1]
#             for seg in self.outlineData:
#                 segments.append(seg + off)
#             self.outlines.set_segments(segments)
#
#         # if self.use_blit:
#         #     self.draw_blit(needs_drawing)
#
#     def show_outlines(self, **kws):
#         from matplotlib.collections import LineCollection
#         from matplotlib._contour import QuadContourGenerator
#
#         segm = self.tracker.segm
#         data = segm.data
#         outlines = []
#         for s in segm.slices:
#             sy, sx = s
#             e = np.array([[sx.start - 1, sx.stop + 1],
#                           [sy.start - 1, sy.stop + 1]])
#             im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]
#             f = lambda x, y: im[int(y), int(x)]
#             g = np.vectorize(f)
#
#             yd, xd = im.shape
#             x = np.linspace(0, xd, xd * 25)
#             y = np.linspace(0, yd, yd * 25)
#             X, Y = np.meshgrid(x[:-1], y[:-1])
#             Z = g(X[:-1], Y[:-1])
#
#             gen = QuadContourGenerator(X[:-1], Y[:-1], Z, None, False, 0)
#             c, = gen.create_contour(0)
#             outlines.append(c + e[:, 0] - 0.5)
#
#         col = LineCollection(outlines, **kws)
#         self.ax.add_collection(col)
#
#         self.outlineData = outlines
#         self.outlines = col
#         return col
#
#     def play(self):
#         if self.playing:
#             return
#
#     def pause(self):
#         'todo'