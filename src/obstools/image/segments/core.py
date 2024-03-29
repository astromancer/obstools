"""
Extensions for segmentation images
"""


# std
import inspect
import numbers
import itertools as itt
from collections import abc

# third-party
import numpy as np
from scipy import ndimage
from loguru import logger
from astropy.utils import lazyproperty
from photutils.segmentation import SegmentationImage, deblend_sources

# local
from recipes import api, dicts
from recipes.functionals import echo0
from recipes.logging import LoggingMixin

# relative
from ...utils import prod
from ...stats import geometric_median
from ..utils import get_overlap
from ..detect import DEFAULT_ALGORITHM, SourceDetectionDescriptor
from .slices import SliceDict
from .trace import trace_boundary
from .stats import MaskedStatsMixin
from .utils import is_lazy
from .groups import LabelGroupsMixin, auto_id
from .display import SegmentPlotter, make_cmap
from .masks import MaskContainer, SegmentMasksMixin

# ---------------------------------------------------------------------------- #
KNOWN_BG_STATS = {'mean', 'median'}


def resolve_bg(stat):
    if (stat := stat.lower()) in KNOWN_BG_STATS:
        return stat

    # if callable(stat):
    #     return stat

    raise ValueError(f'Unknown statistic {stat!r} for computing background '
                     f'flux. Valid choices are: {KNOWN_BG_STATS}, or '
                     f'a custom callable.')


def image_sub(background_estimator):
    if background_estimator in (None, False):
        return echo0

    def sub(image):
        return image - background_estimator(image)

    return sub


# ---------------------------------------------------------------------------- #
def _unpack(label, section, item):
    return item


def _unpack_with_label(label, section, item):
    return label, item


def _unpack_with_slice(label, section, item):
    return section, item


def _unpack_with_label_and_slice(label, section, item):
    return label, section, item


def _get_unpacker(with_label, with_slice):
    return (
        (_unpack, _unpack_with_slice),
        (_unpack_with_label, _unpack_with_label_and_slice)
    )[with_label][with_slice]


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


def _slice2d(array, slices, mask=None, compress=False):
    """
    Slice `np.ndarray` object along last 2 dimensions. If array is None, simply
    return None.

    Parameters
    ----------
    array:  np.ndarray
    slices: tuple of slices
    mask:   bool, np.ndarray or None (default)
    compress: bool

    Returns
    -------
    array or None
    """

    if array is None:
        return None  # propagate `None`s

    # slice along last two dimensions
    cutout = array[tuple((..., *slices))]

    # NOTE next line must work for `mask` an array!
    if (mask is None) or (mask is False):
        return cutout

    if compress:
        # pylint: disable=invalid-unary-operand-type
        return cutout[..., ~mask]

    ma = np.ma.MaskedArray(cutout, copy=True)
    ma[..., mask] = np.ma.masked
    return ma


def _check_arrays(arrays):
    for i, a in enumerate(arrays):
        a = np.asanyarray(a)
        if (a is not None) and a.ndim < 2:
            raise ValueError('All arguments should be (nd)images or `None`.'
                             f' Array {i} is {a.ndim}D.')
        yield a

# ---------------------------------------------------------------------------- #
# class ModelledSegment(Segment):
#     def __init__(self, segment_img, label, slices, area, model=None):
#         super().__init__(segment_img, label, slices, area)
#         self.model = model
#         # self.grid =
#         #


# class Centrality():
#     'TODO: com / gmean / mean / argmax'  # maybe


# def is2d(data):
#     return (nd := data.ndim) == 2 or (nd == 3 and data.shape[0] == 1)


# SegmentImage


class SegmentedImage(SegmentationImage,     # base
                     MaskedStatsMixin,      # stats methods for masked images
                     SegmentMasksMixin,     # handles masks for foreground obj
                     LabelGroupsMixin,      # keeps track of label groups
                     LoggingMixin):         # logger
    """
    Extends `photutils.segmentation.SegmentationImage` functionality.

    Additions to the SegmentationImage class.
        * classmethod for construction from an image of sources

        * support for iterating over segments / slices with `cutouts` method.

        * Calculations on masked arrays:
        * methods for statistics on image segments (min, max, mean, median, etc)
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
          seen as unnecessarily restrictive
    """

    # In terms of modelling, this class also functions as a domain mapping layer
    # that lives on top of images.

    # Source detection
    # ------------------------------------------------------------------------ #
    detection = SourceDetectionDescriptor(DEFAULT_ALGORITHM)

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
    def from_image(cls, image, dilate=0, flux_sort=True, **kws):
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
        obj = cls.detect(image, dilate=dilate, **kws)

        if flux_sort:
            obj.flux_sort(image)

        return obj

    @classmethod
    def detect(cls, image, mask=None, dilate=0, **kws):
        """
        Image object detection that returns a SegmentedImage instance

        Parameters
        ----------
        image
        mask
        dilate


        Returns
        -------

        """

        # Initialize
        seg = cls.detection(image, mask, **kws)

        # dilate
        if dilate != 'auto':
            seg.dilate(iterations=dilate)

        cls.logger.opt(lazy=True).info(
            'Detected {:d} objects covering {:d} pixels.',
            seg.nlabels, lambda: seg.to_binary().sum()
        )

        return seg

    # --------------------------------------------------------------------------
    def _detect_refine(self, image, mask=False, dilate=0,
                       group_id=None, ignore_labels=(),
                       **kws):
        """
        Refine detection by running on image with current segments masked.

        note: This method will be run when calling `detect` from an instance
          of this class. It runs a detection on the background region of the
          image - i.e. all detected objects masked out except those in
          `ignore_labels`

        group_id: hashable object
            Key representing the name of this group. This parameter controls
            whether the new segments will be added to the original
            SegmentedImage. If it is given
        """
        #
        # check if group key ok
        if group_id is auto_id:
            group_id = self.groups.auto_key()

        if not isinstance(group_id, abc.Hashable):
            raise ValueError('Group name {group_id} cannot be used since '
                             'it is not a hashable object.')

        # dilate existing labels
        if self.nlabels and (dilate == 'auto'):
            self.auto_dilate(image, sigma=kws.get('snr', 3))
            dilate = 0

        # update mask, get new labels
        if mask is None:
            mask = False
        mask = self.to_binary(None, ignore_labels) | mask

        # run detect classmethod
        # print(kws)
        new = type(self).detect(image, mask, dilate, **kws)

        # since we dilated the detection masks, we may now be overlapping
        # with previous detections. Remove overlapping pixels here.
        if dilate:
            overlap = self.to_binary() & new.to_binary()
            new.data[overlap] = 0

        # add labels and group
        if group_id:
            new, _ = self.add_segments(new, group_id=group_id)

        return new

    # Initialize
    # ------------------------------------------------------------------------ #
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
        return dicts.pformat({p: getattr(self, p) for p in params},
                             self.__class__.__name__)

    # def __reduce__(self):
    #     # for some reason default object.__reduce__ borks with TypeError when
    #     # unpickling
    #     return self.__class__, (self.data,)

    def __array__(self, *_):
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
        """Reset all lazy properties. Will work for subclasses"""
        for key, _ in inspect.getmembers(self.__class__, is_lazy):
            self.__dict__.pop(key, None)
        # TODO: base class method should suffice in recent versions - remove

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
        """Check if there are any zeros in the segmentation image."""
        return 0 in self.data

    @lazyproperty
    def slices(self):
        """
        Segment bounding boxes as dict of tuple of slices. The object
        returned by this method has builtin vectorized item getting, so you
        can get many slices at once (as a list) by indexing it with a tuple,
        list, or np.ndarray.

        Examples
        --------
        >>> seg.slices[0, 1, 2]

        You can also choose only row or column slices like this:
        >>> seg.slices.x[7, 8] # returns a list of slices which is very nice.

        """
        s = {0: (slice(None),) * self.data.ndim}
        s.update(zip(self.labels, SegmentationImage.slices.fget(self)))
        return SliceDict(s)

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
    def heights(self):  # TODO: manage these through MethodVectorizer api
        """Vector of segment heights"""
        return self.get_heights()

    def get_heights(self, labels=None):
        return self._get_hw(labels, 0)

    @lazyproperty
    def widths(self):
        """Vector of segment widths"""
        return self.get_widths()

    def get_widths(self, labels=None):
        return self._get_hw(labels, 1)

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
        # otherwise `np.max` borks with empty sequence
        return super().max_label if self.labels.size else 0

    def make_cmap(self, background_color='#000000', seed=None):
        return make_cmap(self.max_label + 1, background_color, seed)

    def resolve_labels(self, labels=None, ignore=None, allow_zero=False):
        """

        Get the list of labels from input. Default is to return the
        full list of unique labels in the segmentation image. If a sequence
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

        if (labels is None) or (labels is all) or (labels is ...):
            labels = self.labels

        if isinstance(labels, numbers.Integral):
            # allow passing integers for convenience
            labels = [labels]

        if isinstance(labels, (tuple, set)):
            # interpret tuples as sequences of labels, not as a group name
            labels = list(labels)

        if isinstance(labels, abc.Hashable):
            # interpret as a group label
            if labels not in self.groups:
                raise ValueError(
                    f'Could not interpret object {labels!r} as a set of labels,'
                    ' or as a key to any label group.'
                )

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
            raise ValueError(f'Invalid label(s): {tuple(invalid)}')

        return labels

    @lazyproperty
    def mask0(self):  # todo use `masked_background` now in photutils
        return self.data == 0

    @lazyproperty
    def fractional_areas(self):
        return self.areas / prod(self.shape)

    @lazyproperty
    def areas(self):
        """
        A 1D array of areas (in pixel**2) of the non-zero labeled
        regions.

        The `~numpy.ndarray` starts with the *non-zero* label.  The
        returned array has a length equal to the number of labels and
        matches the order of the ``labels`` attribute.
        """
        slices = iter(self.slices.values())
        next(slices)  # full image slice for zero label
        return np.array([
            np.count_nonzero(self._data[slices] == label)
            for label, slices in zip(self.labels, slices)
        ])

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
        return m if expand else m.any(0)

    def to_binary_3d(self, labels=None, ignore_labels=()):
        """
        Expand segments into 3D sequence of masks, one segment label per image
        """
        labels = self.resolve_labels(labels)
        if len(ignore_labels):
            labels = np.setdiff1d(labels, ignore_labels)
        return self.data[None] == labels[:, None, None]

    @classmethod
    def from_binary(cls, array):
        # data, n_obj
        data, _ = ndimage.label(array)
        return cls(data)

    # aliases
    # from_boolean = from_bool = from_binary
    as_mask = to_boolean = to_binary  # to_mask = to_bool

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
    def get_overlap(self, origin, shape, type_=None):
        """
        Select a sub-region of the segmented image starting at yx-indices
        `origin` and having dimension `shape`. If `origin` and `shape` imply a
        frame which only partialy overlaps the segmentation, zeroes are filled
        in.

        Parameters
        ----------
        origin
        shape
        type_:
            Class of the returned object. Can be any callable that takes an
            array as it's first argument, but will typically be a subclass of
            `SegmentationImage`, or an `np.ndarray`. If None (default), an
            instance of this class is returned.

        Returns
        -------
        SegmentationImage
        """
        if type_ is None:
            type_ = self.__class__
        return type_(get_overlap(self, self.data, origin, shape))

    # alias
    select_region = select_overlap = get_overlap

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
        for _, cutout in self.enum_slices(labels):
            yield cutout

    def enum_slices(self, labels=None):
        """
        Yield (label, slice) pairs optionally filtering None slices
        """

        for label in self.resolve_labels(labels):
            yield label, self.slices[label]

    def iter_segments(self, image, labels=None):
        """
        Yields flattened sub-regions of the image sequentially by label.

        Parameters
        ----------
        image
        labels

        Yields
        ------

        """
        yield from self.cutouts(image, labels=labels, flatten=True)

    @api.synonyms(
        {'enum(erated?)?':  'labelled',
         'with_label':      'labelled',
         'with_slice':      'with_slices',
         'mask':            'masked',
         'flat(ten(ed)?)?': 'compress'},
        action=None
    )
    def cutouts(self, *arrays, labels=None, masked=False, compress=False,
                labelled=False, extend=0, with_slices=False):
        """
        Yields labelled sub-regions of multiple arrays sequentially as tuple of
        (optionally masked or compressed) arrays.

        Parameters
        ----------
        arrays: One or more ndarray(s)
            Arrays can have any dimensionality greater or equal 2 - cutouts are
            made along the last 2 dimensions. May contain None, in which case
            None is returned in the same position instead of the sliced
            sub-array. 
        labels: array-like
            Sequence of integer labels.
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
                No masking is done, resulting image cutouts are `np.ndarray`.
        compress: bool
            Yield compressed 1D arrays of data corresponding to label instead of
            a masked_array. This option is here since it is faster to check the
            sub-array labels than the entire frame if the slices have already
            been calculated. Note that both `mask` and `compress` cannot be
            simultaneously True and will raise a ValueError.
        labelled: bool
            Each (tuple of) cutout(s) is enumerate with `label` value in the
            fashion of the builtin enumerate.


        Yields
        ------
        np.ndarray or tuple
            (tuple of) ((possibly label) and (masked) array(s))
        """

        n = len(arrays)
        # checks
        if n == 0:
            raise ValueError('Need at least one array to slice.')

        # check dimensions
        arrays = list(_check_arrays(arrays))

        # flags which determine which arrays in the sequence are to be mask
        flags = get_masking_flags(arrays, masked)
        if flags.any() and compress:
            raise ValueError("Use either `masked` or `compress`. Can't do both.")

        # function that yields the result. neat little trick avoid unit tuples
        _next = (next if n == 1 else tuple)
        yielder = _get_unpacker(labelled, with_slices)

        # flag which determined arrays in the sequence are to mask
        if compress:
            flags[:] = True

        if extend := int(extend):
            labels = self.resolve_labels(labels)
            sections = self.slices.extend(labels, extend, clip=self.shape)
            itr = zip(labels, sections)
            masks = {l: (self.data[s] != l) for l, s in zip(labels, sections)}
        else:
            itr = self.enum_slices(labels)
            masks = self.masks

        for label, section in itr:
            # NOTE this propagates values that are `None` in `arrays` tuple
            cutouts = (_slice2d(_, section,
                                masks[label] if flag else None,
                                compress)
                       for _, flag in zip(arrays, flags))

            yield yielder(label, section, _next(cutouts))

    # alias
    coslice = cutout = cutouts

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
        # data = np.asanyarray(data)

        # check shapes same
        shape = self.shape
        # print(data.ndim, data.shape, self.shape, data.shape[-len(shape):])
        if (2 > data.ndim > 4) or (data.shape[-len(shape):] != shape):
            raise ValueError('Arrays does not have the correct shape for '
                             'computation with segmentation data. Input data '
                             f'shape: {data.shape}, segmentation data shape '
                             f'{shape}.')

        # TODO: maybe use numpy.broadcast_arrays if you want to do stats on
        #  higher dimensional data sets using this class. You will have to
        #  think carefully on how to managed masked points, since replacing a
        #  label in the seg data will only work for 2d data. May have to
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

        if not np.ma.is_masked(image):
            return self.data

        if masked_pixels_label is None:
            masked_pixels_label = self.max_label + 1
        masked_pixels_label = int(masked_pixels_label)

        # ignore masked pixels
        seg_data = self.data.copy()
        seg_data[image.mask] = masked_pixels_label
        # this label will not be used for statistic computation
        return seg_data

    def thumbnails(self, image=None, labels=None, masked=False, extend=0,
                   same_size=False):
        """
        Thumbnail cutout images based on segmentation.

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
        if not same_size:
            return list(self.cutouts(data, labels=labels, masked=masked,
                                     extend=extend))

        if labels is None:
            labels = self.resolve_labels()

        # TODO: move same_size to cutouts??
        sizes = self.slices.sizes(labels)
        biggest = sizes.max(0) + int(extend)
        slices = self.slices.extend(labels, (biggest - sizes) / 2, self.shape)
        if masked:
            mask = self.to_binary()
            return [np.ma.MaskedArray(image[_], mask[_]) for _ in slices]
        return [image[_] for _ in slices]

    @api.synonyms({'stat': 'statistic_bg'})
    def flux(self, image, labels=None, bg=(0,), statistic_bg='median'):
        """
        An estimate of the net (background subtracted) source counts, and its
        associated uncertainty.

        Parameters
        ----------
        image: ndarray or np.ma.MaskedArray
            2d image or 3d image stack (video).
        labels: sequence of int
            Labels to use.
        bg: sequence of int, np.ndarray, SegmentationImage
            if sequence, must be of length 1 or same length as `labels`
            if array of dimensionality 2, must be same shape as image.
        statistic_bg: str or callable
            Statistic to use for background flux estimate.

        Returns
        -------

        """
        self._check_input_data(image)
        labels = self.resolve_labels(labels)

        # estimate sky counts and noise from data
        counts, counts_bg_pp, n_pix_src, n_pix_bg = self._flux(
            image, labels, bg, statistic_bg)

        # bg subtracted source counts
        signal = counts - counts_bg_pp * n_pix_src

        # revised CCD equation from Merlin & Howell '95
        noise = np.sqrt(signal +  # ← poisson.      sky + instrument ↓
                        n_pix_src * (1 + n_pix_src / n_pix_bg) * counts_bg_pp)

        return signal, noise

    def _flux(self, image, labels, bg=(0,), stat='median'):

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
            n_obj, n_bg = len(labels), len(bg)
            if n_bg not in (n_obj, 1):
                raise ValueError('Unequal number of background / object labels '
                                 f'({n_bg} / {n_obj})')
            seg_bg = self

        # resolve background counts estimate function
        stat = resolve_bg(stat)

        # get background stat
        counts_bg_pp = getattr(seg_bg, stat)(image, bg)
        ones = np.ones_like(image)
        n_pix_bg = seg_bg.sum(ones, bg)
        n_pix_src = self.sum(ones, labels)
        # counts_bg = counts_bg_pp * n_pix_src

        # mean counts per binned pixel (src + bg)
        counts = self.sum(image, labels)
        return counts, counts_bg_pp, n_pix_src, n_pix_bg

    def snr(self, image, labels=None, bg=(0,)):
        """
        A signal-to-noise ratio estimate. Calculates the SNR by taking the
        ratio of total background subtracted flux to the total estimated
        uncertainty (including poisson, sky, and instrumental noise) in each
        segment as per Merlin & Howell '95
        """

        return np.divide(*self.flux(image, labels, bg))
        # return signal / noise

    # def noise(self, image)

    def flux_sort(self, image, labels=None, bg=(0,), bg_stat='median'):
        """
        Re-label segments for highest per-pixel counts in descending order
        """
        labels = self.resolve_labels(labels)
        flx, _ = self.flux(image, labels, bg, bg_stat)
        order = np.argsort(flx)[::-1]

        # re-order segmented image labels
        self.relabel_many(labels[order], labels)

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

    # ------------------------------------------------------------------------ #
    def geometric_median(self, image, labels=None, mask=None, njobs='_ignored'):

        image = np.asanyarray(image).squeeze()
        assert image.ndim == 2, 'Only 2d images suported for now.'

        data = np.array([*np.indices(image.shape), image])  # yxz (3, r, c)
        return np.array([
            geometric_median(yxz.T)  # (3, m)
            for yxz in self.cutouts(data, labels=labels, compress=True)
        ])

    # alias
    gmed = g_med = geometric_median

    def com_std(self, coms, image, std, labels=None):
        """
        Standard deviation uncertainty in center-of-mass measurement (x, y)
        given the pixel values and their standard deviations. Uses linearized
        propegation of uncertainty assuming gaussian noise models.
        """
        if labels is None:
            labels = self.labels
        elif len(coms) != self.nlabels:
            labels = self.resolve_labels(labels)

        if (n := len(coms)) != len(labels):
            raise ValueError('Number of measurements in center-of-mass does not'
                             ' match size of `labels` vector.')

        sigmas = np.empty((n, 2))
        R = coms[..., None, None]
        for i, (xy, z, σ) in enumerate(
                self.cutouts(np.indices(image.shape)[::-1], image, std)):
            sigmas[i] = np.sqrt((((R[i] - xy) * σ) ** 2).sum((1, 2))) / z.sum()

        return sigmas

    def com_bg(self, image, labels=None, mask=None,
               background_estimator=np.ma.median, grid=None, njobs='_ignored'):
        """
        Compute centre of mass of background subtracted (masked) image.
        Default is to use median statistic  as background estimator.
        Pre-subtraction can sometimes improve accuracy of measurements in
        presence of noise.

        Parameters
        ----------
        image
        labels
        mask
        background_estimator # stat
        grid

        Returns
        -------

        """

        # TODO: clarify the conditions under which this function improves upon
        # CoM...

        if self.shape != image.shape:
            raise ValueError(f'Invalid image shape {image.shape} for '
                             f' segmentation of shape {self.shape}.')

        # get background subtraction function
        bg_sub = image_sub(background_estimator)

        if grid is None:
            grid = np.indices(self.shape)

        labels = self.resolve_labels(labels)

        # generalized for higher dimensional data
        axes_sum = tuple(range(1, image.ndim))

        if np.ma.is_masked(image):
            image = image.data
            mask = image.mask

        counter = itt.count()
        com = np.empty((len(labels), image.ndim))
        # u  = np.empty((len(labels), image.ndim))  # uncertainty
        for lbl, (seg, sub, msk, grd) in self.cutouts(
                self.data, image, mask, grid, labels=labels, labelled=True):
            #
            i = next(counter)

            # ignore whatever is in this slice and is nmskot same label as well
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
                # For a zero sum image segment, the center of mass is equivalent
                # to that of a constant segment.
                # warnings.warn(f'Function `com_bg` encountered zero-valued '
                #               f'image segment at label {lbl}.')
                # can skip next
                # com[i] = np.nan
                com[i] = grd.sum(axes_sum) / sub.size
            else:
                # compute centre of mass
                com[i] = (sub * grd).sum(axes_sum) / sum_

        return com

    def peak(self, image, labels=None, upsample=1, filter='lanczos', pixel_centre=0,
             njobs='_ignored'):

        upsample = int(upsample)
        assert upsample > 0

        if upsample == 1:
            return self.maximum_position(image, labels) + pixel_centre

        #
        from PIL import Image

        labels = self.resolve_labels(labels)

        im = Image.fromarray(image.astype('f'))
        newsize = np.multiply(im.size, upsample)
        im1 = im.resize(newsize, getattr(Image, filter.upper()))
        seg1 = Image.fromarray(self.data * 1.).resize(newsize)
        peaks = ndimage.maximum_position(np.array(im1), seg1, labels)

        return (np.array(peaks) / upsample) + pixel_centre

    # def qfactor(self, image, labels, size, connectivity)

    # --------------------------------------------------------------------------

    def relabel_many(self, *label_sets):
        """
        Reassign multiple labels

        Parameters
        ----------
        [old_labels]
        new_labels

        Returns
        -------

        """
        assert len(label_sets) in {1, 2}
        *old, new = label_sets
        old, = old or (None, )
        if isinstance(new, dict):
            old, new = zip(*new.items())

        old = self.resolve_labels(old)

        if len(old) != len(new):
            raise ValueError('Unequal number of labels')

        # catch for empty label vectors
        if old.size == 0:
            return

        # there may be labels missing
        forward_map = np.arange(self.max_label + 1)
        forward_map[old] = new
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

        if iterations == 'auto':
            return self.auto_dilate(labels)

        if not isinstance(iterations, numbers.Integral):
            raise ValueError('`iterations` parameter should be an integer.')

        # expand masks to 3D sequence
        labels = self.resolve_labels(labels)
        masks = self.to_binary(labels, expand=True)

        if structure is None:
            if d := {4: 1, 8: 2}.get(connectivity):
                structure = ndimage.generate_binary_structure(2, d)
            else:
                raise ValueError('Invalid connectivity={0}.  '
                                 'Options are 4 or 8.'.format(connectivity))

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

        self.data = data
        return self

    def auto_dilate(self, image, labels=None, dmax=5, sigma=3, connectivity=1):
        #
        from obstools.stats import mad

        labels = self.resolve_labels(labels)

        if len(labels) == 0:
            return self

        for count in range(dmax + 1):
            self.logger.debug('round {:d}', count)

            mim = self.mask_sources(image, labels)
            m = np.ma.median(mim)
            σ = mad(mim, m)

            # get annuli - test pixels
            b3 = self.to_annuli(0, 1, labels, connectivity=connectivity)
            slices = self.slices.extend(labels, 2)
            for label, b, s in zip(labels, b3, slices):
                bb = b[s]
                pixels = mim[s][bb]
                w = np.array(np.where(bb))
                dark = pixels - m < σ * sigma
                if dark.all():
                    labels = np.setdiff1d(labels, label)
                else:
                    logger.debug('label: {:d}: {:d} pixels added.', label,
                                 sum(~dark))
                    bb[tuple(w[:, dark])] = False
                    self.data[s][bb] = label

    def deblend(self, image, npixels, **kws):
        return self.__class__(
            deblend_sources(image, self.data, npixels, **kws).data
        )

    def blend(self):
        """
        Inverse operation of `deblend`. Merge segments that are touching
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

    def to_annuli(self, buffer=0, width=5, labels=None, remove_sources=True,
                  connectivity=1):
        """
        Create 3D boolean array containing annular masks for sky regions around
        labelled regions in `labels`. Regions containing labelled pixels from
        other sources within sky regions are removed unless
        `remove_sources=False` requested.

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
        struct = ndimage.generate_binary_structure(2, connectivity)[None]
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

    def add_segments(self, data, label_insert=None, copy=False, group_id=None):
        """
        Add segments from another SegmentationImage. New segments will
        over-write old segments in the pixel positions where they overlap.

        Parameters
        ----------
        data:
        label_insert:
            If None; new labels come after current ones.
            If int; value for starting label of new data. Current labels will be
              shifted upwards.
        copy
        group_id

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

        if group_id is not None:
            groups[group_id] = new_labels

        # mask for new label regions
        new_pix = data.astype(bool)
        if copy:
            seg_data_out = self.data.copy()
            seg_data_out[new_pix] = data[new_pix]
            groups.update(self.groups)
            return self.__class__(seg_data_out, groups), new_labels

        self.data[new_pix] = data[new_pix]
        self._reset_lazy_properties()
        self.groups.update(groups)
        return self, new_labels

    def position_to_label(self, pos):
        # check which segmented regions pos is in
        pos = np.asanyarray(pos)
        labels = np.zeros(pos.shape[:-1], int)
        inframe = ((pos > 0) & (pos < self.shape)).all(-1)
        labels[inframe] = self.data[tuple(np.array(pos[inframe], int).T)]
        return labels

    # ------------------------------------------------------------------------ #
    @lazyproperty
    def show(self):
        return SegmentPlotter(self)

    plot = display = show

    def get_boundary(self, label, offset=-0.5):
        """
        Fast boundary trace for segment `label`. This algorithm is designed to
        be general enough to handle disjoint segments with a single label, or
        segments with holes in them, and therefore always returns a list even if
        the segment is monolithic.

        Parameters
        ----------
        label: int
            The segment label for which to get the boundary
        offset: float
            The offset that will be added to the points, by default -0.5, so
            that the contours line up with imaged displayed with bottom left
            pixel center.

        Returns
        -------
        segments: list of arrays
            Line segments tracing each part of the segment.
        perimeter: array
            Length of the circumference of each part of the segment.
        """
        self.check_label(label)
        b = (self.sliced(label) == label)
        origin = self.slices.llc(label)

        count = 0
        segments = []
        perimeter = []
        while b.any():
            pixels, points, p = trace_boundary(b)
            # get outline image, fill it, difference original, repeat.
            # This will trace outer as well as inner boundaries for
            # segments with holes.
            tmp = np.zeros_like(b)
            tmp[tuple(pixels.T)] = True
            b = np.logical_xor(b, ndimage.binary_fill_holes(tmp))
            #
            segments.append((points + origin + offset)[:, ::-1])
            perimeter.append(p)

            if count > 10:
                break
            count += 1

        # correction factor for perimeter length (from
        return segments, np.multiply(0.95, perimeter)

    def get_boundaries(self, labels=None):
        """Get traced outlines for many labels"""
        # TODO systematically cache in traced
        return {label: self.get_boundary(label)
                for label in self.resolve_labels(labels)}

    @lazyproperty
    def traced(self):
        return self.get_boundaries()

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
        # segments, perimeter
        if self.nlabels:
            _, perimeter = zip(*self.traced.values())
            return list(perimeter)
        return []

    @lazyproperty
    def roundness(self):
        """Circularity"""
        return (4 * np.pi * self.areas /
                np.square(list(map(sum, self.perimeter))))

    def circularize(self):
        x = (..., None, None)
        r = np.sqrt(self.areas / np.pi)[x]
        d = np.square(
            np.indices(self.shape)[:, None] - self.com(self.data).T[x]
        ).sum(0)
        return self.__class__(
            np.sum((np.sqrt(d) <= r) * self.labels[x], 0)
        )


class SegmentsModelHelper(SegmentedImage):
    """
    Mixin class for image models with piecewise domains.
    """

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

    def get_coord_grids(self, labels=None):
        return dict(self.cutouts(self.grid, labels=self.resolve_labels(labels),
                                 flatten=True, labelled=True))

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


class SegmentsMasksHelper(SegmentsModelHelper):

    def __init__(self, data, grid=None, domains=None, groups=None,
                 **persistent_masks):
        #
        SegmentsModelHelper.__init__(self, data, grid, domains)
        LabelGroupsMixin.__init__(self, groups)
        self._masks = None
        self._persistent_masks = persistent_masks

    def set_groups(self, groups):
        LabelGroupsMixin.set_groups(self, groups)
        del self.group_masks  #

    @lazyproperty
    # making group_masks a lazyproperty means it will reset auto-magically when
    # the segmentation data changes
    def group_masks(self):
        return MaskContainer(self, self.groups, **self._persistent_masks)


class GlobalSegmentation(SegmentsMasksHelper):

    @classmethod
    def merge(cls, segmentations, delta_xy, extend=True, f_accept=0.5,
              post_dilate=1):
        # zero point correspond to minimum offset from initializer and serves as
        # coordinate reference
        # make int type to avoid `Cannot cast with casting rule 'same_kind'
        # downstream
        return cls(merge_segmentations(segmentations, delta_xy,
                                       extend, f_accept, post_dilate),
                   zero_point=np.floor(delta_xy.min(0)).astype(int))

    def __init__(self, data, zero_point):
        super().__init__(data)
        self.zero_point = zero_point

    def get_start_indices(self, delta_xy):
        if np.ma.is_masked(delta_xy):
            raise ValueError('Cannot get start indices for masked offsets.')

        return np.abs((delta_xy + self.zero_point).round().astype(int))

    def get_overlap(self, start, shape, type_=SegmentedImage):
        return super().get_overlap(start, shape, type_)

    def for_offset(self, delta_xy, shape, type_=SegmentedImage):
        if np.ma.is_masked(delta_xy):
            raise ValueError('Cannot get segmented image for masked offsets.')

        return self.get_overlap(
            self.get_start_indices(delta_xy), shape, type_)

    def flux(self, image, origin, labels=None, labels_bg=(0,), bg_stat='median'):
        sub = self.get_overlap(origin, image.shape)
        return sub.flux(image, labels, labels_bg, bg_stat)

    def sort(self, measure, descend=False):
        if (n := len(measure)) != self.nlabels:
            raise ValueError('Cannot reorder labels for measure with '
                             f'incompatable size {n}. {describe(self)} has '
                             f'{self.nlabels} labels.')

        o = slice(None, None, -1) if descend else ...
        order = np.ma.argsort(measure, endwith=False)[o]
        self.relabel_many(order + 1, self.labels)
        return order


# class SegmentationGroups(SegmentedImage, LabelGroupsMixin):
#
#     def __init__(self, data, use_zero=False):
#         super().__init__(data, use_zero)
#
#     def add_segments(self, data, label_insert=None, group=None, copy=False):
#         seg, new_labels = super().add_segments(data, label_insert, copy)
#         self.groups[group] = new_labels
#         return seg
