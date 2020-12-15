import functools as ftl
import itertools as itt
import logging

import numpy as np

from motley.table import Table

from obstools.modelling import UnconvergedOptimization
from obstools.phot.utils import iter_repeat_last
from photutils import detect_threshold, detect_sources
from recipes.pprint.misc import seq_repr_trunc
from recipes.logging import get_module_logger
from recipes.logging import LoggingMixin

# module level logger
logger = get_module_logger()


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


def detect(image, mask=False, background=None, snr=3., npixels=7,
           edge_cutoff=None, deblend=False):
    """
    Image detection that returns a SegmentedImage instance

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

    # separate pixel mask for threshold calculation (else the mask gets
    # duplicated to threshold array, which will skew the detection stats)
    # calculate threshold without masked pixels so that std accurately measured
    # for region of interest
    if np.ma.isMA(image):
        mask = mask | image.mask
        image = image.data

    # # check mask reasonable
    # if mask.sum() == mask.size:

    # detection
    threshold = detect_threshold(image, snr, background, mask=mask)
    if not np.any(mask):
        mask = None  # annoying photutils #HACK

    seg = detect_sources(image, threshold, npixels, mask=mask)

    # check if anything detected
    no_sources = (seg is None)  # or (np.sum(seg) == 0)
    if no_sources:
        logger.debug('No objects detected')
        return np.zeros_like(image, bool)

    if deblend and not no_sources:
        from photutils import deblend_sources
        seg = deblend_sources(image, seg, npixels)

    if edge_cutoff:
        border = make_border_mask(image, edge_cutoff)
        # labels = np.unique(seg.data[border])
        seg.remove_masked_labels(border)

    # intentionally return an array
    return seg.data


def detect_measure(image, mask=False, background=None, snr=3., npixels=7,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    from .core import SegmentedImage
    seg = SegmentedImage.detect(image, mask, background, snr, npixels,
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


class MultiThresholdBlobDetection(LoggingMixin):
    # algorithm defaults
    snr = (10, 7, 5, 3)
    npixels = (7, 5, 3)
    deblend = (True, False)
    dilate = (4, 2, 1)
    edge_cutoff = None
    max_iter = np.inf

    def __call__(self, image, mask=None, snr=snr, npixels=npixels,
                 deblend=deblend, dilate=dilate, edge_cutoff=edge_cutoff,
                 max_iter=max_iter, group_name_format='sources{count}',
                 model=None, opt_kws=None, report=None):
        #
        'todo ?'


def detect_loop(image, mask=None, snr=3, npixels=3,
                deblend=True, dilate=0, edge_cutoff=None,
                max_iter=np.inf, group_name_format='sources{count}',
                model=None, opt_kws=None, report=None):
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
    image background region after each round of detection.  The model
    optimization can be controlled by passing `opt_kws` dict.

    Parameters
    ----------
    image: array-like
    mask: array-like, same shape as image
    snr: float or sequence of float
    npixels: int or sequence of int
    deblend: bool or sequence of bool

    dilate: int or sequence of int or 'auto'

    edge_cutoff: int or tuple
    max_iter: int
        Maximum number of iteration of the algorithm

    group_name_format: str
    model
    opt_kws: dict
    report: bool

    Returns
    -------
    seg: SegmentedImage
        The segmented image
    groups: dict
        Groups of detected sources.  One group for each iteration of the
        algorithm.
    info: dict
        Detection parameters for each round
    result: np.ndarray or None
        Fit parameters for model
    residual: np.ndarray

    """
    # todo: this could be a method of the SourceDetectionMixin class

    from .core import SegmentedImage

    # short circuit
    if max_iter == 0:
        # seg, info, model, result, residual
        return np.zeros(image.shape, int), [], [], None, image

    if not isinstance(group_name_format, str):
        raise TypeError('`group_name_format` parameter should be a format '
                        'string')

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

    # get empty segmented image
    seg = SegmentedImage.empty_like(image)

    # label groups
    # keep track of group info + detection meta data
    groups = []
    info = []
    gof = []

    # detection loop
    counter = itt.count()
    while True:
        # keep track of iteration number
        count = next(counter)
        logger.debug('count %i', count)

        if count >= max_iter:
            logger.debug('break: max_iter reached')
            break

        # get detect options and assign group name
        # noinspection PyTupleAssignmentBalance
        snr_, npix, dil, debl = opts = next(var_gen)
        info_dict = dict(zip(var_names, opts))
        group_name = group_name_format.format(count=count, **info_dict)

        # detect on residual image
        new_seg = seg.detect(residual, mask, None, snr_, npix, edge_cutoff,
                             debl, dil, group_name)
        # im = new_seg.display()
        # im.ax.set_title(f'new {count}')

        if not new_seg.data.any():
            logger.debug('break: no new detections')
            break

        # aggregate
        if count == 0:
            seg = new_seg
            new_labels = new_seg.labels

            # initialize the model if required
            if type(model) is type:
                model = model(seg)

        else:
            _, new_labels = seg.add_segments(new_seg)

        # debug log!
        if debug:
            logger.debug('detect_loop: round %i: %i new detections: %s',
                         count, len(new_labels),
                         seq_repr_trunc(tuple(new_labels)))

        # update mask
        mask = mask | new_seg.to_binary()

        # group info
        groups.append(new_labels)
        info.append(info_dict)

        if model:
            # fit model, get residuals
            mimage = np.ma.MaskedArray(image, mask)

            try:
                result = model.fit(mimage, **opt_kws)
                residual = model.residuals(result, image)
                if report:
                    gof.append(model.redchi(result, mimage))
            except UnconvergedOptimization as err:
                logger.info('Model optimization unsuccessful. Returning.')
                break

        # if dilate == 'auto':

    # seg.groups.update()

    # def report_detection(groups, info, model, gof):
    if report:
        if len(info):
            # log what you found
            from recipes import pprint

            seq_repr = ftl.partial(seq_repr_trunc, max_items=3)

            # report detections here
            col_headers = ['snr', 'npix', 'dil', 'debl', 'n_obj', 'labels']
            info_list = list(map(list, map(dict.values, info)))
            tbl = np.column_stack([np.array(info_list, 'O'),
                                   list(map(len, groups)),
                                   list(map(seq_repr, groups))])
            if model:
                col_headers.insert(-1, 'χ²ᵣ')
                tbl = np.insert(tbl, -1, list(map(pprint.numeric, gof)), 1)

            title = 'Object detections'
            if model:
                title += f' with {model.__class__.__name__} model'

            msg = Table(tbl,
                        title=title,
                        col_headers=col_headers,
                        totals=(4,), minimalist=True)
        else:
            msg = 'No detections!'
        logger.info(f'\n{msg}')
        # print(logger.name)

    return seg, info, model, result, residual


class SourceDetectionMixin(object):
    """
    Provides the `from_image` classmethod and the `detect` staticmethod that
    can be used to construct image models from images.
    """

    def __init__(self, seg):
        """Base initializer only sets the `seg` attribute"""
        self.seg = seg

    @classmethod
    def detect(cls, image, *args, **kws):
        """
        Default blob detection algorithm.  Subclasses can override as needed

        Parameters
        ----------
        image

        kws:
            Keywords for source detection algorithm

        Returns
        -------

        """

        return detect_loop(image, *args, **kws)
        # return cls._detect(image, **kws)

    @classmethod
    def from_image(cls, image, detect_sources=True, **detect_opts):
        """
        Construct a instance of this class from an image.
        Sources in the image will be identified using `detect` method.
        Segments for detected sources will be added to the segmented image.
        Source groups will be added to `groups` attribute dict.

`
        Parameters
        ----------
        image
        detect_sources: bool
            Controls whether source detection algorithm is run. This argument
            provides a shortcut to the default source detection by setting
            `True`, or alternatively to skip source detection by setting
            `False`

        detect_opts: dict

        Returns
        -------

        """

        # Detect objects & segment image
        if isinstance(detect_sources, dict):
            detect_opts.update(detect_sources)

        if not ((detect_sources is True) or detect_opts):
            # short circuit the detection loop
            detect_opts['max_iter'] = 0

        # Basic constructor that initializes the object from an image. The
        # base version here runs a detection algorithm to separate foreground
        # objects and background, but doesn't actually include any physically
        # useful models. Subclasses can overwrite this method to add useful
        # models to the segments.

        # Detect objects & segment image
        seg, info, model, result, residual = cls.detect(image, **detect_opts)

        # add detected sources
        seg.groups.update(seg.groups)

        # init
        return cls(seg)
