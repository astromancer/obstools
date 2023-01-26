

# std
import functools as ftl
from collections import defaultdict

# third-party
import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from photutils import detect_sources, detect_threshold

# local
import recipes.pprint as pp
from recipes import string, caching
from recipes.logging import LoggingMixin
from recipes.iter import iter_repeat_last
from recipes.oo.property import classproperty
from motley.table import Table

# relative
from ..modelling import UnconvergedOptimization

# TODO: watershed segmentation on the negative image ?
# TODO: detect_gmm():

# ---------------------------------------------------------------------------- #
# defaults
DEFAULT_ALGORITHM = 'sigma_threshold'
NPIXELS = 7
EDGE_CUTOFF = None
MONOLITHIC = True
ROUNDNESS = (0.5, 1.5)
DILATE = 0
DEBLEND = False

# ---------------------------------------------------------------------------- #


def make_border_mask(image, edge_cutoffs):
    if isinstance(edge_cutoffs, int):
        return _make_border_mask(image.shape,
                                 edge_cutoffs, -edge_cutoffs,
                                 edge_cutoffs, -edge_cutoffs)
    edge_cutoffs = tuple(edge_cutoffs)
    if len(edge_cutoffs) == 4:
        return _make_border_mask(image.shape, *edge_cutoffs)

    raise ValueError(f'Invalid edge_cutoffs {edge_cutoffs}')


def _make_border_mask(shape, xlow=0, xhi=None, ylow=0, yhi=None):
    """Edge mask"""
    mask = np.zeros(shape, bool)

    mask[:ylow] = True
    if yhi is not None:
        mask[yhi:] = True

    mask[:, :xlow] = True
    if xhi is not None:
        mask[:, xhi:] = True
    return mask


# ---------------------------------------------------------------------------- #


class DetectionBase(LoggingMixin):
    """Base class for source detection."""

    members = {}
    owner = None
    # `SourceDetectionDescriptor.__set_name__` assigns the descriptor owner here

    @classproperty
    @classmethod
    def name(cls):
        name = cls.__name__
        if any(map(str.islower, name)):
            return string.snake_case(name)
        return name.lower()

    @classmethod
    def resolve(cls, algorithm):
        # resolve detection class from `algorithm` string
        kls = cls.members.get(algorithm.lower().replace(' ', '_'))
        if kls is None:
            raise TypeError(
                f'Invalid source detection algorithm: {algorithm!r}. Valid'
                f' choices are: {tuple(cls.members.values())}'
            )
        return kls

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not cls.__name__.startswith('_'):
            cls.members[cls.name] = cls

    def _get_owner(self):
        from obstools.image import SegmentedImage

        #
        kls = SegmentedImage
        if self.owner and issubclass(self.owner, kls):
            kls = self.owner
        return kls

    def detect(self, image, mask=None, *args, report=False, **kws):

        seg = self.__call__(image, mask, *args, **kws)

        if report:
            if report is True:
                report = {}
            self.report(image, seg, **report)

        return seg

    @caching.cached(typed={'image': caching.hashers.array,
                           'mask': caching.hashers.array})
    def __call__(self, image, mask=None,
                 npixels=NPIXELS, edge_cutoff=EDGE_CUTOFF,
                 monolithic=MONOLITHIC, roundness=ROUNDNESS,
                 dilate=DILATE, deblend=DEBLEND,
                 **kws):
        """
        Image object detection that returns a `SegmentedImage` instance. Post
        processing to remove sources that do not meet criteria (`npixels`,
        `snr`, `monolithic`, `roundness` etc), followed by optional dilation
        (increasing segment sizes).


        Parameters
        ----------
        image
        mask
        dilate


        Returns
        -------
        obstools.image.segmentation.SegmentedImage
        """

        # self.logger.debug('Running source detection algorithm: {!r} {}', )

        # Initialize
        seg_data = self.fit_predict(image, mask, npixels=npixels, **kws)
        return self.post_process(image, seg_data, npixels, edge_cutoff,
                                 monolithic, roundness, dilate, deblend)

    def fit_predict(self, *args, **kws):
        raise NotImplementedError

    def post_process(self, image, seg_data,
                     npixels=NPIXELS, edge_cutoff=EDGE_CUTOFF,
                     monolithic=MONOLITHIC, roundness=ROUNDNESS,
                     dilate=DILATE, deblend=DEBLEND):
        self.logger.info('Post-processing detected sources with criteria: {}',
                         dict(npixels=npixels,
                              edge_cutoff=edge_cutoff,
                              monolithic=monolithic,
                              roundness=roundness,
                              dilate=dilate,
                              deblend=deblend))

        kls = self._get_owner()

        def msg(labels, criterion):
            if len(labels):
                self.logger.opt(depth=1).debug(
                    'Removing {} segments failing {} criterion: {}',
                    len(labels), criterion, labels
                )

        #  shape rejection
        if monolithic:
            mask = seg_data.astype(bool)
            filled = ndimage.binary_fill_holes(mask)
            seg_data, _ = ndimage.label(filled)
            seg = kls(seg_data)

            remove_labels = set(seg_data[filled & ~mask]).union(
                seg.labels[seg.areas < npixels])
            msg(remove_labels, 'monolithic')
            seg.remove_labels(list(remove_labels))
        else:
            seg = kls(seg_data)

        if edge_cutoff:
            border = make_border_mask(image, edge_cutoff)
            msg(np.unique(seg.data[border]), 'edge cutoff')
            seg.remove_masked_labels(border)

        if deblend:  # and not no_sources:
            seg = seg.deblend(image, npixels)

        # shape rejection 2
        if roundness:
            lo, hi = roundness
            r = seg.roundness
            remove_labels = seg.labels[(lo > r) | (r >= hi)]
            msg(remove_labels, 'roundness')
            seg.remove_labels(remove_labels)

        # dilate
        if dilate:
            seg.dilate(iterations=dilate)

        # cleanup
        if seg.nlabels:
            seg.relabel_consecutive()

        return seg

    def report(self, image, seg, cutouts=True, **kws):
        self.logger.opt(lazy=True).info(
            'Detected {0[0]:d} source{0[1]} covering {0[2]} pixels ({0[3]:.2%} '
            'of the image area).',
            lambda: (seg.nlabels, 's' * (seg.nlabels > 1), seg.areas.sum(),
                     sum(seg.fractional_areas))
        )

        if cutouts:
            self.logger.info('Source images:\n{}',
                             seg.show.console.format_cutouts(image, **kws))


class SigmaThreshold(DetectionBase):
    def fit_predict(self, image, mask=None, background=None,
                    snr=3., npixels=7):
        """
        Image detection worker

        Parameters
        ----------
        image
        mask
        background


        Returns
        -------

        """

        self.logger.info('Running detect with: {:s}',
                         str(dict(snr=snr)))  # npixels=npixels

        if mask is None:
            mask = False  # need this for logical operators below to work

        # separate pixel mask for threshold calculation (else the mask gets
        # duplicated to threshold array, which will skew the detection stats)
        # calculate threshold without masked pixels so that std accurately
        # measured for region of interest
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
            self.logger.info('No objects detected.')
            return np.zeros_like(image, bool)

        # intentionally return an array
        return seg.data


class GMM(DetectionBase):
    def __init__(self, n_components=5, **kws):
        self.gmm = GaussianMixture(n_components, **kws)

    def fit_predict(self, image, mask=None):
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

        pixels = ...
        if (mask is not None) or (mask is not False):
            image = np.ma.MaskedArray(image, mask)
            pixels = ~image.mask

        # model
        y = np.ma.compressed(image).reshape(-1, 1)

        seg = np.zeros(image.shape, int)
        seg[pixels] = self.gmm.fit_predict(y)
        return seg

    def __call__(self, image, mask=None, *args, plot=False, **kws):

        if plot:
            from matplotlib import pyplot as plt
            from matplotlib.colors import ListedColormap

            m = self.gmm.means_.T
            v = self.gmm.covariances_.T
            w = self.gmm.weights_ / np.sqrt(2 * np.pi * v)
            x = np.linspace(y.min(), y.max(), 250).reshape(-1, 1)
            components = w * np.exp(-0.5 * np.square((x - m)) / v).squeeze()

            fig, ax = plt.subplots()
            ax.hist(y.squeeze(), bins=100, density=True, log=True)
            for c in components.T:
                ax.plot(x, c, scaley=False)

            cmap = ListedColormap([l.get_color() for l in ax.lines])
            obj.display(cmap=cmap, draw_labels=False)

        return self.seg

# ---------------------------------------------------------------------------- #
# Backbone for looped source detection


class _BackgroundFitter(DetectionBase):
    """Source detection with optional background model"""

    opt_kws = {}  # default

    def __init__(self, algorithm, model=None):
        super().__init__(algorithm)
        self.model = model
        self.result = self.residual = self.gof = None

    def __call__(self, image, mask=False, opt_kws=opt_kws, **kws):
        # detect on residual image. Only previously undetected sources will
        # be present here
        self.residual = image

        self.logger.debug('Running detection: {}.', kws)
        seg = super().__call__(self.residual, mask, **kws)

        if not self.model:
            return seg

        # initialize the model if required
        if isinstance(self.model, type):
            self.model = self.model(seg)

        # fit the background. update state variables
        mimage = np.ma.MaskedArray(image, mask)
        # result = residual = gof = None
        try:
            self.result = self.model.fit(mimage, **opt_kws)
        except UnconvergedOptimization:
            self.logger.info('Model optimization unsuccessful. Breaking loop.')
        else:
            self.residual = self.model.residuals(self.result, image)
            self.gof = self.model.redchi(self.result, mimage)

        return seg


class _SourceAggregator(DetectionBase):
    """
    Aggregate info from multiple detection loops
    """

    def __init__(self, algorithm, model=None):
        self.count = 0
        self.seg = None
        super().__init__(algorithm, model)

    def __call__(self, image, mask=False, group_id=None, **kws):
        # update mask
        if mask is None:
            mask = False

        if self.seg is None:
            # first round
            self.seg = self._get_owner().empty_like(image)

        # ignore previous detections
        new_seg = super().__call__(image, mask | self.seg.to_binary(), **kws)
        self.seg.add_segments(new_seg, group_id=group_id)

        self.count += 1
        return self.seg


class _ResultsAggregator(_SourceAggregator, _BackgroundFitter):
    def __init__(self, algorithm, model=None):
        super().__init__(algorithm, model)
        self.info = defaultdict(list)
        self.fitness = []

    def __call__(self, image, mask=False, *args, **kws):
        # new detections
        seg = super().__call__(image, mask, *args, **kws)

        # log what was found
        # if logger.getEffectiveLevel() == logging.INFO:
        #     logger.info('Detected {:d} sources covering {:.2%} of image.',
        #                      seg.nlabels, sum(seg.areas) / np.prod(image.shape))

        # aggregate info
        if seg.nlabels:
            for k, v in kws.items():
                self.info[k].append(v)

        # aggregate fitting results
        if self.model and self.result is None:
            self.fitness.append(self.gof)


class _SourceDetectionLoop(_ResultsAggregator):

    def __init__(self, algorithm, model=None, *args, max_iter=1, **kws):
        super().__init__(algorithm, model, *args, **kws)
        self.max_iter = max_iter
        self.params = self.iter_params(**kws)

    def __call__(self, image, mask=False, max_iter=None, *args, **kws):

        if max_iter is None:
            max_iter = self.max_iter

        # sourcery skip: remove-empty-nested-block, remove-redundant-pass
        for _ in self:
            # run `__next__` until StopIteration
            pass

        return self.seg

    @staticmethod
    def iter_params(**kws):
        for values in zip(*map(iter_repeat_last, kws.values())):
            yield dict(zip(kws.keys(), values))

    def __next__(self):
        if self.count >= self.max_iter:
            self.logger.debug('break: max_iter {:d} reached.', self.max_iter)
            raise StopIteration

        # detect new
        params = next(self.params)
        new_segs = super().__call__(self.image, self.mask, **params)

        if not new_segs.nlabels:
            self.logger.debug('break: no new detections.')
            raise StopIteration

        # debug log!
        self.logger.debug('Detection iteration {:d}: {:d} new detections: {:s}',
                          self.count, new_segs.nlabels,
                          pp.collection(tuple(new_segs.labels)))

        if self.model and self.result is None:
            self.logger.info('break: Model optimization unsuccessful. Returning.')
            raise StopIteration

        return new_segs

    def report(self, gof):

        if not self.info:
            return 'No detections!'

        # report detections here
        col_headers = list(self.info.keys())
        info_list = list(self.info.values())
        tbl = np.column_stack([
            np.array(info_list, 'O'),
            list(map(len, self.seg.groups)),
            list(map(ftl.partial(pp.collection, max_items=3), self.seg.groups))
        ])

        title = 'Object detections'
        if self.model:
            title += f' with {self.model.__class__.__name__} model'
            col_headers.insert(-1, 'χ²ᵣ')
            tbl = np.insert(tbl, -1, list(map(pp.nr, gof)), 1)

        return Table(tbl,
                     title=title,
                     col_headers=col_headers,
                     totals=(4,),
                     minimal=True)


class MultiThreshold(_SourceDetectionLoop):
    """
    Multi-threshold object detection for image segmentation and segment
    categorisation. This function runs multiple iterations of the detection
    algorithm on the same image, masking new sources after each round so that
    progressively fainter sources may be detected. By default the algorithm will
    continue looping until no new sources are detected. The number of iterations
    in the loop can also be controlled by specifying `max_iter`.  The arguments
    `snr`, `npixels`, `deblend`, `dilate` can be sequences, in which case each
    new detection round will use the next value in the sequence until the last
    value which will then be repeated for subsequent iterations. If scalar, this
    value will be used repeatedly for each round.

    A background model may also be provided.  This model will be fit to the
    image background region after each round of detection.  The model
    optimization parameters can be controlled by passing `opt_kws` dict.
    """

    # Parameter defaults
    snr = (10, 7, 5, 3)
    npixels = (7, 5, 3)
    deblend = (True, False)
    dilate = 'auto'
    edge_cutoff = None
    max_iter = 5

    # group labels
    # auto_key_template = 'sources{count}'

    def __init__(self, max_iter=max_iter, model=None):
        super().__init__('sigma_threshold', model, max_iter)

    def __call__(self, image, mask=False,
                 snr=snr, npixels=npixels, deblend=deblend, dilate=dilate,
                 edge_cutoff=edge_cutoff):
        """

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
        return super().__call__(image, mask,
                                snr=snr,
                                npixels=npixels,
                                deblend=deblend,
                                dilate=dilate,
                                edge_cutoff=edge_cutoff,
                                max_iter=max_iter)


class SourceDetectionDescriptor:
    """
    A descriptor object for managing source detection algorithms.
    """

    def __init__(self, algorithm=DEFAULT_ALGORITHM, *args, **kws):
        self.algorithm = algorithm
        self._algorithm = DetectionBase.resolve(algorithm)(*args, **kws)

    def __call__(self, image, *args, **kws):
        return self._algorithm.detect(image, *args, **kws)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.algorithm})'

    def __set_name__(self, owner, name):
        # set the class th
        # self.logger.debug('Assigned ownership of {!r} detection algorithm to {}.',
        #                   self.algoritm, owner)
        self._algorithm.owner = owner

    def __get__(self, obj, kls=None):
        self.parent = obj
        return self

    def __set__(self, obj, algorithm):
        """
        >>> class MyImage:
        ...     detect = SourceDetectionDescriptor('gmm')
        ... img = MyImage().detect

        later to switch algorithms:
        >>> img.detect.algorithm = 'sigma_threshold'

        """
        self.algorithm = algorithm
        return self._algorithm

    @property
    def algorithm(self):
        return self._algorithm.name

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = DetectionBase.resolve(algorithm)()

    def report(self, image, seg, show=5, **kws):
        return self._algorithm.report(image, seg, show, **kws)


class SourceDetectionMixin:
    """
    Provides the `from_image` classmethod and the `detect` staticmethod that
    can be used to construct image models from images.
    """

    detection = SourceDetectionDescriptor(DEFAULT_ALGORITHM)

    @classmethod
    def from_image(cls, image, detect=True, **detect_opts):
        """
        Construct a instance of this class from an image.
        Sources in the image will be identified using `detect` method.
        Segments for detected sources will be added to the segmented image.
        Source groups will be added to `groups` attribute dict.

        Parameters
        ----------
        image : np.ndarray
            Image with sources
        detect : bool or dict
            Controls whether source detection algorithm is run. This argument
            provides a shortcut to the default source detection by setting
            `True`, or alternatively to skip source detection by setting
            `False`

        detect_opts : Keywords passed to

        Returns
        -------

        """

        # select source detection algorithm
        if isinstance(detect, dict):
            detect_opts = dict(detect, **detect_opts)
            detect = DEFAULT_ALGORITHM

        if isinstance(detect, str) and cls.detection.algorithm != detect:
            # switch algorithms
            cls.detection.algorithm = detect

        # Detect objects & segment image
        # detect_opts = dict(detect if isinstance(detect, dict) else {},
        #                    **detect_opts)
        if not detect:
            # short circuit the detection loop
            detect_opts['max_iter'] = 0

        # Basic constructor that initializes the object from an image. The
        # base version here runs a detection algorithm to separate foreground
        # objects and background, but doesn't actually include any physically
        # useful models. Subclasses can overwrite this method to add useful
        # models to the segments.

        # Detect objects & init with segmented image
        return cls.detection(image, **detect_opts)

    def detect(self, image, *args, report=True, **kws):
        # subclasses to implement specifics by overwriting this method
        return self.detection(image, *args, report=report, **kws)


# alias
DetectionMixin = SourceDetectionMixin
