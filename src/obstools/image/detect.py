

# std
import functools as ftl
from collections import defaultdict

# third-party
import numpy as np
from sklearn.mixture import GaussianMixture
from photutils import detect_threshold, detect_sources

# local
import recipes.pprint as pp
from recipes.logging import LoggingMixin
from recipes.iter import iter_repeat_last
from recipes.string import sub, snake_case
from motley.table import Table

# relative
from ..modelling import UnconvergedOptimization
from .segmentation import groups, SegmentedImage



def make_border_mask(data, edge_cutoffs):
    if isinstance(edge_cutoffs, int):
        return _make_border_mask(data,
                                 edge_cutoffs, -edge_cutoffs,
                                 edge_cutoffs, -edge_cutoffs)
    edge_cutoffs = tuple(edge_cutoffs)
    if len(edge_cutoffs) == 4:
        return _make_border_mask(data, *edge_cutoffs)

    raise ValueError('Invalid edge_cutoffs %s' % edge_cutoffs)


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


class DetectionBase(LoggingMixin):
    """Base class for source detection"""

    members = {}

    def __new__(cls, algorithm, *args, **kws):
        if isinstance(algorithm, str):
            kls = cls.members[snake_case(algorithm)]
            return super().__new__(kls, *args, **kws)

        return super().__new__(cls, algorithm, *args, **kws)

    def __init__(self, algorithm, *args, **kws):
        assert callable(algorithm)
        self.fit_predict = staticmethod(algorithm)

    def __init_subclass__(cls):
        super().__init_subclass__()
        cls.members[snake_case(cls.__name__)] = cls

    def fit_predict(self, *args, **kws):
        raise NotImplementedError

    def __call__(self, image, mask=False, dilate=0, /, *args, **kws):
        """
        Image object detection that returns a SegmentedImage instance


        Parameters
        ----------
        image
        mask
        dilate


        Returns
        -------
        obstools.image.segmentation.SegmentedImage
        """

        # logger.info('{} {}', args, kws)

        # Initialize
        seg = SegmentedImage(self.fit_predict(image, mask, *args, **kws))

        # dilate
        if dilate != 'auto':
            seg.dilate(iterations=dilate)

        self.report(image, seg)
        return seg

    def report(self, image, seg, show=5):

        self.logger.opt(lazy=True).info(
            'Detected {:d} sources covering {:.2%} of image.',
            lambda: seg.nlabels, lambda: sum(seg.fractional_areas)
        )


class GMM(DetectionBase):
    def __init__(self, n_components=5, **kws):
        self.gmm = GaussianMixture(n_components, **kws)

    def fit_predict(self, image, mask=False):
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

    def __call__(self, image, mask=False, *args, plot=False, **kws):

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


class SigmaThreshold(DetectionBase):
    def fit_predict(self, image, mask=False, background=None, snr=3., npixels=7,
                    edge_cutoff=None, deblend=False):
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
                          str(dict(snr=snr,
                                   npixels=npixels,
                                   edge_cutoff=edge_cutoff,
                                   deblend=deblend)))

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
            self.logger.debug('No objects detected.')
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


class BackgroundFitter(DetectionBase):
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


class SourceAggregator(DetectionBase):
    """
    Aggregate info from multiple detection loops
    """

    def __init__(self, algorithm, model=None):
        self.count = 0
        self.seg = None
        super().__init__(algorithm, model)

    def __call__(self, image, mask=False, group_id=groups.auto_id, **kws):
        # update mask
        if mask is None:
            mask = False

        if self.seg is None:
            # first round
            self.seg = SegmentedImage.empty_like(image)

        # ignore previous detections
        new_seg = super().__call__(image, mask | self.seg.to_binary(), **kws)
        self.seg.add_segments(new_seg, group_id=group_id)

        self.count += 1
        return self.seg


class ResultsAggregator(SourceAggregator, BackgroundFitter):
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


class SourceDetection(ResultsAggregator):
    """A descriptor object for managing source detection algorithms"""

    def __get__(self, obj, kls=None):
        if obj is None:
            # called from class. Re-init to start clean.
            return self.__class__(self.fit_predict)
        return self

    def __set__(self, obj, algorithm):
        """
        >>> class MyImage:
        ...     detect = SourceDetection('gmm')
        ... img = MyImage().detect

        later to switch algorithms:
        >>> img.detect = 'sigma_threshold'
        # FIXME: This is not really intuitive img.detect.algorithm better
        """

        # resolve strings
        if isinstance(algorithm, str):
            return type(obj).members[sub(algorithm.lower(), {' ': '', '_': ''})]

        if not callable(algorithm):
            raise TypeError('Invalid algorithm.')

        # create detector class from callable algorithm
        return self.__class__(algorithm)


class SourceDetectionLoop(SourceDetection):

    def __init__(self, algorithm, model=None, *args, max_iter=1, **kws):
        super().__init__(algorithm, model, *args, **kws)
        self.max_iter = max_iter
        self.params = self.iter_params(**kws)

    def __call__(self, image, mask=False, max_iter=None, *args, **kws):
        if max_iter is None:
            max_iter = self.max_iter

        for _ in self:
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


class MultiThreshold(SourceDetectionLoop):
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


# class SourceDetectionDescriptor:
#     """Descriptor class for source detection"""

#     def __init__(self, algorithm):
#         # self.algorithm = algorithm
#         self.algorithm = SourceDetection(algorithm)

#     def __get__(self, instance, objtype=None):
#         if instance is None:  # called from class
#             return self

#         # called from instance
#         return self.detect

# #     def __call__(self, *args, **kws):
#         return self.detect

# class DetectionMeta(type):

#     _algorithm = sigma_threshold  # default
#     """
#     Default blob detection algorithm. Instances of this metaclass can set this
#     with the `detection` property
#     """

#     # todo register the algorithms

#     @property
#     def detection(self):
#         """The source detection algorithm"""
#         return self._algorithm

#     @detection.setter
#     def detection(self, algorithm):
#         # resolve strings
#         if isinstance(algorithm, str):
#             algorithm = algorithms[algorithm]

#         self._algorithm = algorithm


class SourceDetectionMixin:
    """
    Provides the `from_image` classmethod and the `detect` staticmethod that
    can be used to construct image models from images.
    """

    detect = SourceDetection('sigma_threshold')

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
        detect : bool
            Controls whether source detection algorithm is run. This argument
            provides a shortcut to the default source detection by setting
            `True`, or alternatively to skip source detection by setting
            `False`

        detect_opts : Keywords passed to

        Returns
        -------

        """

        # Detect objects & segment image
        detect_opts = dict(detect if isinstance(detect, dict) else {},
                           **detect_opts)
        if not detect_opts:
            # short circuit the detection loop
            detect_opts['max_iter'] = 0

        # Basic constructor that initializes the object from an image. The
        # base version here runs a detection algorithm to separate foreground
        # objects and background, but doesn't actually include any physically
        # useful models. Subclasses can overwrite this method to add useful
        # models to the segments.

        # Detect objects & init with segmented image
        return cls.detect(image, **detect_opts)


# algorithms = {'sigma_threshold': sigma_threshold,
#               }
