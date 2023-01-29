"""
Methods for tracking camera movements in astronomical time-series CCD photometry
"""

# std
import math
import numbers
import tempfile
import functools as ftl
import itertools as itt
import contextlib as ctx
import multiprocessing as mp
from pathlib import Path
from pydoc import describe

# third-party
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from loguru import logger
from joblib import Parallel, delayed
from bottleneck import nanmean, nanstd
from scipy.spatial.distance import cdist
from matplotlib.transforms import AffineDeltaTransform
from astropy.utils import lazyproperty
from astropy.stats import median_absolute_deviation as mad

# local
import motley
from recipes.io import load_memmap
from recipes.pprint import describe
from recipes.dicts import AttrReadItem
from recipes.logging import LoggingMixin
from recipes.parallel.joblib import initialized

# relative
from ..image.noise import CCDNoiseModel
from ..image.detect import SourceDetectionMixin, make_border_mask
from ..image.registration import (ImageRegister, compute_centres_offsets,
                                  report_measurements)
from ..image.segmentation import (LabelGroupsMixin, SegmentedImage,
                                  SegmentsModelHelper, merge_segmentations)
from .gui import SourceTrackerGUI


# from recipes.parallel.synced import SyncedArray, SyncedCounter


# TODO: CameraTrackingModel / CameraOffset / CameraPositionModel
# ConstellationModel
# TODO: filter across frames for better shift determination ???
# TODO: wavelet sharpen / lucky imaging for better relative positions

# TODO
#  simulate how different centre measures performs for sources with decreasing snr
#  super resolution images
#  lucky imaging ?


# ---------------------------------------------------------------------------- #


FILENAMES = dict(measurements='centroids.dat',
                 measure_avg='centroids-mean.dat',
                 measure_std='centroids-std.dat',
                 delta_xy='coords-delta.dat',
                 sigma_xy='coords-std.dat',
                 flux='flux.dat',
                 flux_std='flux-std.dat',
                 snr='snr.dat')

TABLE_STYLE = dict(txt='bold', bg='g')
LABEL_STYLE = dict(offset=(6, 6),
                   color='w',
                   size=10,
                   fontweight='bold')

memory_lock = ctx.nullcontext()

# ---------------------------------------------------------------------------- #
PROGRESS_FMT = motley.stylize(
    '{{desc}: {percentage:3.0f}%{bar}'
    '{n_fmt}/{total_fmt}:|lime} '
    '{rate_fmt:|gold} '
    '{elapsed:|cyan} eta {remaining:|cyan}'
)


def write_tqdm_log(msg):
    tqdm.write(msg, end='')


def set_lock(mem_lock, tqdm_lock):
    """
    Initialize each process with a global variable lock.
    """
    global memory_lock
    memory_lock = mem_lock
    tqdm.set_lock(tqdm_lock)


# ---------------------------------------------------------------------------- #
def check_image_drift(cube, nframes, mask=None, snr=5, npixels=10):
    """Estimate the maximal positional drift for sources"""

    #
    logger.info('Estimating maximal image drift for {:d} frames.', nframes)

    # take `nframes` frames evenly spaced across data set
    step = len(cube) // nframes
    maxImage = cube[::step].max(0)  #

    segImx = SegmentedImage.detect(maxImage, mask, snr=snr, npixels=npixels,
                                   dilate=3)

    mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                      for (xs, ys) in segImx.slices], 0)
    return mxshift, maxImage, segImx


def boundary_proximity(seg, points, labels=None):
    labels = seg.resolve_labels(labels)
    return np.array([np.sqrt(np.square(xy - seg.traced[l][0][0]).sum(1).min())
                     for l, xy in zip(labels, points)])
    # return np.array([np.sqrt(np.square(xy - boundary).sum(1).min())
    #                  for ((boundary,), _), xy in zip(seg.traced.values(), points)])

# class NullSlice():
#     """Null object pattern for getitem"""
#
#     def __getitem__(self, item):
#         return None

# constructor used for pickling
# def _construct_SegmentedArray(*args):
#     return SegmentedArray(*args)


class LabelUser:
    """
    Mixin class for objects that use `SegmentedImage`.

    Adds the `use_label` and `ignore_label` properties.  Whenever a function
    that takes the `label` parameter is called with the default value `None`,
    the labels in `use_labels` will be used instead. This helps with dynamically
    including / excluding labels and with grouping labelled segments together.

    If initialized without any arguments, all labels in the segmentation image
    will be in use.

    """

    def __init__(self, use_labels=None, ignore_labels=None):
        if use_labels is None:
            use_labels = self.seg.labels
        if ignore_labels is None:
            ignore_labels = []

        self._use_labels = np.setdiff1d(use_labels, ignore_labels)
        self._ignore_labels = np.asarray(ignore_labels)

    @property
    def ignore_labels(self):
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, labels):
        self._ignore_labels = np.asarray(labels, int)
        self._use_labels = np.setdiff1d(self.seg.labels, labels)

    @property
    def use_labels(self):
        return self._use_labels

    @use_labels.setter
    def use_labels(self, labels):
        self._use_labels = np.asarray(labels, int)
        self._ignore_labels = np.setdiff1d(self.seg.labels, labels)

    def resolve_labels(self, labels=None):
        return self.use_labels if labels is None else self.seg.has_labels(labels)

    @property
    def nlabels(self):
        return len(self.use_labels)

    @property
    def sizes(self):
        return [len(labels) for labels in self.values()]


# class MaskUser:
#     """Mixin class giving masks property"""
#
#     def __init__(self, groups=None):
#         self._groups = None
#         self.set_groups(groups)
#


class MaskContainer(AttrReadItem):
    def __init__(self, seg, groups, **persistent):
        assert isinstance(groups, dict)  # else __getitem__ will fail
        super().__init__(persistent)
        self.persistent = np.array(list(persistent.values()))
        self.seg = seg
        self.groups = groups

    def __getitem__(self, name):
        if name in self:
            return super().__getitem__(name)

        if name in self.groups:
            # compute the mask of this group on demand
            mask = self[name] = self.of_group(name)
            return mask

        raise KeyError(name)

    @lazyproperty
    def all(self):
        return self.seg.to_binary()

    def for_phot(self, labels=None, ignore_labels=None):
        """
        Select sub-regions of the image that will be used for photometry.
        """
        labels = self.seg.resolve_labels(labels)
        # np.setdiff1d(labels, ignore_labels)
        # indices = np.digitize(labels, self.seg.labels) - 1
        kept_labels = np.setdiff1d(labels, ignore_labels)
        indices = np.digitize(labels, kept_labels) - 1

        m3d = self.seg.to_binary_3d(None, ignore_labels)
        all_masked = m3d.any(0)
        return all_masked & ~m3d[indices]

    def prepare(self, labels=None, ignore_labels=None, sky_buffer=2,
                sky_width=10, edge_cutoffs=None):
        """
        Select sub-regions of the image that will be used for photometry.
        """

        # sky_regions = self.prepare_sky(labels, sky_buffer, sky_width,
        #                                edge_cutoffs)
        self['phot'] = masks_phot = self.for_phot(labels, ignore_labels)
        self['sky'] = masks_phot.any(0)

    def prepare_sky(self, labels, sky_buffer=2, sky_width=10,
                    edge_cutoffs=None):
        sky_regions = self.seg.to_annuli(sky_buffer, sky_width, labels)
        if edge_cutoffs is not None:
            edge_mask = make_border_mask(self.seg.data, edge_cutoffs)
            sky_regions &= ~edge_mask

        return sky_regions

    def of_group(self, g):
        return self.seg.to_binary(self.groups[g])


class SegmentsMasksHelper(SegmentsModelHelper, LabelGroupsMixin):

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

    def select_overlap(self, start, shape, type_=SegmentedImage):
        return super().select_overlap(start, shape, type_)

    def for_offset(self, delta_xy, shape, type_=SegmentedImage):
        if np.ma.is_masked(delta_xy):
            raise ValueError('Cannot get segmented image for masked offsets.')

        return self.select_overlap(
            self.get_start_indices(delta_xy), shape, type_)

    def flux(self, image, origin, labels=None, labels_bg=(0,), bg_stat='median'):
        sub = self.select_overlap(origin, image.shape)
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


class SourceTracker(LabelUser,
                    LabelGroupsMixin,
                    # SourceDetectionMixin,
                    LoggingMixin):
    """
    A class to track sources in a CCD image frame to aid in doing time series
    photometry.

    Stellar positions in each new frame are calculated as the background
    subtracted centroids of the chosen image segments. Centroid positions of
    chosen sources are filtered for bad values (and outliers) and then combined
    by a weighted average. The weights are taken as the SNR of the star
    brightness. Also handles arbitrary pixel masks.

    Constellation of sources in image is assumed to be unchanging. The
    coordinate positions of the sources with respect to the brightest source are
    updated upon each iteration.
    """
    # FIXME: remove redundant code
    # TODO: bayesian version

    centrality = {
        'com':              1,
        # 'com_bg':           0,
        'geometric_median': 0,
        'peak':             1,
        # 'upsample_max':     0
    }

    bg = staticmethod(np.ma.median)

    # params for detecting outliers
    # TODO: manage as properties ?
    snr_cut = 0
    _distance_cut = 10
    # maximum distance allowed for point measurements from centre of
    # distribution. This allows tracking sources with low snr, but not computing
    # their centroids which will have larger scatter.
    _saturation_cut = 95
    # If we know the saturation value of the chip, we discard source
    # measurements with saturated pixels.

    def __init__(self, coords, seg,
                 use_labels=None, label_groups=None,
                 bad_pixel_mask=None, edge_cutoffs=0,
                 weights='snr', update_weights_every=25,
                 update_centres_every=100, measure_centres_after=None,
                 precision=0.5,  # should be ~ diffraction limit
                 reference_index=0,
                 noise_model=None):
        """
        Class for tracking camera movement by measuring location of sources
        in CCD images. The class combines various measures of central tendency
        eg. centroid, projected geometric median, marginal gaussian fit, to
        estimate the source positions. The relative source positions are
        estimated using data from all these measurements, whereupon the
        positional shifts for each image frame are computed by weighted average
        of the offset measurements of the individual sources from their cluster
        centers, optionally using the source signal-to-noise ratios as weights.

        Parameters
        ----------
        coords : array-like
            Initial reference coordinates of the sources, xy. These will be
            updated dynamically.
        seg :  SegmentedImage or array-like
            The segmentation image. Centroids for each source are computed
            within the corresponding labelled region.
        use_labels : array-like, optional
            Only sources corresponding to these labels will be used to calculate
            frame shift. Note that center of mass measurements are for all
            sources (that are on-frame) in order to compute the best relative
            positions for all sources, but that the frame-to-frame shift will be
            computed using only the sources in `use_labels`, by default this
            includes all sources.
        label_groups : str, optional
            Name of the segment label group to use for centroid measurements, by
            default None, which uses all availbale labels.
        bad_pixel_mask : np.ndarray, optional
            Ignore these pixels in centroid computation.
        edge_cutoffs : int or list of int, optional
            Ignore labels that are (at most) this close to the edges of the
            image, by default 0.
        weights : bool, str, optional
            Per-source weights for computing frame offset by weighted average.
            By default all sources are weighted equally. To activate
            signal-to-noise ratio weighting, `use weights='snr'`. The frequency
            of the weight update calculation is controlled by the
            `update_weights_every` parameter.
        update_weights_every : int, optional
            How often the source weights are updated, by default every 10 frames.
        update_centres_every : int, optional
            How often the relative position vector is updated. Once the required
            positional accuracy of sources has been acheived, the relative
            position vector will not be updated any further. The default is 100.
        precision : float, optional
            Required positional accuracy of soures in units of pixels, by
            default 0.25.
        reference_index : int, optional
            Index of source (coords) to use as internal reference, by default 0.
            The `rpos` property will return relative positions with respect to
            this source.
        """

        if isinstance(seg, SegmentsMasksHelper):
            # detecting the class of the SegmentationImage allows some
            # optimization by avoiding unnecessary recompute of lazyproperties.
            # Also allows custom subclasses of SegmentsModelHelper to be
            # used
            self.seg = seg
        else:
            self.seg = SegmentsMasksHelper(seg,
                                           groups=label_groups,
                                           bad_pixels=bad_pixel_mask)

        # label include / exclude logic
        LabelUser.__init__(self, use_labels)
        LabelGroupsMixin.__init__(self, label_groups)

        # counter for incremental update
        self.counter = itt.count()  # SyncedCounter()
        # shared memory. these are just placeholders for now, they are set in
        # `_init_memory`
        self.measurements = self.measure_avg = self.measure_std = self.delta_xy = self.sigma_xy = None

        # reference position (in pixel coordinates) from which the shift will
        # be measured.
        # store originand relative positions separately so we can update
        self.ir = int(reference_index)
        self.xy0 = coords[self.ir]
        self.rpos = coords - coords[self.ir]
        self.origin = np.array([0, 0])
        self.region_centres = self.seg.com(self.seg.data)

        # algorithmic details
        self.edge_cutoffs = edge_cutoffs
        self.precision = float(precision)
        self.snr_weighting = (weights == 'snr')
        self._source_weights = np.ones(len(coords)) if self.snr_weighting else weights
        self._update_weights_every = (int(max(update_weights_every, 0))
                                      if self.snr_weighting else 0)
        self._update_centres_every = int(max(update_centres_every, 0))
        self._measure_centres_after = int(measure_centres_after or update_centres_every)
        self._precision_reached = False

        self.noise_model = noise_model

        # masks for photometry
        # self.bad_pixel_mask = bad_pixel_mask

        # if edge_cutoffs is not None:
        #     self.edge_mask = make_border_mask(self.seg.data, edge_cutoffs)
        # else:
        #     self.edge_mask = None

        # extended masks
        # self.masks = MaskContainer(self.seg, self.groups,
        #                            bad_pixels=bad_pixel_mask)

        # label logic
        # if use_labels is None:
        #     # we want only to use the sources with high snr fro CoM tracking

    def init_memory(self, n, loc=None, overwrite=False):
        """
        Initialize shared memory synchronised access wrappers. Should only be
        run in the main process.

        Parameters
        ----------
        n:
            number of frames
        loc
        overwrite

        Returns
        -------

        """

        if loc is None:
            loc = tempfile.mkdtemp()
        loc = Path(loc)

        # get array shapes
        nstats = len(self.centrality)
        nsources = self.seg.nlabels
        shapes = dict(
            measurements=(n, nstats, nsources, 2),
            measure_avg=(n, nsources, 2),
            delta_xy=(n, 2),
            sigma_xy=(nsources, 2),
            flux=(n, nsources),
            flux_std=(n, nsources)
        )
        if self.snr_weighting or self.snr_cut:
            shapes['snr'] = (math.ceil(n / self._update_weights_every), nsources)
        if self.noise_model:
            shapes['measure_std'] = (n, nsources, 2)

        # load memory
        spec = ('f', np.nan, overwrite)
        for name, shape in shapes.items():
            setattr(self, name, load_memmap(loc / FILENAMES[name], shape, *spec))

    def __call__(self, image, index, mask=None, counter=None, depth=0):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """
        count = next(counter or self.counter)
        self.logger.debug(f'{count = }, {index = }')

        # TODO: use grid and add offset to grid when computing CoM.  Will be
        # more efficient
        # xy = self.measure(image, index,  mask)
        # # off = (xy - self.region_centres).mean(0)

        # # weights
        # if self.should_update_weights(count):
        #     self.update_weights(image, count)
        #     # TODO: maybe aggregate weights and use mean ??

        #     # print(count, 'weights', self._source_weights)

        # # update relative positions from CoM measures
        # # check here if measurements of cluster centres are good enough!
        # if self.should_update_pos(count):
        #     self.logger.info('Measuring positions, offsets at count {}.', count)
        #     self.compute_centres_offsets(self.measurements[:(index + 1)])
        #     off = self.delta_xy[index]
        # else:
        #     # calculate median frame shift using current best relative positions
        #     # optionally using snr weighting scheme
        #     # measurements wrt image, offsets wrt coords
        #     off = self.compute_offset(xy, self._source_weights)
        #     self.delta_xy[index] = off

        # # if np.isnan(off).any() or np.ma.is_masked(off):
        # #     raise ValueError('')
        # #
        # # if np.any(np.abs(off) > 20):
        # #     raise ValueError('')

        # # if np.any(np.abs(self.current_offset - off) > 1):
        # #     self.origin = off.round().astype(int)
        # print('!!!!!'*10)
        # self.logger.info('OFFSET: {}', off)

        # # Update origin
        # if np.any(off >= 1) and depth == 0:
        #     # NOTE: `origin` in image yx coords
        #     self.logger.info('ORIGIN SHIFT: {}', off)
        #     self.origin += np.round(off)[::-1]
        #     self.logger.debug('Updated origin = {}. offset = {}', self.origin, off)

        #     # re-measure
        #     self.logger.debug('re-measuring centroids')
        #     return self(image, index,  mask, iter([count]), depth + 1)

        # finally return the new coordinates
        # return self.coords + off

    def __str__(self):
        nl = '\n'
        pre = f'{type(self).__name__}(coords='
        return pre + str(self.coords).replace(nl, nl.ljust(len(pre) + 1)) + ')'

    __repr__ = __str__

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):

        # ensure correct type
        origin = np.asanyarray(origin)
        if np.ma.is_masked(origin):
            raise ValueError('Lower left corner indices `origin` should not be '
                             'masked.')

        if origin.dtype.kind != 'i':
            origin = origin.round().astype(int)

        if np.any(origin > self.seg.shape):
            raise ValueError(f'Tracker `origin` {origin} outside of image'
                             f' boundary for segmented image: {self.seg}')

        self._origin = origin

    @property
    def coords(self):
        """
        Reference coordinates (xy). Computed as initial coordinates `xy0` +
        displacement array `rpos` to allow updating the relative positions upon
        call.
        """
        return self.xy0 + self.rpos

    @property
    def masks(self):  # extended masks
        return self.seg.group_masks

    @lazyproperty
    def feature_weights(self):
        weights = np.fromiter(self.centrality.values(), float)
        return (weights / weights.sum())[:, None, None]

    def should_update_weights(self, count=0):
        return ((self.snr_weighting or self.snr_cut) and
                ((self._source_weights is None) or
                 (count % self._update_weights_every == 0)))

    def should_update_pos(self, count=0):
        # if count % self._update_centres_every) == 0:
        return (count % self._update_centres_every) == 0  # (count > 0) and
        # ( )#and
        # (self.sigma_xy.mean() > self.precision))

    def get_coords(self, i=None):
        if i is None:
            return self.coords[None] + self.delta_xy[:, None]
        return self.coords + self.delta_xy[i]

    def get_coord(self, i):
        return self.coords + self.delta_xy[i]

    def get_coords_residual(self):
        return self.measurements - self.coords - self.delta_xy[:, None]

    # @api.synonyms({'njobs': 'n_jobs'})
    def run(self, data, indices=None, n_jobs=-1, progress_bar=True):
        """
        Start a worker pool of source trackers. The workload will be split into
        chunks of size ``

        Parameters
        ----------
        data : array-like
            Image stack.
        indices : Iterable, optional
            Indices of frames to compute, the default None, runs through all the
            data.
        n_jobs : int, optional
            Number of concurrent woorker processes to launch, by default -1

        progress_bar : bool, optional
            _description_, by default True

        Raises
        ------
        FileNotFoundError
            If memory has not been initialized prior to calling this method.
        """

    #     if self.measurements is None:
    #         raise FileNotFoundError('Initialize memory first by calling the '
    #                                 '`init_memory` method.')

    #     for i, f in enumerate(self.centrality):
    #         getattr(self.seg, f)(data, None, n_jobs, self.measurements[:, i])

    # def _run(self, data, indices=None, n_jobs=-1, progress_bar=True, **kws):

        if indices is None:
            indices, = np.where(~self.measured)

        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the '
                                    '`init_memory` method.')

        # divide work
        batch_size = self._update_weights_every or self._update_centres_every
        batches = mit.chunked(indices, batch_size)
        n_batches = len(indices) / batch_size

        # triggers for computing coordinate centres and weights as needed
        update_centres, update_weights = (
            itt.chain(
                # burn in
                itt.repeat(False, burn_in - 1),
                # compute every nth batch
                itt.cycle(itt.chain([None], itt.repeat(False, every - 1)))
            )
            for burn_in, every in
            np.array([(self._measure_centres_after, self._update_centres_every),
                      (0, self._update_weights_every)]) // batch_size
        )

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        # setup compute context
        pbar = dict(bar_format=PROGRESS_FMT, ascii=' ╸━', unit=' frames',
                    disable=not progress_bar)
        # workload = zip(batches, update_weights, update_centres)
        workload = tqdm(zip(batches, update_weights, update_centres),
                        total=n_batches, unit_scale=batch_size, **pbar)

        if n_jobs == 1:
            # workload iterable with progressbar if required
            worker = self.loop
            context = ctx.nullcontext(list)
        else:
            tqdm.set_lock(mp.RLock())  # for managing output contention
            memory_lock = mp.Lock()

            # context = Parallel(n_jobs=n_jobs, backend='loky', verbose=1)
            executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', verbose=10)
            context = initialized(executor, set_lock, (memory_lock, tqdm.get_lock()))
            # NOTE: object serialization is about x100-150 times faster with
            # multiprocessing backend. ~0.1s vs 10s.
            # backend = 'multiprocessing'

            worker = delayed(self.loop)
            # workload iterable with progressbar if required

            # setup progressbar
            # logger.remove()
            # pbar_sink = self.logger.add(write_tqdm_log, colorize=True, enqueue=True)

        # execute
        with context as compute:
            # t_start = time.time()
            compute(worker(data, *args) for args in workload)

            # logger.debug('With {} backend, pickle serialization took: {:.3f}s',
            #              backend, time.time() - t_start)
            # else:
            #     worker_pool.starmap(self.loop,
            #                         ((data, idx) for idx in mit.divide(n_jobs, indices)))
            #     worker_pool.close()
            #     worker_pool.join()

        # self.logger.remove(pbar_sink)
        # self.logger.add(sys.stderr)
        # self.logger.debug('{} Processes successfully shut down.', n_jobs)

    def loop(self, data, indices, update_weights=None, update_centres=None):

        if isinstance(indices, numbers.Integral):
            indices = np.array(indices, 'i')

        indices = np.fromiter(indices, 'i')

        # weights
        if update_weights or self.should_update_weights():
            i = indices[0]
            self.update_weights(data[i], i)

        # measure
        for i in indices:
            xym = self.measure(data[i], i)
            off = self.compute_offsets(xym, self._source_weights)  # δxy

            # Update origin
            if np.any(np.abs(shift := off[::-1] - self.origin) >= 1):
                self.logger.info('SHIFT: {}', shift)
                # NOTE: `origin` in image yx coords
                self.origin += np.array(np.round(shift)).astype(int)
                self.logger.debug('Updated origin = {}.', self.origin)

                # re-measure
                self.logger.debug('re-measuring centroids')
                xym = self.measure(data[i], i)
                off = self.compute_offsets(xym, self._source_weights)

            with memory_lock:
                self.delta_xy[i] = off

        xym = self.measure_avg[indices]

        if update_centres or self.should_update_pos():
            self.logger.info('Updating source positions & offsets.')
            self.compute_centres_offsets()
            self.logger.opt(lazy=True).debug(
                'Average pos stddev: {}', lambda: self.sigma_xy.mean(0))
        else:
            # calculate median frame shift using current best relative positions
            # optionally using snr weighting scheme
            # measurements wrt image, offsets wrt coords
            with memory_lock:
                self.sigma_xy = nanstd(self.measure_avg - self.delta_xy[:, None], 0)
                self.logger.opt(lazy=True).debug(
                    'Average pos stddev: {}', lambda: self.sigma_xy.mean(0))

        # self.logger.info('OFFSET: {}', off)
        return self.coords + self.delta_xy[indices, None]

    @property
    def measured(self):
        return ~np.isnan(self.measure_avg).all((1, 2))

    def measure(self, image, index, mask=None, origin=None):
        # self.logger.debug('Measuring centroids')
        self.measurements[index] = xy = self.measure_source_locations(image, mask, origin)
        self.measure_avg[index] = xym = (xy * self.feature_weights).sum(0)

        if self.noise_model:
            self.measure_std[index] = self.seg.com_std(
                xy[0], image, self.noise_model(image), self.use_labels
            )

        self.flux[index], self.flux_std[index] = self.seg.flux(image)

        return xym

    def measure_source_locations(self, image, mask=None, origin=None):
        """
        Calculate measure of central tendency (centre-of-mass) for the objects
        in the segmentation image

        Parameters
        ----------
        image
        mask
        origin:
            CRITICAL XY coordinates!!!!!!!!

        Returns
        -------

        """
        # more robust that individual CoM track.
        # eg: cosmic rays:
        # '/media/Oceanus/UCT/Observing/data/Feb_2015/
        #  MASTER_J0614-2725/20150225.001.fits' : 23

        if mask is None:
            mask = self.masks.bad_pixels  # may be None
        elif self.masks.bad_pixels is not None:
            mask |= self.masks.bad_pixels

        yx = self._measure_source_locations(image, mask, origin)

        # check measurements
        # FIXME: still need this ?? - checking overlap at every frame above
        bad = np.any((yx < self.edge_cutoffs) |
                     (yx > np.subtract(image.shape, self.edge_cutoffs)),
                     -1)
        yx[bad] = np.nan

        return yx[..., ::-1]

    def _measure_source_locations(self, data, mask, origin):
        """
        Measure locations for all sources that are on-frame, ignoring those that
        are partially off-frame.

        Parameters
        ----------
        data : np.ndarray
            Image array for which locations will be measured
        mask : np.ndarray
            Masked pixels to ignore.  Should be broadcastable `data.shape`
        origin : np.ndarray
            Lower left corner of the current data with respect to the global
            segmentation image.
            CRITICAL  XY coordinates!!!!!!!!

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        assert data.ndim in {2, 3}

        if origin is None:
            origin = self.origin

        if np.any((np.abs(origin) > (shape := data.shape[-2:]))):
            raise ValueError(f'Image of shape {shape} does not overlap with'
                             ' segmentation image: Tracker origin (lower left'
                             f' corner) is at {origin}.')

        self.logger.trace('Measuring source locations for frame with origin at '
                          '{}.', origin)

        # select the current active region from global segmentation
        seg = self.seg.select_overlap(origin, shape)

        # work only with sources that are contained in this image region
        # remove detections that are partially off-frame
        # NOTE origin in image coordinates yx!!
        # labels = set(self.seg.labels)
        # for k, corner in zip((1, -1), (origin, origin + image.shape)):
        #     for i, j in  zip((1, -1), np.clip(corner, 0, None)):
        #         print(((slice(*(None, j)[::k]), ...)[::i]))
        #         # off image segment
        #         labels -= set(self.seg.data[((slice(*(None, j)[::k]), ...)[::i])].ravel())

        # if not labels:
        #     from IPython import embed
        #     embed(header="Embedded interpreter at 'src/obstools/phot/tracking.py':1097")
        #     raise ValueError('No segments in image.')

        # compute centroids
        data = np.ma.MaskedArray(data, mask)
        labels = None  # self.use_labels
        yx = np.full((len(self.centrality), self.seg.nlabels, 2), np.nan)
        for i, stat in enumerate(self.centrality):
            # labels - 1
            yx[i] = c = getattr(seg, stat)(data, labels, njobs=1)[:, :2]
            self.logger.trace('stat: {}\n{}', stat, c)

        # NOTE this is ~2x faster than padding image and mask. can probably
        #  be made even faster by using __slots__

        # TODO: check if using grid + offset then com is faster
        # TODO: can do FLUX estimate here!!!

        # self.check_measurement(xy)
        # self.coords - tracker.zero_point < set_lockt[0]

        # check values inside segment labels
        # good = ~self.is_bad(com)
        # self.coms[i, good] = com[good]

        return yx

    # def check_measurement(self, xy):

    # def _compute_centres_offset(self):

    def compute_centres_offsets(self):
        """
        Measure the relative positions of sources, as well as the xy offset
        (dither) of each image frame based on the set of centroid; measurements
        in `xy`.

        Parameters
        ----------
        xy

        Returns
        -------

        """
        # xy = self.measurements
        # ok = np.logical_not(np.isnan(self.measurements).all((1, 2, 3)))

        self.logger.debug('compute_centres_offsets')

        with memory_lock:
            if (self.sigma_xy.mean() < self.precision).all():
                return self._check_precision()

        xym, centres, σ_pos, delta_xy, out = compute_centres_offsets(
            self.measure_avg, self._distance_cut)

        # self.logger.info('compute_centres_offsets OK')

        # origin = np.floor(delta_xy.min(0))

        # self.zero_point
        # note using the first non-masked point will avoid this compute
        #  FIXME: account for difference between zero point and min off here...
        # smallest offset as origin makes sense from global image perspective
        # changing the zero_point means you have to update all offsets

        # make the offsets relative to the global segmentation
        xy0 = centres[self.ir]
        δ = self.xy0 - xy0

        with memory_lock:
            self.delta_xy[:] = delta_xy + δ
            self.rpos = centres - xy0
            self.sigma_xy[:] = σ_pos

        # self.logger.info('compute_centres_offsets SAVED')
        # if (σ_pos < self.precision).all():

        self.report_measurements(xym, centres, σ_pos)

    # TODO Rename this here and in `compute_centres_offsets`

    def _check_precision(self):
        if self._precision_reached:
            return

        n = np.sum(self.measured)
        if n < self._measure_centres_after:
            self.logger.info('Delaying centre compute.')
            return

        self._precision_reached = n
        self.logger.opt(lazy=True).info(
            'Required positional accuracy of {0[0]:} < {0[1]:g} achieved '
            'with {0[2]:} measurements. Relative source positions will '
            'not be updated further.',
            lambda: (self.sigma_xy.mean(0),
                     self.precision,
                     self._precision_reached)
        )
        return

    def report_measurements(self, xy=None, centres=None, σ_xy=None,
                            counts=None, detect_frac_min=None,
                            count_thresh=None):
        # TODO: rename pprint()

        if xy is None:
            nans = np.isnan(self.measurements)
            good = ~nans.all((1, 2))
            xy = self.measurements[good]

        if centres is None:
            centres = self.coords

        if σ_xy is None:
            σ_xy = self.sigma_xy

        return report_measurements(xy, centres, σ_xy, counts, detect_frac_min,
                                   count_thresh, self.logger)

    def plot(self, image=None, ax=None, contours=True, labels=True, **kws):
        im = None
        if image is not None:
            from scrawl.image import ImageDisplay

            im = ImageDisplay(image, ax=ax, **kws)
            ax = im.ax

        if contours is False:
            return im

        if contours is True:
            contours = {}

        contours = self.seg.show.contours(
            ax,
            offsets=self.origin,
            transOffset=AffineDeltaTransform(ax.transData),
            **contours
        )

        # add label artists
        texts = self.seg.show.labels(ax, **LABEL_STYLE) if labels else None
        return im, contours, texts

    # def animate(self, data):

    def plot_position_measures(self, labels, section=..., **kws):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        from .gui import SourceTrackerGUI

        delta = self.measurements[section] - self.coords
        delta_avg = self.measure_avg[section] - self.coords
        delta_xy = self.delta_xy[section]
        comm = dict(lw=1, ls='--', fc='none', ec='k', zorder=10)

        fig, axes = plt.subplots(len(labels), 2, sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        for ax in axes.ravel():
            # add pixel size rect
            ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, **comm))
            ax.add_patch(Circle((0, 0), self.precision, **comm))
            ax.set_aspect('equal')
            ax.tick_params(bottom=True, top=True, left=True, right=True)

        comm = dict(ls='', mfc='none', zorder=1)
        features = {**self.centrality, 'Weigthed Avg': 1}
        for j in range(len(labels)):
            ax1, ax2 = axes[j]

            for i, (measure, use) in enumerate(features.items()):
                if not use:
                    continue

                if items := SourceTrackerGUI.centroid_props.get(measure):
                    data, (marker, label) = delta[:, i, j], items
                else:
                    data, marker, label = delta_avg[:, j], 'x', measure

                comm = {**comm, 'marker': marker, 'label': label}
                ax1.plot(*data.T, **comm)
                ax2.plot(*(data + delta_xy).T, **comm)

        # fig.tight_layout()
        # fig.subplots_adjust(top=0.82, hspace=0, vspace=0)
        axes[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.05))

    # def plot_position_measures(self, **kws):

    #     import matplotlib.pyplot as plt
    #     from matplotlib.patches import Rectangle, Circle
    #     from .gui import SourceTrackerGUI

    #     delta = self.measurements - self.coords

    #     for j in range(len(self.use_labels)):
    #         fig, axes = _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    #         for ax in axes:
    #             # add pixel size rect
    #             comm = dict(lw=1, ls='--', fc='none', ec='k', zorder=10)
    #             ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, **comm))
    #             ax.add_patch(Circle((0, 0), self.precision, **comm))
    #             ax.set_aspect('equal')
    #             ax.tick_params(bottom=True, top=True, left=True, right=True)

    #         for i, (measure, use) in enumerate(self.centrality.items()):
    #             if not use:
    #                 continue

    #             marker, label = SourceTrackerGUI.centroid_props[measure]
    #             comm = dict(marker=marker, label=label, ls='', mfc='none', zorder=1)
    #             ax1.plot(*delta[:, i, j].T, **comm)
    #             ax2.plot(*(delta[:, i, j] + self.delta_xy).T, **comm)

    #         ax1.legend(loc='lower center', bbox_to_anchor=(1.05, 1.05))

    # def plot_position_measures(self, **kws):

        #     self.measurements

        # from obstools.phot.diagnostics import plot_position_measures

        # return plot_position_measures(self.measurements, self.coords,
        #                               -self.delta_xy, **kws)

    # def _pad(self, image, origin, zero=0.):
    #     # make a measurement of some property of image using the segments
    #     # from the tracker. In general, image will be smaller than global
    #     # segmentation, therefore first pad the image with zeros
    #
    #     stop_indices = origin + image.shape
    #
    #     iset_lockt = -np.min([origin, (0, 0)], 0)
    #     istop = np.array([None, None])
    #     thing = np.subtract(self.seg.shape, stop_indices)
    #     l = thing < 0
    #     istop[l] = thing[l]
    #     islice = tuple(map(slice, istart, istop))
    #
    #     origin = np.clip(origin, 0, None)
    #     stop_indices = np.clip(stop_indices, None, self.seg.shape)
    #
    #     section = tuple(map(slice, origin, stop_indices))
    #     im = np.full(self.seg.shape, zero)
    #     im[section] = image[islice]
    #
    #     return im

    def prepare_image(self, image, origin):
        """
        Prepare a background image by masking sources, bad pixels and whatever
        else

        Parameters
        ----------
        image
        origin

        Returns
        -------

        """
        mask = self.get_object_mask(origin, origin + image.shape)
        return np.ma.MaskedArray(image, mask)

    def get_object_mask(self, start, stop):
        i0, j0 = start
        i1, j1 = stop
        return self.masks.all[i0:i1, j0:j1] | self.masks.bad_pixels

    def get_masks(self, start, shape):
        phot_masks = select_rect_pad(self.seg, self.masks.phot, start, shape)
        sky_mask = select_rect_pad(self.seg, self.masks.sky, start, shape)
        bad_pix = self.masks.bad_pixels
        return phot_masks | bad_pix, sky_mask | bad_pix

    def get_segments(self, start, shape):
        """
        Ensure that if the start indices implies that we are beyond the
        limits of the global segmentation array, we return only overlapping data


        Parameters
        ----------
        start
        shape

        Returns
        -------

        """
        return self.seg.select_overlap(start, shape)

    def flux_sort(self, fluxes):

        n_labels = len(self.use_labels)
        assert len(fluxes) == n_labels

        # reorder source labels for descending brightness
        brightness_order = fluxes.argsort()[::-1]  # bright to faint
        new_order = np.arange(n_labels)

        # re-label for ascending brightness
        self.seg.relabel_many(brightness_order + 1,
                              new_order + 1)

        self.ir = new_order[self.ir]
        self.rpos[brightness_order] = self.rpos[new_order]
        self.sigma_xy[brightness_order] = self.sigma_xy[new_order]
        self.use_labels = new_order + 1

        return brightness_order

    def pprint(self):
        from motley.table import Table

        # TODO: say what is being ignored
        # TODO: include uncertainties
        # TODO: print wrt current origin

        # coo = [:, ::-1]  # , dtype='O')
        return Table(self.coords,
                     col_headers=list('xy'),
                     col_head_props=dict(bg='g'),
                     row_headers=self.use_labels)

    def sdist(self):
        coo = self.coords
        return cdist(coo, coo)

    # def mask_image(self, image, mask=None):  # TODO prepare_background better
    #     """
    #     Prepare a background image by masking sources, bad pixels and whatever
    #     else
    #     """
    #     # mask sources
    #     imbg = self.seg.mask_image(image)
    #     if mask is not None:
    #         imbg.mask |= mask
    #
    #     if self.masks.bad_pixels is not None:
    #         imbg.mask |= self.masks.bad_pixels
    #
    #     return imbg

    # def prep_masks_phot(self, labels=None, edge_cutoffs=None, sky_buffer=2,
    #                     sky_width=10):
    #     """
    #     Select sub-regions of the image that will be used for photometry
    #     """
    #     if labels is None:
    #         labels = self.seg.labels
    #         indices = np.arange(self.seg.nlabels)
    #         # note `resolve_labels` only uses the labels in `use_labels`,
    #         #  which is NOT what we want since we want all sources to be masked
    #         #  not just those which are being used for tracking
    #     else:
    #         indices = np.digitize(labels, self.seg.labels) - 1
    #
    #     #
    #     m3d = self.seg.to_bool_3d()
    #     all_masked = m3d.any(0)
    #
    #     sky_regions = self.seg.to_annuli(sky_buffer, sky_width, labels)
    #     if edge_cutoffs is not None:
    #         edge_mask = make_border_mask(self.seg.data, edge_cutoffs)
    #         sky_regions &= ~edge_mask
    #
    #     phot_masks = all_masked & ~m3d[indices]
    #     self.masks.phot, self.masks.sky = phot_masks, sky_regions
    #     self.mask_all = all_masked
    #     return phot_masks, sky_regions, all_masked

    # def mask_all_objects(self):

    # def _flux_estimate(self, image, ij):
    #
    #     seg.sum(image) - seg.median(image, [0]) * seg.areas

    # def flux_estimate_annuli(self, image, sky_buffer=2, sky_width=10):
    #     sky_masks = self.seg.to_annuli(sky_buffer, sky_width)
    #     # sky_masks &= ~self.streak_mask

    def _flux_estimate_annuli(self, image, ij, edge_cutoffs=None):

        # add object segments to model
        stop = ij + image.shape
        slice2d = tuple(map(slice, ij, stop))
        slice3d = (slice(None),) + slice2d
        edge_mask = make_border_mask(image, edge_cutoffs)
        sm = (self.masks.sky[slice3d] & ~(edge_mask | self.masks.bad_pixels))

        n_sources = len(self.masks.sky)
        flx_bg, npix_bg = np.empty((2, n_sources), int)
        counts, npix = np.ma.MaskedArray(np.empty((2, n_sources), int), True)
        for i, sky in enumerate(sm):
            bg_pixels = image[sky]
            flx_bg[i] = self.bg(bg_pixels)
            npix_bg[i] = len(bg_pixels)

        # source counts
        seg = SegmentedImage(self.seg.data[slice2d])
        indices = seg.labels - 1
        counts[indices] = seg.counts(image)
        npix[indices] = seg.counts(np.ones_like(image))
        src = counts - flx_bg * npix

        return src, flx_bg, npix, npix_bg

    def update_weights(self, image, i):
        self.logger.debug('Updating weights')
        self.snr[i // self._update_weights_every] = self.get_snr_weights(image, i)
        self._source_weights = nanmean(self.snr, 0)
        self.logger.debug('Sources SNR: {}', self._source_weights)

    def get_snr_weights(self, image, i=None):
        # snr weighting scheme (for robustness)
        if image.shape != self.seg.shape:
            image = self._pad(image, self.origin)

        # ignore sources with low snr (their positions will still be tracked,
        # but not used to compute frame offsets)
        snr = self.seg.snr(image, self.use_labels)
        low_snr = snr < self.snr_cut
        if low_snr.sum() == len(snr):
            self.logger.warning('{}SNR for all sources below cut: {} < {:.1f}',
                                (f'Frame {i}: ' if i else ''), snr, self.snr_cut)
            low_snr = snr < snr.max()  # else we end up with nans

        snr[low_snr] = 0
        # self.logger.trace('SNR weights: {}', snr / snr.sum())

        if np.all(snr == 0):
            raise ValueError('Could not determine weights for centrality '
                             'measurement from image.')
            # self.logger.warning('Received zero weight vector. Setting to
            # unity')

        return snr

    def is_bad(self, coo):
        """
        improve the robustness of the algorithm by removing centroids that are
        bad.  Bad is inf / nan / outside segment.
        """

        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter COM positions that are outside of detection regions
        lbad2 = ~self.seg.inside_segment(coo, self.use_labels)

        return lbad | lbad2

    def is_outlier(self, coo, mad_thresh=5, jump_thresh=5):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track sources in low snr conditions.
        """
        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter centroid measurements that are obvious errors (large jumps)
        r = np.sqrt(np.square(self.coords - coo).sum(1))
        lj = r > jump_thresh

        if len(coo) < 5:  # scatter large for small sample sizes
            lm = np.zeros(len(coo), bool)
        else:
            lm = r - np.median(r) > mad_thresh * mad(r)

        return lj | lm | lbad

    def compute_offsets(self, xy, weights=None):
        """
        Calculate the xy offset of coordinate `coo` from centre reference

        Parameters`
        ----------
        coo
        weights

        Returns
        -------

        """
        # shift calculated as snr weighted mean of individual CoM shifts
        xy = np.ma.MaskedArray(xy, np.isnan(xy))
        # if weights is None:
        #     weights = self._source_weights
        return np.ma.average(self.coords - xy, 0, weights)

    def update_pos_point(self, coo, weights=None):
        # TODO: bayesian_update
        """"
        Incremental average of relative positions of sources (since they are
        considered static)
        """
        # see: https://math.stackexchange.com/questions/106700/incremental-averageing
        vec = coo - coo[self.ir]
        n = self.count + 1
        if weights is None:
            weights = 1. / n
        else:
            weights = (weights / weights.max() / n)[:, None]

        ix = self.use_labels - 1
        inc = (vec - self.rpos[ix]) * weights
        self.logger.debug('rpos increment:\n{:s}', inc)
        self.rpos[ix] += inc

    # def get_shift(self, image):
    #     com = self.seg.com_bg(image)
    #     l = ~self.is_outlier(com)
    #     shift = np.median(self.coords[l] - com[l], 0)
    #     return shift

    def best_for_tracking(self, image, close_cut=None, snr_cut=snr_cut,
                          saturation=None):
        """
        Find sources that are best suited for centroid tracking based on the
        following criteria:
        """
        too_bright, too_close, too_faint = [], [], []
        msg = 'Stars: {} too {} for tracking'
        if saturation:
            too_bright = self.too_bright(image, saturation)
            if len(too_bright):
                self.logger.debug(msg, str(too_bright), 'bright')
        if close_cut:
            too_close = self.too_close(close_cut)
            if len(too_close):
                self.logger.debug(msg, str(too_close), 'close')
        if snr_cut:
            too_faint = self.too_faint(image, snr_cut)
            if len(too_faint):
                self.logger.debug(msg, str(too_faint), 'faint')

        ignore = ftl.reduce(np.union1d, (too_bright, too_close, too_faint))
        ix = np.setdiff1d(np.arange(len(self.xy0)), ignore)
        if len(ix) == 0:
            self.logger.warning('No suitable sources found for tracking!')
        return ix

    # def auto_window(self):
    #     sdist_b = self.sdist[snr > self._snr_thresh]

    def too_faint(self, image, threshold=snr_cut):
        crude_snr = self.seg.snr(image)
        return np.where(crude_snr < threshold)[0]

    def too_close(self, threshold=_distance_cut):
        # Check for potential interference problems from sources that are close
        #  together
        return np.unique(np.ma.where(self.sdist() < threshold))

    def too_bright(self, data, saturation, threshold=_saturation_cut):
        # Check for saturated sources by flagging pixels withing 1% of saturation
        # level
        # TODO: make exact
        lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        # TODO check if you can improve speed here - dont have to check entire
        # array?

        satpix = np.where((lower < data) & (data < upper))
        b = np.any(np.abs(np.array(satpix)[:, None].T - self.coords) < 3, 0)
        w, = np.where(np.all(b, 1))
        return w

    def gui(self, hdu, **kws):
        return SourceTrackerGUI(self, hdu, **kws)

    @classmethod
    def from_image(cls, image, mask=None, top: int = None, detect=True, **kws):
        """Initialize the tracker from an image."""

        # Detect sources
        assert detect
        seg = super().from_image(image, detect)

        labels = None
        if top:
            flux, sigma = seg.flux(np.ma.MaskedArray(image, mask))
            labels = np.argsort(flux)[:top] + 1

        # Centroid
        xy = seg.com_bg(image, mask)[:, ::-1]

        # cls.best_for_tracking(image)
        obj = cls(xy, seg, labels, bad_pixel_mask=mask, **kws)

        # log nice table with what's been found.
        obj.logger.info('Found the following sources:\n{:s}\n', obj.pprint())

        return obj

    @classmethod
    def from_hdu(cls, hdu, sample_exposure_depth=5, **detect_kws):

        # get image and info from the hdu
        image = hdu.get_sample_image(min_depth=sample_exposure_depth)
        # precision is used to decice when our measurement of the object
        # positions are accurate enough. Look for the telescope info in the fits
        # headers so we can compute the diffraction limit, and use that as a
        # guide.
        kws = {}
        if dl := getattr(hdu, 'diffraction_limit', None):
            kws['precision'] = dl

        kws['noise_model'] = CCDNoiseModel(hdu.readout.noise, hdu.readout.preAmpGain)

        tracker = SourceTracker.from_image(image, detect=detect_kws, **kws)
        return tracker, image

        # idx = np.ogrid[0:hdu.nframes - 1:n*1j].astype(int)
        # reg = ImageRegister.from_images(hdu.calibrated[idx],
        #                                 np.tile(hdu.get_fov(), (n, 1)),
        #                                 fit_rotation=False)
        # clf = reg.get_clf(bin_seeding=True, min_bin_freq=3)
        # reg.register(clf, plot=True)
        # # seg = reg.global_seg()

        # origin = np.floor(reg.delta_xy.min(0)).astype(int)
        # # seg = GlobalSegmentation(reg.global_seg(), zero_point)

        # tracker = cls(reg.xy, reg.global_seg(), origin=origin)
        # tracker.init_memory(hdu.nframes)

    # TODO:
    # def from_fits(cls, filename, snr=3., npixels=7, edge_cutoff=3, deblend=False,
    #                flux_sort=True, dilate=1, bad_pixel_mask=None, edge_mask=None)

    @classmethod
    def from_images(cls, images, mask=None,
                    required_positional_accuracy=0.5, centre_distance_max=1,
                    f_detect_measure=0.5, f_detect_merge=0.2, post_merge_dilate=1,
                    flux_sort=True, worker_pool=itt, report=None, plot=False,
                    **detect_kws):
        """

        Parameters
        ----------
        images
        mask
        required_positional_accuracy
        centre_distance_max
        f_detect_measure
        f_detect_merge
        post_merge_dilate
        flux_sort
        worker_pool
        report
        plot
        detect_kws

        Returns
        -------

        """

        reg = ImageRegister(images)

        from obstools.image.segmentation import detect_measure

        #
        # n, *ishape = np.shape(images)
        # detect sources, measure locations
        segmentations, coms = zip(*map(
            ftl.partial(detect_measure, **detect_kws), images))

        # register constellation of sources
        centres, σ_xy, delta_xy, outliers, xy = register(
            cls.clustering, coms, centre_distance_max, f_detect_measure,
            plot)

        # combine segmentation images
        seg_glb = GlobalSegmentation.merge(segmentations, delta_xy,
                                           True,  # extend
                                           f_detect_merge,
                                           post_merge_dilate)

        # since the merge process re-labels the sources, we have to ensure the
        # order of the labels correspond to the order of the clusters.
        # Do this by taking the label of the pixel nearest measured centers
        # This is also a useful metric for the success of the clustering
        # step: Sometimes multiple sources are put in the same cluster,
        # in which case the centroid will likely be outside the labelled
        # regions in the segmented image (label 0). These sources will then
        # fortuitously be ignored below.

        # clustering algorithm may also identify more sources than in `seg_glb`
        # whether this happens will depend on the `f_detect_merge` parameter.
        # The sources that are not in the segmentation image will get the
        # label 0 in `cluster_labels`
        cxx = np.ma.getdata(centres - seg_glb.zero_point)
        indices = cxx.round().astype(int)
        cluster_labels = seg_glb.data[tuple(indices.T)]
        seg_lbl_omit = (cluster_labels == 0)
        if seg_lbl_omit.any():
            cls.logger.info('{} sources omitted from merged segmentation.',
                            seg_lbl_omit.sum())

        # return seg_glb, xy, centres, σ_xy, delta_xy, outliers, cluster_labels

        # may also happen that sources that are close together get grouped in
        # the same cluster by clustering algorithm.  This is bad, but can be
        # probably be detected by checking labels in individual segmented images
        # if n_clusters != seg_glb.nlabels:

        # return seg_glb, xy, centres, delta_xy  # , counts, counts_med
        if flux_sort:
            # Measure fluxes here. bright objects near edges of the slot
            # have lower detection probability and usually only partially
            # detected merged segmentation usually more complete for these
            # sources which leads to more accurate flux measurement and
            # therefore better change of flagging bright partial sources for
            # photon bleed
            origins = seg_glb.get_start_indices(delta_xy)

            # ======================================================================
            counts = np.ma.empty(xy.shape[:-1])
            ok = np.logical_not(seg_lbl_omit)

            counts[:, ok] = list(worker_pool.starmap(
                seg_glb.flux, ((image, ij0)
                               for ij0, image in
                               zip(origins, images))))

            counts[:, ~ok] = np.ma.masked
            counts_med = np.ma.median(counts, 0)

            # ======================================================================
            # reorder source labels for descending brightness
            order = np.ma.argsort(counts_med, endwith=False)[::-1]
            # order = seg_glb.sort(np.ma.compressed(counts_med), descend=True)
            seg_glb.relabel_many(order[ok] + 1, seg_glb.labels)

            # reorder measurements
            counts = counts[:, order]
            counts_med = counts_med[order]

        # `cluster_labels` maps cluster nr to label in image
        # reorder measurements to match order of labels in image
        cluster_labels = seg_glb.data[tuple(indices.T)]
        order = np.ma.MaskedArray(
            cluster_labels, cluster_labels == 0).argsort()
        centres = centres[order]
        xy = xy[:, order]
        σ_xy = σ_xy[order]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            im = seg_glb.display()
            im.ax.set_title('Global segmentation 0')
            display(im.figure)

        # initialize tracker
        use_source = (σ_xy < required_positional_accuracy).all(1)
        use_source = np.ma.getdata(use_source) & ~np.ma.getmask(use_source)
        if use_source.sum() < 2:
            cls.logger.warning(
                'Measured positions not accurate enough: σ_xy = {} > {:f}',
                σ_xy, required_positional_accuracy
            )
            # TODO: continue until accurate enough!!!

        # FIXME: should use bright sources / high snr here!!
        use_labels = np.where(use_source)[0] + 1

        # init
        tracker = cls(cxx, seg_glb, use_labels=use_labels,
                      bad_pixel_mask=mask)
        tracker.sigma_xy = σ_xy
        # tracker.clustering = clf
        # tracker.xy_off_min = xy_off_min
        tracker.zero_point = seg_glb.zero_point
        # tracker.current_offset = delta_xy[0]
        tracker.origin = (delta_xy[0] - delta_xy.min(0)).round().astype(int)

        if report:
            # TODO: can probs also highlight large uncertainties
            #  and bright targets!
            tracker.report_measurements(xy, centres, σ_xy, counts_med,
                                        f_detect_measure)

        return tracker, xy, centres, delta_xy, counts, counts_med
