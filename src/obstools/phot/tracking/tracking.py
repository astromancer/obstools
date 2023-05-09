"""
Methods for tracking camera movements (dither) in astronomical time-series CCD
photometry.
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

# third-party
import numpy as np
import more_itertools as mit
from tqdm import tqdm
from loguru import logger
from joblib import Parallel, delayed
from bottleneck import nanmean, nanstd
from astropy.utils import lazyproperty

# local
from recipes.io import load_memmap
from recipes.dicts import AttrReadItem
from recipes.parallel.joblib import initialized

# relative
from ...image.noise import CCDNoiseModel
from ...image.registration import ImageRegister
from ...image.segments import LabelUser, SegmentsMasksHelper
from ..proc import ContextStack
from ..logging import TqdmLogAdapter, TqdmStreamAdapter
from . import CONFIG
from .display import SourceTrackerPlots
from .dither import PointSourceDitherModel


# from recipes.parallel.synced import SyncedArray, SyncedCounter

# TODO: CameraTrackingModel / CameraOffset / CameraPositionModel 
# TODO: filter across frames for better shift determination ???
# TODO: wavelet sharpen / lucky imaging for better relative positions

# TODO
#  simulate how different centre measures performs for sources with decreasing snr
#  super resolution images
#  lucky imaging ?

# ---------------------------------------------------------------------------- #
_s0 = slice(None)

# ---------------------------------------------------------------------------- #
# Multiprocessing
sync_manager = mp.Manager()
# check precision of computed source positions
precision_reached = sync_manager.Value('i', -1)
# when was the centroid distribution spread last estimated
_last_checked = sync_manager.Value('i', -1)
_computing_centres = sync_manager.Value('b', 0)
# default lock - does nothing
memory_lock = ctx.nullcontext()


def set_lock(mem_lock, tqdm_lock):
    """
    Initialize each process with a global variable lock.
    """
    global memory_lock
    memory_lock = mem_lock
    tqdm.set_lock(tqdm_lock)


# ---------------------------------------------------------------------------- #

def wrap_int(obj):
    return [obj] if isinstance(obj, numbers.Integral) else obj


def keep_dims(*args):
    return tuple(map(wrap_int, args))


# class NullSlice():
#     """Null object pattern for getitem"""
#
#     def __getitem__(self, item):
#         return None

# ---------------------------------------------------------------------------- #

# FIXME: remove redundant code
# TODO: bayesian version


class SourceTracker(LabelUser,
                    PointSourceDitherModel):
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
    #  This allows tracking sources with low snr, but not
    # computing their centroids which will have larger scatter.
    bg = staticmethod(np.ma.median)

    # params for detecting outliers
    reference_index = 0
    # Index of source (coords) to use as internal reference, by default 0. The
    # `rpos` property will return relative positions with respect to this source.
    noise_model = None

    def __init__(self, coords, seg, labels=None,
                 features=CONFIG.centroids,
                 bad_pixel_mask=None,
                 weights=CONFIG.weights,
                 precision=CONFIG.precision,  # should be ~ diffraction limit
                 cutoffs=CONFIG.cutoffs,
                 compute=CONFIG.compute,
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
        labels : array-like or str, optional
            Only sources corresponding to these labels will be used to calculate
            frame shift. Note that center of mass measurements are for all
            sources (that are on-frame) in order to compute the best relative
            positions for all sources, but that the frame-to-frame shift will be
            computed using only the sources in `labels`, by default this
            includes all sources for the default `snr` weighting scheme.
        bad_pixel_mask : np.ndarray, optional
            Ignore these pixels in centroid computation.
        weights : bool, str, optional
            Per-source weights for computing frame offset by weighted average.
            By default all sources are weighted equally. To activate
            signal-to-noise ratio weighting, `use weights='snr'`. The frequency
            of the weight update calculation is controlled by the
            `update_weights_every` parameter.
        precision : float, optional
            Required positional accuracy of soures in units of pixels, by
            default 0.5.
        """
        # update_weights_every : int, optional
        #     How often the source weights are updated, by default every 10 frames.
        # update_centres_every : int, optional
        #     How often the relative position vector is updated. Once the required
        #     positional accuracy of sources has been acheived, the relative
        #     position vector will not be updated any further. The default is 100.

        if isinstance(seg, SegmentsMasksHelper):
            # detecting the class of the SegmentationImage allows some
            # optimization by avoiding unnecessary recompute of lazyproperties.
            # Also allows custom subclasses of SegmentsModelHelper to be
            # used
            self.seg = seg
        else:
            self.seg = SegmentsMasksHelper(seg,
                                           bad_pixels=bad_pixel_mask)

        # label include / exclude logic
        LabelUser.__init__(self, labels)
        PointSourceDitherModel.__init__(self, getattr(cutoffs, 'distance', None))

        # Measures of central tendency
        self.features = dict(features)

        # shared memory. these are just placeholders for now, they are set in
        # `init_memory`
        self.measurements = self.measure_avg = self.measure_std = \
            self.delta_xy = self.sigma_xy = None

        # reference position (in pixel coordinates) from which the shift will
        # be measured.
        # store origin and relative positions separately so we can update
        self.xy0 = coords[self.reference_index]
        self.rpos = coords - coords[self.reference_index]

        # algorithmic details
        self.precision = float(precision)
        self.snr_weighting = snrw = (weights == 'snr')
        self._source_weights = np.ones(len(self.use_labels)) if snrw else weights

        region_centres = self.seg.com(self.seg.data, self.use_labels)[:, ::-1]
        self.origin = self.compute_offset(region_centres)[::-1].round(0).astype(int)
        self.logger.debug('Origin set to: {}', self.origin)

        # shared memory for internal use
        # self.counter = SyncedCounter()

        self.noise_model = noise_model
        self.cutoffs = AttrReadItem(cutoffs)
        self._compute = AttrReadItem({k: slice(*v) for k, v, in compute.items()})
        if not snrw:
            self._compute['weights'] = slice(0, 0, 0)

        # plotting
        self.show = self.plot = SourceTrackerPlots(self)

        # masks for photometry
        # self.bad_pixel_mask = bad_pixel_mask

        #     edge_cutoffs : int or list of int, optional
        # Ignore labels that are (at most) this close to the edges of the
        # image, by default 0.
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
        nstats = len(self.features)
        nsources = len(self.use_labels)  # self.seg.nlabels
        shapes = dict(
            measurements=(n, nstats, nsources, 2),
            measure_avg=(n, nsources, 2),
            delta_xy=(n, 2),
            sigma_xy=(nsources, 2),
            flux=(n, nsources, 2),
            # _origins=(n, 2)
        )
        if self.snr_weighting or self.cutoffs.snr:
            shapes['snr'] = (math.ceil(n / self._compute.weights.step), nsources)
        if self.noise_model:
            shapes['measure_std'] = (n, nsources, 2)

        # load memory
        filenames = CONFIG.filenames
        spec = ('f', np.nan, overwrite)
        for name, shape in shapes.items():
            setattr(self, name, load_memmap(loc / filenames[name], shape, *spec))

        self._origins = load_memmap(loc / filenames['_origins'], (n, 2), 'i', 0,
                                    overwrite)

    def __call__(self, data, indices=None, mask=None):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """

        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the '
                                    '`init_memory` method.')

        if data.ndim == 2:
            data = [data]

        return self.loop(data, indices, mask)

    def __str__(self):
        nl = '\n'
        pre = f'{type(self).__name__}('  # coords=
        cx = np.array2string(self.coords[self.use_labels - 1],
                             precision=2, separator=', ')
        # ppr.mapping({'coords': cx,
        #              'source_weights'})
        return f'{pre}{cx.replace(nl, nl.ljust(len(pre) + 1))})'

    __repr__ = __str__

    # ------------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------------ #
    @lazyproperty
    def feature_weights(self):
        weights = np.fromiter(self.features.values(), float)
        return (weights / weights.sum())[:, None, None]

    def should_update_weights(self, request=None):
        if request is None:
            return ((self.snr_weighting or self.cutoffs.snr) and
                    (self._source_weights is None))
        return request

    # ------------------------------------------------------------------------ #
    def should_update_pos(self, request=None):
        if request is not None:
            return request

        return not self._check_precision_reached() and _computing_centres.value != 1

    def _check_precision_reached(self):
        self.logger.trace('Checking precision:')

        # with precision_reached:
        if (when := precision_reached.value) != -1:
            self.logger.trace('Required precision reached at frame {}.', when)
            return True

        with memory_lock:
            n = np.sum(self.measured)
            if n < self._compute.centres.start:
                self.logger.info(
                    'Delaying centre compute until {} centroid measurements'
                    ' have been made. This can be configured through the '
                    '`measure_centres_after` parameter.',
                    self._compute.centres.start)
                return False

            if n == _last_checked.value:
                self.logger.debug('No new measurements since previous precision'
                                  ' check - continuing...')
                return True

            if (self.sigma_xy.mean() < self.precision).all():
                # Reached required precision
                precision_reached.value = n
                self.logger.opt(lazy=True).info(
                    'Required positional accuracy of {0[0]:} < {0[1]:g} '
                    'achieved with {0[2]:} measurements. Relative source '
                    'positions will not be updated further.',
                    lambda: (self.sigma_xy.mean(0), self.precision, n)
                )
                self.report()
                return True

            _last_checked.value = n

        self.logger.opt(lazy=True).info(
            'Precision after {0[2]:} measurements: {0[0]:} > {0[1]:g}. ',
            lambda: (self.sigma_xy.mean(0), self.precision, n)
        )

        return False

    # ------------------------------------------------------------------------ #
    def get_coords(self, section=_s0, source=_s0):
        cast = ([np.newaxis] * (source is _s0))
        return self.coords[source] + self.delta_xy[(section, *cast)]

    def get_coords_residual(self, section=_s0, feature=_s0, source=_s0,
                            shifted=True):

        source = (self.use_labels - 1)[source]

        if (feature == 'avg'):
            data = self.measure_avg[keep_dims(section, source)]
        else:
            if feature is not _s0:
                # feature is str - index
                feature = list(self.features).index(feature)

            data = self.measurements[keep_dims(section, feature, source)]

        return np.squeeze(
            data - self.coords[np.newaxis, source]
            - (self.delta_xy[section, np.newaxis, :] if shifted else 0)
        )

        # na = [np.newaxis]
        # updel = na * ((feature is ...) + (source is ...))
        # upcoo = (*(na * (feature is ...)),
        #          (self.use_labels - 1)[source],
        #          *(na * (isinstance(source, numbers.Integral))))

        # return (self.measurements[keep_dims(section, feature, source)]
        #         - self.coords[np.newaxis, source]
        #         - (self.delta_xy[section, np.newaxis, :] if shifted else 0))

    # ------------------------------------------------------------------------ #
    # @api.synonyms({'njobs': 'n_jobs'})
    def run(self, data, indices=None, n_jobs=-1, backend='multiprocessing',
            progress_bar=True):
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
        # preliminary checks
        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the '
                                    '`init_memory` method.')

        if indices is None:
            indices, = np.where(~self.measured)

        if len(indices) == 0:
            self.logger.info('All frames have been measured. To force a rerun, '
                             'you may do >>> tracker.measurements[:] = np.nan')
            return

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        # main compute
        self._run(data, indices, n_jobs, progress_bar, backend)

        # finally, recompute the positions
        self.compute_centres_offsets(report=True)

    def _run(self, data, indices, n_jobs, progress_bar, backend):

        # init counter
        precision_reached.value = -1

        # setup compute context

        context = ContextStack()
        if n_jobs == 1:
            worker = self.loop
            context.add(ctx.nullcontext(list))
        else:
            worker = self._setup_compute(n_jobs, backend, context, progress_bar)

        # execute
        # with ctx.redirect_stdout(TqdmStreamAdapter()):
        logger.remove()
        logger.add(TqdmStreamAdapter(), colorize=True, enqueue=True)

        with context as compute:
            compute(worker(data, *args) for args in
                    self._get_workload(len(data), indices, n_jobs, progress_bar))

        # self.logger.debug('With {} backend, pickle serialization took: {:.3f}s',
        #              backend, time.time() - t_start)

    def _setup_compute(self, n_jobs, backend, context, progress_bar):
        # locks for managing output contention
        tqdm.set_lock(mp.RLock())
        memory_lock = mp.Lock()

        # NOTE: object serialization is about x100-150 times faster with
        # multiprocessing backend. ~0.1s vs 10s for loky.
        worker = delayed(self.loop)
        executor = Parallel(n_jobs, backend)  # verbose=10
        context.add(initialized(executor, set_lock,
                                (memory_lock, tqdm.get_lock())))

        # Adapt logging for progressbar
        if progress_bar:
            # These catch the print statements
            # context.add(TqdmStreamAdapter(sys.stdout))
            # context.add(TqdmStreamAdapter(sys.stderr))
            context.add(TqdmLogAdapter())

        return worker

    def _get_workload(self, total, indices, n_jobs, progress_bar):
        # divide work
        batch_size = min(self._compute.weights.step, self._compute.centres.step)
        if batch_size == 1:
            max(self._compute.weights.step, self._compute.centres.step)

        batches = mit.chunked(indices, batch_size)
        n_batches = round(len(indices) / batch_size)

        #
        self.logger.info('Work split into {} batches of {} frames each, using {} '
                         'concurrent workers.', n_batches, batch_size, n_jobs)

        # triggers for computing coordinate centres and weights as needed
        triggers = np.array(
            [(self._compute.centres.start, self._compute.centres.step),
             (self._compute.weights.start, self._compute.weights.step)]
        ) // batch_size
        update_centres, update_weights = (
            itt.chain(
                # burn in
                itt.repeat(False, burn_in - 1),
                # compute every nth batch
                itt.cycle(itt.chain([val], itt.repeat(False, every - 1)))
            )
            for (burn_in, every), val in zip(triggers, (None, True))
        )

        # workload iterable with progressbar if required
        return tqdm(zip(batches, update_weights, update_centres),
                    initial=(total - len(indices)) // batch_size,
                    total=total // batch_size, unit_scale=batch_size,
                    disable=not progress_bar, **CONFIG.progress)

    def loop(self, data, indices, update_weights=None, update_centres=None,
             report=True):

        indices = np.atleast_1d(indices)

        # # weights
        # self.logger.info(f'{update_weights=}; '
        #                  f'{self.should_update_weights()=}')

        if self.should_update_weights(update_weights):
            self.update_weights(data[(i := indices[0])], i)

        # measure
        for i in indices:
            self.delta_xy[i] = off = self.measure(data, i)

            # with memory_lock:
            #     # next(self.counter)
            #     self.delta_xy[i] = off

        # if indices.size < 10:
        self.logger.trace('OFFSETS frame {}: {}', i, off)
        # xym = self.measure_avg[indices]

        # update relative positions from CoM measures
        # check here if measurements of cluster centres are good enough!

        # self.logger.info(f'{update_centres=}; '
        #                  f'{self.should_update_pos()=}')

        if self.should_update_pos(update_centres):
            self.logger.info('Updating source positions & offsets.')
            if _computing_centres.value:
                self.logger.info('Centres currently being computed by another '
                                 'process. Skipping.')
            else:
                _computing_centres.value = 1
                self.compute_centres_offsets(report)
                _computing_centres.value = 0

            self.logger.opt(lazy=True).trace(
                'Average pos stddev: {}', lambda: self.sigma_xy.mean(0))
        else:
            # calculate median frame shift using current best relative positions
            # optionally using snr weighting scheme
            # measurements wrt image, offsets wrt coords
            with memory_lock:
                self.sigma_xy = nanstd(self.measure_avg - self.delta_xy[:, None], 0)
                self.logger.opt(lazy=True).trace(
                    'Average pos stddev: {}', lambda: self.sigma_xy.mean(0))

        # self.logger.info('OFFSET: {}', off)
        # finally return the new coordinates
        return self.coords + self.delta_xy[indices, None]

    def update_weights(self, image, i):
        self.logger.trace('Updating weights')
        self.snr[i // self._compute.weights.step] = self.get_snr_weights(image, i)
        self._source_weights = nanmean(self.snr, 0)
        self.logger.trace('Sources SNR: {}', self._source_weights)

    def get_snr_weights(self, image, i=None):
        # snr weighting scheme (for robustness)
        if image.shape != self.seg.shape:
            raise ValueError(f'NOPE {image.shape}, {self.seg.shape} ')
            # image = self._pad(image, self.origin)

        # ignore sources with low snr (their positions will still be tracked,
        # but not used to compute frame offsets)
        snr = self.seg.snr(image, self.use_labels)
        low_snr = snr < self.cutoffs.snr
        if low_snr.sum() == len(snr):
            self.logger.warning('{}SNR for all sources below cut: {} < {:.1f}',
                                (f'Frame {i}: ' if i else ''), snr, self.cutoffs.snr)
            low_snr = snr < snr.max()  # else we end up with nans

        snr[low_snr] = 0
        # self.logger.trace('SNR weights: {}', snr / snr.sum())

        if np.all(snr == 0):
            raise ValueError('Could not determine weights for centrality '
                             'measurement from image.')
            # self.logger.warning('Received zero weight vector. Setting to
            # unity')

        return snr

    @property
    def measured(self):
        return ~np.isnan(self.measure_avg).all((1, 2))

    def measure(self, data, i, mask=None):

        if mask is None:
            mask = self.masks.bad_pixels  # NOTE: may be None
        elif self.masks.bad_pixels is not None:
            mask |= self.masks.bad_pixels

        #
        image = np.ma.MaskedArray(data[i], mask)
        xym = self._measure(image, i, self.origin)
        dxy = self.compute_offset(xym, self._source_weights)

        if np.ma.is_masked(dxy) or ~np.isfinite(dxy).all():
            raise ValueError('Masked or nan in xy offsets.')
        #
        # if np.any(np.abs(dxy) > 20):
        #     raise ValueError('')

        self.logger.trace('OFFSET: {}', dxy)

        # Update origin
        if (np.ma.abs(dxy) > 1).any():
            new = -self.update_origin(dxy, i, image)
            self.logger.trace('UPDATED OFFSET {}: {} {}', i, dxy, new)
            return new

        # same origin
        self._origins[i] = self.origin
        return dxy

    def compute_offset(self, xy, weights=None):
        return super().compute_offset(xy, self.coords[self.use_labels - 1], weights)

    def update_origin(self, dxy, i, image):
        self.logger.opt(lazy=True).trace(
            'Frame {[0]}: \norigin = {[1]}\nδxy = {[2]}',
            lambda: i, self.origin, np.array2string(dxy, precision=2))

        # NOTE: `origin` in image yx coords
        self.origin = np.array(np.round(dxy[::-1])).astype(int)
        self._origins[i] = self.origin
        self.logger.trace('Updated origin = {}.', self.origin)

        # re-measure
        self.logger.trace('re-measuring centroids.')
        xym = self._measure(image, i, self.origin)
        return self.compute_offset(xym, self._source_weights)

    def _measure(self, image, index, origin):

        self.logger.trace('Measuring source positions for frame relative to '
                          'array index {}.', origin)
        #
        seg, xy = self.measure_positions(image, origin)

        # TODO: use grid and add offset to grid when computing CoM.  Will be
        self.measurements[index] = xy
        self.measure_avg[index] = xym = (xy * self.feature_weights).sum(0)

        if self.noise_model:
            self.measure_std[index] = seg.com_std(
                xy[0], image, self.noise_model(image), self.use_labels
            )

        # Flux with uncertainty
        self.flux[index] = seg.flux(image, self.use_labels)

        return xym

    def measure_positions(self, image, mask=None, origin=None):
        """
        Calculate measure of central tendency (centre-of-mass) for the objects
        in the segmentation image

        Parameters
        ----------
        image
        mask
        origin:
            Array index of segmented image w.r.t. image data.

        Returns
        -------

        """
        # more robust that individual CoM track.
        # eg: cosmic rays:
        # '/media/Oceanus/UCT/Observing/data/Feb_2015/MASTER_J0614-2725/20150225.001.fits' : 23

        # NOTE origin in array coordinates yx!!
        if origin is None:
            origin = self.origin

        # get segmented image for current origin
        seg = self.get_segments(image, origin)

        image = np.ma.MaskedArray(image, mask)
        yx = self._measure_positions(image, seg)
        yx = self._check_measurement(yx)

        return seg, yx[..., ::-1]

    def _check_measurement(self, yx):
        # check if any measurements out of frame
        ec = np.array(self.cutoffs.edge)
        yx[((yx < ec) | (yx > image.shape - ec)).any(-1)] = np.nan

        return yx

    def _measure_positions(self, data, seg):
        """
        Measure positions for all sources that are on-frame, ignoring those that
        are partially off-frame.

        Parameters
        ----------
        data : np.ndarray
            Image array for which positions will be measured
        mask : np.ndarray
            Masked pixels to ignore.  Should be broadcastable `data.shape`
        origin : np.ndarray
            Array index of segmented image w.r.t. image data.


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

        # NOTE this is ~2x faster than padding image and mask. can probably
        #  be made even faster by using __slots__

        # work only with sources that are contained in this image region
        # remove detections that are partially off-frame
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
        labels = self.use_labels
        yx = np.full((len(self.features), len(labels), 2), np.nan)
        for i, stat in enumerate(self.features):
            # labels - 1
            yx[i] = c = getattr(seg, stat)(data, labels, njobs=1)[:, :2]
            self.logger.trace('stat: {}\n{}', stat, c)

        # TODO: check if using grid + offset then com is faster
        # TODO: can do FLUX estimate here!!!

        # self.check_measurement(xy)
        # self.coords - tracker.zero_point < set_lockt[0]

        # check values inside segment labels
        # good = ~self.is_bad(com)
        # self.coms[i, good] = com[good]

        return yx

    def get_segments(self, origin=None, shape=None):
        """
        Get segmentation image overlapping with data given origin.


        Parameters
        ----------
        start
        shape

        Returns
        -------

        """
        # NOTE origin in array coordinates yx!!
        if origin is None:
            origin = self.origin

        if shape is None:
            shpe = self.seg.shape

        if (np.abs(origin) > shape[-2:]).any():
            raise ValueError(f'Image of shape {shape} does not overlap with '
                             f'segmented image with origin at array index: '
                             f'{origin}.')

        # select the current active region from global segmentation
        return self.seg.get_overlap(origin, shape)
        # NOTE: the segmented image returned here is in the image array
        # coordinates, so image stats below are in image array coords

    def fit(self, report=False):
        """
        Measure the relative positions of sources, as well as the xy shifts
        (dither) of each image frame based on the average centroid measurements
        in `self.measured_avg`.

        Parameters
        ----------
        report: bool
            Whether to print a table with the current coordinates and their 
            measurement uncertainty.
        """

        self.logger.debug('Estimating source positions from average of {} '
                          'features.', len(self.features))

        xym, centres, σ_pos, delta_xy, out = super().fit(
            self.measure_avg, self._distance_cut, report=False
        )

        # shift the coordinate system relative to the new reference point
        xy0 = centres[self.reference_index]
        δ = self.xy0 - xy0

        with memory_lock:
            self.delta_xy[:] = delta_xy - δ
            self.rpos = centres - xy0
            self.sigma_xy[:] = σ_pos

        self.logger.debug('Updated source positions: δ = {}.', δ)

        if report:
            self.report(xym, centres, σ_pos)

    def report(self, xy=None, centres=None, σ_xy=None, counts=None,
               detect_frac_min=None, count_thresh=None):

        if xy is None:
            xy = self.measure_avg[self.measured]

        if centres is None:
            centres = self.coords[self.use_labels - 1]

        if σ_xy is None:
            σ_xy = self.sigma_xy

        return super().report(xy, centres, σ_xy, counts, detect_frac_min,
                              count_thresh, self.logger)

    # alias
    pprint = report

    # def prepare_image(self, image, origin):
    #     """
    #     Prepare a background image by masking sources, bad pixels and whatever
    #     else

    #     Parameters
    #     ----------
    #     image
    #     origin

    #     Returns
    #     -------

    #     """

    #     mask = self.get_object_mask(origin, origin + image.shape)
    #     return np.ma.MaskedArray(image, mask)

    # def get_object_mask(self, start, stop):
    #     i0, j0 = start
    #     i1, j1 = stop
    #     return self.masks.all[i0:i1, j0:j1] | self.masks.bad_pixels

    # def get_masks(self, start, shape):
    #     phot_masks = self.seg.get_overlap(self.masks.phot, start, shape)
    #     sky_mask = self.seg.get_overlap(self.masks.sky, start, shape)
    #     bad_pix = self.masks.bad_pixels
    #     return phot_masks | bad_pix, sky_mask | bad_pix

    # def sdist(self):
    #     coo = self.coords
    #     return cdist(coo, coo)

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

    # def _flux_estimate(self, image, ij):
    #
    #     seg.sum(image) - seg.median(image, [0]) * seg.areas

    # def flux_estimate_annuli(self, image, sky_buffer=2, sky_width=10):
    #     sky_masks = self.seg.to_annuli(sky_buffer, sky_width)
    #     # sky_masks &= ~self.streak_mask

    # def _flux_estimate_annuli(self, seg, image):

    #     # add object segments to model
    #     # stop = ij + image.shape
    #     # slice2d = tuple(map(slice, ij, stop))
    #     # slice3d = (slice(None),) + slice2d
    #     # edge_mask = make_border_mask(image, edge_cutoffs)
    #     # sm = (self.masks.sky[slice3d] & ~(edge_mask | self.masks.bad_pixels))

    #     n_sources = len(self.masks.sky)
    #     flx_bg, npix_bg = np.empty((2, n_sources), int)
    #     counts, npix = np.ma.MaskedArray(np.empty((2, n_sources), int), True)
    #     for i, sky in enumerate(sm):
    #         bg_pixels = image[sky]
    #         flx_bg[i] = self.bg(bg_pixels)
    #         npix_bg[i] = len(bg_pixels)

    #     # source counts
    #     seg = SegmentedImage(self.seg.data[slice2d])
    #     indices = seg.labels - 1
    #     counts[indices] = seg.counts(image)
    #     npix[indices] = seg.counts(np.ones_like(image))
    #     src = counts - flx_bg * npix

    #     return src, flx_bg, npix, npix_bg

    # def is_bad(self, coo):
    #     """
    #     improve the robustness of the algorithm by removing centroids that are
    #     bad.  Bad is inf / nan / outside segment.
    #     """

    #     # flag inf / nan values
    #     lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

    #     # filter COM positions that are outside of detection regions
    #     lbad2 = ~self.seg.inside_segment(coo, self.use_labels)

    #     return lbad | lbad2

    # def is_outlier(self, coo, mad_thresh=5, jump_thresh=5):
    #     """
    #     improve the robustness of the algorithm by removing centroids that are
    #     outliers.  Here an outlier is any point further that 5 median absolute
    #     deviations away. This helps track sources in low snr conditions.
    #     """
    #     # flag inf / nan values
    #     lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

    #     # filter centroid measurements that are obvious errors (large jumps)
    #     r = np.sqrt(np.square(self.coords - coo).sum(1))
    #     lj = r > jump_thresh

    #     if len(coo) < 5:  # scatter large for small sample sizes``
    #         lm = np.zeros(len(coo), bool)
    #     else:
    #         lm = r - np.median(r) > mad_thresh * mad(r)

    #     return lj | lm | lbad

    # def update_pos_point(self, coo, weights=None):
    #     # TODO: bayesian_update
    #     """"
    #     Incremental average of relative positions of sources (since they are
    #     considered static)
    #     """
    #     # see: https://math.stackexchange.com/questions/106700/incremental-averageing
    #     vec = coo - coo[self.reference_index]
    #     n = self.count + 1
    #     if weights is None:
    #         weights = 1. / n
    #     else:
    #         weights = (weights / weights.max() / n)[:, None]

    #     ix = self.use_labels - 1
    #     inc = (vec - self.rpos[ix]) * weights
    #     self.logger.debug('rpos increment:\n{:s}', inc)
    #     self.rpos[ix] += inc

    # def best_for_tracking(self, image, close_cut=None, snr_cut=snr_cut,
    #                       saturation=None):
    #     """
    #     Find sources that are best suited for centroid tracking based on the
    #     following criteria:
    #     """
    #     too_bright, too_close, too_faint = [], [], []
    #     msg = 'Stars: {} too {} for tracking'
    #     if saturation:
    #         too_bright = self.too_bright(image, saturation)
    #         if len(too_bright):
    #             self.logger.debug(msg, str(too_bright), 'bright')
    #     if close_cut:
    #         too_close = self.too_close(close_cut)
    #         if len(too_close):
    #             self.logger.debug(msg, str(too_close), 'close')
    #     if snr_cut:
    #         too_faint = self.too_faint(image, snr_cut)
    #         if len(too_faint):
    #             self.logger.debug(msg, str(too_faint), 'faint')

    #     ignore = ftl.reduce(np.union1d, (too_bright, too_close, too_faint))
    #     ix = np.setdiff1d(np.arange(len(self.xy0)), ignore)
    #     if len(ix) == 0:
    #         self.logger.warning('No suitable sources found for tracking!')
    #     return ix

    # # def auto_window(self):
    # #     sdist_b = self.sdist[snr > self._snr_thresh]

    # def too_faint(self, image, threshold=snr_cut):
    #     crude_snr = self.seg.snr(image)
    #     return np.where(crude_snr < threshold)[0]

    # def too_close(self, threshold=_distance_cut):
    #     # Check for potential interference problems from sources that are close
    #     #  together
    #     return np.unique(np.ma.where(self.sdist() < threshold))

    # def too_bright(self, data, saturation, threshold=_saturation_cut):
    #     # Check for saturated sources by flagging pixels withing 1% of saturation
    #     # level
    #     # TODO: make exact
    #     lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
    #     # TODO check if you can improve speed here - dont have to check entire
    #     # array?

    #     satpix = np.where((lower < data) & (data < upper))
    #     b = np.any(np.abs(np.array(satpix)[:, None].T - self.coords) < 3, 0)
    #     w, = np.where(np.all(b, 1))
    #     return w

    def gui(self, hdu, **kws):
        from obstools.phot.tracking import SourceTrackerGUI

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

        from obstools.image.segments import detect_measure

        #
        # n, *ishape = np.shape(images)
        # detect sources, measure positions
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
            tracker.report(xy, centres, σ_xy, counts_med,
                           f_detect_measure)

        return tracker, xy, centres, delta_xy, counts, counts_med
