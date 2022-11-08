"""
Methods for tracking camera movements in astronomical time-series CCD photometry
"""

# std
import tempfile
import functools as ftl
import itertools as itt
import multiprocessing as mp
from pathlib import Path
from pydoc import describe

# third-party
import numpy as np
import more_itertools as mit
from loguru import logger
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist
from matplotlib.transforms import AffineDeltaTransform
from astropy.utils import lazyproperty
from astropy.stats import median_absolute_deviation as mad
from obstools.phot.gui import SourceTrackerGUI

# local
from recipes.pprint import describe
from recipes.dicts import AttrReadItem
from recipes.logging import LoggingMixin
from recipes.parallel.synced import SyncedArray, SyncedCounter

# relative
from ..io import load_memmap
from ..image.detect import SourceDetectionMixin, make_border_mask
from ..image.registration import (ImageRegister, compute_centres_offsets,
                                  group_features, report_measurements)
from ..image.segmentation import (LabelGroupsMixin, SegmentedImage,
                                  SegmentsModelHelper, merge_segmentations,
                                  select_overlap)


# TODO: CameraTrackingModel / CameraOffset / CameraPositionModel
# ConstellationModel
# TODO: filter across frames for better shift determination ???
# TODO: wavelet sharpen / lucky imaging for better relative positions

# TODO
#  simulate how different centre measures performs for sources with decreasing snr
#  super resolution images
#  lucky imaging ?


TABLE_STYLE = dict(txt='bold', bg='g')
FILENAMES = dict(positions='coord-pos.dat',
                 shifts='coord-delta.dat')
LABEL_STYLE = dict(offset=(6, 6),
                   color='w',
                   size=10,
                   fontweight='bold')


    

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
        super().__init__(data, grid, domains)

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
    def merge(cls, segmentations, xy_offsets, extend=True, f_accept=0.5,
              post_dilate=1):
        # zero point correspond to minimum offset from initializer and serves as
        # coordinate reference
        # make int type to avoid `Cannot cast with casting rule 'same_kind'
        # downstream
        return cls(merge_segmentations(segmentations, xy_offsets,
                                       extend, f_accept, post_dilate),
                   zero_point=np.floor(xy_offsets.min(0)).astype(int))

    def __init__(self, data, zero_point):
        super().__init__(data)
        self.zero_point = zero_point

    def get_start_indices(self, xy_offsets):
        if np.ma.is_masked(xy_offsets):
            raise ValueError('Cannot get start indices for masked offsets')

        return np.abs((xy_offsets + self.zero_point).round().astype(int))

    def select_overlap(self, start, shape, type_=SegmentedImage):
        return super().select_overlap(start, shape, type_)

    def for_offset(self, xy_offsets, shape, type_=SegmentedImage):
        if np.ma.is_masked(xy_offsets):
            raise ValueError('Cannot get segmented image for masked offsets')

        return self.select_overlap(
            self.get_start_indices(xy_offsets), shape, type_)

    def flux(self, image, origin, labels=None, labels_bg=(0,), bg_stat='median'):
        sub = self.select_overlap(origin, image.shape)
        return sub.flux(image, labels, labels_bg, bg_stat)

    def sort(self, measure, descend=False):
        if (n := len(measure)) != self.nlabels:
            raise ValueError('Cannot reorder labels for measure with '
                             f'incompatable size {n}. {describe(self)} has '
                             f'{self.nlabels} labels')

        o = slice(None, None, -1) if descend else ...
        order = np.ma.argsort(measure, endwith=False)[o]
        self.relabel_many(order + 1, self.labels)
        return order


class SourceTracker(LabelUser,
                    LabelGroupsMixin,
                    SourceDetectionMixin,
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
    coordinate positions of the sources with respect to the brightest star are
    updated upon each iteration.


    """
    # FIXME: remove redundant code
    # TODO: bayesian version
    # TODO: make multiprocessing safe

    centrality = ('com',
                  'com_bg', 
                  'geometric_median',
                  'peak',
                  'upsample_max')

    bg = staticmethod(np.ma.median)

    # TODO: manage as properties ?
    snr_weighting = False
    snr_cut = 0
    _distance_cut = 10
    _saturation_cut = 95  # %

    # this allows tracking sources with low snr, but not computing their
    # centroids which will have larger scatter

    # outlier removal
    delta_r = 1  # pixels  # TODO: pass to initializer!!!
    # maximal distance for individual location measurements from centre of
    # distribution

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
    def from_hdu(cls, hdu, n=10, **kws):

        idx = np.ogrid[0:hdu.nframes - 1:n*1j].astype(int)
        reg = ImageRegister.from_images(hdu.calibrated[idx],
                                        np.tile(hdu.get_fov(), (n, 1)),
                                        fit_rotation=False)
        clf = reg.get_clf(bin_seeding=True, min_bin_freq=3)
        reg.register(clf, plot=True)
        # seg = reg.global_seg()

        origin = np.floor(reg.xy_offsets.min(0)).astype(int)
        # seg = GlobalSegmentation(reg.global_seg(), zero_point)

        tracker = cls(reg.xy, reg.global_seg(), origin=origin)
        tracker.init_memory(hdu.nframes)
        return tracker, reg

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
        centres, σ_xy, xy_offsets, outliers, xy = register(
            cls.clustering, coms, centre_distance_max, f_detect_measure,
            plot)

        # combine segmentation images
        seg_glb = GlobalSegmentation.merge(segmentations, xy_offsets,
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

        # return seg_glb, xy, centres, σ_xy, xy_offsets, outliers, cluster_labels

        # may also happen that sources that are close together get grouped in
        # the same cluster by clustering algorithm.  This is bad, but can be
        # probably be detected by checking labels in individual segmented images
        # if n_clusters != seg_glb.nlabels:

        # return seg_glb, xy, centres, xy_offsets  # , counts, counts_med
        if flux_sort:
            # Measure fluxes here. bright objects near edges of the slot
            # have lower detection probability and usually only partially
            # detected merged segmentation usually more complete for these
            # sources which leads to more accurate flux measurement and
            # therefore better change of flagging bright partial sources for
            # photon bleed
            origins = seg_glb.get_start_indices(xy_offsets)

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
            # reorder star labels for descending brightness
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
        use_star = (σ_xy < required_positional_accuracy).all(1)
        use_star = np.ma.getdata(use_star) & ~np.ma.getmask(use_star)
        if use_star.sum() < 2:
            cls.logger.warning(
                'Measured positions not accurate enough: σ_xy = {} > {:f}',
                σ_xy, required_positional_accuracy
            )
            # TODO: continue until accurate enough!!!

        # FIXME: should use bright sources / high snr here!!
        use_labels = np.where(use_star)[0] + 1

        # init
        tracker = cls(cxx, seg_glb, use_labels=use_labels,
                      bad_pixel_mask=mask)
        tracker.sigma_pos = σ_xy
        # tracker.clustering = clf
        # tracker.xy_off_min = xy_off_min
        tracker.zero_point = seg_glb.zero_point
        # tracker.current_offset = xy_offsets[0]
        tracker.origin = (xy_offsets[0] - xy_offsets.min(0)).round().astype(int)

        if report:
            # TODO: can probs also highlight large uncertainties
            #  and bright targets!
            tracker.report_measurements(xy, centres, σ_xy, counts_med,
                                        f_detect_measure)

        return tracker, xy, centres, xy_offsets, counts, counts_med

    def __init__(self, coords, seg,
                 use_labels=None, label_groups=None,
                 bad_pixel_mask=None, edge_cutoffs=0,
                 weights=None, update_weights_every=10,
                 update_centres_every=100,
                 precision=0.25,  # should be diffraction limit
                 reference_index=0):
        """
        Class for tracking camera movement by measuring location of sources 
        in CCD images (by eg. centroid or projected geometric median)

         positions of all sources with good signal-to-noise.


        Parameters
        ----------
        coords : array-like
            Reference coordinates of the sources, xy
        seg :  SegmentedImage or array-like
            The segmentation image. Centroids for each star are computed within
            the corresponding labelled region.
        use_labels : array-like, optional
            Only sources corresponding to these labels will be used to
            calculate frame shift. Note that center of mass measurements are
            for all sources (that are on-frame) in order to compute the best 
            relative positions for all sources, but that the frame-to-frame shift
            will be computed using only the sources in `use_labels`, by default
            this includes all sources.
        label_groups : [type], optional
            [description], by default None
        bad_pixel_mask : np.ndarray, optional
            Ignore these pixels in centroid computation.
        edge_cutoffs : int or list of int, optional
            Ignore labels that are (at most) this close to the edges of the 
            image, by default 0.
        background_estimator : [type], optional
            [description], by default np.ma.median.
        weights : [type], optional
            Per-source weighting for weighted average offset computation, by 
            default all sources are weighted equally.
        update_centres_every : int, optional
            How often the relative position vector is updated. Once the required
            positional accuracy of sources has been acheived, the relative
            position vector will not be updated any further. The default is 100.
        update_weights_every : int, optional
            How often the source weights are updated, by default every 10 frames
        precision : float, optional
            Required positional accuracy of soures in units of pixels, by 
            default 0.25.
        reference_index : int, optional
            [description], by default 0.
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
        self.counter = SyncedCounter()
        self.measurements = None
        self.xy_offsets = None

        # pixel coordinates of reference position from which the shift will
        # be measured.

        n_sources = len(coords)
        self.ir = int(reference_index)
        self.xy0 = coords[self.ir]
        self.rpos = coords - coords[self.ir]
        self.sigma_pos = np.full((n_sources, 2), np.inf)
        self.precision = float(precision)
        # store zero point and relative positions separately so we can update

        # algorithmic details
        # self._measure_source_locations = self.seg.com_bg
        self._update_centres_every = int(update_centres_every)
        self._update_weights_every = int(update_weights_every)

        # background_estimator for location measurement
        # self.bg = background_estimator
        self._weights = weights
        # self.current_offset = np.zeros(2)
        self.origin = np.array((0, 0))
        # self.xy_off_min = None

        self.edge_cutoffs = edge_cutoffs

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

    def init_memory(self, n, loc=None, clobber=False):
        """

        Parameters
        ----------
        n:
            number of frames
        loc
        clobber

        Returns
        -------

        """

        if loc is None:
            loc = tempfile.mkdtemp()
        #
        loc = Path(loc)
        nstats = len(self.centrality)

        common = ('f', np.nan, clobber)
        self.measurements = load_memmap(loc / FILENAMES['positions'],
                                        (n, nstats, self.seg.nlabels, 2),
                                        *common)
        self.xy_offsets = load_memmap(loc / FILENAMES['shifts'],
                                      (n, 2),
                                      *common)

    def __call__(self, image, index, mask=None):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """

        count = next(self.counter)
        self.logger.opt(lazy=True).debug(f'{count = }, {index = }')

        # TODO: use grid and add offset to grid when computing CoM.  Will be
        # more efficient
        xy = self.measure_source_locations(image, mask)

        self.measurements[index] = xy

        # weights
        if self.should_update_weights(count):
            self._weights = self.get_snr_weights(image)
            # TODO: maybe aggregate weights and use mean ??

            # print(count, 'weights', self._weights)

        # # update relative positions from CoM measures
        # check here if measurements of cluster centres are good enough!
        if self.should_update_pos(count):
            self.logger.info('Measuring positions, offsets at count {}.', count)
            self.compute_centres_offsets(self.measurements[:(index + 1)])
            off = self.xy_offsets[index]
        else:
            # calculate median frame shift using current best relative positions
            # optionally using snr weighting scheme
            # measurements wrt image, offsets wrt coords
            off = self.compute_offset(xy, self._weights)
            self.xy_offsets[index] = off

        # Update origin
        self.origin = off[::-1].astype(int)  # NOTE: `origin` in image yx coords
        self.logger.debug('Updated origin = {}. offset = {}', self.origin, off)

        # if np.isnan(off).any():
        #     raise Exception('')
        #
        # if np.ma.is_masked(off):
        #     raise ValueError('')
        #
        # if np.any(np.abs(off) > 20):
        #     raise ValueError('')

        # if np.any(np.abs(self.current_offset - off) > 1):
        #     self.origin = off.round().astype(int)
        # print('OFFSET', off)

        # finally return the new coordinates
        return self.coords + off

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

    def should_update_weights(self, count):
        return ((self.snr_weighting or self.snr_cut) and
                ((self._weights is None) or
                 ((count + 1) % self._update_weights_every == 0)))

    def should_update_pos(self, count):
        return ((count > 0) and
                ((count + 1) % self._update_centres_every) == 0 and
                (self.sigma_pos > self.precision).any())

    def get_coords(self, i=None):
        if i is None:
            return self.coords[None] - self.xy_offsets[:, None]
        return self.coords - self.xy_offsets[i]

    def get_coord(self, i):
        return self.coords - self.xy_offsets[i]

    def get_coords_residual(self):
        return self.measurements - self.coords - self.xy_offsets[:, None]

    def run(self, data, indices=None, worker_pool=None, n_jobs=-1,
            start_offset=None):
        # run `n_jobs` tracking loops concurrently

        if self.measurements is None:
            raise FileNotFoundError('Initialize memory first by calling the'
                                    '`init_memory` method.')

        if indices is None:
            indices = range(len(self.measurements))

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        self.counter = SyncedCounter()
        self.sigma_pos = SyncedArray(self.sigma_pos)

        if worker_pool is None:
            # create pool with n_jobs
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:
                parallel(delayed(self.track_loop)(i, data)
                         for i in mit.divide(n_jobs, indices))

        # TODO: initializer that sets the starting indices for each pickled
        #  clone
        else:
            worker_pool.map(
                self.track_loop, ((i, data)
                                  for i in mit.divide(n_jobs, indices)))

    def track_loop(self, indices, data):
        for i in indices:
            self.track(data[i], i)

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

    def measure_source_locations(self, image, mask=None):  # , origin=(0, 0)
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

        yx = self._measure_source_locations(image, mask)

        # check measurements
        # FIXME: still need this ?? - checking overlap at every frame above
        bad = np.any((yx < self.edge_cutoffs) |
                     (yx > np.subtract(image.shape, self.edge_cutoffs)),
                     -1)
        yx[bad] = np.nan

        return yx[..., ::-1]

    def _measure_source_locations(self, image, mask):
        """
        Measure locations for all sources that are on-frame, ignoring those that
        are partially off-frame.

        Parameters
        ----------
        image : np.ndarray
            Image array for which locations will be measured
        mask : np.ndarray
            Masked pixels to ignore.  Should be the same shape as `image`
        origin : np.ndarray
            Lower left corner of the current image with respect to the global
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

        if np.any((np.abs(self.origin) > image.shape)):
            raise ValueError(f'Image of shape {image.shape} does not overlap'
                             ' with segmentation image: Tracker origin (lower '
                             f'left corner) is at {self.origin}.')

        self.logger.debug('Measuring source locations for frame with origin at '
                          '{}.', self.origin)

        # select the current active region from global segmentation
        seg = self.seg.select_overlap(self.origin, image.shape)

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

        # compute background subtracted CoM
        image = np.ma.MaskedArray(image, mask)
        labels = self.use_labels
        yx = np.full((len(self.centrality), self.seg.nlabels, 2), np.nan)
        for i, stat in enumerate(self.centrality):
            yx[i, labels - 1] = c = getattr(seg, stat)(image, labels)[:, :2]
            logger.debug('stat: {}\n{}', stat, c)
    

        # NOTE this is ~2x faster than padding image and mask. can probably
        #  be made even faster by using __slots__

        # TODO: check if using grid + offset then com is faster
        # TODO: can do FLUX estimate here!!!

        # self.check_measurement(xy)
        # self.coords - tracker.zero_point < start[0]

        # check values inside segment labels
        # good = ~self.is_bad(com)
        # self.coms[i, good] = com[good]

        return yx + self.origin

    # def check_measurement(self, xy):

    def compute_centres_offsets(self, xy=None):
        """
        Measure the relative positions of sources, as well as the xy offset
        (dither) of each image frame based on the set of centroid  measurements
        in `xy`.

        Parameters
        ----------
        xy

        Returns
        -------

        """

        # TODO: many different algorithms can be used here
        # if i is not None:
        #     i = len(self.measurements)

        if xy is None:
            xy = self.measurements

        i = len(xy)
        xym, centres, σ_pos, xy_offsets, out = \
            compute_centres_offsets(xy.mean(1), self.delta_r)
        n_points = xy_offsets.mask.any(-1).sum()

        # origin = np.floor(xy_offsets.min(0))

        # self.zero_point
        # note using the first non-masked point will avoid this compute
        #  FIXME: account for difference between zero point and min off here...
        # smallest offset as origin makes sense from global image perspective
        # changing the zero_point means you have to update all offsets

        # make the offsets relative to the global segmentation
        xy0 = centres[self.ir]
        δ = self.xy0 - xy0
        self.xy_offsets[:i] = xy_offsets + δ
        self.rpos = centres - xy0
        self.sigma_pos[:] = σ_pos

        if (σ_pos < self.precision).all():
            self.logger.info(
                'Required positional accuracy of {:g} achieved with {} '
                'measurements. Relative source positions will not be updated '
                'further.', self.precision, n_points
            )

        self.report_measurements(xym, centres, σ_pos)

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
            σ_xy = self.sigma_pos

        return report_measurements(xy, centres, σ_xy, counts, detect_frac_min,
                                   count_thresh, self.logger)

    def plot_position_measures(self, **kws):

        self.measurements

        # from obstools.phot.diagnostics import plot_position_measures

        # return plot_position_measures(self.measurements, self.coords,
        #                               -self.xy_offsets, **kws)

    # def _pad(self, image, origin, zero=0.):
    #     # make a measurement of some property of image using the segments
    #     # from the tracker. In general, image will be smaller than global
    #     # segmentation, therefore first pad the image with zeros
    #
    #     stop_indices = origin + image.shape
    #
    #     istart = -np.min([origin, (0, 0)], 0)
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

        # reorder star labels for descending brightness
        brightness_order = fluxes.argsort()[::-1]  # bright to faint
        new_order = np.arange(n_labels)

        # re-label for ascending brightness
        self.seg.relabel_many(brightness_order + 1,
                              new_order + 1)

        self.ir = new_order[self.ir]
        self.rpos[brightness_order] = self.rpos[new_order]
        self.sigma_pos[brightness_order] = self.sigma_pos[new_order]
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

    # def get_weights(self, com):
    #     #
    #     weights = np.ones(len(self.use_labels))
    #     # flag outliers
    #     bad = self.is_bad(com)
    #     self.logger.debug('bad: {:s}', np.where(bad)[0])
    #     weights[bad] = 0
    #     return weights

    def get_snr_weights(self, image, i=None):
        # snr weighting scheme (for robustness)

        im = self._pad(image, self.origin)
        weights = snr = self.seg.snr(im, self.use_labels)
        # self.logger.debug('snr: {:s}', snr)

        # ignore sources with low snr (their positions will still be tracked,
        # but not used to compute new positions)
        low_snr = snr < self.snr_cut
        if low_snr.sum() == len(snr):
            self.logger.warning('{}SNR for all sources below cut: {} < {:.1f}',
                                (f'Frame {i}: ' if i else ''), snr, self.snr_cut)
            low_snr = snr < snr.max()  # else we end up with nans

        weights[low_snr] = 0
        self.logger.debug('weights: {:s}', weights)

        if np.all(weights == 0):
            raise ValueError('Could not determine weights for centrality '
                             'measurement from image')
            # self.logger.warning('Received zero weight vector. Setting to
            # unity')

        return weights

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

    def compute_offset(self, xy, weights):
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
        xy = np.ma.MaskedArray(xy, np.isnan(xy)).mean(1)
        idx = self.use_labels - 1
        δ = (self.coords[idx] - xy[idx])
        return np.ma.mean(δ, 0)
        # offset = np.ma.average(δ, 0, weights)
        # this offset already relative to the global segmentation
        # return offset

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


# from matplotlib.widgets import CheckButtons
#
#
# class GraphicalSourceTracker(
#         FitsCubeDisplay):  # TODO: Inherit from mpl Animation??
#
#     trackerClass = SourceTracker
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
#         self.image.set_data(image)
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
#             self._xmarks, = self.ax.plot(*self.tracker.coords[:, ::-1].T,
#                                          **self.marker_properties)
#         return self._xmarks
#
#     def get_image_data(self, i):
#         tr, d = self.image_buttons.get_status()
#         image = self.data[i]
#         trk = self.tracker
#         if (tr and d):
#             mask = trk.seg.to_bool(trk.use_labels)
#             data = np.ma.masked_array(image, mask=mask)
#             return data
#         elif tr:  # and not d
#             return trk.seg.data
#         else:
#             return image
#
#     def set_frame(self, i, draw=True):
#         self.logger.debug('set_frame: {:s}', i)
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
#         seg = self.tracker.seg
#         data = seg.data
#         outlines = []
#         for s in seg.slices:
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
#         'TODO'
