"""
Methods for tracking camera movements in astronomical time-series CCD photometry
"""

# std libs
from typing import Any

from IPython import embed

import logging
import functools as ftl
import itertools as itt
import multiprocessing as mp

# third-party libs
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from astropy.stats import median_absolute_deviation as mad

# local libs
from recipes.logging import LoggingMixin
from recipes.introspection.utils import get_module_name
from obstools.phot.utils import LabelGroupsMixin
from obstools.phot.segmentation import (SegmentationHelper,
                                        SegmentationGridHelper,
                                        make_border_mask,
                                        merge_segmentations, select_rect_pad)
from obstools.stats import geometric_median

from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift  # DBSCAN
from sklearn.preprocessing import StandardScaler
from astropy.utils import lazyproperty

from recipes.dict import AttrReadItem
from recipes.parallel.synced import SyncedCounter, SyncedArray

import more_itertools as mit

# from obstools.phot.utils import id_stars_dbscan, group_features

# from IPython import embed

# TODO: CameraTrackingModel / CameraOffset / CameraPositionModel

# TODO: filter across frames for better shift determination ???
# TODO: wavelet sharpen / lucky imaging for better relative positions
from sklearn.cluster import OPTICS
from scipy.stats import mode

# TODO
#  simulate how different centre measures performs for stars with decreasing snr
#  super resolution images
#  lucky imaging ?


# module level logger
logger = logging.getLogger(get_module_name(__file__))

TABLE_STYLE = dict(txt='bold', bg='g')


def id_stars_kmeans(images, segmentations):
    # TODO: method of tracker ????

    # combine segmentation for sample images into global segmentation
    # this gives better overall detection probability and yields more accurate
    # optimization results

    # this function also uses kmeans clustering to id stars within the overall
    # constellation of stars across sample images.  This is fast but will
    # mis-identify stars if the size of camera dither between frames is on the
    # order of the distance between stars in the image.

    coms = []
    snr = []
    for image, segm in zip(images, segmentations):
        coms.append(segm.com_bg(image))
        snr.append(segm.snr(image))

    ni = len(images)
    # use the mode of cardinality of sets of com measures
    lengths = list(map(len, coms))
    mode_value, counts = mode(lengths)
    k, = mode_value
    # nstars = list(map(len, coms))
    # k = np.max(nstars)
    features = np.concatenate(coms)

    # rescale each feature dimension of the observation set by stddev across
    # all observations
    # whitened = whiten(features)

    # fit
    centroids, distortion = kmeans(features, k)  # centroids aka codebook
    labels = cdist(centroids, features).argmin(0)

    cx = np.ma.empty((ni, k, 2))
    cx.mask = True
    w = np.ma.empty((ni, k))
    w.mask = True
    indices = np.split(labels, np.cumsum(lengths[:-1]))
    for ifr, ist in enumerate(indices):
        cx[ifr, ist] = coms[ifr]
        w[ifr, ist] = snr[ifr]

    # shifts calculated as snr-weighted average
    shifts = np.ma.average(cx - centroids, 1, np.dstack([w, w]))

    return cx, centroids, np.asarray(shifts)


def cluster_id_stars(clf, xy):
    #
    X = np.vstack(xy)
    n = len(X)
    # stars_per_image = list(map(len, xy))
    # no_detection = np.equal(stars_per_image, 0)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(np.vstack(xy))

    logger.info('Grouping %i position measurements using:\n%s', n, clf)
    clf.fit(X)

    labels = clf.labels_
    # core_samples_mask = (clf.labels_ != -1)
    # core_sample_indices_, = np.where(core_samples_mask)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    # n_per_label = np.bincount(db.labels_[core_sample_indices_])
    logger.info('Identified %i stars using %i/%i points (%i noise)',
                n_clusters, n - n_noise, n, n_noise)
    return n_clusters, n_noise


def plot_clusters(ax, clf, features, cmap='tab20b'):
    from matplotlib.cm import get_cmap

    core_sample_indices_, = np.where(clf.labels_ != -1)
    labels_xy = clf.labels_[core_sample_indices_]

    # plot
    cmap = get_cmap(cmap)
    colors = cmap(np.linspace(0, 1, labels_xy.max() + 1))
    c = colors[clf.labels_[core_sample_indices_]]

    # fig, ax = plt.subplots(figsize=im.figure.get_size_inches())
    ax.scatter(*features[core_sample_indices_].T, edgecolors=c,
               facecolors='none')
    ax.plot(*features[clf.labels_ == -1].T, 'kx', alpha=0.3)

    ax.grid()


def measure_positions_offsets(xy, d_cut=None, detect_frac_min=0.9):
    """
    Measure the relative positions of detected stars from the individual
    location measurements in xy

    Parameters
    ----------
    xy:     array, shape (n_points, n_stars, 2)
    d_cut:  float
    detect_frac_min

    Returns
    -------

    """
    assert 0 < detect_frac_min < 1

    n, n_stars, _ = xy.shape
    nans = np.isnan(np.ma.getdata(xy))
    bad = (nans | np.ma.getmask(xy)).any(-1)
    ignore_frames = nans.all((1, 2))
    n_ignore = ignore_frames.sum()
    assert n_ignore < n
    if n_ignore:
        logger.info('Ignoring %i/%i (%.1f%%) nan values',
                    n_ignore, n, (n_ignore / n) * 100)

    # mask nans
    xy = np.ma.MaskedArray(xy, nans)

    # any measure of centrality for cluster centers is only a good estimator of
    # the relative positions of stars when the camera offsets are taken into
    # account.

    # Due to camera/telescope drift, some stars (those near edges of the frame)
    # may not be detected in many frames. Cluster centroids are not an accurate
    # estimator of relative position for these stars since it's an incomplete
    # sample. Only stars that are detected in `detect_frac_min` fraction of
    # the frames will be used to calculate frame xy offset

    n_use = n - n_ignore
    n_detections_per_star = np.empty(n_stars, int)
    w = np.where(~bad)[1]

    n_detections_per_star[np.unique(w)] = np.bincount(w)
    use_stars = (n_detections_per_star / n_use) > detect_frac_min
    i_use, = np.where(use_stars)
    if np.any(~use_stars):
        logger.info(
                'Ignoring %i/%i stars with low (<%.0f%%) detection fraction',
                n_stars - len(i_use), n_stars, detect_frac_min * 100)

    # first estimate of relative positions comes from unshifted cluster centers
    # Compute cluster centres as geometric median
    good = ~ignore_frames
    xyc = xy[good][:, use_stars]
    nansc = nans[good][:, use_stars]
    if nansc.any():
        xyc[nansc] = 0  # prevent warning emit in _measure_positions_offsets
        xyc[nansc] = np.ma.masked
    #
    centres = np.empty((n_stars, 2))  # ma.masked_all
    for i, j in enumerate(i_use):
        centres[j] = geometric_median(xyc[:, i])

    # ensure output same size as input
    δxy = np.ma.masked_all((n, 2))
    σxy = np.empty((n_stars, 2))  # σxy = np.ma.masked_all((n_stars, 2))
    centres[use_stars], σxy[use_stars], δxy[good], out = \
        _measure_positions_offsets(xyc, centres[use_stars], d_cut)

    # compute positions of all sources with offsets from best and brightest
    for i in np.where(~use_stars)[0]:
        recentred = xy[:, i].squeeze() - δxy
        centres[i] = geometric_median(recentred)
        σxy[i] = recentred.std()

    # fix outlier indices
    idxf, idxs = np.where(out)
    idxg, = np.where(good)
    idxu, = np.where(use_stars)
    outlier_indices = (idxg[idxf], idxu[idxs])
    xy[outlier_indices] = np.ma.masked

    return xy, centres, σxy, δxy, outlier_indices


import warnings


def _measure_positions_offsets(xy, centres, d_cut=None):
    # ensure we have at least some centres
    assert not np.all(np.ma.getmask(centres))

    n, n_stars, _ = xy.shape
    n_points = n * n_stars

    out = np.zeros(xy.shape[:-1], bool)
    xym = np.ma.MaskedArray(xy, copy=True)

    counter = itt.count()
    while True:
        count = next(counter)
        if count >= 5:
            raise Exception('Emergency stop')

        # xy position offset in each frame
        xy_offsets = (xym - centres).mean(1, keepdims=True)

        # shifted cluster centers (all stars)
        xy_shifted = xym - xy_offsets
        # Compute cluster centres as geometric median of shifted clusters
        centres = np.ma.empty((n_stars, 2))
        for i in range(n_stars):
            centres[i] = geometric_median(xy_shifted[:, i])

        if d_cut is None:
            return centres, xy_shifted.std(0), xy_offsets.squeeze(), out

        # compute position residuals
        cxr = xy - centres - xy_offsets

        # flag outliers
        with warnings.catch_warnings():  # catch RuntimeWarning for masked
            warnings.filterwarnings('ignore')
            d = np.sqrt((cxr * cxr).sum(-1))

        out_new = (d > d_cut)
        out_new = np.ma.getdata(out_new) | np.ma.getmask(out_new)

        changed = (out != out_new).any()
        if changed:
            out = out_new
            xym[out] = np.ma.masked
            n_out = out.sum()

            if n_out / n_points > 0.5:
                raise Exception('Too many outliers!!')

            logger.info('Ignoring %i/%i (%.3f%%) values with |δr| > %.1f',
                        n_out, n_points, (n_out / n_points) * 100, d_cut)
        else:
            break

    return centres, xy_shifted.std(0), xy_offsets.squeeze(), out


# def measure_positions_dbscan(xy):
#     # DBSCAN clustering
#     db, n_clusters, n_noise = id_stars_dbscan(xy)
#     xy, = group_features(db, xy)
#
#     # Compute cluster centres as geometric median
#     centres = np.empty((n_clusters, 2))
#     for j in range(n_clusters):
#         centres[j] = geometric_median(xy[:, j])
#
#     xy_offsets = (xy - centres).mean(1)
#     xy_shifted = xy - xy_offsets[:, None]
#     # position_residuals = xy_shifted - centres
#     centres = xy_shifted.mean(0)
#     σ_pos = xy_shifted.std(0)
#     return centres, σ_pos, xy_offsets


def group_features(classifier, *features):
    """
    Groups items from multiple data sets in features into classes based on the
    cluster labels taken from `classifier`. Each data set is returned as a
    masked array in which items from a certain class that are omitted in the
    sequence are masked out. It is therefore assumed that items in each
    data set are part of a ordered sequence.


    Parameters
    ----------
    classifier
    features

    Returns
    -------

    """
    labels = classifier.labels_
    # core_samples_mask = np.zeros(len(labels), dtype=bool)
    # core_samples_mask[classifier.core_sample_indices_] = True

    unique_labels = list(set(labels))
    if -1 in unique_labels:
        unique_labels.pop(unique_labels.index(-1))
    n_clusters = len(unique_labels)
    n_samples = len(features[0])

    items_per_frame = list(map(len, features[0]))
    split_indices = np.cumsum(items_per_frame[:-1])
    indices = np.split(labels, split_indices)

    grouped = []
    for f in next(zip(*features)):
        shape = (n_samples, n_clusters)
        if f.ndim > 1:
            shape += (f.shape[-1],)

        g = np.ma.empty(shape)
        g.mask = True
        grouped.append(g)

    for i, j in enumerate(indices):
        ok = (j != -1)
        if ~ok.any():
            # catches case in which f[i] is empty (eg. no object detections)
            continue

        for f, g in zip(features, grouped):
            g[i, j[ok]] = f[i][ok, ...]

    return tuple(grouped)


def report_measurements(xy, centres, σ_xy, counts=None, detect_frac_min=None,
                        count_thresh=None, logger=logger):
    # report on relative position measurement
    import operator as op

    from recipes import pprint
    from motley.table import Table, highlight
    from motley.utils import ConditionalFormatter

    # TODO: probably mask nans....

    #
    n_points, n_stars, _ = xy.shape
    n_points_tot = n_points * n_stars

    if np.ma.is_masked(xy):
        bad = xy.mask.any(-1)
        good = np.logical_not(bad)
        points_per_star = good.sum(0)
        stars_per_image = good.sum(1)
        n_noise = bad.sum()
        no_detection, = np.where(np.equal(stars_per_image, 0))
        if len(no_detection):
            logger.info(f'No stars detected in frames: {no_detection!s}')

        if n_noise:
            extra = f'\nn_noise = {n_noise}/{n_points_tot} ' \
                f'({n_noise / n_points_tot :.1%})'
    else:
        points_per_star = np.tile(n_points, n_stars)
        extra = ''

    col_headers = ['n (%)', 'x (px.)', 'y (px.)']
    fmt = {0: ftl.partial(pprint.decimal_with_percentage,
                          total=n_points, precision=0)}  # right_pad=1

    if detect_frac_min is not None:
        n_min = detect_frac_min * n_points
        fmt[0] = ConditionalFormatter('y', op.lt, n_min, fmt[0])

    # get array with number ± std representations
    columns = [points_per_star,
               pprint.uarray(centres, σ_xy, 2)[:, ::-1]]

    if counts is not None:
        # TODO highlight counts?
        cn = 'counts (e⁻)'
        fmt[cn] = ftl.partial(pprint.numeric, thousands=' ', precision=1,
                              compact=False)
        if count_thresh:
            fmt[cn] = ConditionalFormatter('c', op.ge, count_thresh,
                                           fmt[cn])
        columns.append(counts)
        col_headers += [cn]

    # tbl = np.ma.column_stack(columns)
    tbl = Table.from_columns(*columns,
                             title='Measured star locations',
                             title_props=TABLE_STYLE,
                             col_headers=col_headers,
                             col_head_props=TABLE_STYLE,
                             precision=0,
                             align='r',
                             number_rows=True,
                             total=[0],
                             formatters=fmt)

    logger.info('\n' + str(tbl) + extra)
    return tbl


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


# class NullSlice():
#     """Null object pattern for getitem"""
#
#     def __getitem__(self, item):
#         return None

# constructor used for pickling
# def _construct_SegmentedArray(*args):
#     return SegmentedArray(*args)


class LabelUser(object):
    """
    Mixin class for inheritors that use `SegmentationHelper`.
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
        return self.segm.has_labels(labels)

    @property
    def nlabels(self):
        return len(self.use_labels)

    @property
    def sizes(self):
        return [len(labels) for labels in self.values()]


# class MaskUser(object):
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
        self.segm = seg
        self.groups = groups

    def __getitem__(self, name):
        if name in self:
            return super().__getitem__(name)
        elif name in self.groups:
            # compute the mask of this group on demand
            mask = self[name] = self.of_group(name)
            return mask
        raise KeyError(name)

    @lazyproperty
    def all(self):
        return self.segm.to_bool()

    def for_phot(self, labels=None, ignore_labels=None):
        """
        Select sub-regions of the image that will be used for photometry
        """
        labels = self.segm.resolve_labels(labels)
        # np.setdiff1d(labels, ignore_labels)
        # indices = np.digitize(labels, self.segm.labels) - 1
        kept_labels = np.setdiff1d(labels, ignore_labels)
        indices = np.digitize(labels, kept_labels) - 1

        m3d = self.segm.to_bool_3d(None, ignore_labels)
        all_masked = m3d.any(0)
        return all_masked & ~m3d[indices]

    def prepare(self, labels=None, ignore_labels=None, sky_buffer=2,
                sky_width=10, edge_cutoffs=None):
        """
        Select sub-regions of the image that will be used for photometry
        """

        # sky_regions = self.prepare_sky(labels, sky_buffer, sky_width,
        #                                edge_cutoffs)
        self['phot'] = masks_phot = self.for_phot(labels, ignore_labels)
        self['sky'] = masks_phot.any(0)

    def prepare_sky(self, labels, sky_buffer=2, sky_width=10,
                    edge_cutoffs=None):
        sky_regions = self.segm.to_annuli(sky_buffer, sky_width, labels)
        if edge_cutoffs is not None:
            edge_mask = make_border_mask(self.segm.data, edge_cutoffs)
            sky_regions &= ~edge_mask

        return sky_regions

    def of_group(self, g):
        return self.segm.to_bool(self.groups[g])


class SegmentationMasksHelper(SegmentationGridHelper, LabelGroupsMixin):

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
    # making group_masks a lazyproperty means it will get reset auto-magically when
    # the segmentation data changes
    def group_masks(self):
        return MaskContainer(self, self.groups, **self._persistent_masks)


class GlobalSegmentation(SegmentationMasksHelper):
    @classmethod
    def merge(cls, segmentations, xy_offsets, extend=True, f_accept=0.5,
              post_dilate=1):
        # zero point correspond to minimum offset from initializer and serves as
        # coordinate reference
        # make int type to avoid `Cannot cast with casting rule 'same_kind'
        # downstream
        return cls(merge_segmentations(segmentations, xy_offsets,
                                       extend,
                                       f_accept,
                                       post_dilate),
                   zero_point=np.floor(xy_offsets.min(0)).astype(int))

    def __init__(self, data, zero_point):
        super().__init__(data)
        self.zero_point = zero_point

    def get_start_indices(self, xy_offsets):
        if np.ma.is_masked(xy_offsets):
            raise ValueError('Cannot get start indices for masked offsets')

        return np.abs((xy_offsets + self.zero_point).round().astype(int))

    def select_subset(self, start, shape, type_=SegmentationHelper):
        return self.select_subset(start, shape, type_)

    def for_offset(self, xy_offsets, shape, type_=SegmentationHelper):
        if np.ma.is_masked(xy_offsets):
            raise ValueError('Cannot get segmented image for masked offsets')

        return self.select_subset(
                self.get_start_indices(xy_offsets), shape, type_)

    def flux(self, image, llc, labels=None, labels_bg=(0,), bg_stat='median'):
        sub = self.select_subset(llc, image.shape)
        return sub.flux(image, labels, labels_bg, bg_stat)

    def sort(self, measure, descend=False):
        if len(measure) != self.nlabels:
            raise ValueError('Cannot reorder labels for measure with '
                             'incompatable size %i. %s has %i labels' %
                             (len(measure), self, self.nlabels))

        o = slice(None, None, -1) if descend else ...
        order = np.ma.argsort(measure, endwith=False)[o]
        self.relabel_many(order + 1, self.labels)
        return order


DETECTION_DEFAULTS = dict(snr=3.,
                          npixels=7,
                          edge_cutoffs=None,
                          deblend=False,
                          dilate=1)

from obstools.phot.tracking.core import GlobalSegmentation


class StarTracker(LabelUser, LoggingMixin, LabelGroupsMixin):
    """
    A class to track stars in a CCD image frame to aid in doing time series
    photometry.

    Stellar positions in each new frame are calculated as the background
    subtracted centroids of the chosen image segments. Centroid positions of
    chosen stars are filtered for bad values (or outliers) and then combined
    by a weighted average. The weights are taken as the SNR of the star
    brightness. Also handles arbitrary pixel masks.

    Constellation of stars in image assumed unchanging. The coordinate positions
    of the stars with respect to the brightest star are updated upon each
    iteration.


    """

    # TODO: bayesian version
    # TODO: make multiprocessing safe

    segmClass = SegmentationHelper

    # TODO: manage as properties ?
    snr_weighting = False
    snr_cut = 0
    _distance_cut = 10
    _saturation_cut = 95  # %

    # this allows tracking stars with low snr, but not computing their
    # centroids which will have larger scatter

    # clustering algorithm for construction from ungrouped centroid measurements
    clustering = None  # just a place holder for now

    # todo: put default clustering algorithm here
    # outlier removal
    delta_r = 1  # pixels  # TODO: pass to initializer!!!

    # maximal distance for individual location measurements from centre of
    # distribution

    # # minimum location measurements on individual stars before clustering is
    # # triggered
    # _min_measurements =

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
    def from_images(cls, images, mask=None, required_positional_accuracy=0.5,
                    centre_distance_max=1, f_detect_measure=0.5,
                    f_detect_merge=0.2, post_merge_dilate=1, flux_sort=True,
                    worker_pool=None, report=None, plot=False, **detect_kws):
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

        from obstools.phot.segmentation import detect_measure

        if worker_pool is None:
            # create worker pool
            raise NotImplementedError

        from collections import Callable
        if isinstance(plot, Callable):
            display = plot
            plot = True
        else:
            plot = bool(plot)

            def display(*_):
                pass

        #
        n, *ishape = images.shape
        # detect sources, measure locations
        _detect_measure = ftl.partial(detect_measure, **detect_kws)
        segmentations, coms = zip(
                *map(_detect_measure, images))

        # clustering + relative position measurement
        logger.info('Identifying stars')
        # clf = OPTICS(min_samples=int(f_detect_accept * n))
        clf = MeanShift(bandwidth=2, cluster_all=False)
        n_clusters, n_noise = cluster_id_stars(clf, coms)
        xy, = group_features(clf, coms)

        # 🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(13.9, 2))  # shape for slotmode
            ax.set_title(f'Position Measurements (CoM) {n} frames')
            # fontweight='bold'
            plot_clusters(ax, clf, np.vstack(coms)[:, ::-1])
            ax.set(**dict(zip(map('{}lim'.format, 'yx'),
                              tuple(zip((0, 0), ishape)))))
            display(fig)

        #
        logger.info('Measuring relative positions')
        _, centres, σ_xy, xy_offsets, outliers = \
            measure_positions_offsets(xy, centre_distance_max, f_detect_measure)

        # zero point for tracker (slices of the extended frame) correspond
        # to minimum offset
        # xy_off_min = xy_offsets.min(0)
        # zero_point = np.floor(xy_off_min)

        # 🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            # diagnostic for source location measurements
            from obstools.phot.diagnostics import plot_position_measures

            fig, axes = plot_position_measures(xy, centres, xy_offsets)
            fig.suptitle('Position Measurements (CoM)')  # , fontweight='bold'
            display(fig)

        # combine segmentation images
        seg_glb = GlobalSegmentation.merge(segmentations, xy_offsets,
                                           True,  # extend
                                           f_detect_merge,
                                           post_merge_dilate)

        # since the merge process re-labels the stars, we have to ensure the
        # order of the labels correspond to the order of the clusters.
        # Do this by taking the label of the pixel nearest measured centers
        # This is also a useful metric for the success of the clustering
        # step: Sometimes multiple stars are put in the same cluster,
        # in which case the centroid will likely be outside the labelled
        # regions in the segmented image (label 0). These stars will then
        # fortuitously be ignored below.

        # clustering algorithm may also identify more stars than in `seg_glb`
        # whether this happens will depend on the `f_detect_merge` parameter.
        # The sources that are not in the segmentation image will get the
        # label 0 in `cluster_labels`
        cxx = np.ma.getdata(centres - seg_glb.zero_point)
        indices = cxx.round().astype(int)
        cluster_labels = seg_glb.data[tuple(indices.T)]
        seg_lbl_omit = (cluster_labels == 0)
        if seg_lbl_omit.any():
            cls.logger.info('%i sources omitted from merged segmentation.' %
                            seg_lbl_omit.sum())

        # return seg_glb, xy, centres, σ_xy, xy_offsets, outliers, cluster_labels

        # may also happen that stars that are close together get grouped in
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
            llcs = seg_glb.get_start_indices(xy_offsets)

            # ======================================================================
            counts = np.ma.empty(xy.shape[:-1])
            ok = np.logical_not(seg_lbl_omit)

            counts[:, ok] = worker_pool.starmap(
                    seg_glb.flux, ((image, ij0)
                                   for i, (ij0, image) in
                                   enumerate(zip(llcs, images))))

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
        order = np.ma.MaskedArray(cluster_labels, cluster_labels == 0).argsort()
        centres = centres[order]
        xy = xy[:, order]
        σ_xy = σ_xy[order]

        # 🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            im = seg_glb.display()
            im.ax.set_title('Global segmentation 0')
            display(im.figure)

        # initialize tracker
        use_star = (σ_xy < required_positional_accuracy).all(1)
        use_star = np.ma.getdata(use_star) & ~np.ma.getmask(use_star)
        if use_star.sum() < 2:
            cls.logger.warning('Measured positions not accurate enough: σ_xy '
                               '= %s > %f', σ_xy, required_positional_accuracy)
            # TODO: continue until accurate enough!!!

        # FIXME: should use bright stars / high snr here!!
        use_labels = np.where(use_star)[0] + 1

        # init
        tracker = cls(cxx, seg_glb, use_labels=use_labels,
                      bad_pixel_mask=mask)
        tracker.sigma_rvec = σ_xy
        tracker.clustering = clf
        # tracker.xy_off_min = xy_off_min
        tracker.zero_point = seg_glb.zero_point
        # tracker.current_offset = xy_offsets[0]
        tracker.current_start = \
            (xy_offsets[0] - xy_offsets.min(0)).round().astype(int)

        # log
        if report is None:
            report = cls.logger.getEffectiveLevel() <= logging.INFO
        if report:
            # TODO: can probs also highlight large uncertainties
            #  and bright targets!

            tracker.report_measurements(xy, centres, σ_xy, counts_med,
                                        f_detect_measure)

        return tracker, xy, centres, xy_offsets, counts, counts_med

    @classmethod
    def from_measurements(cls, segmentations, xy, counts=None,
                          merge_accept_frac=0.2, post_merge_dilate=1,
                          required_positional_accuracy=0.5,
                          detect_frac_min=0.9,
                          bad_pixel_mask=None, report=None):
        """
        Initialize from set of segmentation images sampled across the data cube.

        This method will construct a global segmentation image from which the
        source tracker will be initialized.

        TODO: describe position measurement  algorithm

        Combine various image segmentations into single global image segmentation.
        This works better than the single image deep segmentation for a number of
        reasons:
            * The background gradient is large towards the edges of the
            image. Frame dither may therefore shift sources out of this region
            and onto region with flatter background, thereby increase their
            detectability.  This only really works if the maximal
            positional movement of the telescope / camera is large enough to
            move stars significantly away from the edges of the slot.
            * The background level and structure may change throughout the run,
            detection from multiple sample images across the cube improves overall
            sensitivity.
            * Influence of bad pixels on object detectability are smoothed out by
            dither.



        Returns
        -------

        """

        # This method makes more sense when dealing with images that have
        # strong vignetting towards edges than detecting on the mean image
        # since background effects will skew detection statistics towards edges.

        # clustering + relative position measurement
        cls.logger.info('Identifying stars from position measurements')
        db, n_clusters, n_noise = id_stars_dbscan(xy)
        xy, = group_features(db, xy)

        from motley.table import Table
        cls.logger.info('Identified the following stars:\n%s',
                        Table.from_columns(xy.mean(0)[::-1],
                                           np.sum(~xy[..., 0].mask, 0),
                                           title='Detected stars',
                                           title_props=TABLE_STYLE,
                                           col_headers=list('xyn'),
                                           col_head_props=TABLE_STYLE,
                                           number_rows=True, align='r'))

        #
        cls.logger.info('Measuring relative positions')
        _, centres, σ_xy, xy_offsets, outliers = \
            measure_positions_offsets(xy, cls.delta_r, detect_frac_min)
        # TODO: use self.measure_positions_offsets ?!

        # combine segmentation images
        seg_extended = merge_segmentations(segmentations.astype(bool),
                                           xy_offsets,
                                           True,  # extend
                                           merge_accept_frac,
                                           0)  # post_merge_dilate

        # zero point for tracker (slices of the extended frame) correspond to
        # minimum offset
        xy_off_min = xy_offsets.min(0)
        zero_point = np.floor(xy_off_min).astype(int)
        # zero point int type avoids `Cannot cast with casting rule 'same_kind'
        # downstream

        # δ = zero_point - xy_off_min

        # MEASURE FLUXES HERE. bright objects near edges of the slot
        #  have lower detection probability and usually only partially detected
        #  merged segmentation usually more complete for these sources which
        #  leads to more accurate flux measurement and therefore better
        #  change of flagging bright partial sources for photon bleed
        llcs = get_start_indices(xy_offsets, zero_point)
        for start in llcs:
            select_rect_pad(seg_extended, )

        # ordering
        # if counts is not None:
        #     # reorder everything bright to faint
        #     counts, = group_features(db, counts)
        #
        #     # reorder star labels for descending brightness
        #     counts_med = np.ma.median(counts, 0)
        #     brightness_order = counts_med.argsort()[::-1]  # bright to faint
        #
        #     # reorder everything
        #     xy = xy[:, brightness_order]
        #     counts = counts[:, brightness_order]
        #     counts_med = counts_med[brightness_order]
        # else:
        #     counts_med = None

        # Make position measures relative to segmentation zero point
        # xy_offsets_z = xy_offsets - zero_point

        # xy -= zero_point
        # centres -= zero_point
        # xy_offsets += zero_point

        # since the merge process re-labels the stars, we have to ensure the
        # order of the labels correspond to the order of the clusters.
        # Do this by taking the label of the pixel nearest measured centers
        # removed masked points from coordinate measurement
        cxx = centres - zero_point
        if np.ma.is_masked(centres):
            cxx = cxx[~centres.mask.any(-1)].data

        indices = cxx.round().astype(int)
        old_labels = seg_extended.data[tuple(indices.T)]

        # it may happen that there are more labelled object in the merged
        # SegmentationImage than where found by the clustering algorithm.
        n_clusters = len(cxx)
        if n_clusters < seg_extended.nlabels:
            missing = np.setdiff1d(seg_extended.labels, old_labels)
            old_labels = np.append(old_labels, missing)

        # print('relabel', old_labels, seg_extended.labels)
        seg_extended.relabel_many(old_labels, seg_extended.labels)

        # initialize tracker
        use_star = (σ_xy < required_positional_accuracy).all(1)
        use_star = np.ma.getdata(use_star) & ~np.ma.getmask(use_star)
        if use_star.sum() < 2:
            cls.logger.warning('Measured positions not accurate enough: σ_xy '
                               '= %s > %f', σ_xy, required_positional_accuracy)
            # TODO: continue until accurate enough!!!

        # FIXME: should use bright stars / high snr here!!
        use_labels = np.where(use_star)[0] + 1

        # init
        tracker = cls(cxx, seg_extended, use_labels=use_labels,
                      bad_pixel_mask=bad_pixel_mask)
        tracker.sigma_rvec = σ_xy
        tracker.clustering = db
        # tracker.xy_off_min = xy_off_min
        tracker.zero_point = zero_point
        # tracker.current_offset = xy_offsets[0]
        tracker.current_start = (xy_offsets[0] - xy_off_min).round().astype(int)

        # log
        if report is None:
            report = cls.logger.getEffectiveLevel() <= logging.INFO
        if report:
            tracker.report_measurements(xy, centres, σ_xy, counts_med)

        return tracker, xy, centres, xy_offsets, counts, counts_med

    #
    # @classmethod
    # def from_cube_segment(cls, cube, ncomb, subset, nrand=None, mask=None,
    #                       bgmodel=None, snr=(10., 3.), npixels=7,
    #                       edge_cutoffs=None, deblend=False, dilate=1,
    #                       flux_sort=True):
    #
    #     # select `ncomb`` frames randomly from `nrand` in the interval `subset`
    #     if isinstance(subset, int):
    #         subset = (0, subset)  # treat like a slice
    #
    #     i0, i1 = subset
    #     if nrand is None:  # if not given, select from entire subset
    #         nrand = i1 - i0
    #     nfirst = min(nrand, i1 - i0)
    #     ix = np.random.randint(i0, i0 + nfirst, ncomb)
    #     # create ref image for init
    #     image = np.median(cube[ix], 0)
    #     cls.logger.info('Combined %i frames from amongst frames (%i->%i) for '
    #                     'reference image.', ncomb, i0, i0 + nfirst)
    #
    #     # init the tracker
    #     tracker, p0bg = SlotModeTracker.from_image(
    #             image, mask, bgmodel, snr, npixels, edge_cutoffs, deblend,
    #             dilate, flux_sort)
    #     return tracker, image, p0bg

    def __init__(self, coords, segm, label_groups=None, use_labels=None,
                 reference_index=0, bad_pixel_mask=None, edge_cutoffs=3,
                 background_estimator=np.ma.median, weights=None,
                 update_rvec_every=100, update_weights_every=10,
                 precision=0.25):
        """
        Class for tracking CCD camera movement by measuring location of stars in
        images (by eg. centroid or projected geometric median)

         positions of all stars with good signal-to-noise.

        Relative position of stars are
        updated on each call based on new input data. # fixme


        Parameters
        ----------
        coords: array-like
            reference coordinates of the stars
        segm: array or SegmentationHelper (or subclass thereof) instance
            segmentation image. Centroids for each star are computed within
            the corresponding labelled region
        label_groups:

        use_labels: array-like of int
            only stars corresponding to these labels will be used to
            calculate frame shift
        bad_pixel_mask: array-like
            ignore these pixels in centroid computation
        edge_cutoffs: int or list of int
            Ignore labels that are too close to the edges of the image
        reference_index: int

        """

        # label include / exclude logic
        LabelUser.__init__(self, use_labels)
        LabelGroupsMixin.__init__(self, label_groups)

        if isinstance(segm, SegmentationMasksHelper):
            # detecting the class of the SegmentationImage allows some
            # optimization by avoiding unnecessary recompute of lazyproperties.
            # Also allows custom subclasses of SegmentationGridHelper to be
            # used
            self.segm = segm
        else:
            self.segm = SegmentationMasksHelper(segm,
                                                groups=label_groups,
                                                bad_pixels=bad_pixel_mask)

        # counter for incremental update
        self.counter = None
        self.measurements = None
        self.xy_offsets = None

        # pixel coordinates of reference position from which the shift will
        # be measured.

        n_stars = len(coords)
        self.ir = int(reference_index)
        self.yx0 = coords[self.ir]
        self.rvec = coords - coords[self.ir]
        self.sigma_rvec = np.full(n_stars, np.inf)
        self.precision = float(precision)
        # store zero point and relative positions separately so we can update
        # them independently

        # algorithmic details
        # self._measure_star_locations = self.segm.com_bg
        self._update_rvec_every = int(update_rvec_every)
        self._update_weights_every = int(update_weights_every)

        # background_estimator for location measurement
        self.bg = background_estimator
        self._weights = weights
        # self.current_offset = np.zeros(2)
        self.current_start = np.zeros(2, int)
        # self.xy_off_min = None

        self.edge_cutoffs = edge_cutoffs

        # masks for photometry
        # self.bad_pixel_mask = bad_pixel_mask

        # if edge_cutoffs is not None:
        #     self.edge_mask = make_border_mask(self.segm.data, edge_cutoffs)
        # else:
        #     self.edge_mask = None

        # extended masks
        # self.masks = MaskContainer(self.segm, self.groups,
        #                            bad_pixels=bad_pixel_mask)

        # label logic
        # if use_labels is None:
        #     # we want only to use the stars with high snr fro CoM tracking

    def __call__(self, image, mask=None):
        return self.track(image, mask)

    # @property
    # def ngroups(self):
    #     return len(self.groups)

    # def _com_use_labels(self):
    #     return self.segm.com(self, self.use_labels)

    @property
    def masks(self):  # extended masks
        return self.segm.group_masks

    # @property
    # def nsegs(self):
    #     return self.segm.nlabels

    @property
    def rcoo(self):  # todo rename coords / coords_yx
        """
        Reference coordinates (yx). Computed as initial coordinates + relative
        coordinates to allow updating the relative positions upon call
        """
        return self.yx0 + self.rvec

    @property
    def rcoo_xy(self):  # todo rename coords_xy
        """Reference coordinates (xy)"""
        return self.rcoo[:, ::-1]

    # @property
    # def xy_offset_max(self):
    #     return self.xy_offsets.max(0)

    def run(self, data, indices=None, pool=None, njobs=mp.cpu_count(),
            start_offset=None):
        # run concurrently

        if self.measurements is None:
            raise Exception('Initialize memory first')

        if indices is None:
            indices = range(len(self.measurements))

        self.counter = SyncedCounter()
        self.sigma_rvec = SyncedArray(self.sigma_rvec)

        if pool is None:
            'create pool with njobs'
            raise NotImplementedError

        # TODO: initializer that sets the starting indices for each pickled
        #  clone

        pool.map(self.track_loop, ((i, data)
                                   for i in mit.divide(indices, njobs)))

    def track(self, index, image, mask=None):
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

        # TODO: use grid and add offset to grid when computing CoM.  Will be
        #  vastly more efficient
        yx = self.measure_star_locations(image, mask, self.current_start)
        self.measurements[index] = yx

        # weights
        if self.snr_weighting or self.snr_cut:
            if (self._weights is None) or (count // self._update_weights_every):
                self._weights = self.get_snr_weights(image)
                # TODO: maybe aggregate weights and use mean ??

        # # update relative positions from CoM measures
        # check here if measurements of cluster centres are good enough!
        if ((count + 1) % self._update_rvec_every) == 0 and count > 0 and \
                (self.sigma_rvec > self.precision).any():
            self.logger.info('Measuring positions, offsets')
            self.measure_positions_offsets(self.measurements[:(index + 1)])
            off = self.xy_offsets[index]
        else:
            # calculate median frame shift using current best relative positions
            # optionally using snr weighting scheme
            # measurements wrt image, offsets wrt rcoo
            off = self.compute_offset(yx, self._weights)
            self.xy_offsets[index] = off

        # TODO: detect if stars move off frame!!!

        #
        self.current_start = off.round().astype(int)

        # if np.isnan(off).any():
        #     raise Exception('PISS WIT')
        #
        # if np.ma.is_masked(off):
        #     raise ValueError('OFFICIOUS TWAT-WAFFLE')
        #
        # if np.any(np.abs(off) > 20):
        #     raise ValueError('FUCK STICKS')

        # if np.any(np.abs(self.current_offset - off) > 1):
        #     self.current_start = off.round().astype(int)
        # print('OFFSET', off)

        # finally return the new coordinates
        return self.rcoo + off

    def track_loop(self, indices, data):
        for i in indices:
            self.track(i, data[i])

    def get_coords(self, i=None):
        if i is None:
            return self.rcoo[None] - self.xy_offsets[:, None]
        return self.rcoo - self.xy_offsets[i]

    def get_coord(self, i):
        return self.rcoo - self.xy_offsets[i]

    def get_coords_residual(self):
        return self.measurements - self.rcoo - self.xy_offsets[:, None]

    def init_mem(self, n, loc=None, clobber=False):
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
        from obstools.modelling.utils import load_memmap

        if loc is None:
            import tempfile
            from pathlib import Path
            loc = Path(tempfile.mkdtemp())

        common = ('f', np.nan, clobber)
        self.measurements = load_memmap(loc / 'coo.com',
                                        (n, self.nlabels, 2),
                                        *common)
        self.xy_offsets = load_memmap(loc / 'coo.shift',
                                      (n, 2),
                                      *common)

    def measure_star_locations(self, image, mask=None, start_indices=(0, 0)):
        """
        calculate measure of central tendency for the objects in the
        segmentation image

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """
        # more robust that individual CoM track.
        # eg: cosmic rays:
        # '/media/Oceanus/UCT/Observing/data/Feb_2015/
        #  MASTER_J0614-2725/20150225.001.fits' : 23

        if not ((mask is None) or (self.masks.bad_pixels is None)):
            mask |= self.masks.bad_pixels
        else:
            mask = self.masks.bad_pixels  # may be None
            if mask is None:
                mask = False

        # ensure correct type
        start_indices = np.asanyarray(start_indices)
        if np.ma.is_masked(start_indices):
            raise ValueError('Start indices cannot be masked array')

        if not start_indices.dtype.kind == 'i':
            start_indices = start_indices.round().astype(int)

        yx = self._measure_star_locations(image, mask, start_indices)

        # check measurements
        bad = np.any((yx < self.edge_cutoffs) |
                     (yx > np.subtract(image.shape, self.edge_cutoffs)),
                     1)
        yx[bad] = np.nan  # FIXME: better to excise using mask!!

        return yx

    def _measure_star_locations(self, image, mask, start_indices):

        seg = self.segm.select_subset(start_indices, image.shape)
        xy = seg.com_bg(image, self.use_labels, mask, None)

        # note this is ~2x faster than padding image and mask. can probably
        #  be made even faster by using __slots__
        # todo: check if using grid + offset then com is faster

        # TODO: can do FLUX estimate here!!!

        # TODO: filter bad values here ??

        # self.check_measurement(xy)
        # self.rcoo - tracker.zero_point < start[0]

        # check values inside segment labels
        # good = ~self.is_bad(com)
        # self.coms[i, good] = com[good]

        return xy  # + start_indices

    # def check_measurement(self, xy):

    def measure_positions_offsets(self, xy=None):
        """
        Measure the relative positions of stars, as well as the xy offset of
        each image based on the set of measurements in `coms`

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
            measure_positions_offsets(xy, self.delta_r)
        n_points = xy_offsets.mask.any(-1).sum()

        # origin = np.floor(xy_offsets.min(0))

        # self.zero_point
        # note using the first non-masked point will avoid this compute
        #  FIXME: account for difference between zero point and min off here...
        # smallest offset as origin makes sense from global image perspective
        # changing the zero_point means you have to update all offsets

        # make the offsets relative to the global segmentation
        yx0 = centres[self.ir]
        δ = self.yx0 - yx0
        self.xy_offsets[:i] = xy_offsets + δ
        self.rvec = centres - yx0
        self.sigma_rvec[:] = σ_pos

        if (σ_pos < self.precision).all():
            self.logger.info(
                    'Required positional accuracy of %g achieved with %i '
                    'measurements. Relative positions will not be updated '
                    'further.', self.precision, n_points)

        self.report_measurements(xym, centres, σ_pos)

    def report_measurements(self, xy=None, centres=None, σ_xy=None,
                            counts=None, detect_frac_min=None,
                            count_thresh=None):

        if xy is None:
            nans = np.isnan(self.measurements)
            good = ~nans.all((1, 2))
            xy = self.measurements[good]

        if centres is None:
            centres = self.rcoo

        if σ_xy is None:
            σ_xy = self.sigma_rvec

        return report_measurements(xy, centres, σ_xy, counts, detect_frac_min,
                                   count_thresh, self.logger)

    def group_measurements(self, *measurement_sets):
        return group_features(self.clustering, *measurement_sets)

    def plot_position_measures(self):
        from obstools.phot.diagnostics import plot_position_measures

        return plot_position_measures(self.measurements, self.rcoo,
                                      -self.xy_offsets)

    # def _pad(self, image, start_indices, zero=0.):
    #     # make a measurement of some property of image using the segments
    #     # from the tracker. In general, image will be smaller than global
    #     # segmentation, therefore first pad the image with zeros
    #
    #     stop_indices = start_indices + image.shape
    #
    #     istart = -np.min([start_indices, (0, 0)], 0)
    #     istop = np.array([None, None])
    #     thing = np.subtract(self.segm.shape, stop_indices)
    #     l = thing < 0
    #     istop[l] = thing[l]
    #     islice = tuple(map(slice, istart, istop))
    #
    #     start_indices = np.clip(start_indices, 0, None)
    #     stop_indices = np.clip(stop_indices, None, self.segm.shape)
    #
    #     section = tuple(map(slice, start_indices, stop_indices))
    #     im = np.full(self.segm.shape, zero)
    #     im[section] = image[islice]
    #
    #     return im

    def prepare_image(self, image, start_indices):
        """
        Prepare a background image by masking stars, bad pixels and whatever
        else

        Parameters
        ----------
        image
        start_indices

        Returns
        -------

        """
        mask = self.get_object_mask(start_indices, start_indices + image.shape)
        return np.ma.MaskedArray(image, mask)

    def get_object_mask(self, start, stop):
        i0, j0 = start
        i1, j1 = stop
        return self.masks.all[i0:i1, j0:j1] | self.masks.bad_pixels

    def get_masks(self, start, shape):
        phot_masks = select_rect_pad(self.segm, self.masks.phot, start, shape)
        sky_mask = select_rect_pad(self.segm, self.masks.sky, start, shape)
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
        return self.segm.select_subset(start, shape)

    def flux_sort(self, fluxes):

        n_labels = len(self.use_labels)
        assert len(fluxes) == n_labels

        # reorder star labels for descending brightness
        brightness_order = fluxes.argsort()[::-1]  # bright to faint
        new_order = np.arange(n_labels)
        new_labels = new_order + 1

        # re-label for ascending brightness
        # new_labels = np.arange(1, len(self.use_labels) + 1)
        self.segm.relabel_many(brightness_order + 1,
                               new_labels)

        self.ir = new_order[self.ir]
        self.rvec[brightness_order] = self.rvec[new_order]
        self.sigma_rvec[brightness_order] = self.sigma_rvec[new_order]
        self.use_labels = new_labels

        return brightness_order

    # def offsets_to_indices(self, xy_offsets):

    # def reset_counter(self):
    #     self.count = 0

    # def update_rvec(self):

    # def measure_offset(self, com, weights):
    #     pass

    # def update_rvec_batch(self, coms=None):
    #     #
    #     # self._pre_batch = coms
    #     # self._pre_batch_size = len(coms)
    #
    #     # TODO: many different algorithms can be used here
    #
    #     if coms is None:
    #         coms = np.ma.MaskedArray(self.measurements,
    #                                  np.isnan(self.measurements))
    #
    #     # cluster centroids
    #     cen = coms.mean(0)
    #     # mean frame shift measured from CoM
    #     shifts = np.ma.average(coms - cen, 1, self._weights)
    #     # since the above combines measurements, stddev is same for all stars
    #     sigma_rvec = (coms[:, 0] - shifts).std(0)
    #
    #     rvec = cen - cen[self.ir]
    #     self.rvec[self.use_labels - 1] = rvec
    #     return rvec, sigma_rvec, shifts

    # def centroids(self, image, mask=None):
    #
    #     if not ((mask is None) or (self.bad_pixel_mask is None)):
    #         mask |= self.bad_pixel_mask
    #     else:
    #         mask = self.bad_pixel_mask  # may be None
    #
    #     # calculate CoM
    #     com = self.segm.com_bg(image, self.use_labels, mask, self.bg)
    #     return com

    def pprint(self):
        from motley.table import Table

        # FIXME: not working

        # TODO: say what is being ignored
        # TODO: include uncertainties

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

    # def mask_segments(self, image, mask=None):  # todo prepare_background better
    #     """
    #     Prepare a background image by masking stars, bad pixels and whatever
    #     else
    #     """
    #     # mask stars
    #     imbg = self.segm.mask_segments(image)
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
    #         labels = self.segm.labels
    #         indices = np.arange(self.segm.nlabels)
    #         # note `resolve_labels` only uses the labels in `use_labels`,
    #         #  which is NOT what we want since we want all stars to be masked
    #         #  not just those which are being used for tracking
    #     else:
    #         indices = np.digitize(labels, self.segm.labels) - 1
    #
    #     #
    #     m3d = self.segm.to_bool_3d()
    #     all_masked = m3d.any(0)
    #
    #     sky_regions = self.segm.to_annuli(sky_buffer, sky_width, labels)
    #     if edge_cutoffs is not None:
    #         edge_mask = make_border_mask(self.segm.data, edge_cutoffs)
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
    #     sky_masks = self.segm.to_annuli(sky_buffer, sky_width)
    #     # sky_masks &= ~self.streak_mask

    def _flux_estimate_annuli(self, image, ij, edge_cutoffs=None):

        # add object segments to model
        stop = ij + image.shape
        slice2d = tuple(map(slice, ij, stop))
        slice3d = (slice(None),) + slice2d
        edge_mask = make_border_mask(image, edge_cutoffs)
        sm = (self.masks.sky[slice3d] & ~(edge_mask | self.masks.bad_pixels))

        n_stars = len(self.masks.sky)
        flx_bg, npix_bg = np.empty((2, n_stars), int)
        counts, npix = np.ma.MaskedArray(np.empty((2, n_stars), int), True)
        for i, sky in enumerate(sm):
            bg_pixels = image[sky]
            flx_bg[i] = self.bg(bg_pixels)
            npix_bg[i] = len(bg_pixels)

        # source counts
        seg = SegmentationHelper(self.segm.data[slice2d])
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
    #     self.logger.debug('bad: %s', np.where(bad)[0])
    #     weights[bad] = 0
    #     return weights

    def get_snr_weights(self, image, i=None):
        # snr weighting scheme (for robustness)

        im = self._pad(image, self.current_start)
        weights = snr = self.segm.snr(im, self.use_labels)
        # self.logger.debug('snr: %s', snr)

        # ignore stars with low snr (their positions will still be tracked,
        # but not used to compute new positions)
        low_snr = snr < self.snr_cut
        if low_snr.sum() == len(snr):
            self.logger.warning(('Frame %i: ' if i else '') +
                                'SNR for all stars below cut: %s < %.1f',
                                snr, self.snr_cut)
            low_snr = snr < snr.max()  # else we end up with nans

        weights[low_snr] = 0
        self.logger.debug('weights: %s', weights)

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
        xym = np.ma.MaskedArray(xy, np.isnan(xy))
        δ = (self.rcoo[self.use_labels - 1] - xym)
        offset = np.ma.average(δ, 0, weights)
        # this offset already relative to the global segmentation
        return offset

    def update_rvec_point(self, coo, weights=None):
        # TODO: bayesian_update
        """"
        Incremental average of relative positions of stars (since they are
        considered static)
        """
        # see: https://math.stackexchange.com/questions/106700/incremental-averageing
        vec = coo - coo[self.ir]
        n = self.count + 1
        if weights is not None:
            weights = (weights / weights.max() / n)[:, None]
        else:
            weights = 1. / n

        ix = self.use_labels - 1
        inc = (vec - self.rvec[ix]) * weights
        self.logger.debug('rvec increment:\n%s', str(inc))
        self.rvec[ix] += inc

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
            too_faint = self.too_faint(image, snr_cut)
            if len(too_faint):
                self.logger.debug(msg, str(too_faint), 'faint')

        ignore = ftl.reduce(np.union1d, (too_bright, too_close, too_faint))
        ix = np.setdiff1d(np.arange(len(self.yx0)), ignore)
        if len(ix) == 0:
            self.logger.warning('No suitable stars found for tracking!')
        return ix

    # def auto_window(self):
    #     sdist_b = self.sdist[snr > self._snr_thresh]

    def too_faint(self, image, threshold=snr_cut):
        crude_snr = self.segm.snr(image)
        return np.where(crude_snr < threshold)[0]

    def too_close(self, threshold=_distance_cut):
        # Check for potential interference problems from stars that are close
        #  together
        return np.unique(np.ma.where(self.sdist() < threshold))

    def too_bright(self, data, saturation, threshold=_saturation_cut):
        # Check for saturated stars by flagging pixels withing 1% of saturation
        # level
        # TODO: make exact
        lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        # TODO check if you can improve speed here - dont have to check entire
        # array?

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
