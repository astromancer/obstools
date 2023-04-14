"""
Image registration (point set registration) for astronomicall images.
"""

# Helper functions to infer World Coordinate System given a target name or
# coordinates of a object in the field. This is done by matching the image
# with the DSS image for the same field via image registration.  A number of
# methods are implemented for doing this:
#   coherent point drift
#   matching via locating dense cluster of displacements between points
#   direct image-to-image matching
#   brute force search with gaussian mixtures on points


# std
import re
import numbers
import warnings
import operator as op
import itertools as itt
from collections import abc

# third-party
import numpy as np
import aplpy as apl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D
from loguru import logger
from mpl_multitab import MplTabs
from joblib import Parallel, delayed
from astropy.utils import lazyproperty
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from scipy.stats import binned_statistic_2d, mode
from scipy.interpolate import NearestNDInterpolator

# local
import recipes.pprint as pp
from recipes.string import indent
from recipes.functionals import echo0
from recipes.logging import LoggingMixin
from recipes.lists import cosort, split_like
from recipes.utils import duplicate_if_scalar

# relative
from .. import transforms as tf
from ..campaign import ImageHDU
from ..stats import geometric_median
from ..modelling import UnconvergedOptimization
from ..utils import STScIServerError, get_coordinates, get_dss
from .utils import non_masked
from .mosaic import MosaicPlotter
from .gmm import CoherentPointDrift
from .segmentation import SegmentedImage
from .image import ImageContainer, SkyImage


# ---------------------------------------------------------------------------- #
TABLE_STYLE = dict(txt=('bold', 'underline'), bg='g')

# ---------------------------------------------------------------------------- #
# defaults
HOP = True
REFINE = True
SAMPLE_STAT = 'median'
DEPTH = 5
PLOT = False


# ---------------------------------------------------------------------------- #


def normalize_image(image, centre=np.ma.median, scale=np.ma.std):
    """Recenter and scale"""
    image = image - centre(image)
    if scale:
        return image / scale(image)
    return image


def objective_pix(target, values, xy, bins, p):
    """Objective for direct image matching"""
    xy = tf.rigid(xy, p)
    bs = binned_statistic_2d(*xy.T, values, 'mean', bins)
    rs = np.square(target - bs.statistic)
    return np.nansum(rs)


def interpolate_nn(a, where):
    good = ~where
    grid = np.moveaxis(np.indices(a.shape), 0, -1)
    nn = NearestNDInterpolator(grid[good], a[good])
    a[where] = nn(grid[where])
    return a


# def fit_pixels(self, image, fov, p0):
#     """Match pixels directly"""
#     sy, sx = self.data.shape
#     dx, dy = self.fov
#     hx, hy = 0.5 * self.scale
#     by, bx = np.ogrid[-hx:(dx + hx):complex(sy + 1),
#                       -hy:(dy + hy):complex(sx + 1)]
#     bins = by.ravel(), bx.ravel()
#
#     sy, sx = image.shape
#     dx, dy = fov
#     yx = np.mgrid[0:dx:complex(sy), 0:dy:complex(sx)].reshape(2, -1).T  # [::-1]
#
#     target = normalize_image(self.data)
#     values = normalize_image(image).ravel()
#     result = minimize(ftl.partial(objective_pix, target, values, yx, bins),
#                       p0)
#     return result


def dist_flat(coo):
    """lower triangle of (symmetric) distance matrix"""
    # n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between sources
    # since the distance matrix is symmetric, ignore lower half
    return sdist[np.tril_indices(len(coo), -1)]


def id_sources_kmeans(images, segmentations):
    # combine segmentation for sample images into global segmentation
    # this gives better overall detection probability and yields more accurate
    # optimization results

    # this function also uses kmeans clustering to id sources within the overall
    # constellation of sources across sample images.  This is fast but will
    # mis-identify sources if the size of camera dither between frames is on the
    # order of the distance between sources in the image.

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
    # nsources = list(map(len, coms))
    # k = np.max(nsources)
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


def plot_clusters(ax, features, labels, colours=None, cmap=None, nrs=False,
                  **scatter_kws):
    """

    Parameters
    ----------
    ax
    labels
    features
    cmap

    Returns
    -------

    """

    ok = (labels != -1)
    core_sample_indices_, = np.where(ok)
    n = labels[core_sample_indices_].max() + 1

    # plot
    if colours is None:
        if cmap is None:
            from photutils.utils.colormaps import make_random_cmap
            cmap = make_random_cmap(n)
            # this cmap has good distinction between colours for clusters nearby
            # each other
        else:
            from matplotlib.cm import get_cmap
            cmap = get_cmap(cmap)

        colours = cmap(np.linspace(0, 1, n))[labels[core_sample_indices_]]
        # colours = labels[core_sample_indices_]

    # fig, ax = plt.subplots(figsize=im.figure.get_size_inches())
    dflt = dict(s=25, marker='*')
    scatter_kws = {**dflt, **scatter_kws}
    if scatter_kws.get('marker') in '.o*':
        scatter_kws.setdefault('edgecolors', colours)
        scatter_kws.setdefault('facecolors', 'none')
    else:
        scatter_kws.setdefault('facecolors', colours)
        scatter_kws.setdefault('edgecolors', 'face')

    art = ax.scatter(*features[core_sample_indices_].T,
                     **{**dflt, **scatter_kws})

    ax.plot(*features[~ok].T, 'rx', alpha=0.5)

    if nrs:
        from scrawl.utils import emboss

        o = 3
        # alpha = 0.75
        for lbl, i in zip(*np.unique(labels[ok], return_index=True)):
            xy = features[labels == lbl]

            # print(i, xy, colours[i])
            nr = ax.annotate(f'${lbl}$', xy.mean(0), (o, o),
                             textcoords='offset points', size=8,
                             color=colours[i])

            # nr, = ax.plot(*(xy.max(0) + offset), marker=f'${i}$', ms=7,
            #               color=colours[i])
            emboss(nr, 1.5, color='0.2')

    ax.grid()
    return art


# def _sanitize_positions(xy):
#     n, n_sources, _ = xy.shape
#     nans = np.isnan(np.ma.getdata(xy))
#     bad = (nans | np.ma.getmask(xy)).any(-1)
#     ignore_frames = bad.all(-1)
#     n_ignore = ignore_frames.sum()
#
#     assert n_ignore != n
#     if n_ignore:
#         logger.info(
#                 'Ignoring %i/%i ({:.1%}) nan / masked values for position '
#                 'measurement', n_ignore, n, (n_ignore / n) * 100)
#
#     # mask nans.  masked
#     return np.ma.MaskedArray(xy, nans)

# estimate_source_locations / estimate_source_positions / measure_position_dither
def compute_centres_offsets(xy, d_cut=None, detect_freq_min=0.9, report=True):
    """
    Measure the relative positions of detected sources from the individual
    location measurements in xy. Use the locations of the most-often
    detected individual objects.

    Parameters
    ----------
    xy:     array, shape (n_points, n_sources, 2)
    d_cut:  float
        distance cutoff 
    detect_freq_min: float
        Required detection frequency of individual sources in order for
        them to be used

    Returns
    -------
    xy, centres, σxy, δxy, outlier_indices
    """
    assert 0 < detect_freq_min < 1

    n, n_sources, _ = xy.shape
    nans = np.isnan(np.ma.getdata(xy))

    # mask nans.  masked
    xy = np.ma.MaskedArray(xy, nans)

    # Due to varying image quality and or camera/telescope drift,
    # some  sources (those near edges of the frame, or variable ones) may
    # not be detected in many frames. Cluster centroids are not an accurate
    # estimator of relative position for these sources since it's an
    # incomplete sample. Only sources that are detected in at least
    # `detect_freq_min` fraction of the frames will be used to calculate
    # frame xy offset.

    # Any measure of centrality for cluster centers is only a good estimator
    # of the relative positions of sources when the camera offsets are
    # taken into account.

    bad = (nans | np.ma.getmask(xy)).any(-1)
    ignore_frames = nans.all((1, 2))
    n_ignore = ignore_frames.sum()
    n_use = n - n_ignore
    if n_ignore == n:
        raise ValueError('All points are masked!')

    if n_ignore:
        logger.info('Ignoring {:d}/{:d} ({:.1%}) nan values in position '
                    'measurements.', n_ignore, n, n_ignore / n)

    n_detections_per_source = np.zeros(n_sources, int)
    w = np.where(~bad)[1]
    u = np.unique(w)
    n_detections_per_source[u] = np.bincount(w)[u]

    f_det = (n_detections_per_source / n_use)
    use_sources = f_det > detect_freq_min
    i_use, = np.where(use_sources)
    if not len(i_use):
        raise ValueError(
            'Detected frequency for all sources appears to be too low. There '
            'are {n_sources} objects across {n} images. Their detection '
            'frequencies are: {fdet}.'
        )

    if np.any(~use_sources):
        logger.info('Ignoring {:d}/{:d} sources with low (<={:.0%}) detection '
                    'frequency for frame shift measurement.',
                    n_sources - len(i_use), n_sources, detect_freq_min)

    # NOTE: we actually want to know where the cluster centres would be
    #  without a specific point to measure delta better

    # first estimate of relative positions comes from unshifted cluster centers
    # Compute cluster centres as geometric median
    good = ~ignore_frames
    xyc = xy[good][:, use_sources]
    nansc = nans[good][:, use_sources]
    if nansc.any():
        # prevent warning emit in _measure_positions_offsets
        xyc[nansc] = 0
        xyc[nansc] = np.ma.masked

    # delay centre compute for fainter sources until after re-centering
    # np.empty((n_sources, 2))  # ma.masked_all
    centres = δxy = np.ma.masked_all((n_sources, 2))
    for i, j in enumerate(i_use):
        centres[j] = geometric_median(xyc[:, i])

    # ensure output same size as input
    δxy = np.ma.masked_all((n, 2))
    σxy = np.empty((n_sources, 2))
    # σxy = np.ma.masked_all((n_sources, 2))

    # compute positions of all sources with frame offsets measured from best
    # and brightest sources
    centres[use_sources], σxy[use_sources], δxy[good], out = \
        _measure_positions_offsets(xyc, centres[use_sources], d_cut)
    #
    for i in np.where(~use_sources)[0]:
        # mask for bad frames in δxy will propagate here
        recentred = xy[:, i].squeeze() - δxy
        centres[i] = geometric_median(recentred)
        σxy[i] = recentred.std()

    # fix outlier indices
    idxf, idxs = np.where(out)
    idxg, = np.where(good)
    idxu, = np.where(use_sources)
    outlier_indices = (idxg[idxf], idxu[idxs])
    xy[outlier_indices] = np.ma.masked

    # pprint!
    if report:
        try:
            #                                          counts
            report_measurements(xy, centres, σxy, δxy, None, detect_freq_min)

        except Exception as err:
            logger.exception('Report failed')

    return xy, centres, σxy, δxy, outlier_indices


def _measure_positions_offsets(xy, centres, d_cut=None, centroid=geometric_median):

    # ensure we have at least some centres
    assert not np.all(np.ma.getmask(centres))

    n, n_sources, _ = xy.shape
    n_points = n * n_sources

    outliers = np.zeros(xy.shape[:-1], bool)
    xym = np.ma.MaskedArray(xy, copy=True)

    counter = itt.count()
    while True:
        count = next(counter)
        if count >= 5:
            raise ValueError('Emergency stop!')

        # xy position offset in each frame
        xy_offsets = (xym - centres).mean(1, keepdims=True)

        # shifted cluster centers (all sources)
        xy_shifted = xym - xy_offsets
        # Compute cluster centres as geometric median of shifted clusters
        centres = np.ma.empty((n_sources, 2))
        for i in range(n_sources):
            centres[i] = centroid(xy_shifted[:, i])

        if d_cut is None:
            # break out here without removing any points
            return centres, xy_shifted.std(0), xy_offsets.squeeze(), outliers

        # FIXME: break out if there are too few points for the concept of
        #  "outlier" to be meaningful

        # compute position residuals
        cxr = xy - centres - xy_offsets

        # flag outliers
        with warnings.catch_warnings():
            # catch RuntimeWarning for masked elements
            warnings.filterwarnings('ignore')
            d = np.sqrt((cxr * cxr).sum(-1))

        out_new = (d > d_cut)
        out_new = np.ma.getdata(out_new) | np.ma.getmask(out_new)

        changed = (outliers != out_new).any()
        if not changed:
            break

        out = out_new
        xym[out] = np.ma.masked
        n_out = out.sum()

        if n_out / n_points > 0.5:
            raise ValueError('Too many outliers!!')

        logger.info('Ignoring {:d}/{:d} ({:.1%}) values with |δr| > {:.3f}',
                    n_out, n_points, (n_out / n_points), d_cut)

    return centres, xy_shifted.std(0), xy_offsets.squeeze(), outliers


def group_features(labels, *features):
    """
    Read multiple sets of features into a uniform array where the first
    dimension selects the class to which those features belong. Each data set is
    returned as a masked array where missing elements have been masked out to
    match the size of the largest class. The returned feature array is easier to
    work with than lists of unevenly sized features for further analysis.

    Parameters
    ----------
    labels 
    features

    Returns
    -------

    """
    # labels = classifier.labels_
    # core_samples_mask = np.zeros(len(labels), dtype=bool)
    # core_samples_mask[classifier.core_sample_indices_] = True

    assert np.size(labels)
    assert features

    #
    unique_labels = list(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)
    zero = unique_labels[0]
    n_clusters = len(unique_labels)
    n_samples = len(features[0])

    # init arrays, one per feature
    grouped = []
    for data in next(zip(*features)):
        shape = (n_samples, n_clusters)
        if data.ndim > 1:
            shape += (data.shape[-1], )

        g = np.ma.empty(shape)
        g.mask = True
        grouped.append(g)

    # group
    for i, j in enumerate(split_like(labels, features[0])):
        ok = (j != -1)
        if ~ok.any():
            # catch for data[i] is empty (eg. no source detections)
            continue

        jj = j[ok] - zero
        for f, g in zip(features, grouped):
            g[i, jj] = f[i][ok, ...]

    return tuple(grouped)


def report_measurements(xy, centres, σ_xy, xy_offsets=None, counts=None,
                        detect_frac_min=None, count_thresh=None, logger=logger):
    # report on relative position measurement
    import operator as op
    import math

    from recipes import pprint
    from motley.table import Table
    from motley.formatters import Decimal, Conditional, Numeric
    # from obstools.stats import mad
    # TODO: probably mask nans....

    #

    n_points, n_sources, _ = xy.shape
    n_points_tot = n_points * n_sources

    if np.ma.is_masked(xy):
        bad = xy.mask.any(-1)
        good = np.logical_not(bad)
        points_per_source = good.sum(0)
        sources_per_image = good.sum(1)
        n_bad = bad.sum()
        no_detection, = np.where(np.equal(sources_per_image, 0))
        if len(no_detection):
            logger.debug('There are no sources in frames: {!s}', no_detection)

        if n_bad:
            extra = (f'\nn_masked = {n_bad}/{n_points_tot} '
                     f'({n_bad / n_points_tot :.1%})')
    else:
        points_per_source = np.tile(n_points, n_sources)
        extra = ''

    # check if we managed to reduce the variance
    # sigma0 = xy.std(0)
    # var_reduc = (sigma0 - σ_xy) / sigma0
    # mad0 = mad(xy, axis=0)
    # mad1 = mad((xy - xy_offsets[:, None]), axis=0)
    # mad_reduc = (mad0 - mad1) / mad0

    # # overall variance report
    # s0 = xy.std((0, 1))
    # s1 = (xy - xy_offsets[:, None]).std((0, 1))
    # # Fractional variance change
    # logger.info('Differencing change overall variance by {!r:}',
    #             np.array2string((s0 - s1) / s0, precision=3))

    # FIXME: percentage format in total wrong
    # TODO: align +- values
    col_headers = ['x', 'y', 'n']  # '(px.)'
    n_min = (detect_frac_min or -math.inf) * n_points
    formatters = {
        'n': Conditional('y', op.le, n_min,
                         Decimal.as_percentage_of(total=n_points,
                                                  precision=0))}

    # get array with number ± std representations
    columns = [pprint.uarray(centres, σ_xy, 2)[:, ::-1], points_per_source]
    # FIXME: don't print uncertainties if less than 6 measurement points

    # x, y = pprint.uarray(centres, σ_xy, 2)
    # columns = {
    #     'x': Column(x, unit='pixels'),
    #     'y': Column(y, unit='pixels'),
    #     'n': Column(points_per_source,
    #                 fmt=Conditional('y', op.le, n_min,
    #                                     Decimal.as_percentage_of(total=n_points,
    #                                                                  precision=0)),
    #                 total='{}')
    # }

    if counts is not None:
        # columns['counts'] = Column(
        #     counts, unit='ADU',
        #     fmt=Conditional('y', op.le, n_min,
        #                         Percentage(total=n_points,
        #                                        precision=0))
        #     )

        formatters['counts'] = Conditional(
            'c', op.ge, (count_thresh or math.inf),
            Numeric(thousands=' ', precision=1, shorten=False))
        columns.append(counts)
        col_headers += ['counts']

    # add variance columns
    # col_headers += ['r_σx', 'r_σy'] + ['mr_σx', 'mr_σy']
    # columns += [var_reduc[:, ::-1], mad_reduc[:, ::-1]]

    #
    tbl = Table.from_columns(*columns,
                             title='Measured source locations',
                             title_style=TABLE_STYLE,
                             units=['pixels', 'pixels', ''],
                             col_headers=col_headers,
                             col_head_style=TABLE_STYLE,
                             col_head_align='^',
                             precision=3,
                             align='r',
                             row_nrs=True,
                             totals=[-1],
                             formatters=formatters,
                             max_rows=25)

    # fix formatting with percentage in total.
    # TODO Still need to think of a cleaner solution for this
    tbl.data[-1, 0] = re.sub(r'\(\d{3,4}%\)', '', tbl.data[-1, 0])
    # tbl.data[-1, 0] = tbl.data[-1, 0].replace('(1000%)', '')

    logger.info('\n{:s}{:s}', tbl, extra)

    return tbl


class ImageRegister(ImageContainer, LoggingMixin):
    """
    A class for image registration and basic astrometry of astronomical images.
    Typical usage pattern is as follows:

    >>> reg = ImageRegister()
    # add a reference image
    >>> reg(image, fov)
    # match a new (partially overlapping) image to the reference image:
    >>> *xy_offset, rotation = reg(new_image, new_fov)
    # cross identify sources across images
    >>> reg.register()
    # plot a mosaic of the overlapping, images
    >>> mos = reg.mosaic()

    Alternatively, if you have a set of images:
    >>> reg = ImageRegister.from_images(images, fovs)
    # this will pick the highest resolution image as the reference image of
    # choice

    Internally, positions of sources are measured as centre-of-mass. The
    position coordinates of all measurements are in units of pixels of the
    reference image.

    """

    # TODO: say something about the matching algorithm and that it's lightning
    #  fast for nearly_aligned images with few sources

    # TODO: move to config
    find_kws = dict(snr=3.,
                    npixels=5,
                    edge_cutoff=2,
                    deblend=False)
    """default source finding keywords"""

    refining = True
    """Refine fitting by running gradient descent after basin-hopping search."""

    # Expansion factor for grid search
    # _search_area_stretch = 1.25

    # Sigma for GMM as a fraction of minimal distance between sources
    _dmin_frac_sigma = 3
    _sigma_fallback = 10  # TODO: move to config

    # TODO: switch for units to arcminutes or whatever ??
    # TODO: uncertainty on center of mass from pixel noise!!!

    @classmethod
    def from_hdus(cls, run, sample_stat=SAMPLE_STAT, depth=DEPTH, primary=None,
                  fit_angle=True, **kws):
        # get sample images etc
        # `from_hdu` is used since the sample image and source detection results
        # are cached (persistantly), so this should be fast on repeated calls.
        images = [SkyImage.from_hdu(hdu, sample_stat, depth,
                                    **{**cls.find_kws, **kws})
                  for hdu in run]
        return cls(images, primary=primary, fit_angle=fit_angle, **kws)

    # @classmethod                # p0 -----------
    # def from_images(cls, images, fovs, p0=(0,0,0), primary=None, plot=PLOT,
    #                 fit_angle=True, **kws):

    #     # initialize workers
    #     with Parallel(n_jobs=1) as parallel:
    #         # detect sources
    #         images = parallel(
    #             delayed(SkyImage.from_image)(image, fov, **kws)
    #             for image, fov in zip(images, fovs)
    #         )

    #     return cls._from_images(images, fovs, angles=0, primary=None,
    #                             plot=PLOT, fit_angle=True, **kws)

    # @classmethod                # p0 -----------
    # def _from_images(cls, images, fovs, angles=0, primary=None, plot=PLOT,
    #                  fit_angle=True, **kws):

    #     n = len(images)
    #     assert 0 < n == len(fovs)

    #     # message
    #     logger.info('Aligning {:d} images on image {:d}.', n, primary)

    #     # initialize and fit
    #     return cls(images,
    #                primary=primary,
    #                fit_angle=fit_angle).fit(plot=plot)

    def __init__(self, images=(), fovs=(), params=(), fit_angle=True,
                 primary=None, **kws):
        """
        Initialize an image register. This class should generally be initialized
        without arguments. The model of the constellation of sources is built
        iteratively by calling an instance of this class on an image. eg:
        >>> ImageRegister()(image)

        If arguments are provided, they are the sequences of images,
        their field-of-views, and transformation parameters for the
        initial (zero-point) transform.


        Parameters
        ----------
        images:
        fovs:
            Field of view for the image. Order of the dimensions should be
            the same as that of the image, ie. rows first
        params:
        find_kws:
        """
        # TODO: init from single image???

        if kws:
            self.find_kws.update(kws)

        # init container
        ImageContainer.__init__(self, images, fovs)

        # NOTE passing a reference index `primary` is only meaningful if the
        #  class is initialized with a set of images
        self._idx = 0
        self.count = 0
        if primary is None:
            if len(images):
                # align on image with highest source density if not specified
                source_density = self.scales.mean(1) * self.attrs('seg.nlabels')
                primary = source_density.argmax(0)
            else:
                primary = 0

        elif len(images):
            if (self.attrs('seg.nlabels')[primary]) < 2:
                warnings.warn('Primary image {primary} contains too few sources.')

        self.fit_angle = bool(fit_angle)
        self._xy = None
        self.primary = int(primary)
        self.target_coords_pixels = None
        self.sigmas = None
        # self._xy = None
        # keep track of minimal separation between sources
        # self._min_dist = np.inf

        # self.grids = TransformedImageGrids(self)
        # self._sigma_guess = self.guess_sigma(self.xy)

        # state variable placeholders
        self.labels = None
        self._colour_sequence_cache = ()

    def __repr__(self):
        return (f'{super().__repr__()}; '
                + ('unregistered' if self.labels is None else
                   f'{self.labels.max()} sources'))

    @property
    def dof(self):
        return 2 + self.fit_angle

    @property
    def image(self):
        return self.images[self.primary]

    @property
    def fov(self):
        return self.fovs[self.primary]

    @property
    def scale(self):
        return self.scales[self.primary]

    pixel_scale = scale

    @property
    def rscale(self):
        return self.scales / self.pixel_scale

    @property
    def primary(self):
        return self._idx

    @primary.setter
    def primary(self, idx):
        idx = int(idx)
        n = len(self)
        # wrap index
        if idx < 0:
            idx += n

        if idx > n:
            raise ValueError(f'Invalid index ({n}) for `primary` reference '
                             f'image.')

        if self._idx == idx:
            return

        self._idx = idx
        del self.order

        if self._xy is None:
            return

        params = self.params
        rscale = self.rscale[idx]
        self.xy = (self.xy - params[idx, :2]) / rscale
        params[:, :2] /= rscale
        params -= params[idx]

    @lazyproperty
    def order(self):
        pri = self.primary
        return np.r_[pri, np.delete(np.arange(len(self)), pri)]

    @property
    def xy(self):
        """
        The reference coordinates for model constellation in units of
        arcminutes. These will be updated when more images are added to the
        register and we can make a more accurate determination of the source
        positions by the :meth:`recentre` and :meth:`refine` methods.
        """
        if self._xy is None:
            # Make initial reference coordinates.
            if not self:
                raise ValueError('No images available. Add images by calling '
                                 'the `ImageRegister` object on an image.')

            # `xy` are the initial target coordinates for matching. Theses will
            # be updated in `recentre` and `refine`
            self.xy = self[self.primary].xy

        return self._xy

    @xy.setter
    def xy(self, xy):
        # Ensure xy coordinates are always a plain numpy array. Masked arrays
        # don't seem to work well with the matrix product `@` operator used in
        # :func:`rigid`
        self._xy = non_masked(xy)
        del self.model

    # @lazyproperty
    @property
    def xyt(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters
        """
        return list(map(tf.affine, self.coms, self.params, self.rscale))

    @property
    def n_points(self):
        return sum(map(len, self.coms))

    # @property
    def n_sources(self):
        self.check_has_labels()
        return len(set(self.labels) - {-1})

    def n_noise(self):
        self.check_has_labels()
        return list(self.labels).count(-1)

    # @lazyproperty
    @property
    def xyt_block(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters and group them according to the cluster identified labels
        """
        return group_features(self.labels, self.xyt)[0]

    @property
    def min_dist(self):
        if len(self.xy) <= 1:
            raise ValueError('Too few (< 2) sources in primary image frame.')

        return dist_flat(self.xy).min()

    # def min_dist(self):
    #     return min(dist_flat(xy).min() for xy in self.xyt)

    def guess_sigma(self):
        """choose sigma for GMM based on distances between detections"""
        if len(self.xy) <= 1:
            logger.warning('Too few (< 2) sources in primary image frame to '
                           'initialize GMM with an informed default. Using '
                           'fallback value: {}', self._sigma_fallback)
            return self._sigma_fallback

        return self.min_dist / self._dmin_frac_sigma

    @lazyproperty
    def sigma(self):
        return self.guess_sigma()

    @lazyproperty
    def model(self):
        model = CoherentPointDrift(self.xy, self.sigma)
        model.fit_angle = self.fit_angle
        return model

    def __call__(self, obj, *args, **kws):
        if (obj is None) and (self.count == 0):
            obj = self.data

        result = self.fit(obj, *args, **kws)

        if (obj is None):
            # refit => don't aggregate
            return

        # aggregate
        collect = (self.append, self.extend)[isinstance(result, abc.Collection)]
        collect(result)
        # print(result)

        # reset model.
        del self.model

        # self._sigma_guess = min(self._sigma_guess, self.guess_sigma(xy))
        # logger.debug('sigma guess: {:s}', self._sigma_guess)

        # update minimal source seperation
        # self._min_dist = min(self._min_dist, dist_flat(xy).min())

    def check_has_labels(self):
        if self.labels is None:
            raise ValueError(
                'No cluster labels available. Run `register` to fit '
                'clustering model to the measured centre-of-mass points.'
            )

    def check_has_data(self):
        if len(self):
            return

        raise ValueError('No data. Please add some images first. eg: '
                         '{self.__class__.__name__}()(image, fov)')

    # TODO: refine_mcmc
    def fit(self, obj=None, p0=None, hop=HOP, refine=None, plot=PLOT, **kws):
        """
        Flexible fitting method that dispatches fitting method based on the
        type of `obj`, and aggregates results. If this is the first time
        calling this method, the reference image is set.

        Parameters
        ----------
        obj
        fov
        p0
        plot

        Returns
        -------
        params : np.ndarray
            Fitted transform parameters (Δx, Δy, θ); the xy offsets in pixels,
            rotation angle in radians.
        """

        # dispatch fit
        refine = refine or self.refining
        fitters = {type(None):              self._fit_internal,
                   ImageRegister:           self.fit_register,
                   (np.ndarray, SkyImage):  self.fit_image,
                   abc.Collection:          self.fit_sequence,
                   ImageHDU:                self.fit_hdu}
        for types, fit in fitters.items():
            if isinstance(obj, types):
                break
        else:
            raise TypeError(f'Cannot fit object of type {type(obj)}.')

        self.logger.opt(lazy=True).debug(
            'Fitting:{!s}',
            lambda: indent(
                f'\n{pp.caller(fit, (type(obj), p0, hop, refine, plot), kws)}'
            )
        )
        result = fit(obj, p0, hop, refine, plot, **kws)
        self.count += 1
        return result

    def _fit_internal(self, obj, p0, hop, refine, plot, **kws):
        # this method for api consistency

        assert obj is None
        self.check_has_data()

        # (re)fit all images
        # i = int(isinstance(self, ImageRegisterDSS)) and (self._xy is not None)
        hop = (self._xy is None)
        refine = refine or not hop
        obj = self[self.order]
        p0 = self.params[self.order]

        return self.fit_sequence(obj, p0, hop, refine, plot, **kws)

        # i = int(isinstance(self, ImageRegisterDSS)) and (self._xy is not None)
        # hop = (self._xy is None)
        # refine = self.count and not hop
        # obj = self[self.order[i:]]
        # p0 = self.params[self.order[i:]] if p0 is None else p0

        # self[i:] = self.fit_sequence(obj, p0, hop, refine, plot, **kws)
        # return self

    def fit_register(self, reg, p0=None, hop=HOP, refine=REFINE, plot=PLOT, **kws):
        """
        cross match with another `ImageRegister` and aggregate points
        """

        # convert coordinates to pixel coordinates of reference image
        reg.check_has_data()
        xy, params = reg.convert_to_pixels_of(self)

        # use `fit_points` so we don't aggregate images just yet
        p = self.fit_points(xy, p0, hop, refine, plot)

        # finally aggregate the results from the new register
        reg.params = params + p
        self.extend(reg.data)

        # convert back to original coords
        # reg.convert_to_pixels_of(reg)

        # fig, ax = plt.subplots()
        # for xy in reg.xyt:
        #     ax.plot(*xy.T, 'x')
        # ax.plot(*reg.xy.T, 'o', mfc='none', ms=8)

    def fit_sequence(self, items, p0=None, hop=HOP, refine=REFINE,
                     plot=PLOT, njobs=1, **kws):
        #
        assert len(items)

        if p0 is None:
            p0 = ()
        else:
            p0 = np.array(p0)
            assert p0.shape == (len(items), self.dof)

        # run fitting concurrently
        # with Parallel(n_jobs=njobs) as parallel:
        #     images = parallel(
        #         delayed(self.fit)(image, p00, hop, refine, plot, **kws)
        #         for image, p00 in itt.zip_longest(items, p0)
        #     )

        return [self.fit(image, p0, hop, refine, plot, **kws)
                for image, p0 in itt.zip_longest(items, p0)]

        # images.insert(primary, self[primary])
        # self.data[:] = images

        # fit clusters to points
        # self.register()
        # return images

    def fit_hdu(self, hdu, p0=None, hop=HOP, refine=REFINE,
                plot=PLOT, sample_stat=SAMPLE_STAT, depth=DEPTH, **kws):
        """

        Parameters
        ----------
        hdu
        depth
        sample_stat

        Returns
        -------

        """
        return self.fit_image(
            SkyImage.from_hdu(hdu, sample_stat, depth,
                              **{**self.find_kws, **kws}),
            p0, None, hop, refine, plot
        )

    def fit_image(self, image, p0=None, hop=HOP, refine=REFINE,
                  plot=PLOT, fov=None, **kws):
        """
        If p0 is None:
            Search heuristic for image offset and rotation.
        else:


        Parameters
        ----------
        image
        p0: 
            (δy, δx, θ)
        fov
        rotation

        Returns
        -------
        p: parameter values (δy, δx, θ)
        yx: center of mass coordinates
        seg: SegmentationImage
        """

        # p0 = () if p0 is None else p0
        # p0 = *xy0, angle = p0

        image = SkyImage(image, fov)
        if image.xy is None:
            # source detection
            image.detect(**{**self.find_kws, **kws})

        if self.count:
            # get xy in units of `primary` image pixels
            xy = image.xy * image.scale / self.pixel_scale
            image.params = self.fit_points(xy, p0, hop, refine, plot)

        return image

    # @timer
    def fit_points(self, xy, p0=None, hop=HOP, refine=REFINE, plot=PLOT):

        if p0 is None:
            p0 = np.zeros(self.dof)

        if hop:
            # get shift (translation) via brute force search through point to
            # point displacements. This is gauranteed to find the optimal
            # alignment between the two point clouds in the absense of rotation
            # between them

            # get coords.
            xyr = tf.rigid(xy, p0)
            dxy = self._dxy_hop(xyr, plot)
            p = np.add(p0, [*dxy, 0])

        if (not refine) and hop:
            return p

        if len(xy) == 1:
            self.logger.info('Ignoring request to refine fit since the '
                             'image contains only a single source.')
            return p

        self.logger.debug('Refining fit via gradient descent.')
        # do mle via gradient decent
        self.model.fit_angle = self.fit_angle
        pr = self.model.fit(xy, p0=p if hop else p0)

        if (pr is not None):
            return pr

        if hop:
            self.logger.debug('Gradient descent failed, falling back to previous result.')
            return p

        raise UnconvergedOptimization()

    def _dxy_hop(self, xy, plot=PLOT):
        # This is a strategic brute force search along all the offset values
        # that will align pairs of points in the two fields for a known
        # rotation. One of the test offsets is the true offset value. Each
        # offset value represents a local basin in the parameter space, we are
        # checking which one is deepest. This algorithm is N^2 with number of
        # points, so will not be appropriate for dense fields with many points,
        # but tends to be faster than a more general basin-hopping heuristic
        # (such as `scipy.optimize.basin_hopping`) for moderate to low number of
        # points, and more robust than other gradient decent methods since it
        # avoids getting stuck in a local minimum. This search heuristic is only
        # effective when the fields have roughly the same rotation.

        # create search grid.
        trials = (self.xy[None] - xy[:, None]).reshape(-1, 2)
        # Ignore extremal points in grid search.  These represent single
        # point matches at the edges of the frame which are almost certainly
        # not the best match
        # extrema = np.ravel([trials.argmin(0), trials.argmax(0)])
        # trials = np.delete(trials, extrema, 0)

        # This animates the search!
        # line, = im.ax.plot((), (), 'rx')
        # for i, po in enumerate(points.T):
        #     line.set_data(*(xyr - po).T)
        #     im.figure.canvas.draw()
        #     input(i)
        #     if i > 25:
        #          break

        # find minimum
        state = self.model.fit_angle
        self.model.fit_angle = False

        # parallelize
        # r = gridsearch_mp(self.model.loss_mle, trials.T, (xy, ))
        r = [self.model.loss_mle(p, xy) for p in trials]
        p = trials[np.argmin(r)]

        logger.debug('Grid search optimum: {!s}', p)

        if plot:
            im = self.model.gmm.plot(show_peak=False)
            im.ax.plot(*(xy + p).T, 'ro', ms=12, mfc='none')

        # restore sigma
        self.model.fit_angle = state
        return np.array(p)

    # TODO: relative brightness

    def refine(self, fit_angle=True, plot=PLOT):
        """
        Refine alignment parameters by fitting transform parameters for each
        image using maximum likelihood objective for gmm model with peaks
        situated at cluster centers

        """
        if self.labels is None:
            self.register()

        params = self.params
        # guess sigma:  needs to be larger than for brute search `_dxy_hop`
        # since we are searching a smaller region of parameter space
        # self.model.sigma = dist_flat(self.xy).min() / 3

        failed = []
        # TODO: multiprocess
        self.model.fit_angle = fit_angle

        n_jobs = 1  # make sure you can pickle everything before you change
        # this value
        with Parallel(n_jobs=n_jobs) as parallel:
            for i, p in enumerate(parallel(
                    delayed(self.model.fit)(xy, p0=(0, 0, 0))
                    for xy in self.xyt)):
                if p is None:
                    failed.append(i)
                else:
                    params[i] += p

        # for i in range(len(self.images)):
        #     p = self.model.fit(xyt[i], p0=(0, 0, 0))
        #     if p is None:
        #         failed.append(i)
        #     else:
        #         params[i] += p
        logger.log(('SUCCESS', 'INFO')[bool(failed)],
                   'Fitting successful {:d} / {:d}', i - len(failed), i)

        # likelihood ratio test
        xyn = list(map(tf.affine, self.coms, params, self.rscale))
        lhr = self.lh_ratio(np.vstack(self.xyt), np.vstack(xyn))
        better = (lhr > 1)
        # decide whether to accept new params!
        if better:
            # recompute cluster centers
            self.params = params
            self.update_centres()

        if plot:
            fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
            ax.set_title(f'Refined Positions (CoM) {len(self)} frames')
            plot_clusters(ax, np.vstack(self.xyt), self.labels,
                          self._colour_sequence_cache)
        return params, lhr

    def lh_ratio(self, xy0, xy1):
        ratio = self.model.lh_ratio(xy0, xy1)
        logger.info('Likelihood ratio: {:.5f}\n\t'
                    + ('Keeping same', 'Accepting new')[ratio > 1]
                    + ' parameters.', ratio)
        return ratio

    def register(self, clf=None, plot=PLOT):

        self.check_has_data()

        # clustering + relative position measurement
        clf = clf or self.clustering
        self.cluster_points(clf)
        # make the cluster centres the target constellation
        self.xy = self.xyt_block.mean(0)

        if plot:
            #
            art = self.plot_clusters(nrs=True, frames=True)

    # ------------------------------------------------------------------------ #

    @property
    def labels_per_image(self):
        if self.labels is None:
            raise ValueError('Unregistered!')

        return split_like(self.labels, self.coms)

    @property
    def images_per_label(self):
        return [self._images_per_label(lbl)
                for lbl in sorted(set(self.labels) - {-1})]

    def _images_per_label(self, label):
        return [i for i, iml in enumerate(self.labels_per_image) if (label in iml)]

    @property
    def xy_per_label(self):
        xy = np.vstack(self.xyt)
        # np.unique(self.labels)[None].T == self.labels
        return [xy[self.labels == i] for i in sorted(set(self.labels) - {-1})]

    def _trim_labels(self):
        return []

    def remap_labels(self, target, flux_sort=False):
        """
        Re-order the *cluster* labels so that our target is 0, the rest follow
        in descending order of brightness first listing those that occur within
        our science images, then those in the survey image.
        """
        assert isinstance(target, numbers.Integral)

        # get the labels of sources that are detected in at least one of the
        # science frames
        _, nb = cosort(*np.unique(self.labels, return_counts=True)[::-1], order=-1)
        xb, = np.where(self._trim_labels())  # unimportant labels
        [nb.remove(r) for r in (target, -1, *xb) if r in nb]

        if flux_sort:
            # get cluster labels bright to faint
            counts, = group_features(self.labels, self.attrs.counts)

            # measure relative brightness (scale by one of the stars, to account for
            # gain differences between instrumental setups)
            nb = np.ma.median(counts[:, nb] / counts[:, [nb[0]]], 0).argsort()[::-1]

        # these are the new labels!
        return np.argsort([target, *nb, *xb])

    def relabel(self, target=None, flux_sort=False):
        # new desired segment labels
        #  +1 ensure non-zero labels for SegmentedImage
        new = self.remap_labels(target, flux_sort)
        labels = np.full_like(self.labels, -1)
        labels[self.labels != -1] = new[self.labels]
        self.labels = labels

        # relabel all segmentedImages for cross image consistency
        new_labels = []
        for cluster_labels, image in zip(self.labels_per_image, self):
            # relabel image segments
            image_labels = cluster_labels + 1
            reorder = ...
            if np.any(image.seg.labels != image_labels):
                image.seg.relabel_many(image_labels)
                # have to reorder the features
                use = (image_labels != 0)
                reorder = [*image_labels[use].argsort(), *np.where(~use)[0]]
                image.xy = image.xy[reorder]
                image.counts = image.counts[reorder]

            new_labels.extend(cluster_labels[reorder])

        self.labels = np.array(new_labels)
        return new_labels

    # def relabel_clusters(self):
        # match cluster labels to segment labels

    # def to_pixel_coords(self, xy):
    #     # internal coordinates are in arcmin origin at (0,0) for image
    #     return np.divide(xy, self.pixel_scale)

    def convert_to_pixels_of(self, reg):  # to_pixels_of
        # convert coordinates to pixel coordinates of reference image
        ratio = self.pixel_scale / reg.pixel_scale
        xy = self.xy * ratio

        params = self.params
        params[:, :2] *= ratio
        # NOTE: this does not edit image `params` since array is made
        # through the `params` property

        # note this means reg.mosaic will no longer work without fov keyword
        #  since we have changed the scale and it substitutes image shape for
        #  fov if not given

        return xy, params

    # ------------------------------------------------------------------------ #
    def get_centres(self, func=geometric_median):
        """
        Cluster centers via geometric median ignoring noise points

        Returns
        -------

        """
        # this ignores noise points from clustering
        centres = np.empty((self.n_sources(), 2))  # ma.masked_all
        for i, xy in enumerate(np.rollaxis(self.xyt_block, 1)):
            centres[i] = func(xy)

        return centres

    def update_centres(self):
        """
        Measure cluster centers and set those as updated target positions for
        the constellation model
        """
        self.xy = self.get_centres()

    # ------------------------------------------------------------------------ #
    @lazyproperty
    def clustering(self, *args, **kws):
        """
        Classifier for clustering source coordinates in order to cross identify
        sources.
        """
        from sklearn.cluster import MeanShift

        # choose bandwidth based on minimal distance between sources
        return MeanShift(**{**kws,
                            **dict(bandwidth=self.min_dist / 2,
                                   cluster_all=False)})

    def cluster_points(self, clf=None):
        """
        Run clustering algorithm on the set of position measurements in order to 
        cross identify sources.
        """
        # clustering to cross-identify sources
        self.check_has_data()
        if len(self) == 1:
            logger.warning('{} contains only one image. Using position '
                           'measurements from this image directly.',
                           self.__class__.__name__)
            self.xy = self[0].xy
            self.labels = np.arange(len(self.xy))
            return

        clf = clf or self.clustering

        X = np.vstack(self.xyt)
        n = len(X)
        # sources_per_image = list(map(len, xy))
        # no_detection = np.equal(sources_per_image, 0)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(np.vstack(xy))

        self.logger.info('Clustering {:d} position measurements to cross '
                         'identify sources using:{:s}', n, indent(f'\n{clf}'))
        #
        self.labels = clf.fit(X).labels_

        # core_samples_mask = (clf.labels_ != -1)
        # core_sample_indices_, = np.where(core_samples_mask)

        # Number of clusters in labels, ignoring noise if present.
        n_sources = self.n_sources()
        n_noise = self.n_noise()
        # n_per_label = np.bincount(db.labels_[core_sample_indices_])
        logger.info('Identified {:d} sources using {:d}/{:d} points ({:d} noise)',
                    n_sources, n - n_noise, n, n_noise)

        # sanity check
        n_sources_most = max(map(len, self.coms))
        if n_sources > n_sources_most * 1.5:
            self.logger.info(
                'Looks like we may be overfitting clusters. Image with most '
                'sources has {n_sources_most}, while clustering produced '
                '{n_sources} clusters. Maybe reduce bandwidth.'
            )

    # def check_labelled(self):
        # if not len(self):
        #     raise Exception('No data. Please add some images first. eg: '
        #                     '{self.__class__.__name__}()(image, fov)')

        # if self.labels is None:
        #     raise Exception('No clusters identified. Run '
        #                     '`register` to fit clustering model '
        #                     'to the measured centre-of-mass points')

    def recentre(self, centre_distance_cut=None, f_detect_measure=0.25,
                 plot=PLOT):
        """
        Measure frame dither, recenter, and recompute object positions.
        This re-centering algorithm can accurately measure the position of
        sources in the global frame taking frame-to-frame offsets into
        account. The xy offset of each image is measured from the mean
        xy-offset of the brightest sources from their respective cluster centres.
        Once all the frames have been shifted, we re-compute the cluster
        centers using the geometric median.  We check that the new positions
        yield clusters that are more tightly bound, (ie. offset transformation
        parameters are closer the the global minimum) by taking the likelihood
        ratio of the constellation model for before and after the shift.

        Parameters
        ----------
        centre_distance_cut
        f_detect_measure
        plot

        Returns
        -------

        """
        if self.labels is None:
            self.register()

        xy = self.xyt_block

        logger.info('Measuring cluster centres, frame xy-offsets')
        _, centres, xy_std, xy_offsets, outliers = \
            compute_centres_offsets(xy, centre_distance_cut, f_detect_measure)

        # decide whether to accept new params! likelihood ratio test
        lhr = self.lh_ratio(
            np.vstack(xy),
            np.vstack(xy - xy_offsets[:, None]))
        if lhr > 1:
            # some of the offsets may be masked. ignore those
            good = ~xy_offsets.mask.any(1)
            for i in np.where(good)[0]:
                self[i].origin -= xy_offsets[i]

            self.xy = centres
            self.sigmas = xy_std
            del self.model

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            # diagnostic for source location measurements
            from obstools.phot.diagnostics import plot_position_measures

            # plot CoMs
            fig1, axes = plot_position_measures(xy, centres, xy_offsets)
            # pixel_size=self.pixel_scale[0])
            fig1.suptitle('Raw Positions (CoM)')  # , fontweight='bold'

            #
            fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
            ax.set_title(f'Re-centred Positions (CoM) {len(self)} frames')
            plot_clusters(ax, np.vstack(self.xyt), self.labels,
                          self._colour_sequence_cache)

        return xy_offsets, outliers, xy

    def reset(self):
        """
        Reset image container to empty list

        Returns
        -------

        """

        self.data = []
        self.params = []

    def solve_rotation(self):
        # for point clouds that are already quite well aligned. Refine by
        # looking for rotation differences
        xy = self.xyt_block

        # get source that is in most of the frames (likely the target)
        i = (~xy.mask).any(-1).sum(-1).argmax()
        # idx = xy.std((0, -1)).argmin()
        c = self.xy[i]
        aref = np.arctan2(*(self.xy - c).T[::-1])
        a = np.arctan2(*(xy - c).T[::-1])
        # angular offset
        return np.mean(a - aref[:, None], 0)

    def _marginal_histograms(self):

        grid, pixels = self._stack_pixels()
        x, y = grid.T

        fig, axes = plt.subplots(2, 1)
        axes[0].hist(x, pixels, bins=250)
        axes[1].hist(y, pixels, bins=250)

    # @profile(report='bars')
    def _stack_pixels(self, images=None, image_func=echo0):

        if images is None:
            images = np.arange(len(self))

        # polymorphic `images`: arrays or ints
        types = set(map(type, images))
        assert len(types), 'List of images is empty'

        if isinstance(types.pop(), numbers.Integral):
            indices = np.asarray(images)
            images = self.images
        else:
            indices = np.arange(len(images))

        # check if image func expects an integer
        # image_func = image_func or echo0
        # image func expects an image, so we wrap it so we can simply
        # pass an integer below
        ann = list(image_func.__annotations__.values())
        if len(ann) and isinstance(ann[0], int):
            def image_func(i, *args, **kws):
                return image_func(images[i], *args, **kws)

        sizes = self.shapes[indices].prod(1)
        n = sizes.sum()
        idx = np.cumsum(sizes)
        sec = map(slice, [0] + list(idx), idx)

        grid = np.empty((n, 2))
        pixels = np.empty(n)
        for i, s in zip(indices, sec):
            grid[s] = self[i].grid(self.params[i], self.rscale[i])
            pixels[s] = image_func(i).ravel()

        return grid, pixels

    # @profile(report='bars')
    def binned_statistic(self, images=None, stat='mean', bins=None,
                         image_func=None, interpolate=False):

        grid, pixels = self._stack_pixels(images, image_func)

        if bins is None:
            bins = np.floor(grid.ptp(0)).astype(int)

        # compute stat
        bs = binned_statistic_2d(*grid.T, pixels, stat, bins)

        # interpolate nans
        if interpolate:
            interpolate_nn(bs.statistic, np.isnan(bs.statistic))

        return grid, bs

    def drizzle(self, path, outwcs, pixfrac, ignore=()):
        
        from drizzle.drizzle import Drizzle
        
        if len(self.wcss) != len(self) - len(ignore):
            raise ValueError('First do `reg.build_wcs(run)`.')

        # Get the WCS for the output image
        drizzle = Drizzle(outwcs=outwcs, pixfrac=pixfrac)

        # Add the input images to the existing output image
        wcss = iter(self.wcss)
        for i, image in enumerate(self):
            if i not in ignore:
                fscale = image.counts[image.seg.labels == 2].item()
                data = (image.data - image.seg.mean(image.data, 0)) / fscale
                drizzle.add_image(data, next(wcss),
                                  expin=image.meta['EXPOSURE'])

        drizzle.write(path)

    def global_seg(self, circularize=True):
        """
        Get the global segmentation for all sources across all frames in this
        register. The returned SegmentedImage will be relative to the same
        reference image as the register (as given by the attribute `idx`)

        Parameters
        ----------
        circularize : bool, optional
            Whether the source segments should be made circular, by default True

        Returns
        -------
        obstools.image.segmentation.SegmentedImage
            The global segmented image
        """

        # get pixel labels
        def relabel(i: int):
            """
            Relabel segmented image for global segmentation
            """
            seg = self[i].seg.clone()  # copy!
            seg.relabel_many(seg.labels, sidx[i] + 1)
            return seg.data

        sidx = self.source_indices
        grid, pixels = self._stack_pixels(image_func=relabel)

        xn, yn = np.ceil(grid.ptp(0) / self.rscale[self.primary]).astype(int)
        x0, y0 = grid.min(0)
        x1, y1 = grid.max(0)
        bins = np.ogrid[x0:x1:xn * 1j], np.ogrid[y0:y1:yn * 1j]
        # np.ogrid[y0:y1:yn * 1j]
        # bins = tuple(map(np.squeeze, np.ogrid[x0:x1:xn * 1j, y0:y1:yn * 1j]))

        r = np.full((xn + 1, yn + 1), -1)
        indices = (np.digitize(g, b) for g, b in zip(grid.T, bins))
        for i, j, px in zip(*indices, pixels):
            r[i, j] = max(r[i, j], px)

        #
        interpolate_nn(r, r == -1)

        seg = SegmentedImage(r.T)
        return seg.circularize() if circularize else seg

    # def fit_pixels(self, index):

    #     indices = set(range(len(self.images))) - {index}
    #     grid = []
    #     pixels = []
    #     for i in indices:
    #         image = self.images[i]
    #         r = np.divide(self.fovs[i], image.shape) / self.pixel_scale
    #         p = self.params[i]
    #         seg = self.detections[i].dilate(2, copy=True)
    #         for sub, g in seg.cutouts(image, np.indices(image), flatten=True):
    #             g = tf.rigid(g.reshape(-1, 2) * r, p)
    #             grid.extend(g)
    #             pixels.extend(sub.ravel())

    #     bs = binned_statistic_2d(*grid, pixels, 'mean', bins)

    # def _fit_pixels(self, image, fov, p0):
    #     """Match pixels directly"""
    #     sy, sx = self.image.shape
    #     dx, dy = self.fov
    #     hx, hy = 0.5 * self.pixel_scale
    #     by, bx = np.ogrid[-hx:(dx + hx):complex(sy + 1),
    #                       -hy:(dy + hy):complex(sx + 1)]
    #     bins = by.ravel(), bx.ravel()

    #     sy, sx = image.shape
    #     dx, dy = fov
    #     yx = np.mgrid[:dx:complex(sy), :dy:complex(sx)].reshape(2, -1).T

    #     return minimize(
    #         ftl.partial(objective_pix,
    #                     normalize_image(self.data),
    #                     normalize_image(image).ravel(),
    #                     yx, bins),
    #         p0)

    # ------------------------------------------------------------------------ #

    def mosaic(self, axes=None, names=(), scale='sky',
               show_ref_image=True, number_sources=False,
               **kws):

        mos = MosaicPlotter.from_register(self, axes, scale, show_ref_image)
        mos.mosaic(names, **kws)

        if number_sources:
            off = -4 * self.scales.min(0)

            mos.mark_sources(self.xy, marker=None,
                             nrs=True,
                             xy_offset=off)

        return mos

    def plot_detections(self, coords='pixel', **kws):
        gui = ImageRegistrationGUI(self, coords=coords,
                                   **{**dict(regions=True, labels=True), **kws})

        gui.show()
        return gui

    def plot_clusters(self, centres='k+', frames=False, nrs=False,
                      show_bandwidth=True, trim=False, **kws):
        """
        Plot the identified sources (clusters) in a single frame.
        """
        from matplotlib.patches import Rectangle

        self.check_has_data()
        self.check_has_labels()

        n = len(self)
        fig, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        # ax.set_title(f'Position Measurements (CoM) {n} frames')
        labels = self.labels.copy()
        if trim and len(trim := self._trim_labels()):
            xx = (self.labels == np.transpose(np.where(trim))).any(0)
            labels[xx] = -1
        art = plot_clusters(ax, np.vstack(self.xyt), labels, nrs=nrs,
                            label=f'Centroids ({n} images)', **kws)
        self._colour_sequence_cache = art.get_edgecolors()

        if frames:
            for img in self:
                frame = Rectangle(img.origin,
                                  *(img.shape[::-1] * img.scale / self.scale),
                                  angle=np.degrees(img.angle),
                                  fc='none', lw=1, ec='0.5')
                ax.add_artist(frame)

        if centres:
            ax.plot(*self.xy.T, centres, ms=2.5, alpha=0.7)

        # bandwidth size indicator.
        if show_bandwidth:
            # todo: get this to remain anchored lower left but scale with zoom..
            xy = self.xy.min(0)  # - bw * 0.6
            cir = Circle(xy, self.clustering.bandwidth, ec='0.2', alpha=0.75)
            ax.add_artist(cir)

            fig.subplots_adjust(top=0.82)
            proxy = Line2D([], [], marker='o', ls='', ms=10, color=cir.get_fc(), mec='0.2',
                           label=f'clustering bandwidth = {self.clustering.bandwidth:.3f}')
            ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), handles=[art, proxy])

        # TODO: plot position error ellipses
        ax.set_aspect('equal')
        fig.tight_layout()
        return art


class ImageRegistrationGUI(MplTabs):
    def __init__(self, reg, **kws):

        super().__init__()

        self.reg = reg
        self.style = kws
        self.art = {}

        for _ in reg:
            fig = self.add_tab()

        # self._plot(len(reg) - 1)
        self.add_callback(self.plot)

    def plot(self, fig, i):
        i, = i
        self.art[i] = self.reg[i].plot(fig=fig, **self.style)

        # fig.canvas.draw() # WHY? blit?


class ImageRegisterDSS(ImageRegister):
    """
    Align images with the Digitized Sky Survey archival plate images and infer
    WCS parameters.
    """
    #
    _servers = ('all',
                'poss2ukstu_blue', 'poss1_blue',
                'poss2ukstu_red', 'poss1_red',
                'poss2ukstu_ir',
                )

    # TODO: more modern sky images
    # TODO: can you warn users about this possibility if you only have access to
    # old DSS images

    # def __new__(cls, maybe_images=(), *stuff, **kws):
    #     # for slice support
    #     if isinstance(maybe_images, str):
    #         return super().__new__(cls)
    #       # NOTE: BAD UNPICKLE WRONG CLASS
    #     return ImageRegister(maybe_images, *stuff, **kws)

    def __init__(self, name_or_coords, fov=(3, 3), **kws):
        """

        Parameters
        ----------
        name_or_coords: str
            name of object or coordinate string
        fov: int or 2-tuple
            field of view in arcmin
        """

        #
        fov = duplicate_if_scalar(fov)

        coords = get_coordinates(name_or_coords)
        if coords is None:
            raise ValueError('Need object name or coordinates')

        # TODO: proper motion correction
        # TODO: move some of this code to utils

        for srv in self._servers:
            try:
                self.hdu = get_dss(srv, coords.ra.deg, coords.dec.deg, fov)
                break
            except STScIServerError:
                logger.warning('Failed to retrieve image from server: '
                               '%r', srv)

        # DSS data array
        data = self.hdu[0].data.astype(float)
        kws.setdefault('deblend', True)
        ImageRegister.__init__(self, **kws)
        self(data, fov=fov)

        # TODO: print some header info for DSS image - date!
        # DATE-OBS
        # TELESCOP
        # INSTRUME
        # FILTER
        # EXPOSURE
        # SHAPE
        # FOV

        # save target coordinate position
        self.target_coords_world = coords

    @property
    def target_coords_pixel(self):
        hdr = self.hdu[0].header
        return np.subtract([hdr['crpix1'], hdr['crpix2']], 0.5)

    def remap_labels(self, target=None, flux_sort=False, trim=False):
        if target is None:
            target, = self.clustering.predict([self.target_coords_pixels])

        return super().remap_labels(target, flux_sort)

        # if trim:
        #     new_labels[self._trim_labels()] = -1

        # return new_labels

    def mosaic(self, names=(), **kws):

        header = self.hdu[0].header
        name = ' '.join(filter(None, map(header.get, ('ORIGIN', 'FILTER'))))

        # use aplpy to setup figure
        ff = apl.FITSFigure(self.hdu)

        # rescale images to DSS pixel scale
        return super().mosaic(ff.ax, [name, *names], 'pixels', **kws)

        # for art, frame in mos.art

    def get_rotation(self):
        # transform pixel to ICRS coordinate
        h = self.hdu[0].header
        return np.pi / 2 - np.arctan(-h['CD1_1'] / h['CD1_2'])

    # todo: def proper_motion_correction(self, coords):

    # def get_labels(self, xy):
    #     # transform
    #     ix = tuple(self.to_pixel_coords(xy).round().astype(int).T)[::-1]
    #     return self.seg.data[ix]
    #
    #     #
    #     use = np.ones(len(xy), bool)
    #     # ignore sources not detected in dss, but detected in sample image
    #     use[labels == 0] = False
    #     # ignore labels that are detected as 2 (or more) sources in the
    #     # sample image, but only resolved as 1 in dss
    #     for w in where_duplicate(labels):
    #         use[w] = False
    #     return labels, use

    def register(self, clf=None, plot=PLOT):
        # trim=True
        # if trim:
        #     self._trim_labels()

        super().register(clf, plot)
        self.relabel()

    def _trans_to_image(self, index, unit='pixel'):
        assert unit in {'fraction', 'pixel', 'arcmin'}

        scale = self.scale
        image = self[index]
        if unit == 'fraction':
            scale *= 1 / image.fov
        elif unit == 'pixel':
            scale /= image.scale
        # elif unit == 'arcmin':
        #     scale *= 1

        return Affine2D().translate(*-image.origin) \
                         .rotate(-image.angle) \
                         .scale(*scale)

    def _trans_to_dss(self, index, unit='pixel'):
        return self._trans_to_image(index, unit).inverted()

    def _trim_labels(self, index=0):
        xy = np.ma.median(self.xyt_block, 0)
        out = np.ones((len(self), self.n_sources()), bool)
        data = list(self.data)
        data.pop(index)
        for i in range(len(data)):
            tr = self._trans_to_image(i + int(i >= index), unit='fraction')
            xyi = tr.transform(xy)
            out[i] = ((0 > xyi) | (xyi > 1)).any(1)
            # print(i+1, np.where(l)[0])

        return out.all(0)
        # return np.where(outside.all(0))[0]
        # return np.array(sorted(set(self.labels) - {-1}))[outside.all(0)]

    def build_wcs(self, run):
        assert len(run) == len(self) - 1

        self.wcss = []
        for i, hdu in enumerate(run, 1):
            wcs = self._build_wcs(i)
            # TODO obsgeo, dateobs, mjdobs
            hdu.header.update(wcs.to_header())  # NOTE: NOT SAVED
            self.wcss.append(wcs)

        return self.wcss

    def _build_wcs(self, i):
        """
        Create tangential plane WCS for image at index `i`.

        Returns
        -------
        astropy.wcs.WCS instance

        """

        # survey_image_info
        hdr = self.hdu[0].header
        image = self[i]

        # transform target coordinates in DSS image to target in SHOC image
        # convert to pixel llc coordinates then to arcmin
        # target coordinates DSS [pixel]
        xyDSS = np.subtract([hdr['crpix1'], hdr['crpix2']], 0.5)
        # target coordinates SHOC [pixel]
        xyIMG = self._trans_to_image(i).transform(xyDSS).squeeze()
        # target coordinates celestial [degrees]
        xySKY = self.target_coords_world

        # WCS
        # for parameter definitions see:
        # https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html
        w = wcs.WCS(naxis=2)
        # axis type
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        # coordinate increment at reference point
        w.wcs.cdelt = image.scale / 60.  # image scale in degrees
        # target coordinate in degrees reference point
        w.wcs.crval = [xySKY.ra.value, xySKY.dec.value]
        # rotation from stated coordinate type.
        w.wcs.crota = np.degrees([-image.angle, -image.angle])
        # array location of the reference point in pixels
        w.wcs.crpix = xyIMG
        # need for drizzle
        w.pixel_shape = image.shape

        return w

    def drizzle(self, path, pixfrac, outwcs=None):

        #drizzle.add_image(self[0].data, outwcs, expin=hdr['EXPOSURE'])
        if outwcs is not None:
            return super().drizzle(path, outwcs, pixfrac, (0,))

        UNIT_CORNERS = np.array([[0., 0.],
                                 [1., 0.],
                                 [1., 1.],
                                 [0., 1.]])

        cmn = np.full(2, +np.inf)
        cmx = np.full(2, -np.inf)
        for i in range(1, len(self)):
            to_dss = self._trans_to_image(i).inverted()
            corners = to_dss.transform(UNIT_CORNERS * self[i].shape - 0.5)
            cmn = np.min([cmn, corners.min(0)], 0)
            cmx = np.max([cmx, corners.max(0)], 0)

        rscale = np.median(self.rscale[1:])
        shape = (cmx - cmn) / rscale

        # outwcs = wcs.WCS(self.hdu[0].header)
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        outwcs.wcs.cdelt = rscale * self.scale / 60.  # image scale in degrees
        outwcs.wcs.crpix = (self.target_coords_pixel - cmn) / rscale
        xySKY = self.target_coords_world
        outwcs.wcs.crval = [xySKY.ra.value, xySKY.dec.value]
        outwcs.pixel_shape = np.round(shape, 0).astype(int)

        return super().drizzle(path, outwcs, pixfrac, (0,))

# class TransformedImage():
#     def set_p(self):
#         self.art.set_transform()
