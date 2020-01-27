"""
Helper functions to infer World Coordinate System given a target name or
coordinates of a object in the field. This is done by matching the image
with the DSS image for the same field via image registration.  A number of
methods are implemented for doing this:
  point cloud drift
  matching via locating dense cluster of displacements between points
  direct image-to-image matching
  brute force search with gaussian mixtures on points
"""

import functools as ftl
import logging
import multiprocessing as mp
import itertools as itt
import warnings

import numpy as np
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
# from astropy.coordinates import SkyCoord
from obstools.stats import geometric_median

from recipes.containers.dicts import AttrDict
from obstools.phot.segmentation import SegmentationHelper

from pySHOC.utils import get_coordinates, get_dss, STScIServerError

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from recipes.logging import LoggingMixin
from recipes.transformations.rotation import rotation_matrix_2d

from scipy.stats import binned_statistic_2d, mode

from motley.profiling.timers import timer
from recipes.introspection.utils import get_module_name
from scipy.spatial import cKDTree
# from sklearn.cluster import MeanShift
from graphing.imagine import ImageDisplay

from astropy.utils import lazyproperty

# TODO: might be faster to anneal sigma param rather than blind grid search.
#  test this.

logger = logging.getLogger(get_module_name(__file__))

TABLE_STYLE = dict(txt='bold', bg='g')


def roto_translate(X, p):
    """rotate and translate"""
    # https://en.wikipedia.org/wiki/Rigid_transformation
    rot = rotation_matrix_2d(p[-1])
    Xnew = (rot @ X.T).T + p[:2]
    return Xnew


def roto_translate2(X, xy_off=(0., 0.), theta=0.):
    """rotate and translate"""
    if theta:
        X = (rotation_matrix_2d(theta) @ X.T).T
    return X + xy_off


def roto_translate_yx(X, p):
    """rotate and translate"""
    rot = rotation_matrix_2d(p[-1])
    Xnew = (rot @ X[:, ::-1].T).T + p[1::-1]
    return Xnew[:, ::-1]


def roto_translate2_yx(X, yx_off=(0., 0.), theta=0.):
    """rotate and translate"""
    if theta:
        X = (rotation_matrix_2d(theta) @ X[:, ::-1].T).T[:, ::-1]

    return X + yx_off


def check_transforms(yx, p):
    z0 = roto_translate_yx(yx, p)

    pxy = np.r_[p[1::-1], p[-1]]
    z1 = roto_translate(yx[:, ::-1], pxy)[:, ::-1]

    return np.allclose(z0, z1)


def normalize_image(image, centre=np.ma.median, scale=np.ma.std):
    """Recenter and scale"""
    image = image - centre(image)
    if scale:
        return image / scale(image)
    return image


# def get_sample_image(hdu, stat='median', depth=5):
#     # get sample image
#     n = int(np.ceil(depth // hdu.timing.t_exp))
#
#     logger.info(f'Computing {stat} of {n} images (exposure depth of '
#                 f'{float(depth):.1f} seconds) for sample image from '
#                 f'{hdu.filepath.name!r}')
#
#     sampler = getattr(hdu.sampler, stat)
#     image = hdu.calibrated(sampler(n, n))
#     return normalize_image(image, scale=False)


def prob_gmm(xy_trg, xy, sigma):
    # not a true probability (not normalized)
    # *_, d = xy.shape
    # f = 2 * sigma * sigma  #  / np.pow(np.pi * f, d)
    return np.exp(
            -np.square(xy_trg[None] - xy[:, None]).sum(-1) / (2 * sigma * sigma)
    ).sum(-1)


def loglike_gmm(xy_trg, xy, sigma):
    # add arbitrary offset to avoid nans!
    return np.log(prob_gmm(xy_trg, xy, sigma) + 1)


def objective_gmm(xy_trg, xy, sigma, p):
    """
    Objective for gaussian mixture model

    Parameters
    ----------
    xy_trg: array (n_components, n_dims)
    xy
    sigma
    p

    Returns
    -------

    """
    xy_new = roto_translate(xy, p)
    # xy_new = xy + p[:2]
    return -loglike_gmm(xy_trg, xy_new, sigma).sum()


def objective_gmm_yx(yx_trg, yx, sigma, p):
    """Objective for gaussian mixture model"""
    yx_new = roto_translate_yx(yx, p)
    # xy_new = xy + p[:2]
    return -loglike_gmm(yx_trg, yx_new, sigma).sum()


def objective_gmm2_yx(yx_trg, yx, sigma, dyx, theta):
    """Objective for gaussian mixture model"""
    yx_new = roto_translate2_yx(yx, dyx, theta)
    return -loglike_gmm(yx_trg, yx_new, sigma).sum()


def objective_gmm2(xy_trg, xy, sigma, xy_off, theta=0):
    xy_new = roto_translate2(xy, xy_off, theta)
    return -loglike_gmm(xy_trg, xy_new, sigma).sum()


def objective_gmm3(x_trg, x, sigma, xy_off, theta=0):
    xy, counts = x
    xy_new = roto_translate2(xy, xy_off, theta)
    return -loglike_gmm(x_trg, xy_new, sigma).sum()


def objective_pix(target, values, yx, bins, p):
    """Objective for direct image matching"""
    yx = roto_translate_yx(yx, p)
    bs = binned_statistic_2d(*yx.T, values, 'mean', bins)
    rs = np.square(target - bs.statistic)
    return np.nansum(rs)


# def match_pixels(self, image, fov, p0):
#     """Match pixels directly"""
#     sy, sx = self.data.shape
#     dx, dy = self.fov
#     hx, hy = 0.5 * self.pixel_scale
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


# def objective0(xy_trg, xy, p, thresh=0.1):
#     """
#     Objective function that highlights small distances for crude grid search
#     """
#     xy_new = transform(xy, p)
#     ifd = cdist(xy_trg, xy_new)
#     # return np.percentile(ifd, thresh)
#     return np.sum(ifd < thresh)
#
#
# def objective1(xy_trg, xy, p):
#     """
#     Objective function that highlights small distances for crude grid search
#     """
#     xy_new = transform(xy, p)
#     d = cdist(xy_trg, xy_new)
#     return np.sum(1 / (d * d))

class GaussianMixture(object):  # Model  # !
    def __init__(self, xy, sigma):
        """

        Parameters
        ----------
        xy: array (n_components, n_dims)
        sigma
        """

        self.xy = xy
        self.sigma = sigma

    def __call__(self, xy):
        """

        Parameters
        ----------
        xy: array (n_components, n_dims)

        Returns
        -------

        """
        return loglike_gmm(self.xy, xy, self.sigma)

    def objective_trans(self, p, xy):
        xy_new = roto_translate(xy, p)
        return -loglike_gmm(self.xy, xy_new, self.sigma).sum()

    def fit_trans(self, p0, xy, fit_angle=False):
        """
        Fit for roto-translation parameters with gradient descent on Gaussian
        mixture model likelihood

        Parameters
        ----------
        xy: array
            xy coordinates of image features (center of mass of stars)
        p0: array-like
            parameter starting values


        Returns
        -------

        """

        if fit_angle:
            result = minimize(self.objective_trans, p0, xy)
        else:
            theta = p0[-1]
            xy = roto_translate2(xy, [0, 0], theta)
            f = ftl.partial(objective_gmm2, self.xy, xy, self.sigma)
            result = minimize(f, p0[:2])

        if not result.success:
            return None

        if fit_angle:
            return result.x

        # noinspection PyUnboundLocalVariable
        return np.r_[result.x, theta]

    def _auto_grid(self, size, stretch=1.1):

        try:
            xyl = self.xy.min(0)
            xyu = self.xy.max(0)
            xyr = np.mean((xyl, xyu)) + np.multiply((-1, 1),
                                                    (xyu - xyl)) * stretch
        except Exception as err:
            import sys
            from IPython import embed
            from IPython.core.ultratb import ColorTB
            import textwrap
            embed(header=textwrap.dedent(
                    """\
                    Caught the following %s:
                    ------ Traceback ------
                    %s
                    -----------------------
                    Exception will be re-raised upon exiting this embedded interpreter.
                    """) %
                         (err.__class__.__name__,
                          ColorTB().text(*sys.exc_info())))
            raise

        x0, y0 = self.xy.min(0)
        x1, y1 = self.xy.max(0)

        rx, ry = duplicate_if_scalar(size)
        return np.mgrid[x0:x1:(rx * 1j),
               y0:y1:(ry * 1j)]

    def gridsearch_max(self, grid=None, gridsize=100):
        if grid is None:
            grid = self._auto_grid(gridsize)

        xy = grid.reshape(2, -1).T
        ll = self(xy).reshape(grid.shape[1:])
        return grid, ll, xy[ll.argmax()]

    def plot(self, grid=None, gridsize=100, show_peak=True):

        grid, ll, peak = self.gridsearch_max(grid, gridsize)

        extent = np.c_[grid[:, 0, 0], grid[:, -1, -1]].ravel()
        im = ImageDisplay(ll.T, extent=extent,
                          cmap='Blues', alpha=0.5)

        im.ax.plot(*self.xy.T, '.')
        if show_peak:
            im.ax.plot(*peak, 'rx')
        return im, peak


class ConstellationModel(GaussianMixture):
    """ """

    # def register(self, xy, centre_distance_max=None,
    #              f_detect_measure=0.5, plot=False, **plot_kws):
    #
    #     clustering = MeanShift(bandwidth=4 * pixel_scale, cluster_all=False)
    #
    #     centres, σ_xy, xy_offsets, outliers, xy = register_constellation(
    #             clustering, xy, centre_distance_max, f_detect_measure, plot=plot, **plot_kws)


# def find_objects(image, mask=False, background=None, snr=3., npixels=7,
#                  edge_cutoff=None, deblend=False, dilate=0):
#     seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
#                                     edge_cutoff, deblend, dilate)
#     return seg, seg.com_bg(image),


def detect_measure(image, mask=False, background=None, snr=3., npixels=5,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentationHelper.detect(image, mask, background, snr, npixels,
                                    edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    return seg, seg.com_bg(image), counts


# @timer
def offset_disp_cluster(xy0, xy1, sigma=0.03, plot=False):
    # find offset (translation) between frames by looking for the dense
    # cluster of inter-frame point-to-point displacements that represent the
    # xy offset between the two images. Cannot handle rotations.
    # NOTE: also not good for large number of points in terms of performance and
    # accuracy...
    # TODO: parent that decides which registration algorithm to try based on
    #  the number of input points

    points = (xy0[None] - xy1[:, None]).reshape(-1, 2).T

    # vals, xe, ye = np.histogram2d(*points, bins)
    # # todo probably raise if there is no significantly dense cluster
    # i, j = np.unravel_index(vals.argmax(), vals.shape)
    # l = ((xe[i] < points[0]) & (points[0] <= xe[i + 1]) &
    #      (ye[j] < points[1]) & (points[1] <= ye[j + 1]))
    # sub = points[:, l].T
    # yay = sub[np.sum(cdist(sub, sub) > dmax, 0) < (len(sub) // 2)]
    # off = np.mean(yay, 0)

    gmm = GaussianMixture(points.T, sigma)
    grid, ll, peak = gmm.gridsearch_max()

    tree = cKDTree(points.T)
    idx_nn = tree.query_ball_point(peak, 3 * sigma)
    off = points[:, idx_nn].mean(1)

    if plot:
        from matplotlib.patches import Rectangle
        im, _ = gmm.plot()
        im.ax.plot(*points[:, idx_nn], 'o', ms=7, mfc='none')

    return off


def dist_tril(coo, masked=False):
    """distance matrix with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    ix = np.tril_indices(n, -1)
    # since the distance matrix is symmetric, ignore lower half
    if masked:
        sdist = np.ma.masked_array(sdist)
        sdist[ix] = np.ma.masked
    return sdist[ix]


# @timer
def gridsearch_mp(objective, grid, args, **kws):
    # grid search
    f = ftl.partial(objective, *args, **kws)
    ndim, *rshape = grid.shape

    with mp.Pool() as pool:
        r = pool.map(f, grid.reshape(ndim, -1).T)
    pool.join()

    return np.reshape(r, rshape)


def worker(func, args, input_, output, i, indexer, axes):
    indexer[axes] = i
    output[i] = func(*args, input_[tuple(indexer)])


def gridsearch_alt(func, args, grid, axes=..., output=None):
    # from joblib import Parallel, delayed

    # grid search
    if axes is not ...:
        axes = list(axes)
        out_shape = tuple(np.take(grid.shape, axes))
        indexer = np.full(grid.ndim, slice(None))

    if output is None:
        output = np.empty(out_shape)

    indices = np.ndindex(out_shape)
    # with Parallel(max_nbytes=1e3, prefer='threads') as parallel:
    #     parallel(delayed(worker)(func, args, grid, output, ix, indexer, axes)
    #              for ix in indices)
    # note: seems about twice as slow when testing for small datasets due to
    #  additional overheads

    with mp.Pool() as pool:
        pool.starmap(worker, ((func, args, grid, output, ix, indexer, axes)
                              for ix in indices))

    i, j = np.unravel_index(output.argmax(), out_shape)
    return output, (i, j), grid[:, i, j]


def cross_index(coo_trg, coo, dmax=0.2):
    # identify points matching points between sets by checking distances
    # this will probably only work if the frames are fairly well aligned
    dr = cdist(coo_trg, coo)

    # if thresh is None:
    #     thresh = np.percentile(dr, dr.size / len(coo))

    dr[(dr > dmax) | (dr == 0)] = np.nan

    ur, uc = [], []
    for i, d in enumerate(dr):
        dr[i, uc] = np.nan
        if ~np.isnan(d).all():
            # closest star not yet matched with another
            jmin = np.nanargmin(d)
            ur.append(i)
            uc.append(jmin)

    return ur, uc


# def match_cube(self, filename, object_name=None, coords=None):
#     from .core import shocObs
#
#     cube = shocObs.load(filename, mode='update')  # ff = FitsCube(fitsfile)
#     if coords is None:
#         coords = cube.get_coords()
#     if (coords is None):
#         if object_name is None:
#             raise ValueError('Need object name or coordinates')
#         coords = get_coords_named(object_name)
#
#     image = np.fliplr(cube.data[:5].mean(0))
#     fov = cube.get_FoV()
#
def get_corners(p, fov):
    """Get corners relative to DSS coordinates. xy coords anti-clockwise"""
    c = np.array([[0, 0], fov])  # lower left, upper right yx
    corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
    corners = roto_translate_yx(corners, p)
    return corners[:, ::-1]  # return xy !


def get_ulc(params, fovs):
    ulc = np.empty((len(params), 2))
    for i, (p, fov) in enumerate(zip(params, fovs)):
        ulc_ = np.array([[fov[0], 0]])
        ulc[i] = roto_translate_yx(ulc_, p)
    return ulc[:, 0].max(), ulc[:, 1].min()  # yx


def plot_transformed_image(ax, image, fov, p=(0, 0, 0), frame=True,
                           set_lims=True, **kws):
    """"""

    kws.setdefault('hist', False)
    kws.setdefault('sliders', False)

    # plot
    im = ImageDisplay(image, ax=ax, **kws)
    art = im.imagePlot

    # set extent
    extent = np.c_[[0., 0.], fov[::-1]]
    pixel_size = np.divide(fov, image.shape)
    half_pixel_size = pixel_size / 2
    extent -= half_pixel_size[None].T  # adjust to pixel centers...
    art.set_extent(extent.ravel())

    # Rotate the image by setting the transform
    xy, theta = p[1::-1], p[-1]
    art.set_transform(Affine2D().rotate(theta).translate(*xy) +
                      art.get_transform())

    if bool(frame):
        from matplotlib.patches import Rectangle

        frame_kws = dict(fc='none', lw=1.5, ec='0.5')
        if isinstance(frame, dict):
            frame_kws.update(frame)

        ax.add_patch(
                Rectangle(xy - half_pixel_size, *fov[::-1], np.degrees(theta),
                          **frame_kws)
        )

    if set_lims:
        delta = 1 / 100
        c = get_corners(p, fov)
        xlim, ylim = np.vstack([c.min(0), c.max(0)]).T * (1 - delta, 1 + delta)
        im.ax.set(xlim=xlim, ylim=ylim)

    return art


def plot_coords_nrs(cooref, coords):
    fig, ax = plt.subplots()

    for i, yx in enumerate(cooref):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

    for i, yx in enumerate(coords):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


def display_multitab(images, fovs, params, coords):
    from graphing.multitab import MplMultiTab
    from graphing.imagine import ImageDisplay

    import more_itertools as mit

    ui = MplMultiTab()
    for i, (image, fov, p, yx) in enumerate(zip(images, fovs, params, coords)):
        xy = yx[:, ::-1]  # roto_translate_yx(yx, np.r_[-p[:2], 0])[:, ::-1]
        ex = mit.interleave((0, 0), fov)
        im = ImageDisplay(image, extent=list(ex))
        im.ax.plot(*xy.T, 'kx', ms=5)
        ui.add_tab(im.figure)
        plt.close(im.figure)
        # if i == 1:
        #    break
    return ui


from recipes.misc import duplicate_if_scalar


# todo: move to diagnostics


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


def register_constellation(clustering, coms, centre_distance_max=1,
                           f_detect_measure=0.5, plot=False, **plot_kws):
    #
    from collections import Callable
    if isinstance(plot, Callable):
        display = plot
        plot = True
    else:
        plot = bool(plot)

        def display(*_):
            pass
    #  🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # clustering + relative position measurement
    logger.info('Identifying stars')
    n_clusters, n_noise = cluster_id_stars(clustering, coms)
    xy, = group_features(clustering, coms)

    if plot:
        import matplotlib.pyplot as plt

        fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        ax.set_title(f'Position Measurements (CoM) {len(coms)} frames')

        # TODO: plot boxes on this image corresponding to those below
        cmap = plot_clusters(ax, np.vstack(coms)[:, ::-1], clustering)
        # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
        #                   tuple(zip((0, 0), ishape)))))
        display(fig0)

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

        # plot CoMs
        fig1, axes = plot_position_measures(xy, centres, xy_offsets, **plot_kws)
        fig1.suptitle('Raw Positions (CoM)')  # , fontweight='bold'
        display(fig1)

        fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        ax.set_title(f'Re-centred Positions (CoM) {len(coms)} frames')
        rcx = np.vstack([c - o for c, o in zip(coms, xy_offsets)])
        plot_clusters(ax, rcx[:, ::-1], clustering, cmap=cmap)

        # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
        #                   tuple(zip((0, 0), ishape)))))
        display(fig0)

    return centres, σ_xy, xy_offsets, outliers, xy


def id_stars_kmeans(images, segmentations):
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
    """
    fit the clustering model

    Parameters
    ----------
    clf
    xy

    Returns
    -------

    """
    X = np.vstack(xy)
    n = len(X)
    # stars_per_image = list(map(len, xy))
    # no_detection = np.equal(stars_per_image, 0)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(np.vstack(xy))

    logger.info('Clustering %i position measurements using:\n%s', n,
                str(clf).replace('\n', '\t\n'))
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

    # todo: check for clusters that are too close together

    return n_clusters, n_noise


def plot_clusters(ax, features, labels, colours=(), cmap=None):
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

    core_sample_indices_, = np.where(labels != -1)
    labels_xy = labels[core_sample_indices_]
    n = labels_xy.max() + 1

    # plot
    if not len(colours):
        if cmap is None:
            from photutils.utils.colormaps import make_random_cmap
            cmap = make_random_cmap(n)
            # this cmap has good distinction between colours for clusters nearby
            # each other
        else:
            from matplotlib.cm import get_cmap
            cmap = get_cmap(cmap)

        colours = cmap(np.linspace(0, 1, n))[labels[core_sample_indices_]]

    # fig, ax = plt.subplots(figsize=im.figure.get_size_inches())
    art = ax.scatter(*features[core_sample_indices_].T, s=15,
                     marker='.', edgecolors=colours, facecolors='none')
    ax.plot(*features[labels == -1].T, 'kx', alpha=0.3)
    ax.grid()
    return art


def measure_positions_offsets(xy, d_cut=None, detect_frac_min=0.9, report=True):
    # Todo: integrate into ImageRegistration

    # check post and pre variance for sanity

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
    xy, centres, σxy, δxy, outlier_indices
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

    if report:
        # pprint!
        report_measurements(xy, centres, σxy, δxy, None, detect_frac_min)
        #                                        # counts
    return xy, centres, σxy, δxy, outlier_indices


def _measure_positions_offsets(xy, centres, d_cut=None):
    # THIS IS ESSENTIALLY SIGMA CLIPPING....

    # ensure we have at least some centres
    assert not np.all(np.ma.getmask(centres))

    n, n_stars, _ = xy.shape
    n_points = n * n_stars

    outliers = np.zeros(xy.shape[:-1], bool)
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
            return centres, xy_shifted.std(0), xy_offsets.squeeze(), outliers

        # compute position residuals
        cxr = xy - centres - xy_offsets

        # flag outliers
        with warnings.catch_warnings():  # catch RuntimeWarning for masked
            warnings.filterwarnings('ignore')
            d = np.sqrt((cxr * cxr).sum(-1))

        out_new = (d > d_cut)
        out_new = np.ma.getdata(out_new) | np.ma.getmask(out_new)

        changed = (outliers != out_new).any()
        if changed:
            out = out_new
            xym[out] = np.ma.masked
            n_out = out.sum()

            if n_out / n_points > 0.5:
                raise Exception('Too many outliers!!')

            logger.info('Ignoring %i/%i (%.1f%%) values with |δr| > %.3f',
                        n_out, n_points, (n_out / n_points) * 100, d_cut)
        else:
            break

    return centres, xy_shifted.std(0), xy_offsets.squeeze(), outliers


def group_features(labels, *features):
    """
    Read multiple sets of features into a uniform array where the first
    dimension select the class to which those features belong according to
    `classifier` Each data set is returned as a masked array where missing
    elements have been masked out to match the size of the largest class.
    The returned feature array is easier to work with than lists of unevenly
    sized features ito further statistical analysis.


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


def report_measurements(xy, centres, σ_xy, xy_offsets, counts=None,
                        detect_frac_min=None, count_thresh=None, logger=logger):
    # report on relative position measurement
    import operator as op

    from recipes import pprint
    from motley.table import Table
    from motley.utils import ConditionalFormatter
    from obstools.stats import mad
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

    # check if we managed to reduce the variance
    sigma0 = xy.std(0)
    var_reduc = (sigma0 - σ_xy) / sigma0
    mad0 = mad(xy, axis=0)
    mad1 = mad((xy - xy_offsets[:, None]), axis=0)
    mad_reduc = (mad0 - mad1) / mad0

    # overall variance report
    s0 = xy.std((0, 1))
    s1 = (xy - xy_offsets[:, None]).std((0, 1))
    # Fractional variance change
    logger.info('Differencing change overall variance by %r',
                np.array2string((s0 - s1) / s0, precision=3))

    col_headers = ['n (%)', 'x', 'y']  # '(px.)'
    fmt = {0: ftl.partial(pprint.decimal_with_percentage,
                          total=n_points, precision=0, right_pad=1)}  #

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

    # add variance columns
    col_headers += ['r_σx', 'r_σy'] + ['mr_σx', 'mr_σy']
    columns += [var_reduc[:, ::-1], mad_reduc[:, ::-1]]

    # tbl = np.ma.column_stack(columns)
    tbl = Table.from_columns(*columns,
                             title='Measured star locations',
                             title_props=TABLE_STYLE,
                             col_headers=col_headers,
                             col_head_props=TABLE_STYLE,
                             precision=3,
                             align='r',
                             row_nrs=True,
                             totals=[0],
                             formatters=fmt)

    logger.info('\n' + str(tbl) + extra)
    return tbl


#

class Image(object):
    """helper class for image registration currently not in use"""

    def __init__(self, data, fov):
        # data array
        self.data = np.asarray(data)
        self.fov = np.array(fov)

        # pixel size in arcmin xy
        self.pixel_scale = self.fov / self.data.shape

    def __array__(self):
        return self.data


# from obstools.phot.segmentation import SourceDetectionMixin

class ImageRegistration(LoggingMixin):  #
    _fov_search_stretch = 1.0  # _fov_search_stretch
    _dflt_find_kws = dict(snr=3.,
                          npixels=5,
                          edge_cutoff=3,
                          deblend=False)

    # FIXME: is it better to keep coordinates at pixel scale (units are more
    #  readable and intuitive than arcminutes)

    # @classmethod
    # def from_image(cls, fov, **find_kws):

    @staticmethod
    def detect(image, fov, **kws):
        # find stars
        seg, coo, counts = detect_measure(image, **kws)
        yx = (coo * fov / image.shape)
        return seg, yx, counts

    def __init__(self, image, fov, **find_kws):
        """

        Parameters
        ----------
        image:
        fov:
            Field of view for the image. Order of the dimensions should be
            the same as that of the image
        find_kws
        """

        # defaults
        for k, v in self._dflt_find_kws.items():
            find_kws.setdefault(k, v)

        # data array
        # im = Image(image, fov[::-1])
        self.image = np.asarray(image)
        self.fov = np.array(fov)[::-1]
        # pixel size in arcmin xy
        self.pixel_scale = self.fov / image.shape

        # Detect stars in dss frame
        seg, yx, counts = detect_measure(image, **find_kws)
        self.yx = yx * self.pixel_scale

        # containers for matched images
        self.fovs = [self.fov]
        self.images = [self.image]
        self.coms = [self.yx]
        self.detections = [seg]
        self.params = [np.zeros(3)]
        self.centres = self.sigmas = None
        self.targetCoordsPix = None

        # state variables
        self.labels = self.n_stars = self.n_noise = None
        self._colour_sequence_cache = None

        # ashape = np.array(self.data.shape)
        # self.grid = np.mgrid[tuple(map(slice, (0, 0), self.fov, 1j * ashape))]
        # self.ogrid = np.array([self.grid[0, :, 0], self.grid[1, 0, :]])

    def __call__(self, image, fov, rotation=0., plot=False):
        """
        Run match_image and aggregate

        Parameters
        ----------
        image
        fov
        rotation
        plot

        Returns
        -------

        """

        p, yx, seg = self.match_image(image, fov, rotation, plot)

        # aggregate
        self.images.append(image)
        self.detections.append(seg)
        self.coms.append(yx)
        self.fovs.append(fov[::-1])
        self.params.append(p)

        return p, yx

    @lazyproperty
    def xy(self):
        return self.yx[:, ::-1]

    # @lazyproperty
    @property
    def yxt(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters
        """
        return [roto_translate_yx(coords, p)
                for p, coords in zip(self.params, self.coms)]

    # @lazyproperty
    @property
    def yxt_block(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters and group them according to the cluster identified labels
        """
        return group_features(self.labels, self.yxt)[0]

    def cluster_id(self, clustering):
        # clustering to cross-identify stars
        # assert len(self.params)

        logger.info('Identifying stars')
        self.n_stars, self.n_noise = cluster_id_stars(clustering, self.yxt)
        self.labels = clustering.labels_

    def get_clf(self):
        from sklearn.cluster import MeanShift

        pixel_size = self.pixel_scale[0]
        return MeanShift(bandwidth=5 * pixel_size,
                         cluster_all=False)

    def register_constellation(self, clustering=None, plot=False):

        # clustering + relative position measurement
        # clustering =
        self.cluster_id(clustering or self.get_clf())

        if plot:
            import matplotlib.pyplot as plt

            n = len(self.coms)
            fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
            ax.set_title(f'Position Measurements (CoM) {n} frames')

            # TODO: plot boxes on this image corresponding to those below
            art = plot_clusters(ax, np.vstack(self.yxt)[:, ::-1], self.labels)
            self._colour_sequence_cache = art.get_edgecolors()
            # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
            #                   tuple(zip((0, 0), ishape)))))

        # return centres, σ_xy, xy_offsets, outliers, xy
        # return xy

    def recentre(self, centre_distance_cut=None, f_detect_measure=0.25,
                 plot=False, **plot_kws):
        #
        # current = self.yxt
        # FIXME: THESE ARE ALL YX COORDINATES VARIABLE NAMES ARE WRONG!!!

        xy = self.yxt_block

        logger.info('Measuring cluster centres, image offsets')
        _, centres, xy_std, xy_offsets, outliers = \
            measure_positions_offsets(xy, centre_distance_cut,
                                      f_detect_measure)

        self.yx = centres
        self.sigmas = xy_std

        # some of the offsets may be masked. ignore those
        good = ~xy_offsets.mask.any(1)
        for i in np.where(good)[0]:
            self.params[i][:-1] -= xy_offsets[i]

        # self.params -= params[idx]  # relative to reference image

        # set transformed coordinated to recompute
        # del self.yxt

        # 🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if plot:
            # diagnostic for source location measurements
            from obstools.phot.diagnostics import plot_position_measures

            # plot CoMs
            fig1, axes = plot_position_measures(xy, centres, xy_offsets,
                                                pixel_size=self.pixel_scale[0])
            fig1.suptitle('Raw Positions (CoM)')  # , fontweight='bold'

            #
            fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
            ax.set_title(f'Re-centred Positions (CoM) {len(self.coms)} frames')
            rcx = np.vstack(self.yxt)
            plot_clusters(ax, rcx[:, ::-1], self.labels,
                          self._colour_sequence_cache)

        return xy_offsets, outliers, xy

    def solve_rotation(self):
        ''
        # for point clouds that ate already quite well aligned. Refine by
        # looking for rotation differences
        yx = self.yxt_block
        idx = yx.std((0, -1)).argmin()
        c = self.centres[idx]
        aref = np.arctan2(*(self.centres - c).T)
        a = np.arctan2(*(yx - c).T)
        angles = np.mean(a - aref[:, None], 0)

    def reset(self):
        """
        Reset image, fov, detection  containers to empty

        Returns
        -------

        """

        self.fovs = []
        self.images = []
        self.coms = []
        self.detections = []

    def to_pixel_coords(self, xy):
        # internal coordinates are in arcmin origin at (0,0) for image
        return np.divide(xy, self.pixel_scale)

    def match_image(self, image, fov, rotation=0., plot=False):
        """
        Search heuristic for image offset and rotation.


        Parameters
        ----------
        image
        fov
        rotation

        Returns
        -------
        p: parameter values (δy, δx, θ)
        yx: center of mass coordinates
        seg: SegmentationImage
        """

        # detect
        seg, yx, counts = self.detect(image, fov)

        # match images directly
        # mr = self.match_pixels(image, fov[::-1], p)

        #
        p = self.match_points(yx, fov, rotation, plot)

        # return yx (image) coordinates
        # yx = roto_translate_yx(yx, p)

        return p, yx, seg

    def match_points(self, yx, fov, rotation=0., plot=False):

        if rotation is None:
            raise NotImplementedError
            #  match gradient descent
            p = self._match_points_gd_yx(yx, pGs, fit_angle=True)

        # ratio of field of views is an indicator of which algorithm will be
        # more suited to match the fields. If the image is much smaller than
        # the reference image in one dimension, brute force search is more
        # likely to succeed
        do_brute = np.any((self.fov / fov) > 10)
        if not do_brute:
            self.logger.debug('Attempting fast match with field-to-field '
                              'displacement density search.')
            # get shift (translation) via peak detection in displacements (outer
            # difference) between two point clouds
            yxt = roto_translate2_yx(yx, theta=rotation)
            dyx = offset_disp_cluster(self.yx, yxt, plot=False)
            do_brute = np.isnan(dyx).any()
            if do_brute:
                self.logger.info('Fast match failed, falling back to grid '
                                 'search')
            else:
                p = np.r_[dyx, rotation]

        if do_brute:
            # et `tu brute??
            p0 = self.match_points_brute(yx, fov, rotation, plot=plot)

            # final match via gradient descent
            p = self._match_points_gd_yx(yx, p0, fit_angle=False) or p0

        # pyx = np.r_[p[1::-1], p[-1]]
        # if return_coords:
        #     return pyx#, xy[:, ::-1]  # roto_translate(xy, p)
        return p

    def match_pixels(self, image, fov, p0):
        """Match pixels directly"""
        sy, sx = self.data.shape
        dx, dy = self.fov
        hx, hy = 0.5 * self.pixel_scale
        by, bx = np.ogrid[-hx:(dx + hx):complex(sy + 1),
                 -hy:(dy + hy):complex(sx + 1)]
        bins = by.ravel(), bx.ravel()

        sy, sx = image.shape
        dx, dy = fov
        yx = np.mgrid[:dx:complex(sy), :dy:complex(sx)].reshape(2, -1).T

        return minimize(
                ftl.partial(objective_pix,
                            normalize_image(self.data),
                            normalize_image(image).ravel(),
                            yx, bins),
                p0)

    # def match_image_brute(self, image, fov, rotation=0., step_size=0.05,
    #                       return_coords=False, plot=False, sigma_gmm=0.03):
    #
    #     seg, yx, counts = self.find_stars(image, fov)
    #     return self.match_points_brute(yx, fov, rotation, step_size,
    #                                    return_coords, plot, sigma_gmm)

    @timer
    def match_points_brute(self, yx, fov, rotation=0., step_size=0.05,
                           plot=False, sigma_gmm=0.03):
        # grid search with gmm loglikelihood objective
        g, r, ix, pGs = self._match_points_brute_yx(yx, fov, step_size,
                                                    rotation, objective_gmm,
                                                    sigma_gmm)
        pGs[-1] = rotation

        if plot:
            from graphing.imagine import ImageDisplay

            # plot xy coords
            ggfig, ax = plt.subplots()
            ax.scatter(*self.xy.T, self.counts / self.counts.max() * 200)
            ax.plot(*roto_translate_yx(yx, pGs).T[::-1], 'r*')

            ext = np.r_[g[1, 0, [0, -1]],
                        g[0, [0, -1], 0]]
            im = ImageDisplay(r, extent=ext)
            im.ax.plot(*pGs[1::-1], 'ro', ms=15, mfc='none', mew=2)

        return pGs

    def _match_points_brute_yx(self, yx, fov, step_size, rotation=0.,
                               objective=objective_gmm_yx, *args, **kws):

        # create grid
        yx0 = y0, x0 = -np.multiply(fov, self._fov_search_stretch)
        y1, x1 = self.fov - yx0

        xres = int((x1 - x0) / step_size)
        yres = int((y1 - y0) / step_size)
        grid = np.mgrid[y0:y1:complex(yres),
               x0:x1:complex(xres)]

        # add 0s for angle grid
        z = np.full((1,) + grid.shape[1:], rotation)
        grid = np.r_[grid, z]  # FIXME: silly to stack rotation here..
        self.logger.info(
                "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
                *fov, yres, xres)

        # parallel
        r = gridsearch_mp(objective, grid, (self.yx, yx) + args, **kws)
        ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
        pGs = grid[:, i, j]
        self.logger.debug('Grid search optimum: %s', pGs)
        return grid, r, ix, pGs

    # def _match_points_brute(self, xy, fov, step_size, rotation=0.,
    #                         objective=objective_gmm, *args, **kws):
    #
    #     # create grid
    #     xy0 = x0, y0 = -np.multiply(fov, self._fov_search_stretch)
    #     x1, y1 = self.fov - xy0
    #
    #     xres = int((x1 - x0) / step_size)
    #     yres = int((y1 - y0) / step_size)
    #     grid = np.mgrid[x0:x1:complex(xres),
    #            y0:y1:complex(yres)]
    #
    #     # add 0s for angle grid
    #     z = np.full((1,) + grid.shape[1:], rotation)
    #     grid = np.r_[grid, z]  # FIXME: silly to stack rotation here..
    #     self.logger.info(
    #             "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
    #             *fov, yres, xres)
    #
    #     # parallel
    #     r = gridsearch_mp(objective, grid, (self.xy, xy) + args, **kws)
    #     ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
    #     pGs = grid[:, i, j]
    #     self.logger.debug('Grid search optimum: %s', pGs)
    #     return grid, r, ix, pGs

    # @timer
    # def _match_points_gd(self, xy, p0, fit_angle=True, sigma_gmm=0.03):
    #     """
    #     Match points with gradient descent on Gaussian mixture model likelihood
    #
    #     Parameters
    #     ----------
    #     xy: array
    #         xy coordinates of image features (center of mass of stars)
    #     p0: array-like
    #         parameter starting values
    #     sigma_gmm: float
    #          standard deviation of gaussian kernel for gmm
    #
    #     Returns
    #     -------
    #
    #     """
    #
    #     if fit_angle:
    #         f = ftl.partial(objective_gmm, self.xy, xy, sigma_gmm)
    #     else:
    #         theta = p0[-1]
    #         xy = roto_translate2(xy, [0, 0], theta)
    #         f = ftl.partial(objective_gmm2, self.xy, xy, sigma_gmm)
    #         p0 = p0[:2]
    #
    #     result = minimize(f, p0)
    #     if not result.success:
    #         return None
    #
    #     if fit_angle:
    #         return result.x
    #
    #     # noinspection PyUnboundLocalVariable
    #     return np.r_[result.x, theta]

    @timer
    def _match_points_gd_yx(self, yx, p0, fit_angle=True, sigma_gmm=0.03):
        """
        Match points with gradient descent on Gaussian mixture model likelihood

        Parameters
        ----------
        yx: array
            yx coordinates of image features (center of mass of stars)
        p0: array-like
            parameter starting values
        sigma_gmm: float
             standard deviation of gaussian kernel for gmm

        Returns
        -------

        """

        if fit_angle:
            f = ftl.partial(objective_gmm_yx, self.yx, yx, sigma_gmm)
        else:
            theta = p0[-1]
            yx = roto_translate_yx(yx, [0, 0, theta])
            f = ftl.partial(objective_gmm2_yx, self.yx, yx, sigma_gmm,
                            theta=theta)
            p0 = p0[:2]

        result = minimize(f, p0)
        if not result.success:
            return None

        if fit_angle:
            return result.x

        # noinspection PyUnboundLocalVariable
        return np.r_[result.x, theta]

    def _match_final(self, xy):

        from scipy.spatial.distance import cdist
        from recipes.containers.lists import where_duplicate

        # transform
        xy = self.to_pixel_coords(xy)
        ix = tuple(xy.round().astype(int).T)[::-1]
        labels = self.seg.data[ix]
        #
        use = np.ones(len(xy), bool)
        # ignore stars not detected in dss, but detected in sample image
        use[labels == 0] = False
        # ignore labels that are detected as 2 (or more) sources in the
        # sample image, but only resolved as 1 in dss
        for w in where_duplicate(labels):
            use[w] = False

        assert use.sum() > 3, 'Not good enough'

        d = cdist(self.to_pixel_coords(self.xy), xy[use])
        iq0, iq1 = np.unravel_index(d.argmin(), d.shape)
        xyr = self.to_pixel_coords(self.xy[iq0])
        xyn = xy[use] - xyr

        xyz = self.to_pixel_coords(self.xy[labels[use] - 1]) - xyr
        a = np.arctan2(*xyz.T[::-1]) - np.arctan2(*xyn.T[::-1])
        return np.median(a)

    def display(self, show_coms=True, number=True, cmap=None, marker_color='r'):
        """
        Display the image and the coordinates of the stars

        Returns
        -------

        """
        from graphing.imagine import ImageDisplay
        import more_itertools as mit

        ex = mit.interleave((0, 0), self.fov)
        im = ImageDisplay(self.images[0], cmap=cmap, extent=list(ex))

        if show_coms:
            if number:
                s = []
                for i, xy in enumerate(self.xy):
                    s.extend(
                            im.ax.plot(*xy, marker='$%i$' % i,
                                       color=marker_color)
                    )
            else:
                s, = im.ax.plot(*self.xy.T, 'rx')

        return im, s

    def mosaic(self, names=(), **kws):
        mp = MosaicPlotter(self)
        mp.mosaic(self.images, self.fovs, self.params, names,  # self.coms,
                  **kws)
        return mp


class ImageRegistrationDSS(ImageRegistration):
    """
    Image registration using Digitized Sky Survey images
    """
    #
    _servers = ('poss2ukstu_blue', 'poss1_blue',
                'poss2ukstu_red', 'poss1_red',
                'poss2ukstu_ir',
                'all')

    def __init__(self, name_or_coords, fov=(3, 3), **find_kws):
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

        for srv in self._servers:
            try:
                self.hdu = get_dss(srv, coords.ra.deg, coords.dec.deg, fov)
                break
            except STScIServerError:
                self.logger.warning('Failed to retrieve image from server: '
                                    '%r', srv)

        # DSS data array
        data = self.hdu[0].data.astype(float)
        find_kws.setdefault('deblend', True)
        ImageRegistration.__init__(self, data, fov, **find_kws)

        # TODO: print some header info for DSS image - date!
        # DATE-OBS
        # TELESCOP
        # INSTRUME
        # FILTER
        # EXPOSURE
        # SHAPE
        # FOV

        # save target coordinate position
        self.targetCoords = coords
        self.targetCoordsPix = np.divide(self.images[0].shape, 2) + 0.5

    # todo: def proper_motion_correction(self, coords):

    # def get_labels(self, xy):
    #     # transform
    #     ix = tuple(self.to_pixel_coords(xy).round().astype(int).T)[::-1]
    #     return self.seg.data[ix]
    #
    #     #
    #     use = np.ones(len(xy), bool)
    #     # ignore stars not detected in dss, but detected in sample image
    #     use[labels == 0] = False
    #     # ignore labels that are detected as 2 (or more) sources in the
    #     # sample image, but only resolved as 1 in dss
    #     for w in where_duplicate(labels):
    #         use[w] = False
    #     return labels, use

    def build_wcs(self, hdu, p, telescope=None):
        """
        Create tangential plane wcs
        Parameters
        ----------
        p
        cube
        telescope

        Returns
        -------
        astropy.wcs.WCS instance

        """

        from astropy import wcs

        *yxoff, theta = p
        fov = hdu.get_fov(telescope)
        pxscl = fov / hdu.shape[-2:]
        # transform target coordinates in DSS image to target in SHOC image
        h = self.hdu[0].header
        # target object pixel coordinates
        crpix = np.array([h['crpix1'], h['crpix2']])
        crpixDSS = crpix - 0.5  # convert to pixel llc coordinates
        cram = crpixDSS / self.pixel_scale  # convert to arcmin
        rot = rotation_matrix_2d(-theta)
        crpixSHOC = (rot @ (cram - yxoff)) / pxscl
        # target coordinates in degrees
        xtrg = self.targetCoords
        # coordinate increment
        cdelt = pxscl / 60  # in degrees
        flip = np.array(hdu.flip_state[::-1], bool)
        cdelt[flip] = -cdelt[flip]

        # see:
        # https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html
        # for parameter definitions
        w = wcs.WCS(naxis=2)
        # array location of the reference point in pixels
        w.wcs.crpix = crpixSHOC
        # coordinate increment at reference point
        w.wcs.cdelt = cdelt
        # coordinate value at reference point
        w.wcs.crval = xtrg.ra.value, xtrg.dec.value
        # rotation from stated coordinate type.
        w.wcs.crota = np.degrees([-theta, -theta])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type

        return w

    def plot_coords_nrs(self, coords):
        fig, ax = plt.subplots()

        for i, yx in enumerate(self.xy):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

        for i, yx in enumerate(coords):
            ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


# class TransformedImage():
#     def set_p(self):
#         self.art.set_transform()

class MosaicPlotter(object):
    """Plot the results from image registration run"""

    # TODO: make so that editing params, fovs changes plot live!!

    default_cmap_ref = 'Greys'
    alpha_cycle_value = 0.65

    def __init__(self, reg, use_aplpy=None, name=None):

        self.reg = reg
        if isinstance(self.reg, ImageRegistrationDSS):
            if use_aplpy is None:
                use_aplpy = True

            if name is None:
                header = self.reg.hdu[0].header
                name = ' '.join(
                        filter(None, map(header.get, ('ORIGIN', 'FILTER'))))

        self.use_aplpy = use_aplpy
        self.name = name
        self.names = []
        self.params = []
        self.fovs = []

        self.art = AttrDict()  # art
        self.image_label = None

        self._counter = itt.count()
        self._ff = None
        self._fig = None
        self._ax = None
        self._low_lims = (np.inf, np.inf)
        self._up_lims = (-np.inf, -np.inf)
        self._idx_active = 0
        self.alpha_cycle = []

        # connect image scrolling
        self.fig.canvas.mpl_connect('scroll_event', self._scroll_safe)
        self.fig.canvas.mpl_connect('button_press_event', self.reset)

    def setup(self):
        if self.use_aplpy:
            import aplpy as apl
            self._ff = ff = apl.FITSFigure(self.reg.hdu)
            self._ax = ff.ax
            self._fig = ff.ax.figure
            # f.add_grid()
        else:
            self._fig, self._ax = plt.subplots()

        self._fig.tight_layout()

    def _world2pix(self, p, fov):  # FIXME: this name is inappropriate
        # convert fov to the DSS pixel coordinates (aplpy)
        if self.use_aplpy:
            # scale_ratio = (fov / self.reg.fov)
            dsspix = (fov / self.reg.fov) * self.reg.data.shape
            y, x = p[:2] / self.reg.pixel_size + 0.5
            return (y, x, p[-1]), dsspix
        return p, fov

    @property
    def fig(self):
        if self._fig is None:
            self.setup()
        return self._fig

    @property
    def ax(self):
        if self._ax is None:
            self.setup()
        return self._ax

    def plot_image(self, image=None, fov=None, p=(0, 0, 0), name=None,
                   coords=None, frame=True, **kws):
        """

        Parameters
        ----------
        image
        fov
        p
        name
        frame
        kws

        Returns
        -------

        """

        if np.isnan(p).any():
            raise ValueError('Invalid parameter value(s)')

        update = True
        if image is None:
            assert len(self.reg.images)
            image = self.reg.images[0]
            fov = self.reg.fovs[0]
            update = False
            if name is None:
                name = self.name

        if name is None:
            name = 'image%i' % next(self._counter)

        # convert fov to the DSS pixel coordinates (aplpy)
        p, fov = self._world2pix(p, fov)
        self.params.append(p)
        self.fovs.append(fov)

        # plot
        # image = image / image.max()
        self.art[name] = plot_transformed_image(self.ax, image, fov, p, frame,
                                                set_lims=False, cbar=False,
                                                **kws)

        if coords is not None:
            # cxx = roto_translate_yx(coords, p)
            line, = self.ax.plot(*coords.T[::-1], 'x')
            # plot_points.append(line)

        self.names.append(name)
        if update:
            self.update_axes_limits(p, fov)

    def mosaic(self, images, fovs, params, names=(), coords=(), **kws):
        # mosaic plot

        cmap = kws.pop('cmap_ref', self.default_cmap_ref)
        cmap_other = kws.pop('cmap', None)

        n = len(images)
        alpha_magic = min(1 / n, 0.5)
        alpha = kws.setdefault('alpha', alpha_magic)

        for image, fov, p, name, coo in itt.zip_longest(
                images, fovs, params, names, coords):
            self.plot_image(image, fov, p, name, coo, cmap=cmap, **kws)
            cmap = cmap_other

        # always keep reference image for comparison when scrolling images
        self.alpha_cycle = np.vstack([np.eye(n) * self.alpha_cycle_value,
                                      np.ones(n) * alpha])
        self.alpha_cycle[:, 0] = 1

    def update_axes_limits(self, p, fov, delta=0.01):
        corners = get_corners(p, fov)
        self._low_lims = np.min([corners.min(0), self._low_lims], 0)
        self._up_lims = np.max([corners.max(0), self._up_lims], 0)
        # expand limits slightly beyond corners for aesthetic
        xlim, ylim = zip(self._low_lims * (1 - delta),
                         self._up_lims * (1 + delta))
        self.ax.set(xlim=xlim, ylim=ylim)

    def mark_target(self, name, colour='forestgreen', arrow_size=10,
                    arrow_head_distance=2.5, arrow_offset=(0, 0),
                    text_offset=3, **text_props):

        # TODO: determine arrow_offset automatically by looking for peak

        """

        Parameters
        ----------
        name
        colour
        arrow_size
        arrow_head_distance:
            distance in arc seconds from source location to the arrow head
            point for both arrows
        arrow_offset:
            xy offset in arc seconds for arrow point location
        text_offset:
            xy offset in reg pixels


        Returns
        -------

        """
        assert self.reg.targetCoordsPix is not None

        import numbers
        import matplotlib.patheffects as path_effects

        assert isinstance(arrow_head_distance, numbers.Real), \
            '`arrow_offset` should be float'

        # target indicator arrows
        # pixel_size_arcsec =
        arrow_offset = arrow_offset / self.reg.pixel_size / 60
        xy_target = self.reg.targetCoordsPix + arrow_offset

        arrows = []
        for i in np.eye(2):
            # quick and easy way to create arrows with annotation
            xy = xy_target + arrow_head_distance * i
            ann = self.ax.annotate('', xy, xy + arrow_size * i,
                                   arrowprops=dict(arrowstyle='simple',
                                                   fc=colour)
                                   )
            arrows.append(ann.arrow_patch)
        # text
        txt = self.ax.text(*(xy_target + text_offset), name,
                           color=colour, **text_props)
        # add border around text to make it stand out (like the arrows)
        txt.set_path_effects(
                [path_effects.Stroke(linewidth=2, foreground='black'),
                 path_effects.Normal()])

        return txt, arrows

    def label(self, indices, xy_offset=(0, 0), **text_props):
        texts = {}
        params = np.array(self.params[1:])
        fovs = np.array(self.fovs[1:])
        for name, idx in indices.items():
            xy = np.add(get_ulc(params[idx], fovs[idx])[::-1],
                        xy_offset)
            angle = np.degrees(params[idx, -1].mean())
            texts[name] = self.ax.text(*xy, name,
                                       rotation=angle,
                                       rotation_mode='anchor',
                                       **text_props)
        return texts

    def _scroll(self, event):
        """
        This method allows you to scroll through the images in the mosaic
        using the mouse.
        """
        if event.inaxes and len(self.art):
            # set alphas
            self._idx_active += [-1, +1][event.button == 'up']
            self._idx_active %= (len(self.art) + 1)  # wrap
            alphas = self.alpha_cycle[self._idx_active]
            for i, pl in enumerate(self.art.values()):
                pl.set_alpha(alphas[i])
                if i == self._idx_active:
                    # position -1 represents the original image
                    z = -1 if (i == -1) else 1
                    pl.set_zorder(z)
                else:
                    pl.set_zorder(0)

            if self.image_label is not None:
                self.image_label.remove()
                self.image_label = None

            # set tiles
            if self._idx_active != len(self.art):
                # position -1 represents the original image
                name = f'{self._idx_active}: {self.names[self._idx_active]}'
                p = self.params[self._idx_active]
                yx = np.atleast_2d([self.fovs[self._idx_active][0], 0])
                xy = roto_translate_yx(yx, p)[0, ::-1]
                self.image_label = self.ax.text(*xy, name, color='w',
                                                rotation=np.degrees(p[-1]),
                                                rotation_mode='anchor',
                                                va='top')

            # redraw
            self.fig.canvas.draw()

    def _scroll_safe(self, event):
        try:
            # print(vars(event))
            self._scroll(event)

        except Exception as err:
            import traceback

            print('Scroll failed:')
            traceback.print_exc()
            print('len(names)', len(self.names))
            print('self._idx_active', self._idx_active)

            self.image_label = None
            self.fig.canvas.draw()

    def reset(self, event):
        # reset original alphas
        if event.button == 2:
            self._idx_active = 0
            alphas = self.alpha_cycle[0]
            for i, pl in enumerate(self.art.values()):
                pl.set_alpha(alphas[i])
                if i == self._idx_active:
                    # position -1 represents the original image
                    z = -1 if (i == -1) else 1
                    pl.set_zorder(z)
                else:
                    pl.set_zorder(0)

        # redraw
        self.fig.canvas.draw()

# class DSSMosaic(MosaicPlotter):
#
#     def __init__(self, reg, use_aplpy=True, name=None):
#
#         if name is None:
#             header = self.reg.hdu[0].header
#             name = ' '.join(
#                     filter(None, map(header.get, ('ORIGIN', 'FILTER'))))
#
#         super().__init__(reg, use_aplpy, name)
#
#     def mosaic(self, images, fovs, params, names=(), coords=(), **kws):
#
#         super().mosaic(images, fovs, params, names, coords, **kws)
#
#         self.art[self.names[0]].set_cmap(self.default_cmap_ref)
#
#
