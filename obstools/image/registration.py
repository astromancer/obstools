"""
Helper functions to infer World Coordinate System given a target name or
coordinates of a object in the field. This is done by matching the image
with the DSS / SkyMapper image for the same field via image registration.  
A number of methods are implemented for doing this:
  point cloud drift
  matching via locating dense cluster of displacements between points
  direct image-to-image matching 
  brute force search with gaussian mixtures on points
"""

import collections as col
from recipes.misc import duplicate_if_scalar
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
from astropy.utils import lazyproperty

from recipes.containers.dicts import AttrDict
from obstools.image.segmentation import SegmentedImage

from pySHOC.utils import get_coordinates, get_dss, STScIServerError

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from recipes.logging import LoggingMixin
from recipes.transformations.rotation import rotation_matrix_2d, rotate

from scipy.stats import binned_statistic_2d, mode

from motley.profiling.timers import timer
from recipes.introspection.utils import get_module_name
from scipy.spatial import cKDTree
# from sklearn.cluster import MeanShift
from graphing.imagine import ImageDisplay
import matplotlib.pyplot as plt

# TODO: might be faster to anneal sigma param rather than blind grid search.
#  test this.

logger = logging.getLogger(get_module_name(__file__))

TABLE_STYLE = dict(txt='bold', bg='g')


def roto_translate(X, p):
    """
    rotate and translate a g1set of coordinate points in 2d

    Parameters
    ----------
    X: array, shape (n_samples, 2)
    p: x, y, θ

    Returns
    -------

    """
    # https://en.wikipedia.org/wiki/Rigid_transformation
    rot = rotation_matrix_2d(p[-1])
    Xnew = (rot @ np.transpose(X)).T + p[:2]
    return Xnew


def roto_translate2(X, xy_off=(0., 0.), theta=0.):
    """rotate and translate"""
    if theta:
        X = (rotation_matrix_2d(theta) @ X.T).T
    return X + xy_off


# def roto_translate_yx(X, p):
#     """rotate and translate"""
#     rot = rotation_matrix_2d(p[-1])
#     Xnew = (rot @ X[:, ::-1].T).T + p[1::-1]
#     return Xnew[:, ::-1]
#
#
# def roto_translate2_yx(X, yx_off=(0., 0.), theta=0.):
#     """rotate and translate"""
#     if theta:
#         X = (rotation_matrix_2d(theta) @ X[:, ::-1].T).T[:, ::-1]
#
#     return X + yx_off


# def check_transforms(yx, p):
#     z0 = roto_translate_yx(yx, p)
#
#     pxy = np.r_[p[1::-1], p[-1]]
#     z1 = roto_translate(yx[:, ::-1], pxy)[:, ::-1]
#
#     return np.allclose(z0, z1)


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

    # for higher dimensions
    # *_, n = xy.shape
    # volume = len(xy_trg) * (np.sqrt(2 * np.pi) * sigma) ** n
    return np.exp(
        -np.square(xy_trg[None] - xy[:, None]).sum(-1) / (2 * sigma * sigma)
    ).sum(-1)


def non_masked(xy):
    if np.ma.is_masked(xy):
        return xy[~xy.mask.any(-1)].data
    # elif np.ma.isMA(xy):
    #     return xy.data
    return np.array(xy)


def _objective_gmm(xy_trg, xy, sigma):
    # sum of gmm log likelihood

    # ignore masked points
    xy_trg = non_masked(xy_trg)
    xy = non_masked(xy)

    prob = prob_gmm(xy_trg, xy, sigma)
    bad = (prob == 0)
    # :              substitute 0 --> 1e-300         to avoid warnings and infs
    return -np.log(prob[~bad]).sum() + 300 * bad.sum()


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
    return _objective_gmm(xy_trg, xy_new, sigma)


def loglike_gmm(xy_trg, xy, sigma):
    # replace 0s with arbitrarily small number to avoid negative infinities
    prob = prob_gmm(xy_trg, xy, sigma)
    prob[prob == 0] = 1e-100  # ~ e ** -230
    return np.log(prob)


# def objective_gmm_yx(yx_trg, yx, sigma, p):
#     """Objective for gaussian mixture model"""
#     yx_new = roto_translate_yx(yx, p)
#     # xy_new = xy + p[:2]
#     return _objective_gmm(yx_trg, yx_new, sigma)
#
#
# def objective_gmm2_yx(yx_trg, yx, sigma, dyx, theta):
#     """Objective for gaussian mixture model"""
#     yx_new = roto_translate2_yx(yx, dyx, theta)
#     return _objective_gmm(yx_trg, yx_new, sigma)


def objective_gmm2(xy_trg, xy, sigma, xy_off, theta=0):
    xy_new = roto_translate2(xy, xy_off, theta)
    return _objective_gmm(xy_trg, xy_new, sigma)


def objective_gmm3(x_trg, x, sigma, xy_off, theta=0):
    xy, counts = x
    xy_new = roto_translate2(xy, xy_off, theta)
    return _objective_gmm(x_trg, xy_new, sigma)


def objective_pix(target, values, xy, bins, p):
    """Objective for direct image matching"""
    xy = roto_translate(xy, p)
    bs = binned_statistic_2d(*xy.T, values, 'mean', bins)
    rs = np.square(target - bs.statistic)
    return np.nansum(rs)


# def stack_stat()

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
        return np.exp(-np.square(self.xy[None] - xy[:, None]).sum(-1) /
                      (2 * self.sigma * self.sigma)).sum(-1)

    def objective(self, xy):
        """
        Negative log likelihood summed over all points xy

        Parameters
        ----------
        xy

        Returns
        -------
        #
        """

        return _objective_gmm(self.xy, xy, self.sigma)

    def objective_trans(self, p, xy):
        xy_new = roto_translate(xy, p)
        return _objective_gmm(self.xy, xy_new, self.sigma)

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
            result = minimize(self.objective_trans, p0, (xy,))
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

        xyl = self.xy.min(0)
        xyu = self.xy.max(0)
        (x0, y0), (x1, y1) = \
            np.mean((xyl, xyu), 0) + \
            np.array((-1, 1), ndmin=2).T * (xyu - xyl) * stretch / 2

        rx, ry = duplicate_if_scalar(size)
        return np.mgrid[x0:x1:(rx * 1j),
                        y0:y1:(ry * 1j)]

    def gridsearch_max(self, grid=None, gridsize=100):
        if grid is None:
            grid = self._auto_grid(gridsize)

        xy = grid.reshape(2, -1).T
        ll = self(xy).reshape(grid.shape[1:])
        return grid, ll, xy[ll.argmax()]

    def plot(self, grid=None, gridsize=100, show_peak=False, cmap='Blues',
             alpha=0.5):

        grid, ll, peak = self.gridsearch_max(grid, gridsize)

        extent = np.c_[grid[:, 0, 0], grid[:, -1, -1]].ravel()
        im = ImageDisplay(ll.T, extent=extent,
                          cmap=cmap, alpha=alpha)

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
#     seg = SegmentedImage.detect(image, mask, background, snr, npixels,
#                                     edge_cutoff, deblend, dilate)
#     return seg, seg.com_bg(image),


def detect_measure(image, mask=False, background=None, snr=3., npixels=5,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentedImage.detect(image, mask, background, snr, npixels,
                                edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    coords = seg.com_bg(image)
    return seg, coords, counts


# @timer
def offset_disp_cluster(xy0, xy1, sigma=None, plot=False):
    # find offset (translation) between frames by looking for the dense
    # cluster of inter-frame point-to-point displacements that represent the
    # xy offset between the two images. Cannot handle rotations.

    # Finding density peak in inter-frame distances is fast and accurate
    # under certain conditions only.
    #  - Images have large overlap. There should be more shared points
    #    than points that are in one frame only
    # -  Images are roughly to the same depth (i.e roughly the same point
    #    density (this is done by `hdu.sampler.to_depth`)

    # NOTE: also not good for large number of points in terms of performance and
    #   accuracy...
    # NOTE: this was a quaint first attempt, but much better way to find the
    #  offset is to use `ImageRegistration._dxy_brute

    points = (xy0[None] - xy1[:, None]).reshape(-1, 2).T

    if sigma is None:
        # size of gaussians a third of closest distance between stars
        sigma = min(dist_flat(xy0).min(), dist_flat(xy1).min()) / 3

    gmm = GaussianMixture(points.T, sigma)
    grid, ll, peak = gmm.gridsearch_max()

    tree = cKDTree(points.T)
    idx_nn = tree.query_ball_point(peak, 2 * sigma)
    off = points[:, idx_nn].mean(1)

    if plot:
        from matplotlib.patches import Rectangle
        im, _ = gmm.plot()
        im.ax.plot(*points[:, idx_nn], 'o', ms=7, mfc='none')

    # old algorithm using histogram. Not robust since bins may cut dense
    # clusters of points
    # vals, xe, ye = np.histogram2d(*points, bins)
    # # todo probably raise if there is no significantly dense cluster
    # i, j = np.unravel_index(vals.argmax(), vals.shape)
    # l = ((xe[i] < points[0]) & (points[0] <= xe[i + 1]) &
    #      (ye[j] < points[1]) & (points[1] <= ye[j + 1]))
    # sub = points[:, l].T
    # yay = sub[np.sum(cdist(sub, sub) > dmax, 0) < (len(sub) // 2)]
    # off = np.mean(yay, 0)

    return off


def dist_flat(coo):
    """lower triangle of (symmetric) distance matrix"""
    n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    ix = np.tril_indices(n, -1)
    return sdist[ix]


# @timer
def gridsearch_mp(objective, grid, args=(), **kws):
    # grid search

    from joblib import Parallel, delayed

    # f = ftl.partial(objective, *args, **kws)
    ndim, *rshape = grid.shape

    n_jobs = 1  # make sure you can pickle everything before you change
    # this value
    with Parallel(n_jobs=n_jobs) as parallel:
        results = parallel(delayed(objective)(p, *args, **kws)
                           for p in grid.reshape(ndim, -1).T)

    return np.reshape(results, rshape)


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
    c = np.array([[0, 0], fov[::-1]])  # lower left, upper right xy
    # corners = np.c_[c[0], c[:, 1], c[1], c[::-1, 0]].T  # / clockwise yx
    # corners = roto_translate_yx(corners, p)
    corners = np.c_[c[0], c[::-1, 0], c[1], c[:, 1]].T  # / clockwise xy
    corners = roto_translate(corners, p)
    return corners  # return xy ! [:, ::-1]


def _get_ulc(p, fov):
    return roto_translate([0, fov[0]], p)


def get_ulc(params, fovs):
    """
    Get upper left corners of off all frames given roto-translation
    parameters and field of view
    """
    ulc = np.empty((len(params), 2))
    for i, (p, fov) in enumerate(zip(params, fovs)):
        ulc_ = np.array([[0, fov[0]]])
        ulc[i] = roto_translate(ulc_, p)
    return ulc[:, 0].min(), ulc[:, 1].max()  # xy


def plot_transformed_image(ax, image, fov=None, p=(0, 0, 0), frame=True,
                           set_lims=True, **kws):
    """

    Parameters
    ----------
    ax
    image
    fov
    p
    frame
    set_lims
    kws

    Returns
    -------

    """

    kws.setdefault('hist', False)
    kws.setdefault('sliders', False)

    # plot
    im = ImageDisplay(image, ax=ax, **kws)
    art = im.imagePlot

    # set extent
    if fov is None:
        fov = image.shape

    extent = np.c_[[0., 0.], fov[::-1]]
    pixel_size = np.divide(fov, image.shape)
    half_pixel_size = pixel_size / 2
    extent -= half_pixel_size[None].T  # adjust to pixel centers...
    art.set_extent(extent.ravel())

    # Rotate the image by setting the transform
    *xy, theta = p  # * fov, p[-1]
    art.set_transform(Affine2D().rotate(theta).translate(*xy) +
                      art.get_transform() # ax.transData
                      )

    if bool(frame):
        from matplotlib.patches import Rectangle

        frame_kws = dict(fc='none', lw=1, ec='0.5', alpha=kws.get('alpha', 0.3))
        if isinstance(frame, dict):
            frame_kws.update(frame)

        frame = Rectangle(xy - half_pixel_size, *fov[::-1], np.degrees(theta),
                          **frame_kws)
        ax.add_patch(frame)

    if set_lims:
        delta = 1 / 100
        c = get_corners(p, fov)
        xlim, ylim = np.vstack([c.min(0), c.max(0)]).T * (1 - delta, 1 + delta)
        im.ax.set(xlim=xlim, ylim=ylim)

    return art, frame


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


# def register_constellation(clustering, coms, centre_distance_max=1,
#                            f_detect_measure=0.5, plot=False, **plot_kws):
#     #
#     from collections import Callable
#     if isinstance(plot, Callable):
#         display = plot
#         plot = True
#     else:
#         plot = bool(plot)
#
#         def display(*_):
#             pass
#     #  🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#     # clustering + relative position measurement
#     logger.info('Identifying stars')
#     n_clusters, n_noise = cluster_id_stars(clustering, coms)
#     xy, = group_features(clustering, coms)
#
#     if plot:
#         import matplotlib.pyplot as plt
#
#         fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
#         ax.set_title(f'Position Measurements (CoM) {len(coms)} frames')
#
#         # TODO: plot boxes on this image corresponding to those below
#         cmap = plot_clusters(ax, np.vstack(coms)[:, ::-1], clustering)
#         # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
#         #                   tuple(zip((0, 0), ishape)))))
#         display(fig0)
#
#     #
#     logger.info('Measuring relative positions')
#     _, centres, σ_xy, xy_offsets, outliers = \
#         measure_positions_offsets(xy, centre_distance_max, f_detect_measure)
#
#     # zero point for tracker (slices of the extended frame) correspond
#     # to minimum offset
#     # xy_off_min = xy_offsets.min(0)
#     # zero_point = np.floor(xy_off_min)
#
#     # 🎨🖌~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     if plot:
#         # diagnostic for source location measurements
#         from obstools.phot.diagnostics import plot_position_measures
#
#         # plot CoMs
#         fig1, axes = plot_position_measures(xy, centres, xy_offsets, **plot_kws)
#         fig1.suptitle('Raw Positions (CoM)')  # , fontweight='bold'
#         display(fig1)
#
#         fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
#         ax.set_title(f'Re-centred Positions (CoM) {len(coms)} frames')
#         rcx = np.vstack([c - o for c, o in zip(coms, xy_offsets)])
#         plot_clusters(ax, rcx[:, ::-1], clustering, cmap=cmap)
#
#         # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
#         #                   tuple(zip((0, 0), ishape)))))
#         display(fig0)
#
#     return centres, σ_xy, xy_offsets, outliers, xy


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

    logger.info('Clustering %i position measurements using:\n\t%s', n,
                str(clf).replace('\n', '\n\t'))
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


def plot_clusters(ax, features, labels, colours=(), cmap=None, **scatter_kws):
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
    for k, v in dict(s=25,
                     marker='.',
                     facecolors='none').items():
        scatter_kws.setdefault(k, v)
    #
    art = ax.scatter(*features[core_sample_indices_].T,
                     edgecolors=colours, **scatter_kws)
    ax.plot(*features[labels == -1].T, 'kx', alpha=0.3)
    ax.grid()
    return art


# def _sanitize_positions(xy):
#     n, n_stars, _ = xy.shape
#     nans = np.isnan(np.ma.getdata(xy))
#     bad = (nans | np.ma.getmask(xy)).any(-1)
#     ignore_frames = bad.all(-1)
#     n_ignore = ignore_frames.sum()
#
#     assert n_ignore != n
#     if n_ignore:
#         logger.info(
#                 'Ignoring %i/%i (%.1f%%) nan / masked values for position '
#                 'measurement', n_ignore, n, (n_ignore / n) * 100)
#
#     # mask nans.  masked
#     return np.ma.MaskedArray(xy, nans)


def compute_centres_offsets(xy, d_cut=None, detect_freq_min=0.9, report=True):
    """
    Measure the relative positions of detected stars from the individual
    location measurements in xy.  Use the locations of the most-often
    detected individual objects

    Parameters
    ----------
    xy:     array, shape (n_points, n_stars, 2)
    d_cut:  float
    detect_freq_min: float
        Required detection frequency of individual sources in order for
        them to be used

    Returns
    -------
    xy, centres, σxy, δxy, outlier_indices
    """
    assert 0 < detect_freq_min < 1

    n, n_stars, _ = xy.shape
    nans = np.isnan(np.ma.getdata(xy))

    # mask nans.  masked
    xy = np.ma.MaskedArray(xy, nans)

    # Due to varying image quality and or camera/telescope drift,
    # some  stars (those near edges of the frame, or variable ones) may
    # not be detected in many frames. Cluster centroids are not an accurate
    # estimator of relative position for these stars since it's an
    # incomplete sample. Only stars that are detected in at least
    # `detect_freq_min` fraction of the frames will be used to calculate
    # frame xy offset.
    # Any measure of centrality for cluster centers is only a good estimator
    # of the relative positions of stars when the camera offsets are
    # taken into account.

    bad = (nans | np.ma.getmask(xy)).any(-1)
    ignore_frames = nans.all((1, 2))
    n_ignore = ignore_frames.sum()
    n_use = n - n_ignore
    if n_ignore == n:
        raise Exception('All points are masked!')

    if n_ignore:
        logger.info('Ignoring %i/%i (%.1f%%) nan values for position '
                    'measurement', n_ignore, n, (n_ignore / n) * 100)

    n_detections_per_star = np.empty(n_stars, int)
    w = np.where(~bad)[1]

    n_detections_per_star[np.unique(w)] = np.bincount(w)
    f_det = (n_detections_per_star / n_use)
    use_stars = f_det > detect_freq_min
    i_use, = np.where(use_stars)
    if not len(i_use):
        raise Exception('Detected frequently for all sources appears to '
                        'be too low. There are %i objects across %i images.'
                        ' Their detection frequencies are: %s' %
                        (n_stars, n, f_det))

    if np.any(~use_stars):
        logger.info('Ignoring %i / %i stars with low (<=%.0f%%) detection '
                    'frequency for frame offset measurement.',
                    n_stars - len(i_use), n_stars, detect_freq_min * 100)

    # NOTE: we actually want to know where the cluster centres would be
    #  without a specific point to measure delta better

    # first estimate of relative positions comes from unshifted cluster centers
    # Compute cluster centres as geometric median
    good = ~ignore_frames
    xyc = xy[good][:, use_stars]
    nansc = nans[good][:, use_stars]
    if nansc.any():
        # prevent warning emit in _measure_positions_offsets
        xyc[nansc] = 0
        xyc[nansc] = np.ma.masked

    # delay centre compute for fainter stars until after re-centering
    centres = np.empty((n_stars, 2))  # ma.masked_all
    for i, j in enumerate(i_use):
        centres[j] = geometric_median(xyc[:, i])

    # ensure output same size as input
    δxy = np.ma.masked_all((n, 2))
    σxy = np.empty((n_stars, 2))  # σxy = np.ma.masked_all((n_stars, 2))

    # compute positions of all sources with frame offsets measured from best
    # and brightest stars
    centres[use_stars], σxy[use_stars], δxy[good], out = \
        _measure_positions_offsets(xyc, centres[use_stars], d_cut)
    #
    for i in np.where(~use_stars)[0]:
        # mask for bad frames in δxy will propagate here
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
        report_measurements(xy, centres, σxy, δxy, None, detect_freq_min)
        #                                        # counts
    return xy, centres, σxy, δxy, outlier_indices


def _measure_positions_offsets(xy, centres, d_cut=None):
    # FIXME: THIS IS ESSENTIALLY SIGMA CLIPPING

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
            # break out here without removing any points
            return centres, xy_shifted.std(0), xy_offsets.squeeze(), outliers

        # FIXME: break out if there are too few points for the concept of
        #  "outlier" to be meaningful

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
    # sigma0 = xy.std(0)
    # var_reduc = (sigma0 - σ_xy) / sigma0
    # mad0 = mad(xy, axis=0)
    # mad1 = mad((xy - xy_offsets[:, None]), axis=0)
    # mad_reduc = (mad0 - mad1) / mad0

    # # overall variance report
    # s0 = xy.std((0, 1))
    # s1 = (xy - xy_offsets[:, None]).std((0, 1))
    # # Fractional variance change
    # logger.info('Differencing change overall variance by %r',
    #             np.array2string((s0 - s1) / s0, precision=3))

    # FIXME: percentage format in total wrong
    # TODO: align +- values
    col_headers = ['n (%)', 'x', 'y']  # '(px.)'
    formatters = {0: ftl.partial(pprint.decimal_with_percentage,
                                 total=n_points, precision=0, right_pad=1)}  #

    if detect_frac_min is not None:
        n_min = detect_frac_min * n_points
        formatters[0] = ConditionalFormatter('y', op.le, n_min, formatters[0])

    # get array with number ± std representations
    columns = [points_per_star,
               pprint.uarray(centres, σ_xy, 2)[:, ::-1]]
    # FIXME: don't print uncertainties if less than 6 measurement points

    if counts is not None:
        # TODO highlight counts?
        cn = 'counts (e⁻)'
        formatters[cn] = ftl.partial(pprint.numeric, thousands=' ', precision=1,
                                     compact=False)
        if count_thresh:
            formatters[cn] = ConditionalFormatter('c', op.ge, count_thresh,
                                                  formatters[cn])
        columns.append(counts)
        col_headers += [cn]

    # add variance columns
    # col_headers += ['r_σx', 'r_σy'] + ['mr_σx', 'mr_σy']
    # columns += [var_reduc[:, ::-1], mad_reduc[:, ::-1]]

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
                             formatters=formatters)

    # fix formatting with percentage in total.  Still need to think of a
    # cleaner solution for this
    tbl.data[-1, 0] = tbl.data[-1, 0].replace('(1000%)', '')

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


class TransformedImageGrids(col.defaultdict, LoggingMixin):
    """
    Container for segment masks
    """

    def __init__(self, reg):
        self.reg = reg
        col.defaultdict.__init__(self, None)

    def __missing__(self, i):
        # the grid is computed at lookup time here and inserted into the dict
        # after this func executes
        shape = self.reg.images[i].shape
        r = (self.reg.fovs[i] / shape) / self.reg.pixel_scale
        g = np.indices(shape).reshape(2, -1).T * r
        g = roto_translate(g[:, ::-1], self.reg.params[i])
        return g  # .reshape ??


# from obstools.phot.image import SourceDetectionMixin

class ImageRegistration(LoggingMixin):  # ImageRegister
    """
    A class for image registration and basic astrometry of astronomical images
    """

    # default source finding
    _dflt_find_kws = dict(snr=3.,
                          npixels=5,
                          edge_cutoff=2,
                          deblend=False)

    # Expansion factor for grid search
    _search_area_stretch = 1.25

    # Sigma for GMM as a fraction of minimal distance between stars
    _dmin_frac_sigma = 3

    # TODO: switch for units to arcminutes or whatever ??

    # @classmethod
    # def from_image(cls, fov, **find_kws):

    def detect(self, image, fov, **kws):
        # find stars
        # defaults
        for k, v in self._dflt_find_kws.items():
            kws.setdefault(k, v)

        seg, coo, counts = detect_measure(image, **kws)
        # TODO: uncertainty on center of mass from pixel noise!!!

        # transform to the pixel coordinates of the reference image by scaling
        ratio = np.divide(fov, image.shape) / self.pixel_scale
        xy = (coo * ratio)[:, ::-1]
        return seg, xy, counts

    # def to_pixel_coords(self, xy):
    #     # internal coordinates are in arcmin origin at (0,0) for image
    #     return np.divide(xy, self.pixel_scale)

    def convert_to_pixels_of(self, imr):  # to_pixels_of
        # convert coordinates to pixel coordinates of reference image
        ratio = self.pixel_scale / imr.pixel_scale
        self.xy *= ratio

        params = np.array(self.params)
        params[:, :2] *= ratio
        self.params = list(params)

        for cx in self.coms:
            cx *= ratio

        # NOTE this means reg.mosaic will no longer work without fov keyword
        #  since we have changed the scale and it substitutes image shape for
        #  fov if not given

        # return xy, coms, params

    @classmethod  # .            p0 -----------
    def from_images(cls, images, fovs, angles=(), ridx=None, plot=False,
                    **find_kws):

        n = len(images)
        assert n
        assert len(fovs)
        assert n == len(fovs)

        # align on highest res image if not specified
        shapes = list(map(np.shape, images))
        pixel_scales = np.divide(fovs, shapes)
        if ridx is None:
            ridx = pixel_scales.argmin(0)[0]

        indices = np.arange(len(images))
        if ridx:  # put this index first
            indices[0], indices[ridx] = indices[ridx], indices[0]

        cls.logger.info('Aligning %i images on image %i', n, ridx)
        if not len(angles):
            angles = np.zeros(n)
        else:
            angles = np.array(angles) - angles[ridx]  # relative angles!

        reg = cls()
        for i in indices:
            reg(images[i], fovs[i], angles[i], plot=plot)

        # re-order everything
        for at in 'images, fovs, detections, coms, counts, params'.split(', '):
            setattr(reg, at, list(map(getattr(reg, at).__getitem__, indices)))

        reg.register_constellation()
        return reg

    @property
    def image(self):
        return self.images[self.idx]

    @property
    def fov(self):
        return self.fovs[self.idx]

    @property
    def pixel_scale(self):
        return self.fov / self.image.shape

    @property
    def xy(self):
        # Ensure xy coordinates are always a plain numpy array.  Masked arrays
        # don't seem to work well with the matrix product `@` operator
        return self._xy

    @xy.setter
    def xy(self, xy):
        self._xy = non_masked(xy)
        del self.model

    # @lazyproperty
    @property
    def xyt(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters
        """
        return [roto_translate(xy, p)
                for p, xy in zip(self.params, self.coms)]

    # @lazyproperty
    @property
    def xyt_block(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters and group them according to the cluster identified labels
        """
        return group_features(self.labels, self.xyt)[0]

    @lazyproperty
    def model(self):
        return ConstellationModel(self.xy, self.guess_sigma())


    def __init__(self, images=(), fovs=(), params=(), **find_kws):
        """
        Initialize an image register.
        This class should generally be initialized without arguments. The model
        of the constellation of stars is built iteratively by calling an
        instance of this class on an image.
        eg:
        >>> ImageRegistration()(image)

        If arguments are provided, they are the sequences of images,
        their field-of-views, and transformation parameters for the
        initial (zero-point) transform

        Parameters
        ----------
        image:
        fov:
            Field of view for the image. Order of the dimensions should be
            the same as that of the image, ie. rows first
        find_kws
        """

        # TODO: FOV is equivalent to the scaling parameters of an affine
        #  transform. Add scaling to params elliminates need for keeping fov.

        # data array
        # im = Image(image, fov[::-1])
        # self.image = np.asarray(image)
        # self.fov = np.array(fov)  # yx!!
        # # pixel size in arcmin yx
        # self.pixel_scale = (self.fov / image.shape)
        n = len(images)
        assert n == len(fovs) == len(params)

        # NOTE passing a reference index is only meaningful if the class is
        #  initialized with a set of images
        self.idx = 0

        # containers for matched images
        self.targetCoordsPix = None
        self.sigmas = None
        self._xy = None
        self.images = []
        self.fovs = []
        self.detections = []
        self.coms = []
        self.counts = []
        self.params = []
        self.grids = TransformedImageGrids(self)

        # choose sigma for GMM based on distances between detections
        # self._sigma_guess = self.guess_sigma(self.xy)
        # self.model = ConstellationModel(self.xy, self._sigma_guess)

        # state variables
        self.labels = self.n_stars = self.n_noise = None
        self._colour_sequence_cache = ()

        # parameters for zero point transforms
        # indices = list(set(range(len(images))) - {ridx})
        if len(params):
            params = np.array(params)
            segs, coords, counts = zip(*map(detect_measure, images))
            self.aggregate(images, fovs, segs, coords, counts, params)
            return

        # if n:
        #     shapes = list(map(np.shape, self.images))
        #     pixel_scales = np.divide(fovs, shapes)
        #
        #     # align on highest res image if not specified
        #     idx = ridx
        #     if ridx is None:
        #         idx = pixel_scales.argmin(0)[0]
        #
        #     others = set(range(n)) - {idx}

        # self.aggregate(self.image, self.fov, seg, self.xy, counts, np.zeros(3))

    def __call__(self, image, fov, rotation=0., plot=False, **find_kws):
        """
        Run `match_image` and aggregate results. If this is the first time
        calling this method, the reference image is set.

        Parameters
        ----------
        image
        fov
        rotation
        plot

        Returns
        -------

        """

        # first image will be the de facto reference frame
        if len(self.images) == 0:
            # Detect stars in reference frame
            # np.divide(fov, image.shape)
            seg, yx, counts = detect_measure(image, **find_kws)
            # `xy` are the initial target coordinates for matching. Theses will
            # be updated in `recentre`
            p = np.zeros(3)
            good = np.isfinite(yx).all(1)
            self._xy = xy = yx[good, ::-1]
            counts = counts[good]
            self._min_dist = dist_flat(self.xy).min()
        else:
            p, xy, counts, seg = self.match_image(image, fov, rotation, plot)

        # aggregate
        self.aggregate(image, fov, seg, xy, counts, p)

        # update
        self._min_dist = min(self._min_dist, dist_flat(xy).min())
        # self._sigma_guess = min(self._sigma_guess, self.guess_sigma(xy))
        # self.logger.debug('sigma guess: %s' % self._sigma_guess)

        # reset model
        del self.model

        return p, xy

    def __len__(self):
        return len(self.params)

    def aggregate(self, *args):
        """
        Aggregate the results from another `ImageRegister`, or explicitly
        append image, fov, seg, xy, counts, params
        """
        if len(args) == 1 and isinstance(self, ImageRegistration):
            reg = args[0]
            images, fovs, segs, xy, counts, params = \
                reg.images, reg.fovs, reg.detections, reg.coms, reg.counts, reg.params
        elif len(args) == 6:
            # detect if sequences provided
            images = args[0]
            if isinstance(images, np.ndarray) and images.ndim == 2:
                # if (set(map(type, args)) - {list, tuple, np.ndarray}):
                args = tuple(zip(args))

            # check consistent sizes
            assert len(set(map(len, args))) == 1, \
                'Unequally sized input containers'

            images, fovs, segs, xy, counts, params = args
        else:
            raise ValueError('Invalid number of arguments')

        self.images.extend(images)
        self.fovs.extend(np.array(fovs))
        self.detections.extend(segs)
        self.coms.extend(xy)  # [good]
        self.counts.extend(counts)  # [good]
        self.params.extend(params)

    def min_dist(self):
        return min(dist_flat(xy).min() for xy in self.coms)

    def guess_sigma(self):
        return self._min_dist / self._dmin_frac_sigma

    def match_hdu(self, hdu, depth=10, sample_stat='median', **findkws):
        """

        Parameters
        ----------
        hdu
        depth
        sample_stat

        Returns
        -------

        """
        image = hdu.get_sample_image(depth, sample_stat)
        return self(image, hdu.fov, **findkws)

    def match_reg(self, reg, angle=0):
        # cross match with another image register (instance of ImageRegistration)

        # convert coordinates to pixel coordinates of reference image
        reg.convert_to_pixels_of(self)

        reg.copy()


        # use `match points` so we don't aggregate data just yet
        xy = rotate(reg.xy.T, angle).T
        p = self.match_points(xy, 0)
        params = np.array(reg.params)
        params[:, :2] += p[:2]
        params[:, -1] += angle
        reg.params = list(params)

        # finally aggregate the results from all the registers
        self.aggregate(reg)

        # convert back to original coords
        reg.convert_to_pixels_of(reg)

        # fig, ax = plt.subplots()
        # for xy in reg.xyt:
        #     ax.plot(*xy.T, 'x')
        # ax.plot(*imr.xy.T, 'o', mfc='none', ms=8)

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
        seg, xy, counts = self.detect(image, fov)

        # match images directly
        # mr = self.match_pixels(image, fov[::-1], p)

        #
        p = self.match_points(xy, rotation, plot)
        # p = self.match_points(xy, fov, rotation, plot)

        # return yx (image) coordinates
        # yx = roto_translate_yx(yx, p)

        return p, xy, counts, seg

    # @timer
    def match_points(self, xy, rotation=0., plot=False):

        if rotation is None:
            raise NotImplementedError
            #  match gradient descent
            # p = self.model.fit_trans(pGs, xy, fit_angle=True)

        # get shift (translation) via brute force search through point to
        # point distances. This is gauranteed to find the optimal alignment
        # between the two point clouds in the absense of rotation between them
        dxy = self._dxy_brute(xy, rotation, plot=plot)
        p = np.r_[dxy, rotation]
        return p

    def register_constellation(self, clustering=None, plot=False):

        # clustering + relative position measurement
        clustering = clustering or self.get_clf()
        self.cluster_id(clustering)
        # make the cluster centres the target constellation
        self.xy = self.xyt_block.mean(0)

        if plot:
            from matplotlib.patches import Circle

            #
            art = self.plot_clusters()
            ax = art.axes

            # bandwidth size indicator.
            # TODO: get this to remain anchored lower left but scale with zoom..
            bw = clustering.bandwidth
            xy = self.xy.min(0)  # - bw * 0.6
            cir = Circle(xy, bw, alpha=0.5)
            ax.add_artist(cir)

    def get_centres(self):
        """
        Cluster centers via geometric median ignoring noise points

        Returns
        -------

        """
        # this ignores noise points from clustering
        centres = np.empty((self.n_stars, 2))  # ma.masked_all
        for i, xy in enumerate(np.rollaxis(self.xyt_block, 1)):
            centres[i] = geometric_median(xy)

        return centres

    def update_centres(self):
        """
        Measure cluster centers and set those as updated target positions for
        the constellation model
        """
        self.xy = self.get_centres()

    def get_clf(self):
        from sklearn.cluster import MeanShift
        # minimal distance between stars distanc
        return MeanShift(bandwidth=self.min_dist() / 3, cluster_all=False)

    def cluster_id(self, clustering):
        # clustering to cross-identify stars
        # assert len(self.params)

        logger.info('Identifying stars')
        self.n_stars, self.n_noise = cluster_id_stars(clustering, self.xyt)

        # sanity check
        # n_stars_most = np.vectorize(len, (int,))(xy).max()
        # if n_stars_most < 2 * self.n_stars:
        #     raise Exception('Looks like clustering produced too many clusters.'
        #                     'Image with most detections has %i; clustering '
        #                     'has %i clusters.' % (n_stars_most, self.n_stars))
        #

        self.labels = clustering.labels_

    def plot_clusters(self):

        n = len(self.coms)
        fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        ax.set_title(f'Position Measurements (CoM) {n} frames')

        art = plot_clusters(ax, np.vstack(self.xyt), self.labels)
        self._colour_sequence_cache = art.get_edgecolors()

        # TODO: plot position error ellipses

        # ax.set(**dict(zip(map('{}lim'.format, 'yx'),
        #                   tuple(zip((0, 0), ishape)))))
        return art

    def recentre(self, centre_distance_cut=None, f_detect_measure=0.25,
                 plot=False):
        """
        Measure frame dither, recenter, and recompute object positions.
        This re-centering algorithm can accurately measure the position of
        sources in the global frame taking frame-to-frame offsets into
        account. The xy offset of each image is measured from the mean
        xy-offset of the brightest stars from their respective cluster centres.
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
        xy = self.xyt_block

        logger.info('Measuring cluster centres, frame xy-offsets')
        _, centres, xy_std, xy_offsets, outliers = \
            compute_centres_offsets(xy, centre_distance_cut, f_detect_measure)

        # decide whether to accept new params! likelihood ratio test
        lhr = self._lh_ratio(np.vstack(xy),
                             np.vstack(xy - xy_offsets[:, None]))
        if lhr > 1:
            # some of the offsets may be masked. ignore those
            good = ~xy_offsets.mask.any(1)
            for i in np.where(good)[0]:
                self.params[i][:-1] -= xy_offsets[i]

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
            ax.set_title(f'Re-centred Positions (CoM) {len(self.coms)} frames')
            plot_clusters(ax, np.vstack(self.xyt), self.labels,
                          self._colour_sequence_cache)

        return xy_offsets, outliers, xy

    def solve_rotation(self):
        # for point clouds that ate already quite well aligned. Refine by
        # looking for rotation differences
        xy = self.xyt_block

        # get star that is in most of the frames (likely the target)
        i = (~xy.mask).any(-1).sum(-1).argmax()
        # idx = xy.std((0, -1)).argmin()
        c = self.xy[i]
        aref = np.arctan2(*(self.xy - c).T[::-1])
        a = np.arctan2(*(xy - c).T[::-1])
        # angular offset
        return np.mean(a - aref[:, None], 0)

    # TODO: refine_mcmc

    def refine(self, fit_angle=True, plot=False):
        """
        Refine alignment parameters by fitting transfor parameters for each
        image using maximum likelihood objective for gmm model with peaks
        situated at cluster centers

        """
        xyt = self.xyt
        params = np.array(self.params)
        # guess sigma:  needs to be larger than for brute search `_dxy_brute`
        # since we are searching a smaller region of parameter space
        # self.model.sigma = dist_flat(self.xy).min() / 3

        failed = []
        for i in range(len(self.images)):
            p = self.model.fit_trans((0, 0, 0), xyt[i], fit_angle)
            if p is None:
                failed.append(i)
            else:
                params[i] += p

        self.logger.info('Fitting successful %i / %i', i - len(failed), i)

        # likelihood ratio test
        xyn = [roto_translate(xy, p)
               for p, xy in zip(params, self.coms)]
        lhr = self._lh_ratio(np.vstack(xyt), np.vstack(xyn))
        better = (lhr > 1)
        # decide whether to accept new params!
        if better:
            # recompute cluster centers
            self.params = list(params)
            self.update_centres()

        if plot:
            fig2, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
            ax.set_title(f'Refined Positions (CoM) {len(self.coms)} frames')
            plot_clusters(ax, np.vstack(self.xyt), self.labels,
                          self._colour_sequence_cache)
        return params, better

    def _lh_ratio(self, xy0, xy1):
        # likelihood ratio
        ratio = np.exp(self.model.objective(xy0) - self.model.objective(xy1))
        self.logger.info('Likelihood ratio: %.5f', ratio)

        # decide whether to accept new params!
        if ratio > 1:
            self.logger.info('Accepting new parameters.')
        else:
            self.logger.info('Keeping same parameters.')
        return ratio

    def reset(self):
        """
        Reset image, fov, detection  containers to empty

        Returns
        -------

        """

        self.images = []
        self.fovs = []
        self.detections = []
        self.coms = []
        self.counts = []
        self.params = []

    @property
    def scales(self):
        return np.divide(self.fovs, list(map(np.shape, self.images)))

    def _marginal_histograms(self):

        grid, pixels = self._stack_pixels()
        x, y = grid.T

        fig, axes = plt.subplots(2, 1)
        axes[0].hist(x, pixels, bins=250)
        axes[1].hist(y, pixels, bins=250)

    def _stack_pixels(self, indices=...):
        if indices is ...:
            indices = range(len(self.images))

        sizes = [self.images[i].size for i in indices]
        secs = map(slice, sizes, sizes[1:] + [None])
        n = sum(sizes)
        grid = np.empty((n, 2))
        pixels = np.empty(n)
        for i, sec in zip(indices, secs):
            grid[sec] = self.grids[i]
            pixels[sec] = self.images[i].ravel()

        return grid, pixels

    def stack_stat(self, stat='mean', indices=...):
        if indices is ...:
            indices = range(len(self.images))

        grid = []
        pixels = []
        scales = self.scales
        for i in indices:
            grid.extend(self.grids[i])
            pixels.extend(self.images[i].ravel())

        grid = np.array(grid)
        bins = np.round(grid.ptp(0) * scales.min() / scales[0]).astype(int)

        bs = binned_statistic_2d(*grid.T, pixels, stat, bins)

    def match_pixels(self, index):

        indices = set(range(len(self.images))) - {index}
        grid = []
        pixels = []
        for i in indices:
            image = self.images[i]
            r = np.divide(self.fovs[i], image.shape) / self.pixel_scale
            p = self.params[i]
            seg = self.detections[i].dilate(2, copy=True)
            for sub, g in seg.coslice(image, np.indices(image), flatten=True):
                g = roto_translate(g.reshape(-1, 2) * r, p)
                grid.extend(g)
                pixels.extend(sub.ravel())

        bs = binned_statistic_2d(*grid, pixels, 'mean', bins)

    def _match_pixels(self, image, fov, p0):
        """Match pixels directly"""
        sy, sx = self.image.shape
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
    def match_points_brute(self, xy, rotation=0., gridsize=(50, 50),
                           plot=False):
        # grid search with gmm loglikelihood objective
        g, r, ix, pGs = self._match_points_brute(xy, gridsize, rotation, )
        pGs[-1] = rotation

        if plot:
            from graphing.imagine import ImageDisplay

            # plot xy coords
            # ggfig, ax = plt.subplots()
            # sizes = self.counts[0] / self.counts[0].max() * 200
            # ax.scatter(*self.xy.T, sizes)
            # ax.plot(*roto_translate(xy, pGs).T, 'r*')

            im, peak = self.model.plot(show_peak=False)
            im.ax.plot(*roto_translate(xy, pGs).T, 'rx')

            extent = np.c_[g[:2, 0, 0], g[:2, -1, -1]].ravel()
            im = ImageDisplay(r.T, extent=extent)
            im.ax.plot(*pGs[:-1], 'ro', ms=15, mfc='none', mew=2)

        return pGs

    # def _match_points_brute_yx(self, yx, fov, step_size, rotation=0.,
    #                            objective=objective_gmm_yx, *args, **kws):
    #
    #     # create grid
    #     yx0 = y0, x0 = -np.multiply(fov, self._search_area_stretch)
    #     y1, x1 = self.fov - yx0
    #
    #     xres = int((x1 - x0) / step_size)
    #     yres = int((y1 - y0) / step_size)
    #     grid = np.mgrid[y0:y1:complex(yres),
    #            x0:x1:complex(xres)]
    #
    #     # add 0s for angle grid
    #     z = np.full((1,) + grid.shape[1:], rotation)
    #     grid = np.r_[grid, z]  # FIXME: silly to stack rotation here..
    #     self.logger.info(
    #             "Doing search on (%.1f' x %.1f') (%d x %d pix) sky grid",
    #             *fov, yres, xres)
    #
    #     # parallel
    #     r = gridsearch_mp(objective, grid, (self.yx, yx) + args, **kws)
    #     ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
    #     pGs = grid[:, i, j]
    #     self.logger.debug('Grid search optimum: %s', pGs)
    #     return grid, r, ix, pGs

    # def _make_grid(self, xy):
    #

    def _dxy_brute(self, xy, rotation=0., plot=False):  # _dxy_hop
        # This is a strategic brute force search along all the offset values
        # that will align pairs of stars in the two field.  One of them is
        # the true offset value.  Each offset value represents a local
        # basin in the parameter space, we are checking which one is deepest.
        # This algorithm is N^2 with number of stars.

        # get coords
        xyr = roto_translate(xy, (0, 0, rotation))
        # create search grid.
        points = (self.xy[None] - xyr[:, None]).reshape(-1, 2).T
        # Ignore extremal points in grid search.  These represent single
        # point matches at the edges of the frame which are most certainly
        # not the best match
        extrema = np.ravel([(p.argmin(), p.argmax()) for p in points])
        points = np.delete(points, extrema, 1)

        # This animates the search!
        # line, = im.ax.plot((), (), 'rx')
        # for i, po in enumerate(points.T):
        #     line.set_data(*(xyr - po).T)
        #     im.figure.canvas.draw()
        #     input(i)
        #     if i > 25:
        #          break

        # note: For correct identification of the deepest basin (around the
        #  global minimum), the sigma parameter (spread) of the gaussians
        #  need to be quite small so that erroneous near-alignments do not
        #  contribute significantly to the likelihood.
        self.model.sigma /= 20
        # compute trial points
        trials = xyr.T[..., None] + points[:, None]
        # remove trial points that are far outside the reference image so
        # that they do not artificially down weight the likelihood for
        # regions of the sky that we have no a priori knowledge about and
        # should therefore not penalise unnecessarily
        # xy_max = self.xy.max(0)
        # xy_min = self.xy.min(0)
        # xy_rng = ((xy_max + xy_min) + (xy_max - xy_min) * [[-1], [1]]) / 2
        # xy_rng = xy_rng[..., None, None]
        # far_out = (xy_rng[0] < trials) & (trials < xy_rng[1])
        # tm = np.ma.MaskedArray(trials, far_out)

        r = list(map(self.model.objective, trials.T))
        p = points[:, np.argmin(r)]
        self.logger.debug('Grid search optimum: %s', p)

        if plot:
            im, peak = self.model.plot(show_peak=False)
            im.ax.plot(*(xyr + p).T, 'ro', ms=12, mfc='none')

        # restore sigma
        self.model.sigma *= 20

        return np.array(p)

    def _match_points_brute(self, xy, size, rotation=0.):

        # create grid
        # shape = np.array(self.image.shape)
        dmin = self.xy.min(0) - xy.min(0)
        dmax = self.xy.max(0) - xy.max(0)
        rng = np.sort(np.vstack([dmin, dmax]), 0)
        step_size = rng.ptp(0) / size
        steps = 1j * np.array(duplicate_if_scalar(size))
        grid = np.mgrid[tuple(map(slice, *rng, steps))]

        # span = (fov / self.pixel_scale)
        # ovr = (span * (self._search_area_stretch - 1)) / 2
        # y0, x0 = -ovr
        # y1, x1 = shape + ovr
        # xr, yr = duplicate_if_scalar(size)
        # grid = np.mgrid[x0:x1:complex(xr), y0:y1:complex(yr)]

        # add 0s for angle grid
        z = np.full((1,) + grid.shape[1:], rotation)
        grid = np.r_[grid, z]
        self.logger.info(
            '\nDoing grid search on ' """
                δx = [{0:.1f} : {1:.1f} : {4:.1f}]; 
                δy = [{2:.1f} : {3:.1f} : {5:.1f}] 
                ({6:d} x {6:d}) offset grid"""
            ''.format(*rng.T.ravel(), *step_size, *grid.shape[-2:]))

        # parallel
        r = gridsearch_mp(self.model.objective_trans, grid, (xy,))
        # r = gridsearch_mp(objective, grid, (self.xy, xy) + args, **kws)
        ix = (i, j) = np.unravel_index(r.argmin(), r.shape)
        pGs = grid[:, i, j]
        self.logger.debug('Grid search optimum: %s', pGs)
        return grid, r, ix, pGs

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
    #     self.model.fit_trans(p0, xy, fit_angle)

    # @timer
    # def _match_points_gd_yx(self, yx, p0, fit_angle=True, sigma_gmm=0.03):
    #     """
    #     Match points with gradient descent on Gaussian mixture model likelihood
    #
    #     Parameters
    #     ----------
    #     yx: array
    #         yx coordinates of image features (center of mass of stars)
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
    #         f = ftl.partial(objective_gmm_yx, self.yx, yx, sigma_gmm)
    #     else:
    #         theta = p0[-1]
    #         yx = roto_translate_yx(yx, [0, 0, theta])
    #         f = ftl.partial(objective_gmm2_yx, self.yx, yx, sigma_gmm,
    #                         theta=theta)
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

    # def _match_final(self, xy):
    #
    #     from scipy.spatial.distance import cdist
    #     from recipes.containers.lists import where_duplicate
    #
    #     # transform
    #     xy = self.to_pixel_coords(xy)
    #     ix = tuple(xy.round().astype(int).T)[::-1]
    #     labels = self.seg.data[ix]
    #     #
    #     use = np.ones(len(xy), bool)
    #     # ignore stars not detected in dss, but detected in sample image
    #     use[labels == 0] = False
    #     # ignore labels that are detected as 2 (or more) sources in the
    #     # sample image, but only resolved as 1 in dss
    #     for w in where_duplicate(labels):
    #         use[w] = False
    #
    #     assert use.sum() > 3, 'Not good enough'
    #
    #     d = cdist(self.to_pixel_coords(self.xy), xy[use])
    #     iq0, iq1 = np.unravel_index(d.argmin(), d.shape)
    #     xyr = self.to_pixel_coords(self.xy[iq0])
    #     xyn = xy[use] - xyr
    #
    #     xyz = self.to_pixel_coords(self.xy[labels[use] - 1]) - xyr
    #     a = np.arctan2(*xyz.T[::-1]) - np.arctan2(*xyn.T[::-1])
    #     return np.median(a)

    def display(self, show_coms=True, number=True, cmap=None, marker_color='r'):
        """
        Display the image and the coordinates of the stars

        Returns
        -------

        """
        from graphing.imagine import ImageDisplay
        # import more_itertools as mit

        # ex = mit.interleave((0, 0), self.fov)
        im = ImageDisplay(self.image, cmap=cmap)  # extent=list(ex)

        s = []
        if show_coms:
            if number:
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
    _servers = ('all',
                'poss2ukstu_blue', 'poss1_blue',
                'poss2ukstu_red', 'poss1_red',
                'poss2ukstu_ir',
                )

    # TODO: more modern sky images
    # TODO: can you warn users about this possibility if you only have access to
    # old DSS images

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

        # TODO: proper motion correction
        # TODO: move some of this code to utils

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
        ImageRegistration.__init__(self)
        self(data, fov, **find_kws)

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
        self.targetCoordsPix = np.divide(self.image.shape, 2) + 0.5

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

    # TODO: optional normalize and same clims

    default_cmap_ref = 'Greys'
    alpha_cycle_value = 0.65
    label_props = dict(color='w')

    def __init__(self, reg, use_aplpy=None, name=None, keep_ref_image=True):

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

        # todo: _fov_internal = self.reg.image.shape
        self.art = {}  # art
        self.image_label = None

        self._counter = itt.count()
        self._ff = None
        self._fig = None
        self._ax = None
        self._low_lims = (np.inf, np.inf)
        self._up_lims = (-np.inf, -np.inf)

        # scrolling: index -1 represents the mosaic.  Scrolling forward
        # will show each image starting at 0
        self.alpha_cycle = []
        self._idx_active = -1
        self.keep_ref_image = bool(keep_ref_image)

        # connect image scrolling
        # fixme: this should happen only after first
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

    # def _world2pix(self, p, fov):  # FIXME: this name is inappropriate
    # convert fov to the DSS pixel coordinates (aplpy)
    # if self.use_aplpy:
    # scale_ratio = (fov / self.reg.fov)

    # dsspix = (fov / self.reg.fov) * self.reg.data.shape
    # x, y = p[:2] / self.reg.pixel_size + 0.5
    # return (x, y, p[-1]), dsspix
    # return p, fov

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
        # p, fov = self._world2pix(p, fov)
        if fov is None:
            fov = image.shape
        else:
            fov = np.divide(fov, self.reg.pixel_scale)

        # plot
        # image = image / image.max()
        art = self.art[name] = plot_transformed_image(
            self.ax, image, fov, p, frame,
            set_lims=False, cbar=False,
            **kws)

        # save upper left corner positions (for labels)

        if coords is not None:
            line, = self.ax.plot(*coords.T, 'x')
            # plot_points.append(line)

        self.names.append(name)
        if update:
            self.update_axes_limits(p, fov)

        return art

    def mosaic(self, images, fovs, params, names=(), coords=(), **kws):
        # mosaic plot

        cmap = kws.pop('cmap_ref', self.default_cmap_ref)
        # 
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
                                      np.ones(n) * alpha, ])
        if self.keep_ref_image:
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

    def label_image(self, name='', p=(0, 0, 0), fov=(0, 0), **kws):
        # default args for init
        _kws = {}
        _kws.update(self.label_props)
        _kws.update(kws)
        return self.ax.text(*_get_ulc(p, fov), name,
                            rotation=np.degrees(p[-1]),
                            rotation_mode='anchor',
                            va='top',
                            **_kws)

    # def label(self, indices, xy_offset=(0, 0), **text_props):
    #     texts = {}
    #     params = np.array(self.reg.params[1:])
    #     fovs = np.array(self.reg.fovs[1:])
    #     for name, idx in indices.items():
    #         xy = np.add(get_ulc(params[idx], fovs[idx])[::-1],
    #                     xy_offset)
    #         angle = np.degrees(params[idx, -1].mean())
    #         texts[name] = self.ax.text(*xy, name,
    #                                    rotation=angle,
    #                                    rotation_mode='anchor',
    #                                    **text_props)
    #     return texts

    # def label_images(self):
    #     for i, im in enumerate(self.reg.images):
    #         name = f'{i}: {self.names[i]}'
    #         p = self.reg.params[i]
    #         xy = np.atleast_2d([0, im.shape[0]])  # self.reg.fovs[i][0]
    #         xy = roto_translate(xy, p).squeeze()
    #
    #         # print(xy.shape)
    #         assert xy.shape[0] == 2
    #
    #         self.image_labels.append(
    #                 self.ax.text(*xy, name, color='w', alpha=0,
    #                              rotation=np.degrees(p[-1]),
    #                              rotation_mode='anchor',
    #                              va='top')
    #         )

    def _set_alphas(self):
        """highlight a particular image in the stack"""
        alphas = self.alpha_cycle[self._idx_active]

        # position 0 represents the mosaic
        self.image_label.set_visible(self._idx_active != len(self.art))

        for i, artists in enumerate(self.art.values()):
            for art in artists:
                art.set_alpha(alphas[i])
                art.set_zorder(-(i == self._idx_active == -1))
            # if i == self._idx_active:
            #     im.set_zorder(-(i == -1))
            # else:
            #     im.set_zorder(0)

    def _scroll(self, event):
        """
        This method allows you to scroll through the images in the mosaic
        using the mouse.
        """
        # if self.image_label is None:
        # self.image_label = self.ax.text(0, 0, '', color='w', alpha=0,
        #                                 rotation_mode='anchor',
        #                                 va='top')
        n = len(self.art)
        if event.inaxes and n:  #
            # set alphas
            self._idx_active += [-1, +1][event.button == 'up']  #
            self._idx_active %= (n + 1)  # wrap

            #
            i = self._idx_active
            if self.image_label is None:
                self.image_label = self.label_image()

            txt = self.image_label
            if i < n:
                txt.set_text(f'{i}: {self.names[i]}')
                xy = _get_ulc(self.reg.params[i],
                              (0.98 * self.reg.fovs[i] / self.reg.pixel_scale))
                txt.set_position(xy)

            #
            self._set_alphas()

            # if self.image_label is not None:
            #     self.image_label.remove()
            #     self.image_label = None

            # set tiles
            # if self._idx_active != n:
            #     # position -1 represents the original image

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
            self._idx_active = -1
            self._set_alphas()

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
