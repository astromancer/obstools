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
import functools as ftl
import itertools as itt
import multiprocessing as mp
from collections import abc

# third-party
import numpy as np
import aplpy as apl
import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from loguru import logger
from joblib import Parallel, delayed
from astropy import wcs
from astropy.utils import lazyproperty
from scipy.cluster.vq import kmeans
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.ckdtree import cKDTree
from scipy.stats import binned_statistic_2d, mode
from scipy.interpolate import NearestNDInterpolator

# local
import recipes.pprint as pp
from recipes.string import indent
from recipes.lists import split_like
from recipes.functionals import echo0
from recipes.logging import LoggingMixin
from recipes.utils import duplicate_if_scalar
from scrawl.imagine import ImageDisplay

# relative
from .. import transforms as transform
from ..modelling import Model
from ..campaign import HDUExtra
from ..stats import geometric_median
from ..utils import get_coordinates, get_dss, STScIServerError
from .mosaic import MosaicPlotter
from .segmentation import SegmentedImage
from .image import SkyImage, ImageContainer


# from motley.profiling.timers import timer
# from sklearn.cluster import MeanShift
# from motley.profiling import profile


#
TABLE_STYLE = dict(txt=('bold', 'underline'), bg='g')


def normalize_image(image, centre=np.ma.median, scale=np.ma.std):
    """Recenter and scale"""
    image = image - centre(image)
    if scale:
        return image / scale(image)
    return image


def non_masked(xy):
    xy = np.asanyarray(xy)
    if np.ma.is_masked(xy):
        return xy[~xy.mask.any(-1)].data
    return np.array(xy)


def objective_pix(target, values, xy, bins, p):
    """Objective for direct image matching"""
    xy = transform.rigid(xy, p)
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


class MultiGauss(Model):
    """
    This class implements a model that consists of the sum of multivariate
    Gaussian distributions
    """

    dof = 0  # TODO: maybe expand so this is more dynamic

    def __init__(self, xy, sigmas, amplitudes=1.):
        """
        Create a Gaussian "mixture" with components at locations `xy` with
        standard deviations `sigmas` and relative amplitudes `amplitudes`. The
        locations, stdandard deviations and amplitudes are hyperparameters.
        This is different from the classic Gaussian Mixture Model in that we are
        not imposing any constraints on the mixture weights. Values returned by
        the `eval` method are therefore not probabilities, but that's OK for
        optimization using maximum likelihood.

        Parameters
        ----------
        xy : array-like
            The locations of the component Gaussians. The expected shape is
            (n, n_dims) where `n` is the number of sources and `n_dims` the
            number of spatial dimensions in the problem.
            (eg. Modelling an image n_dims =2:
                n=1 in the case of a point source model
                n=2 in the case of modelling a field of sources
        sigmas : float or array-like
            The standard deviation of the gaussians in the model. If array-like,
            it must have the same size in (at least) the first dimension as
            `xy`.
        amplitudes : array-like
            The relative amplitude of each of the Gaussians components. Must
            have the same size as locations parameter `xy`

        Note that the `amplitudes` are degenerate with the `sigmas` parameter
        since σ² :-> σ² / len(A) expresses an identical equation. The amplitudes
        will be absorbed in the sigmas in the internal representation, but are
        allowed here as a parameter for convenience.
        """

        # TODO: equations in docstring

        # TODO: full covariance matrices

        # cast the xy points to 3d here to avoid doing so every time during the
        # call. Shaves a few microseconds off the call run time :)).  Also make
        # sure we remove all masked data and have a plain np.ndarray object.
        self.xy = non_masked(xy)
        self.n, self.n_dims = (n, k) = self.xy.shape

        # set hyper parameters
        self._exp = None  # the exponential pre-factor
        # TODO: will need to change for full covariance implementation
        self._amplitudes = np.broadcast_to(amplitudes, n).astype(float)
        self.sigmas = sigmas
        self.amplitudes = amplitudes
        self._pre = np.sqrt((2 * np.pi) ** k * self.sigmas.prod(-1))

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.n} sources'

    def pprint(self, **kws):
        """
        Pretty print a tabular representation of the Gaussians

        Returns
        -------
        str
            Tabulated representation of the point sources like
                __________________________
                ⎪_______MultiGauss_______⎪
                ⎪x ⎪ y ⎪  σₓ  ⎪ σᵥ  ⎪  A  ⎪
                ⎪——⎪———⎪—————⎪—————⎪—————⎪
                ⎪ 5⎪ 10⎪ 1   ⎪ 1   ⎪ 1   ⎪
                ⎪ 5⎪  6⎪ 2   ⎪ 2   ⎪ 2   ⎪
                ⎪ 8⎪  2⎪ 3   ⎪ 3   ⎪ 3   ⎪
                ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        """
        from motley.table import Table
        tbl = Table.from_columns(self.xy, self.sigmas, self.amplitudes,
                                 title=self.__class__.__name__,
                                 chead='x y σₓ σᵥ A'.split(),
                                 # unicode subscript y doesn't exist!!
                                 minimalist=True,
                                 **kws)
        return str(tbl)

    @property
    def amplitudes(self):
        return self._amplitudes.squeeze()

    @amplitudes.setter
    def amplitudes(self, amplitudes):
        self.set_amplitudes(amplitudes)

    def set_amplitudes(self, amplitudes):
        self._amplitudes = self._check_prop(
            amplitudes, 'amplitudes', (self.n,))

    @property
    def flux(self):
        """
        Flux is an alternative parameterization for amplitude and corresponds to
        the volumne under the surface in two dimension. This computes the
        corresponding amplitude and sets that. The sigma terms remain unchanged
        by setting the fluxes.
        """
        return self.amplitudes * self._pre

    @flux.setter
    def flux(self, flux):
        self._amplitudes = flux / self._pre

    @property
    def sigmas(self):
        return self._sigmas

    @sigmas.setter
    def sigmas(self, sigmas):
        # TODO: full covariance matices
        self._sigmas = self._check_prop(
            sigmas, 'sigmas', (self.n, self.n_dims))

        # absorb sigmas and amplitude into single exponent
        self._exp = -1 / 2 / self._sigmas ** 2
        # pre-factors so the individual gaussians integrate to 1
        self._pre = np.sqrt((2 * np.pi) ** self.n_dims * self.sigmas.prod(-1))

    def _check_prop(self, a, name, shape):
        # check if masked
        if np.ma.is_masked(a):
            raise ValueError(f'{name!r} cannot be masked')

        # make vanilla array
        a = np.array(a, float, ndmin=len(shape))

        if not np.isfinite(a).all():
            raise ValueError(f'{name!r} must be finite.')

        # check positive definite
        if (a <= 0).any():
            raise ValueError(f'{name!r} must be positive non-zero.')

        # check shape
        if a.ndim > 1:
            r, c = a.shape
            # allows easy setting like >>> self.sigmas = [1, 2, 3]
            if (c == self.n) and (c != self.n_dims) and (r in (1, self.n_dims)):
                a = a.T

        # broadcast
        try:
            a = np.broadcast_to(a, shape)
        except ValueError:
            raise ValueError(
                f'{name!r} parameter has incorrect shape: {np.shape(a)}'
            ) from None

        return a.copy()

    # def __call__(self, p, xy=None):
    #     # just setting the default which is a cheat
    #     return super().__call__(p, xy)

    def _checks(self, p, xy=None, *args, **kws):
        return self._check_params(p), (self._check_grid(xy), *args)

    def _check_params(self, p):
        if (p is None) or (p == ()):
            # default parameter values for evaluation
            return np.zeros(self.dof)
        return p

    def _check_grid(self, grid):
        #
        grid = non_masked(grid)
        shape = grid.shape
        # make sure last dimension is same as model dimension
        if (grid.ndim < 2) and (shape[-1] != self.n_dims):
            raise ValueError(
                'Compute grid has incorrect size in last dimension: shape is '
                f'{shape}, last axis should have size {self.n_dims}.'
            )
        return grid.reshape(np.insert(shape, -1, 1))

    def eval(self, p, xy, *args, **kws):
        """
        Compute the model values at points `xy`. Parameter `p` is ignored.

        Parameters
        ----------
        xy: np.ndarray
            (n, n_dims)

        Returns
        -------
        np.ndarray
            shape (n, ) for n points


        """
        return (self._amplitudes * np.exp(
            (self._exp * np.square(self.xy - xy)).sum(-1)
        )).sum(-1)

    def _auto_grid(self, size, dsigma=3.):
        """
        Create a evaluation grid for the model that encompasses all components
        of the model up to `dsigma` in Mahalanobis distance from the extremal
        points

        Parameters
        ----------
        size : float or array-like
            Size of the grid array.  If float, number will be duplicated to be
            the size along each axis. If array-like, the size aling each
            dimension and it must have the same number of dimensions as the
            model
        dsigma : float, optional
            Controls the coverage (range) of the grid by specifying the maximal
            Mahalanobis distance from extremal points (component locations), by
            default 3

        Returns
        -------
        np.ndarray
            compute grid of size (nx, ny, n_dims)
        """
        xyl, xyu = self.xy + dsigma * self.sigmas * \
            np.array([-1, 1], ndmin=3).T
        size = duplicate_if_scalar(size, self.n_dims)
        slices = map(slice, xyl.min(0), xyu.max(0), size * 1j)
        return np.moveaxis(np.mgrid[tuple(slices)], 0, -1)

    def plot(self, grid=None, size=100, show_xy=True, show_peak=True, **kws):
        """Image the model"""

        ndims = self.n_dims
        if ndims != 2:
            raise Exception(f'Can only image 2D models. Model is {ndims}D.')

        if grid is None:
            size = duplicate_if_scalar(size, ndims, raises=False)
            grid = self._auto_grid(size)

        # compute model values
        z = self((), grid)

        # plot an image
        kws_ = dict(cmap='Blues', alpha=0.5)  # defaults
        kws_.update(**kws)
        im = ImageDisplay(z.T,
                          extent=grid[[0, -1], [0, -1]].T.ravel(),
                          **kws_)

        # plot locations
        if show_xy:
            im.ax.plot(*self.xy.T, '.')

        # mark peak
        if show_peak:
            xy_peak = grid[np.unravel_index(z.argmax(), z.shape)]
            im.ax.plot(*xy_peak, 'rx')
        return im


class GaussianMixtureModel(MultiGauss):

    def __init__(self, xy, sigmas, weights=1):
        if np.all(weights == 1) or (weights is None):
            # shortcut to equal weights. avoid warnings
            amplitudes = 1 / len(xy)
        else:
            amplitudes = weights

        super().__init__(xy, sigmas, amplitudes)

    def set_amplitudes(self, amplitudes):
        super().set_amplitudes(amplitudes)

        # normalize mixture weights so we have a probability distribution not
        # just a field of gaussians
        t = self._amplitudes.sum()
        if not np.allclose(t, 1):
            warnings.warn('Normalizing mixture weights')
            self._amplitudes /= t

    def llh(self, p, xy, data=None, stddev=None):
        """
        Log likelihood of independently drawn observations `xy`. ie.
        Sum of log-likelihoods
        """
        # sum of gmm log likelihood ignoring incoming masked points
        prob = np.atleast_1d(self(p, xy).sum(-1))
        # NOTE THIS OBVIATES ANY OPTIMIZATION through eval but is needed for
        # direct call to this method

        # :           substitute 0 --> 1e-300      to avoid warnings and infs
        prob[(prob == 0)] = 1e-300
        return np.log(prob).squeeze()


# class CoherentPointDrift(GaussianMixtureModel):
#     """
#     Model that fits for the transformation parameters to match point clouds.
#     This is a partial implementation of the full CPD algorithm since we
#     generally know the scale of the feature coordinates (field-of-view) and
#     therefore don't need to fit for the scale parameters.
#     """

#     _fit_angle = True  # default

#     @property
#     def dof(self):
#         return 2 + self._fit_angle

#     @property
#     def fit_angle(self):
#         return self._fit_angle

#     @fit_angle.setter
#     def fit_angle(self, tf):
#         self._fit_angle = tf

#     @property
#     def transform(self):
#         return transforms.rigid if self._fit_angle else np.add

#     def eval(self, p, xy, stddev=None):
#         return super().eval((), self.transform(xy, p))

#     def p0guess(self, data, *args):
#         # starting values for parameter optimization
#         return np.zeros(self.dof)

#     def fit(self, xy, p0=None):
#         # This will evaluate the
#         return super().fit(None, None, self.loss_mle, (xy,))


class CoherentPointDrift(Model):
    """
    Model that fits for the transformation parameters to match point clouds.
    This is a partial implementation of the full CPD algorithm since we
    generally know the scale of the feature coordinates (field-of-view) and
    therefore don't need to fit for the scale parameters.
    """

    # FIXME: this should be a GMM subclass. Need to tweak some of the `Model`
    # features to get that to work

    _fit_angle = True  # default

    @property
    def dof(self):
        return 2 + self._fit_angle

    @property
    def fit_angle(self):
        return self._fit_angle

    @fit_angle.setter
    def fit_angle(self, tf):
        self._fit_angle = tf

    @property
    def transform(self):
        return transform.rigid if self._fit_angle else np.add

    def __init__(self, xy, sigmas, weights=None):
        self.gmm = GaussianMixtureModel(xy, sigmas, weights)

    def __call__(self, p, xy):
        return self.gmm((), self.transform(xy, p))

    def llh(self, p, xy, stddev=None):
        return self.gmm.llh((), self.transform(xy, p))

    def p0guess(self, data, *args):
        # starting values for parameter optimization
        return np.zeros(self.dof)

    def fit(self, xy, stddev=None, p0=None):
        # This will evaluate the
        return super().fit(xy, p0, loss=self.loss_mle)

    def lh_ratio(self, xy0, xy1):
        # likelihood ratio
        return np.exp(self.gmm.llh((), xy0) - self.gmm.llh((), xy1))


# @timer
def offset_disp_cluster(xy0, xy1, sigma=None, plot=False):
    # NOTE: this method is disfavoured compared to the more stable
    # `ImageRegister._dxy_hop`

    # find offset (translation) between frames by looking for the dense
    # cluster of inter-frame point-to-point displacements that represent the
    # xy offset between the two images. Cannot handle rotations.

    # Finding density peak in inter-frame distances is fast and accurate
    # under certain conditions only.
    #  - Images have the same rotation
    #  - Images have large overlap. There should be more shared points
    #    than points that are in one frame only
    #  - Images are roughly to the same depth (i.e roughly the same point
    #    density (this is done by `hdu.sampler.to_depth`)

    # NOTE: also not good for large number of points in terms of performance and
    #   accuracy...

    points = (xy0[None] - xy1[:, None]).reshape(-1, 2).T

    if sigma is None:
        # size of gaussians a third of closest distance between stars
        sigma = min(dist_flat(xy0).min(), dist_flat(xy1).min()) / 3

    gmm = GaussianSourceField(points.T, sigma)
    grid = gmm._auto_grid(100)
    peak = grid[np.unravel_index(gmm(grid).argmax())]

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
    # n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    return sdist[np.tril_indices(len(coo), -1)]


# @timer


def gridsearch_mp(objective, grid, args=(), **kws):
    # grid search

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


# def cross_index(coo_trg, coo, dmax=0.2):
#     # identify points matching points between sets by checking distances
#     # this will probably only work if the frames are fairly well aligned
#     dr = cdist(coo_trg, coo)

#     # if thresh is None:
#     #     thresh = np.percentile(dr, dr.size / len(coo))

#     dr[(dr > dmax) | (dr == 0)] = np.nan

#     ur, uc = [], []
#     for i, d in enumerate(dr):
#         dr[i, uc] = np.nan
#         if ~np.isnan(d).all():
#             # closest star not yet matched with another
#             jmin = np.nanargmin(d)
#             ur.append(i)
#             uc.append(jmin)

#     return ur, uc


# def fit_cube(self, filename, object_name=None, coords=None):
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


def plot_coords_nrs(cooref, coords):
    fig, ax = plt.subplots()

    for i, yx in enumerate(cooref):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='r')

    for i, yx in enumerate(coords):
        ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


def display_multitab(images, fovs, params, coords):
    from scrawl.multitab import MplMultiTab
    from scrawl.imagine import ImageDisplay

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


# def register(clustering, coms, centre_distance_max=1,
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
#     n_clusters, n_noise = cluster_points(clustering, coms)
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
                     marker='.').items():
        scatter_kws.setdefault(k, v)

    if scatter_kws.get('marker') in '.o':
        scatter_kws.setdefault('facecolors', 'none')
    else:
        scatter_kws.setdefault('facecolors', colours)
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
#                 'Ignoring %i/%i ({:.1%}) nan / masked values for position '
#                 'measurement', n_ignore, n, (n_ignore / n) * 100)
#
#     # mask nans.  masked
#     return np.ma.MaskedArray(xy, nans)


def compute_centres_offsets(xy, d_cut=None, detect_freq_min=0.9, report=True):
    """
    Measure the relative positions of detected stars from the individual
    location measurements in xy. Use the locations of the most-often
    detected individual objects.

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
        logger.info('Ignoring {:d}/{:d} ({:.1%}) nan values for position '
                    'measurement', n_ignore, n, n_ignore / n)

    n_detections_per_star = np.zeros(n_stars, int)
    w = np.where(~bad)[1]
    u = np.unique(w)
    n_detections_per_star[u] = np.bincount(w)[u]

    f_det = (n_detections_per_star / n_use)
    use_stars = f_det > detect_freq_min
    i_use, = np.where(use_stars)
    if not len(i_use):
        raise Exception(
            'Detected frequently for all sources appears to be too low. There '
            'are {n_stars} objects across {n} images. Their detection '
            'frequencies are: {fdet}.'
        )

    if np.any(~use_stars):
        logger.info('Ignoring {:d} / {:d} stars with low (<={:.0%}) detection '
                    'frequency for frame offset measurement.',
                    n_stars - len(i_use), n_stars, detect_freq_min)

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
    # np.empty((n_stars, 2))  # ma.masked_all
    centres = δxy = np.ma.masked_all((n_stars, 2))
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

    # pprint!
    if report:
        #                                          counts
        report_measurements(xy, centres, σxy, δxy, None, detect_freq_min)

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
            raise Exception('Too many outliers!!')

        logger.info('Ignoring {:d}/{:d} ({:.1%}) values with |δr| > {:.3f}',
                    n_out, n_points, (n_out / n_points), d_cut)

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
        unique_labels.remove(-1)
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


def report_measurements(xy, centres, σ_xy, xy_offsets=None, counts=None,
                        detect_frac_min=None, count_thresh=None, logger=logger):
    # report on relative position measurement
    import operator as op

    from recipes import pprint
    from motley.table import Table
    from motley.utils import ConditionalFormatter
    # from obstools.stats import mad
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
            extra = (f'\nn_noise = {n_noise}/{n_points_tot} '
                     f'({n_noise / n_points_tot :.1%})')
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
    # logger.info('Differencing change overall variance by {!r:}',
    #             np.array2string((s0 - s1) / s0, precision=3))

    # FIXME: percentage format in total wrong
    # TODO: align +- values
    col_headers = ['x', 'y', 'n']  # '(px.)'
    formatters = {'n': ftl.partial(pprint.decimal_with_percentage,
                                   total=n_points, precision=0)}  #

    if detect_frac_min is not None:
        n_min = detect_frac_min * n_points
        formatters['n'] = ConditionalFormatter('y', op.le, n_min,
                                               formatters['n'])

    # get array with number ± std representations
    columns = [pprint.uarray(centres, σ_xy, 2)[:, ::-1], points_per_star]
    # FIXME: don't print uncertainties if less than 6 measurement points

    if counts is not None:
        # TODO highlight counts?
        cn = 'counts'  # (ADU)
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
                             units=['pixels', 'pixels', ''],
                             col_head=col_headers,
                             col_head_props=TABLE_STYLE,
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
    # add a refernce image
    >>> reg(image, fov)
    # match a new (partially overlapping) image to the reference image:
    >>> xy_offset, rotation = reg(new_image, new_fov)
    # cross identify stars across images
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

    find_kws = dict(snr=3.,
                    npixels=5,
                    edge_cutoff=2,
                    deblend=False)
    """default source finding keywords"""

    refining = True
    """Refine fitting by running gradient descent after basin-hopping search."""

    # Expansion factor for grid search
    _search_area_stretch = 1.25

    # Sigma for GMM as a fraction of minimal distance between stars
    _dmin_frac_sigma = 3

    # TODO: switch for units to arcminutes or whatever ??
    # TODO: uncertainty on center of mass from pixel noise!!!

    @classmethod
    def from_hdus(cls, run, sample_stat='median', depth=10, primary=None,
                  fit_angle=True, **find_kws):
        # get sample images etc
        # `from_hdu` is used since the sample image and source detection results
        # are cached (persistantly), so this should be fast on repeated calls.
        images = []
        for i, hdu in enumerate(run):
            im = SkyImage.from_hdu(hdu, sample_stat, depth,
                                   **{**cls.find_kws, **find_kws})
            images.append(im)

        return cls(images, primary=primary, fit_angle=fit_angle, **find_kws)

    # @classmethod                # p0 -----------
    # def from_images(cls, images, fovs, p0=(0,0,0), primary=None, plot=False,
    #                 fit_angle=True, **find_kws):

    #     # initialize workers
    #     with Parallel(n_jobs=1) as parallel:
    #         # detect sources
    #         images = parallel(
    #             delayed(SkyImage.from_image)(image, fov, **find_kws)
    #             for image, fov in zip(images, fovs)
    #         )

    #     return cls._from_images(images, fovs, angles=0, primary=None,
    #                             plot=False, fit_angle=True, **find_kws)

    # @classmethod                # p0 -----------
    # def _from_images(cls, images, fovs, angles=0, primary=None, plot=False,
    #                  fit_angle=True, **find_kws):

    #     n = len(images)
    #     assert 0 < n == len(fovs)

    #     # message
    #     logger.info('Aligning {:d} images on image {:d}.', n, primary)

    #     # initialize and fit
    #     return cls(images,
    #                primary=primary,
    #                fit_angle=fit_angle).fit(plot=plot)

    def __init__(self, images=(), fovs=(), params=(), fit_angle=True,
                 primary=None, **find_kws):
        """
        Initialize an image register. This class should generally be initialized
        without arguments. The model of the constellation of stars is built
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

        if find_kws:
            self.find_kws.update(find_kws)

        # init container
        ImageContainer.__init__(self, images, fovs)

        # NOTE passing a reference index `primary` is only meaningful if the
        #  class is initialized with a set of images
        self._idx = 0
        self.count = 0
        if primary is None:
            if (len(images) == 0):
                primary = 0
            else:
                # align on image with highest source density if not specified
                source_density = self.scales.mean(1) * self.attrs('seg.nlabels')
                primary = source_density.argmax(0)

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
        self.labels = self.n_stars = self.n_noise = None
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
        return list(map(transform.affine, self.coms, self.params, self.rscale))

    @property
    def n_points(self):
        return sum(map(len, self.coms))

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
        return dist_flat(self.xy).min()

    # def min_dist(self):
    #     return min(dist_flat(xy).min() for xy in self.xyt)

    def guess_sigma(self):
        """choose sigma for GMM based on distances between detections"""
        return self.min_dist / self._dmin_frac_sigma

    @lazyproperty
    def model(self):
        model = CoherentPointDrift(self.xy, self.guess_sigma())
        model.fit_angle = self.fit_angle
        return model

    def __call__(self, obj, *args, **kws):
        if (obj is None) and (count == 0):
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

    def fit(self, obj=None, p0=None, hop=True, refine=None, plot=False,
            **kws):
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

        if (obj is None) and (self.count == 0):
            # obj = self.data
            # (re)fit all images
            obj = self[self.order]
            p0 = self.params[self.order]
            hop = (self._xy is None)
            refine = not hop

        #
        refine = refine or self.refining
        fitters = {ImageRegister:           self.fit_register,
                   (np.ndarray, SkyImage):  self.fit_image,
                   abc.Collection:          self.fit_sequence,
                   HDUExtra:                self.fit_hdu}
        for types, fit in fitters.items():
            if isinstance(obj, types):
                break
        else:
            raise TypeError(f'Cannot fit object of type {type(obj)}.')

        self.logger.opt(lazy=True).debug(
            'Fitting: {!s}',
            lambda: pp.caller(fit, (type(obj), p0, hop, refine, plot), kws)
        )
        result = fit(obj, p0, hop, refine, plot, **kws)
        self.count += 1
        return result

    def fit_sequence(self, items, p0=None, hop=True, refine=True,
                     plot=False, njobs=1, **kws):
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

        images = [self.fit(image, p00, hop, refine, plot, **kws)
                  for image, p00 in itt.zip_longest(items, p0)]

        # images.insert(primary, self[primary])
        # self.data[:] = images

        # fit clusters to points
        # self.register()
        return images

    def fit_register(self, reg, p0=None, hop=True, refine=True, plot=False,
                     **kws):
        """
        cross match with another `ImageRegister` and aggregate points
        """

        # convert coordinates to pixel coordinates of reference image
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

    def fit_hdu(self, hdu, p0=None, hop=True, refine=True,
                plot=False, sample_stat='median', depth=10, **find_kws):
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
                              **{**self.find_kws, **find_kws}),
            p0, None, hop, refine, plot
        )

    def fit_image(self, image, p0=None, hop=True, refine=True,
                  plot=False, fov=None, **find_kws):
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
            image.detect(**{**self.find_kws, **find_kws})

        if self.count:
            # get xy in units of `primary` image pixels
            xy = image.xy * image.scale / self.pixel_scale
            image.params = self.fit_points(xy, p0, hop, refine, plot)

        return image

    # @timer
    def fit_points(self, xy, p0=None, hop=True, refine=True, plot=False):

        if p0 is None:
            p0 = np.zeros(self.dof)

        if hop:
            # get shift (translation) via brute force search through point to
            # point displacements. This is gauranteed to find the optimal
            # alignment between the two point clouds in the absense of rotation
            # between them

            # get coords.
            xyr = transform.rigid(xy, p0)
            dxy = self._dxy_hop(xyr, plot)
            p = p0 + [*dxy, 0]

        if not hop or refine:
            self.logger.debug('Refining fit via gradient descent.')
            # do mle via gradient decent
            self.model.fit_angle = self.fit_angle
            p = self.model.fit(xy, p0=p0)

        return p

    def register(self, clf=None, plot=False):
        # clustering + relative position measurement
        clf = clf or self.clustering
        self.cluster_points(clf)
        # make the cluster centres the target constellation
        self.xy = self.xyt_block.mean(0)

        if plot:
            #
            art = self.plot_clusters()
            ax = art.axes

            # bandwidth size indicator.
            # todo: get this to remain anchored lower left but scale with zoom..
            xy = self.xy.min(0)  # - bw * 0.6
            cir = Circle(xy, clf.bandwidth, alpha=0.5)
            ax.add_artist(cir)

    @property
    def labels_per_image(self):
        if self.labels is None:
            raise Exception('Unregistered!')

        # group_features(self.labels, self.labels)[0]
        return split_like(self.labels, self.coms)

    def remap_labels(self, target, flux_sort=True):
        """
        Re-order the *cluster* labels so that our target is 0, the rest follow
        in descending order of brightness first listing those that occur within
        our science images, then those in the survey image.
        """

        # get cluster labels bright to faint
        counts, = group_features(self.labels, self.attrs.counts)

        # get the labels of stars that are detected in at least one of the
        # science frames
        detected = ~counts[1:].mask
        nb, = np.where(detected.any(0))
        old = [target,
               *np.setdiff1d(nb, target),
               *np.setdiff1d(np.where(detected.all(0)), nb)]

        if flux_sort:
            # measure relative brightness (scale by one of the frames, to account for
            # gain differences between instrumental setups)
            bright2faint = list(np.ma.median(
                counts / counts[:, [old[0]]], 0).argsort()[::-1])
            list(map(bright2faint.remove, old))
            old += bright2faint
        return np.argsort(old)  # these are the new labels!

    def relabel_segments(self, target=None, flux_sort=True):
        # desired new cluster labels
        new_labels = self.remap_labels(target, flux_sort)

        # relabel all segmentedImages for cross image consistency
        for cluster_labels, image in zip(self.labels_per_image, self):
            # we want non-zero labels for SegmentedImage
            image.seg.relabel_many(new_labels[cluster_labels] + 1)

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

    @lazyproperty
    def clustering(self, *args, **kws):
        """
        Classifier for clustering source coordinates in order to cross identify
        stars.
        """
        from sklearn.cluster import MeanShift

        # choose bandwidth based on minimal distance between stars
        return MeanShift(**{**kws,
                            **dict(bandwidth=self.min_dist / 2,
                                   cluster_all=False)})

    def cluster_points(self, clf=None):
        """
        Run clustering algorithm on the set of position measurements in order to 
        cross identify stars.
        """
        # clustering to cross-identify stars
        assert len(self) > 1

        clf = clf or self.clustering

        X = np.vstack(self.xyt)
        n = len(X)
        # stars_per_image = list(map(len, xy))
        # no_detection = np.equal(stars_per_image, 0)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(np.vstack(xy))

        self.logger.info('Clustering {:d} position measurements to cross '
                         'identify sources using:\n\t{:s}', n, indent(str(clf)))
        clf.fit(X)
        labels = clf.labels_
        # core_samples_mask = (clf.labels_ != -1)
        # core_sample_indices_, = np.where(core_samples_mask)

        # Number of clusters in labels, ignoring noise if present.
        self.n_stars = n_stars = len(set(labels)) - int(-1 in labels)
        self.n_noise = n_noise = list(labels).count(-1)
        # n_per_label = np.bincount(db.labels_[core_sample_indices_])
        logger.info('Identified {:d} stars using {:d}/{:d} points ({:d} noise)',
                    n_stars, n - n_noise, n, n_noise)

        # sanity check
        n_stars_most = max(map(len, self.coms))
        if n_stars > n_stars_most * 1.5:
            warnings.warn(f"Looks like we're overfitting clusters. Image with "
                          f"most sources has {n_stars_most}, while clustering "
                          f"produced {n_stars} clusters. Reduce bandwidth.")

        #
        self.labels = clf.labels_

    # def check_labelled(self):
        # if not len(self):
        #     raise Exception('No data. Please add some images first. eg: '
        #                     '{self.__class__.__name__}()(image, fov)')

        # if self.labels is None:
        #     raise Exception('No clusters identified. Run '
        #                     '`register` to fit clustering model '
        #                     'to the measured centre-of-mass points')

    def plot_clusters(self, show_bandwidth=True, **kws):
        """
        Plot the identified sources (clusters) in a single frame
        """
        if not len(self):
            raise Exception(
                'No data. Please add some images first. eg: '
                '{self.__class__.__name__}()(image, fov)'
            )

        if self.labels is None:
            raise Exception(
                'No clusters identified. Run `register` to fit '
                'clustering model to the measured centre-of-mass points'
            )

        n = len(self)
        fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        ax.set_title(f'Position Measurements (CoM) {n} frames')

        art = plot_clusters(ax, np.vstack(self.xyt), self.labels, **kws)
        self._colour_sequence_cache = art.get_edgecolors()

        # bandwidth size indicator.
        # todo: get this to remain anchored lower left but scale with zoom..
        xy = self.xy.min(0)  # - bw * 0.6
        cir = Circle(xy, self.clustering.bandwidth, alpha=0.5)
        ax.add_artist(cir)

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
                self[i].offset -= xy_offsets[i]

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

    def solve_rotation(self):
        # for point clouds that are already quite well aligned. Refine by
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

        n_jobs = -1  # make sure you can pickle everything before you change
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
        xyn = list(map(transform.affine, self.coms, params, self.rscale))
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

    def reset(self):
        """
        Reset image container to empty list

        Returns
        -------

        """

        self.data = []
        self.params = []

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
        if circularize:
            return seg.circularize()
        return seg

    def _dxy_hop(self, xy, plot=False):
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

    # @timer
    def fit_points_brute(self, xy, rotation=0., gridsize=(50, 50),
                         plot=False):
        # grid search with gmm loglikelihood objective
        g, r, ix, pGs = self._fit_points_brute(xy, gridsize, rotation, )
        pGs[-1] = rotation

        if plot:
            from scrawl.imagine import ImageDisplay

            # plot xy coords
            # ggfig, ax = plt.subplots()
            # sizes = self.counts[0] / self.counts[0].max() * 200
            # ax.scatter(*self.xy.T, sizes)
            # ax.plot(*roto_translate(xy, pGs).T, 'r*')

            im, peak = self.model.gmm.plot(show_peak=False)
            im.ax.plot(*transform.rigid(xy, pGs).T, 'rx')

            extent = np.c_[g[:2, 0, 0], g[:2, -1, -1]].ravel()
            im = ImageDisplay(r.T, extent=extent)
            im.ax.plot(*pGs[:-1], 'ro', ms=15, mfc='none', mew=2)

        return pGs

    def _fit_points_brute(self, xy, size, rotation=0.):

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
        logger.info(
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
        logger.debug('Grid search optimum: {:s}', pGs)
        return grid, r, ix, pGs

    def fit_pixels(self, index):

        indices = set(range(len(self.images))) - {index}
        grid = []
        pixels = []
        for i in indices:
            image = self.images[i]
            r = np.divide(self.fovs[i], image.shape) / self.pixel_scale
            p = self.params[i]
            seg = self.detections[i].dilate(2, copy=True)
            for sub, g in seg.coslice(image, np.indices(image), flatten=True):
                g = transform.rigid(g.reshape(-1, 2) * r, p)
                grid.extend(g)
                pixels.extend(sub.ravel())

        bs = binned_statistic_2d(*grid, pixels, 'mean', bins)

    def _fit_pixels(self, image, fov, p0):
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

    # def fit_image_brute(self, image, fov, rotation=0., step_size=0.05,
    #                       return_coords=False, plot=False, sigma_gmm=0.03):
    #
    #     seg, yx, counts = self.find_stars(image, fov)
    #     return self.fit_points_brute(yx, fov, rotation, step_size,
    #                                    return_coords, plot, sigma_gmm)

        # return im, s

    def mosaic(self, axes=None, names=(), scale='sky',
               show_ref_image=True, number_sources=False,
               **kws):

        mos = MosaicPlotter.from_register(self, axes, scale, show_ref_image)
        mos.mosaic(names, **kws)

        if number_sources:
            off = -4 * self.scales.min(0)
            mos.mark_sources(self.xy,
                             marker=None,
                             xy_offset=off)

        return mos

    def plot_detections(self, dilate=2,
                        image=dict(cmap=cmr.voltage_r),
                        contour=dict(cmap='hot', lw=1.5),
                        label=dict(color='w', size='xx-small'),
                        tabbed=True):

        reorder = op.itemgetter(*(self.order + 1))
        segments = reorder(self.detections)
        images = reorder(self.images)

        # overlay ragged apertures
        figures = []
        for im, seg in zip(images, segments):
            seg = seg.dilate(dilate, copy=True)
            img = ImageDisplay(im, **image)
            seg.show_contours(img.ax, **contour)
            seg.show_labels(img.ax, **label)
            figures.append(img.figure)
            #img.save(loc / f'{hdu.file.stem}-ragged.png')

        if tabbed:
            from scrawl.multitab import MplMultiTab
            
            ui = MplMultiTab(figures=figures)
            ui.show()
            return ui

        return figures


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
                logger.warning('Failed to retrieve image from server: '
                               '%r', srv)

        # DSS data array
        data = self.hdu[0].data.astype(float)
        find_kws.setdefault('deblend', True)
        ImageRegister.__init__(self, **find_kws)
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
        self.target_coords_pixels = np.divide(self.image.shape, 2) + 0.5

    def remap_labels(self, target=None, flux_sort=True):
        if target is None:
            target, = self.clustering.predict([self.target_coords_pixels])
        return super().remap_labels(target, flux_sort)

    def mosaic(self, axes=None, names=(), **kws):

        header = self.hdu[0].header
        name = ' '.join(filter(None, map(header.get, ('ORIGIN', 'FILTER'))))

        # use aplpy to setup figure
        ff = apl.FITSFigure(self.hdu)

        # rescale images to DSS pixel scale
        return super().mosaic(ff.ax, [name], 'pixels', **kws)

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
    #     # ignore stars not detected in dss, but detected in sample image
    #     use[labels == 0] = False
    #     # ignore labels that are detected as 2 (or more) sources in the
    #     # sample image, but only resolved as 1 in dss
    #     for w in where_duplicate(labels):
    #         use[w] = False
    #     return labels, use

    def register(self, clf=None, plot=False):
        super().register(clf, plot)
        self.relabel_segments()

    def build_wcs(self, run):
        assert len(run) == len(self) - 1
        for hdu, p, scale in zip(run, self.params[1:], self.scales[1:]):
            self._build_wcs(hdu, p, scale)

    def _build_wcs(self, hdu, p, scale):
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
        from recipes.transforms.rotation import rotation_matrix

        hdr = self.hdu[0].header
        *xyoff, theta = p
        # transform target coordinates in DSS image to target in SHOC image
        # convert to pixel llc coordinates then to arcmin
        # target coordinates DSS [pixel]
        xyDSS = np.subtract([hdr['crpix1'], hdr['crpix2']], 0.5) / self.scale
        # target coordinates SHOC [pixel]
        xySHOC = (rotation_matrix(-theta) @ (xyDSS - xyoff)) / scale
        # target coordinates celestial [degrees]
        xySKY = self.target_coords_world

        # coordinate increment
        # flip = np.array(hdu.flip_state[::-1], bool)
        # cdelt = scale / 60.  # image scale in degrees
        # cdelt[flip] = -cdelt[flip]

        # see:
        # https://www.aanda.org/articles/aa/full/2002/45/aah3859/aah3859.html
        # for parameter definitions
        w = wcs.WCS(naxis=2)
        # array location of the reference point in pixels
        w.wcs.crpix = xySHOC
        # coordinate increment at reference point
        w.wcs.cdelt = scale / 60.  # image scale in degrees
        # target coordinate in degrees reference point
        xtrg = self.target_coords_world
        w.wcs.crval = xtrg.ra.value, xtrg.dec.value
        # rotation from stated coordinate type.
        w.wcs.crota = np.degrees([-theta, -theta])
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # axis type

        return w

    def plot_coords_nrs(self, ax=None,  color='c', ms=10, **kws):
        if ax is None:
            fig, ax = plt.subplots()

        for i, xy in enumerate(self.xy):
            ax.plot(*xy, marker='$%i$' % i, color=color, ms=ms, **kws)

        # for i, yx in enumerate(coords):
        #     ax.plot(*yx[::-1], marker='$%i$' % i, color='g')


# class TransformedImage():
#     def set_p(self):
#         self.art.set_transform()
