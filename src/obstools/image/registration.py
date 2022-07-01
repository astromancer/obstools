# Image registrarion algorithms (point set registration)

"""


Helper functions to infer World Coordinate System given a target name or
coordinates of a object in the field. This is done by matching the image
with the DSS image for the same field via image registration.  A number of
methods are implemented for doing this:
  coherent point drift
  matching via locating dense cluster of displacements between points
  direct image-to-image matching 
  brute force search with gaussian mixtures on points
"""

from scipy.interpolate import NearestNDInterpolator
from obstools.modelling import Model
import numbers
from pyxis.containers import ItemGetter, OfType, AttrMapper, AttrProp
import collections as col
from recipes.misc import duplicate_if_scalar
import functools as ftl
import logging
import multiprocessing as mp
import itertools as itt
import warnings
import re

import numpy as np

from scipy.cluster.vq import kmeans
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from obstools.stats import geometric_median
from astropy.utils import lazyproperty

from recipes.dicts import AttrDict
from obstools.image.segmentation import SegmentedImage
from obstools.phot.campaign import HDUExtra
from ..utils import get_coordinates, get_dss, STScIServerError

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from recipes.logging import LoggingMixin


from scipy.stats import binned_statistic_2d, mode

# from motley.profiling.timers import timer
from scipy.spatial import cKDTree
# from sklearn.cluster import MeanShift
from scrawl.imagine import ImageDisplay
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from . import transforms as trans

# from motley.profiling import profile

from recipes.logging import logging, get_module_logger

# module level logger
logger = get_module_logger()
logging.basicConfig()
logger.setLevel(logging.INFO)


TABLE_STYLE = dict(txt='bold', bg='g')


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
    xy = trans.rigid(xy, p)
    bs = binned_statistic_2d(*xy.T, values, 'mean', bins)
    rs = np.square(target - bs.statistic)
    return np.nansum(rs)


def detect_measure(image, mask=False, background=None, snr=3., npixels=5,
                   edge_cutoff=None, deblend=False, dilate=0):
    #
    seg = SegmentedImage.detect(image, mask, background, snr, npixels,
                                edge_cutoff, deblend, dilate)

    counts = seg.sum(image) - seg.median(image, [0]) * seg.areas
    coords = seg.com_bg(image)
    return seg, coords, counts


def interpolate_nn(a, where):
    good = ~where
    grid = np.moveaxis(np.indices(a.shape), 0, -1)
    nn = NearestNDInterpolator(grid[good], a[good])
    a[where] = nn(grid[where])
    return a


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


class MultivariateGaussians(Model):
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
            number of spatial dimensions in the problem (eg 2 in the case of an
            source field model, 1 in the case of a point source model).
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
                ⎪__MultivariateGaussians_⎪                     
                ⎪x ⎪ y ⎪ σₓ  ⎪ σᵥ  ⎪  A  ⎪
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
                                 minimalist=True)
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
        return self._check_params(p), self._check_grid(xy)

    def _check_params(self, p):
        return np.zeros(self.dof) if (p is None) or (p == ()) else p

    def _check_grid(self, grid):
        #
        grid = non_masked(grid)
        shape = grid.shape
        # make sure last dimension is same as model dimension
        if (grid.ndim < 2) and (shape[-1] != self.n_dims):
            raise ValueError(
                'compute grid has incorrect size in last dimension: shape is '
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

    def plot(self, grid=100, show_xy=True, show_peak=True, **kws):
        """Image the model"""

        if self.n_dims != 2:
            raise Exception('Can only image 2D models')

        grid = duplicate_if_scalar(grid, self.n_dims, raises=False)
        if grid.size == self.n_dims:
            grid = self._auto_grid(grid)
        elif (grid.ndim != 3) or (grid.shape[-1] != self.n_dims):
            raise ValueError('Invalid grid')

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


class GaussianMixtureModel(MultivariateGaussians):

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

    def logLikelihood(self, p, xy, data=None, stddev=None):
        """
        Log likelihood of independently drawn observations `xy`. ie.
        Sum of log-likelihoods
        """
        # sum of gmm log likelihood

        # ignore incoming masked points
        prob = np.atleast_1d(self(p, xy).sum(-1))
        # NOTE THIS OBVIATES ANY OPTIMIZATION through eval but is needed for
        # direct call to this method

        # :           substitute 0 --> 1e-300      to avoid warnings and infs
        prob[(prob == 0)] = 1e-300
        return np.log(prob).squeeze()

    llh = logLikelihood

# class CoherentPointDrift(GaussianMixtureModel):
#     """
#     Model that fits for the transformation parameters to match point clouds.
#     This is a partial implementation of the full CPD algorithm since we
#     generally know the scale of the feature coordinates (field-of-view) and
#     therefore don't need to fit for the scale parameters.
#     """

#     _fit_rotation = True  # default

#     @property
#     def dof(self):
#         return 2 + self._fit_rotation

#     @property
#     def fit_rotation(self):
#         return self._fit_rotation

#     @fit_rotation.setter
#     def fit_rotation(self, tf):
#         self._fit_rotation = tf

#     @property
#     def transform(self):
#         return trans.rigid if self._fit_rotation else np.add

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

    _fit_rotation = True  # default

    @property
    def dof(self):
        return 2 + self._fit_rotation

    @property
    def fit_rotation(self):
        return self._fit_rotation

    @fit_rotation.setter
    def fit_rotation(self, tf):
        self._fit_rotation = tf

    @property
    def transform(self):
        return trans.rigid if self._fit_rotation else np.add

    def __init__(self, xy, sigmas, weights=None):
        self.gmm = GaussianMixtureModel(xy, sigmas, weights)

    def __call__(self, p, xy):
        return self.gmm((), self.transform(xy, p))

    def logLikelihood(self, p, xy, stddev=None):
        return self.gmm.logLikelihood((), self.transform(xy, p))

    llh = logLikelihood

    def p0guess(self, data, *args):
        # starting values for parameter optimization
        return np.zeros(self.dof)

    def fit(self, xy, stddev=None, p0=None):
        # This will evaluate the
        return super().fit(xy, p0, self.loss_mle)


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
    for image, fov, p, yx in zip(images, fovs, params, coords):
        xy = yx[:, ::-1]  # roto_translate_yx(yx, np.r_[-p[:2], 0])[:, ::-1]
        ex = mit.interleave((0, 0), fov)
        im = ImageDisplay(image, extent=list(ex))
        im.ax.plot(*xy.T, 'kx', ms=5)
        ui.add_tab(im.figure)
        plt.close(im.figure)
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
#                 'Ignoring %i/%i (%.1f%%) nan / masked values for position '
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
        logger.info('Ignoring %i/%i (%.1f%%) nan values for position '
                    'measurement', n_ignore, n, (n_ignore / n) * 100)

    n_detections_per_star = np.zeros(n_stars, int)
    w = np.where(~bad)[1]
    u = np.unique(w)
    n_detections_per_star[u] = np.bincount(w)[u]

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

        if not (changed := (outliers != out_new).any()):
            break

        out = out_new
        xym[out] = np.ma.masked
        n_out = out.sum()

        if n_out / n_points > 0.5:
            raise Exception('Too many outliers!!')

        logger.info('Ignoring %i/%i (%.1f%%) values with |δr| > %.3f',
                    n_out, n_points, (n_out / n_points) * 100, d_cut)
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


def report_measurements(xy, centres, σ_xy,  xy_offsets=None,
                        counts=None, detect_frac_min=None, count_thresh=None, logger=logger):
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
                             col_heade=col_headers,
                             col_head_props=TABLE_STYLE,
                             precision=3,
                             align='r',
                             row_nrs=True,
                             totals=[0],
                             formatters=formatters)

    # fix formatting with percentage in total.
    # TODO Still need to think of a cleaner solution for this
    tbl.data[-1, 0] = re.sub(r'\(\d{3,4}%\)', '', tbl.data[-1, 0])
    # tbl.data[-1, 0] = tbl.data[-1, 0].replace('(1000%)', '')

    logger.info('\n' + str(tbl) + extra)
    return tbl


class SkyImage(object):
    """
    Helper class for image registration. Represents an image with some
    associated meta data like size as well as detected sources and their counts.
    """

    def __init__(self, data, fov=None, scale=None):
        """
        Create and SkyImage object with a know size on sky.

        Parameters
        ----------
        data : array-like
            The image data as a 2d
        fov : float or array_like of float, optional
            Field-of-view of the image in arcminutes. The default None, however 
            either `fov` or `scale` must be given.
        scale : float or array_like of float, optional
            Pixel scale in arcminutes/pixel. The default None, however 
            either `fov` or `scale` must be given.

        Raises
        ------
        ValueError
            If both `fov` and `scale` are None
        """

        if (fov is scale is None):
            raise ValueError('Either field-of-view, or pixel scale must be '
                             'given and not be `None`')

        # data array
        self.data = np.asarray(data)
        self.fov = np.array(duplicate_if_scalar(fov))

        # pixel size in arcmin xy
        self.pixel_scale = self.fov / self.data.shape

        # segmentation data
        self.seg = None  # : SegmentedArray:
        # self.xyp = None  # : np.ndarray: center-of-mass coordinates pixels
        self.xy = None  # : np.ndarray: center-of-mass coordinates arcmin
        self.counts = None  # : np.ndarray: pixel sums for segmentation

    def detect(self, **kws):
        self.seg, yx, counts = detect_measure(self.data, **kws)

        # sometimes, we get nans from the center-of-mass calculation
        ok = np.isfinite(yx).all(1)
        if not ok.any():
            warnings.warn('No detections for image')

        self.xy = yx[ok, ::-1]
        # self.xy = self.xyp * self.pixel_scale
        self.counts = counts[ok]
        return self.seg, self.xy, self.counts

    def __array__(self):
        return self.data

    def get_transform(self, params, scale='pixels'):
        if scale == 'pixels':
            scale = 1
        elif scale == 'sky':
            scale = self.pixel_scale
        elif not isinstance(scale, numbers.Real):
            raise ValueError(f'scale value {scale!r} not understood')

        *xyo, theta = params
        return Affine2D().scale(scale).rotate(theta).translate(*xyo)

    # @lazyproperty
    # def grid(self):
    #     return np.indices(self.data.shape)

    def grid(self, p, scale):
        """transformed pixel location grid cartesian xy coordinates"""
        g = np.indices(self.data.shape).reshape(2, -1).T[:, ::-1]
        return trans.affine(g, p, scale)

    def plot(self, ax=None, p=(0, 0, 0), scale='fov', frame=True, **kws):
        """
        Display the image in the axes, applying the affine transformation for
        parameters `p`
        """

        kws.setdefault('hist', False)
        kws.setdefault('sliders', False)
        kws.setdefault('cbar', False)

        if scale == 'fov':
            # get extent - adjusted to pixel centers.
            extent = np.c_[[0., 0.], self.fov[::-1]]
            half_pixel_size = self.pixel_scale / 2
            extent -= half_pixel_size[None].T
            urc = extent[:, 1]
            kws['extent'] = extent.ravel()
        else:
            half_pixel_size = 0.5
            urc = self.fov[::-1]

        # plot
        im = ImageDisplay(self.data, ax=ax, **kws)

        # Rotate + offset the image by setting the transform
        im.imagePlot.set_transform(self.get_transform(p) + im.ax.transData)

        # Add frame
        if frame:
            from matplotlib.patches import Rectangle

            frame_kws = dict(fc='none', lw=1, ec='0.5',
                             alpha=kws.get('alpha'))
            if isinstance(frame, dict):
                frame_kws |= frame

            *xy, theta = p
            frame = Rectangle(np.subtract(xy, half_pixel_size), *urc,
                              np.degrees(theta), **frame_kws)
            im.ax.add_patch(frame)

        return im.imagePlot, frame


# class TransformedImageGrids(col.defaultdict, LoggingMixin):
#     """
#     Container for segment masks
#     """

#     def __init__(self, reg):
#         self.reg = reg
#         col.defaultdict.__init__(self, None)

#     def __missing__(self, i):
#         # the grid is computed at lookup time here and inserted into the dict
#         # after this func executes
#         shape = self.reg.images[i].shape
#         r = (self.reg.fovs[i] / shape) / self.reg.pixel_scale
#         g = np.indices(shape).reshape(2, -1).T * r
#         g = roto_translate(g[:, ::-1], self.reg.params[i])
#         return g  # .reshape ??


# from obstools.phot.image import SourceDetectionMixin

def _echo(_):
    return _


class ImageContainer(col.UserList, OfType(SkyImage), ItemGetter, AttrMapper):
    def __init__(self, images=(), fovs=()):
        """
        A container of `SkyImages`'s

        Parameters
        ----------
        images : sequence, optional
            A sequence of `SkyImages` or 2d `np.ndarrays`, by default ()
        fovs : sequence, optional
            A sequence of field-of-views, each being of size 1 or 2. If each
            item in the sequence is of size 2, it is the field-of-view along the
            image array dimensions (rows, columns). It an fov is size 1, it is
            as being the field-of-view along each dimension of a square image.
            If `images` is a sequence of `np.ndarrays`, `fovs` is a required 
            parameter

        Raises
        ------
        ValueError
            If `images` is a sequence of `np.ndarrays` and `fovs` is not given.
        """
        # check init parameters.  If `images` are arrays, also need `fovs`
        n = len(images)
        if n != len(fovs):
            # items = self.checks_type(images, silent=True)
            types = set(map(type, images))

            if len(types) == 1 and issubclass(types.pop(), SkyImage):
                fovs = [im.fov for im in images]
            else:
                raise ValueError(
                    'When initializing this class from a stack of images, '
                    'please also proved the field-of-views `fovs`.')
                # as well as the set transformation parameters `params`.')

            # create instances of `SkyImage`
            images = map(SkyImage, images, fovs)

        # initialize container
        super().__init__(images)

        # ensure we get lists back from getitem lookup since the initializer
        # works differently to standard containers
        self.set_returned_type(list)

    # properties: vectorized attribute getters on `SkyImage`
    images = AttrProp('data')
    shapes = AttrProp('data.shape', np.array)
    detections = AttrProp('seg')
    coms = AttrProp('xy')
    fovs = AttrProp('fov', np.array)
    scales = AttrProp('pixel_scale', np.array)

    # @property
    # def params(self):
    #     return np.array(self._params)


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
    >>> reg.register_constellation()
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
    #  fast for nearly_aligned images

    find_kws = dict(snr=3.,
                    npixels=5,
                    edge_cutoff=2,
                    deblend=False)
    """default source finding keywords"""

    # Expansion factor for grid search
    _search_area_stretch = 1.25

    # Sigma for GMM as a fraction of minimal distance between stars
    _dmin_frac_sigma = 3

    # TODO: switch for units to arcminutes or whatever ??
    # TODO: uncertainty on center of mass from pixel noise!!!

    @property
    def image(self):
        return self.images[self.idx]

    @property
    def fov(self):
        return self.fovs[self.idx]

    @property
    def pixel_scale(self):
        return self.scales[self.idx]

    @property
    def params(self):
        return np.array(self._params)

    @params.setter
    def params(self, params):
        self._params = list(params)

    @property
    def rscale(self):
        return self.scales / self.pixel_scale

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, idx):
        idx = int(idx)
        n = len(self)
        if idx < 0:
            idx += n

        if idx > n:
            raise ValueError('Invalid index for reference frame')

        par = self.params
        rscale = self.rscale[idx]

        self._xy = (self._xy - par[idx, :2]) / rscale

        par[:, :2] /= rscale
        par -= par[idx]
        self._params = list(par)

        self._idx = idx

    @property
    def xy(self):
        """
        The reference coordinates for model constellation in units of
        arcminutes. These will be updated when more images are added to the
        register and we can make a more accurate determination of the source
        positions by the :meth:`recentre` and :meth:`refine` methods.
        """
        return self._xy

    @xy.setter
    def xy(self, xy):
        # Ensure xy coordinates are always a plain numpy array. Masked arrays
        # don't seem to work well with the matrix product `@` operator used in
        # :func:`roto_translate`
        self._xy = non_masked(xy)
        del self.model

    # @lazyproperty
    @property
    def xyt(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters
        """
        return list(map(trans.affine, self.coms, self.params, self.rscale))

    # @lazyproperty
    @property
    def xyt_block(self):
        """
        Transform raw center-of-mass measurements with the current set of
        parameters and group them according to the cluster identified labels
        """
        return group_features(self.labels, self.xyt)[0]

    @property
    def source_indices(self):
        if self.labels is None:
            raise Exception('Unregistered')

        # group_features(self.labels, self.labels)[0]
        return np.split(self.labels, np.cumsum(list(map(len, self.coms))))[:-1]

    @property
    def corners(self):
        corners = np.empty((len(self), 4, 2))
        frame_size = self.fovs / self.pixel_scale
        for i, (p, fov) in enumerate(zip(self.params, frame_size)):
            c = np.array([[0, 0], fov[::-1]])
            xy = np.c_[c[0], c[::-1, 0], c[1], c[:, 1]].T  # / clockwise xy
            corners[i] = trans.affine(xy, p)
        return corners

    @lazyproperty
    def model(self):
        return CoherentPointDrift(self.xy, self.guess_sigma())

    @classmethod  # .            p0 -----------
    def from_images(cls, images, fovs, angles=(), ridx=None, plot=False,
                    **find_kws):

        n = len(images)
        assert n
        # assert len(fovs)
        assert n == len(fovs)

        # align on highest res image if not specified
        shapes = list(map(np.shape, images))
        pixel_scales = np.divide(fovs, shapes)
        if ridx is None:
            ridx = pixel_scales.argmin(0)[0]

        indices = np.arange(n)
        if ridx:  # put this index first
            indices[0], indices[ridx] = indices[ridx], indices[0]

        # message
        cls.logger.info('Aligning %i images on image %i', n, ridx)

        angles = np.array(angles) - angles[ridx] if len(angles) else np.zeros(n)
        reg = cls(**find_kws)
        for i in indices:
            reg(images[i], fovs[i], angles[i], plot=plot)

        # # re-order everything
        # for at in 'images, fovs, detections, coms, counts, params'.split(', '):
        #     setattr(reg, at, list(map(getattr(reg, at).__getitem__, indices)))

        reg.register_constellation()
        return reg

    def __init__(self, images=(), fovs=(), params=(), **find_kws):
        """
        Initialize an image register.
        This class should generally be initialized without arguments. The model
        of the constellation of stars is built iteratively by calling an
        instance of this class on an image.
        eg:
        >>> ImageRegister()(image)

        If arguments are provided, they are the sequences of images,
        their field-of-views, and transformation parameters for the
        initial (zero-point) transform


        Parameters
        ----------
        images:
        fovs:
            Field of view for the image. Order of the dimensions should be
            the same as that of the image, ie. rows first
        params:
        find_kws:
        """

        if find_kws:
            self.find_kws.update(find_kws)

        # TODO: init from single image???

        # init container
        ImageContainer.__init__(self, images, fovs)

        # NOTE passing a reference index is only meaningful if the class is
        #  initialized with a set of images
        self._idx = 0
        self.targetCoordsPix = None
        self.sigmas = None
        self._xy = None
        # keep track of minimal separation between sources
        self._min_dist = np.inf

        self._params = []
        # self.grids = TransformedImageGrids(self)

        # self._sigma_guess = self.guess_sigma(self.xy)

        # state variables
        self.labels = self.n_stars = self.n_noise = None
        self._colour_sequence_cache = ()

        # for image, fov, params

        # parameters for zero point transforms
        # indices = list(set(range(len(images))) - {ridx})
        # if len(params):
        #     params = np.array(params)
        #     segs, coords, counts = zip(*map(detect_measure, images))
        #     self.aggregate(images, fovs, segs, coords, counts, params)
        #     # return

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

    def __call__(self, image, fov=None, rotation=0., refine=True, plot=False,
                 **find_kws):
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
        # -------

        """
        if not isinstance(image, SkyImage):
            image = SkyImage(image, fov)

        # defaults
        for k, v in self.find_kws.items():
            find_kws.setdefault(k, v)

        # source detection
        image.detect(**find_kws)

        if len(self.images):
            # NOTE: work internally in units of pixels of the reference image
            # since it makes plotting the images easier without having to pass
            # through the reference scale to the plotting routines...
            xy = (image.pixel_scale / self.pixel_scale) * image.xy
            p = self.match_points(xy, rotation, refine, plot)
        else:
            # Make initial reference coordinates.
            # `xy` are the initial target coordinates for matching. Theses will
            # be updated in `recentre` and `refine`
            self._xy = image.xy
            p = np.zeros(3)

        # aggregate
        self.append(image)
        self._params.append(p)
        # self._sigma_guess = min(self._sigma_guess, self.guess_sigma(xy))
        # self.logger.debug('sigma guess: %s' % self._sigma_guess)

        # update minimal source seperation
        self._min_dist = min(self._min_dist, dist_flat(image.xy).min())

        # reset model
        del self.model

        return p

    def __repr__(self):
        return (f'{self.__class__.__name__}: {len(self)} images' +
                ('' if self.labels is None else f'; {self.labels.max()} sources')
                )

    # def to_pixel_coords(self, xy):
    #     # internal coordinates are in arcmin origin at (0,0) for image
    #     return np.divide(xy, self.pixel_scale)

    def convert_to_pixels_of(self, imr):  # to_pixels_of
        # convert coordinates to pixel coordinates of reference image
        ratio = self.pixel_scale / imr.pixel_scale
        xy = self.xy * ratio

        params = self.params
        params[:, :2] *= ratio
        # NOTE: this does not edit internal `_params` since array is made
        # through the `params` property

        # note this means reg.mosaic will no longer work without fov keyword
        #  since we have changed the scale and it substitutes image shape for
        #  fov if not given

        return xy, params

    def min_dist(self):
        return min(dist_flat(xy).min() for xy in self.xyt)

    def guess_sigma(self):
        """choose sigma for GMM based on distances between detections"""
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
        if not isinstance(hdu, HDUExtra):
            raise TypeError('Can only match HDUs if they inherit from '
                            f'`{HDUExtra.__module__}.{HDUExtra.__name__}`. '
                            'Alternatively use the `match_image` or `__call__` '
                            'methods to match an image array directly.')

        image = hdu.get_sample_image(depth, sample_stat)
        return self(image, hdu.fov, **findkws)

    def match_reg(self, reg, rotation=0):
        """
        cross match with another `ImageRegister` and aggregate points
        """

        # convert coordinates to pixel coordinates of reference image
        # reg.convert_to_pixels_of(self)
        xy, params = reg.convert_to_pixels_of(self)

        reg.copy()

        # use `match points` so we don't aggregate data just yet
        # xy = rotate(xy.T, rotation).T
        p = self.match_points(xy, rotation)
        # params = np.array(reg.params)
        params[:, :2] += p[:2]
        params[:, -1] += rotation
        #reg.params = list(params)

        # finally aggregate the results from the new register
        self.extend(reg.data)
        self._params.extend(params)

        # convert back to original coords
        # reg.convert_to_pixels_of(reg)

        # fig, ax = plt.subplots()
        # for xy in reg.xyt:
        #     ax.plot(*xy.T, 'x')
        # ax.plot(*imr.xy.T, 'o', mfc='none', ms=8)

    def match_image(self, image, fov, rotation=0., refine=True, plot=False,
                    **find_kws):
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
        seg, xy, counts = SkyImage(image, fov).detect(**find_kws)

        # match images directly
        # mr = self.match_pixels(image, fov[::-1], p)

        #
        p = self.match_points(xy, rotation, refine, plot)
        # p = self.match_points(xy, fov, rotation, plot)

        # return yx (image) coordinates
        # yx = roto_translate_yx(yx, p)

        return p, xy, counts, seg

    # @timer
    def match_points(self, xy, rotation=0., refine=True, plot=False):

        if rotation is None:
            raise NotImplementedError
            #  match gradient descent
            # p = self.model.fit_trans(pGs, xy, fit_angle=True)

        # get shift (translation) via brute force search through point to
        # point distances. This is gauranteed to find the optimal alignment
        # between the two point clouds in the absense of rotation between them
        dxy = self._dxy_hop(xy, rotation, plot)
        p = np.r_[dxy, rotation]

        if refine:
            # do mle via gradient decent to refine
            r = self.model.fit(xy, p0=p)
            if r is not None:
                return r

        return p

    def register_constellation(self, clustering=None, plot=False):
        # TODO: rename register / cluster
        # clustering + relative position measurement
        clustering = clustering or self.get_clf()
        self.cluster_id(clustering)
        # make the cluster centres the target constellation
        self.xy = self.xyt_block.mean(0)

        if plot:
            #
            art = self.plot_clusters()
            ax = art.axes

            # bandwidth size indicator.
            # todo: get this to remain anchored lower left but scale with zoom..
            xy = self.xy.min(0)  # - bw * 0.6
            cir = Circle(xy, clustering.bandwidth, alpha=0.5)
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

    def get_clf(self, *args, **kws):
        from sklearn.cluster import MeanShift

        # minimal distance between stars distance
        return MeanShift(**{**kws, **dict(bandwidth=self.min_dist() / 2,
                                          cluster_all=False)})

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

    # def check_labelled(self):
        # if not len(self):
        #     raise Exception('No data. Please add some images first. eg: '
        #                     '{self.__class__.__name__}()(image, fov)')

        # if self.labels is None:
        #     raise Exception('No clusters identified. Run '
        #                     '`register_constellation` to fit clustering model '
        #                     'to the measured centre-of-mass points')

    def plot_clusters(self, show_bandwidth=True, **kws):
        """
        Plot the identified sources (clusters) in a single frame
        """
        if not len(self):
            raise Exception('No data. Please add some images first. eg: '
                            '{self.__class__.__name__}()(image, fov)')

        if self.labels is None:
            raise Exception('No clusters identified. Run '
                            '`register_constellation` to fit clustering model '
                            'to the measured centre-of-mass points')

        n = len(self)
        fig0, ax = plt.subplots()  # shape for slotmode figsize=(13.9, 2)
        ax.set_title(f'Position Measurements (CoM) {n} frames')

        art = plot_clusters(ax, np.vstack(self.xyt), self.labels, **kws)
        self._colour_sequence_cache = art.get_edgecolors()

        # bandwidth size indicator.
        # todo: get this to remain anchored lower left but scale with zoom..
        xy = self.xy.min(0)  # - bw * 0.6
        cir = Circle(xy, self.get_clf().bandwidth, alpha=0.5)
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
            self.register_constellation()

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
                self._params[i][:-1] -= xy_offsets[i]

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

    def refine(self, fit_rotation=True, plot=False):
        """
        Refine alignment parameters by fitting transform parameters for each
        image using maximum likelihood objective for gmm model with peaks
        situated at cluster centers

        """
        if self.labels is None:
            self.register_constellation()

        xyt = self.xyt
        params = self.params
        # guess sigma:  needs to be larger than for brute search `_dxy_hop`
        # since we are searching a smaller region of parameter space
        # self.model.sigma = dist_flat(self.xy).min() / 3

        failed = []
        # TODO: multiprocess
        self.model.fit_rotation = fit_rotation
        for i in range(len(self.images)):
            p = self.model.fit(xyt[i], p0=(0, 0, 0))
            if p is None:
                failed.append(i)
            else:
                params[i] += p

        self.logger.info('Fitting successful %i / %i', i - len(failed), i)

        # likelihood ratio test
        xyn = list(map(trans.affine, self.coms, params, self.rscale))
        lhr = self._lh_ratio(np.vstack(xyt), np.vstack(xyn))
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

    def _lh_ratio(self, xy0, xy1):
        # likelihood ratio
        ratio = np.exp(self.model.gmm.llh((), xy0) -
                       self.model.gmm.llh((), xy1))
        self.logger.info('Likelihood ratio: %.5f', ratio)

        # decide whether to accept new params!
        if ratio > 1:
            self.logger.info('Accepting new parameters.')
        else:
            self.logger.info('Keeping same parameters.')
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
    def _stack_pixels(self, images=None, image_func=_echo):

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
        # image_func = image_func or _echo
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

        xn, yn = np.ceil(grid.ptp(0) / self.rscale[self.idx]).astype(int)
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

    def _dxy_hop(self, xy, rotation=0., plot=False):
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

        # get coords.
        xyr = trans.rigid(xy, (0, 0, rotation))
        # create search grid.

        trials = (self.xy[None] - xyr[:, None]).reshape(-1, 2)
        # Ignore extremal points in grid search.  These represent single
        # point matches at the edges of the frame which are most certainly
        # not the best match
        extrema = np.ravel([trials.argmin(0), trials.argmax(0)])
        trials = np.delete(trials, extrema, 0)

        # This animates the search!
        # line, = im.ax.plot((), (), 'rx')
        # for i, po in enumerate(points.T):
        #     line.set_data(*(xyr - po).T)
        #     im.figure.canvas.draw()
        #     input(i)
        #     if i > 25:
        #          break

        # find minimum
        state = self.model.fit_rotation
        self.model.fit_rotation = False
        r = [self.model.loss_mle(p, xy) for p in trials]
        p = trials[np.argmin(r)]

        self.logger.debug('Grid search optimum: %s', p)

        if plot:
            im = self.model.gmm.plot(show_peak=False)
            im.ax.plot(*(xyr + p).T, 'ro', ms=12, mfc='none')

        # restore sigma
        self.model.fit_rotation = state
        return np.array(p)

    # TODO: relative brightness

    # @timer
    def match_points_brute(self, xy, rotation=0., gridsize=(50, 50),
                           plot=False):
        # grid search with gmm loglikelihood objective
        g, r, ix, pGs = self._match_points_brute(xy, gridsize, rotation, )
        pGs[-1] = rotation

        if plot:
            from scrawl.imagine import ImageDisplay

            # plot xy coords
            # ggfig, ax = plt.subplots()
            # sizes = self.counts[0] / self.counts[0].max() * 200
            # ax.scatter(*self.xy.T, sizes)
            # ax.plot(*roto_translate(xy, pGs).T, 'r*')

            im, peak = self.model.gmm.plot(show_peak=False)
            im.ax.plot(*trans.rigid(xy, pGs).T, 'rx')

            extent = np.c_[g[:2, 0, 0], g[:2, -1, -1]].ravel()
            im = ImageDisplay(r.T, extent=extent)
            im.ax.plot(*pGs[:-1], 'ro', ms=15, mfc='none', mew=2)

        return pGs

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
                g = trans.rigid(g.reshape(-1, 2) * r, p)
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

        # return im, s

    def mosaic(self, names=(), number_sources=False, **kws):

        from obstools.image.mosaic import MosaicPlotter

        mos = MosaicPlotter.from_register(self)
        # params are in units of pixels convert to units of `fov` (arcminutes)
        params = self.params
        params[:, :2] *= self.pixel_scale
        mos.mosaic(params, names,  **kws)

        if number_sources:
            off = -4 * self.rscale.min(0) * self.pixel_scale
            mos.mark_sources(self.xy * self.pixel_scale,
                             marker=None,
                             xy_offset=off)

        return mos


class ImageRegistrationDSS(ImageRegister):
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
        ImageRegister.__init__(self, **find_kws)
        self(data, fov)

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

    def mosaic(self, names=(), **kws):

        from obstools.image.mosaic import MosaicPlotter

        header = self.hdu[0].header
        name = ' '.join(filter(None, map(header.get, ('ORIGIN', 'FILTER'))))
        names = (name, )

        # import aplpy as apl
        # ff = apl.FITSFigure(self.hdu)

        mos = super().mosaic(names, **kws)

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
        from recipes.transforms.rotation import rotation_matrix

        *yxoff, theta = p
        fov = hdu.get_fov(telescope)
        pxscl = fov / hdu.shape[-2:]
        # transform target coordinates in DSS image to target in SHOC image
        h = self.hdu[0].header
        # target object pixel coordinates
        crpix = np.array([h['crpix1'], h['crpix2']])
        crpixDSS = crpix - 0.5  # convert to pixel llc coordinates
        cram = crpixDSS / self.pixel_scale  # convert to arcmin
        rot = rotation_matrix(-theta)
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
