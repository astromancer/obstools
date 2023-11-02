

# std
import warnings

# third-party
import numpy as np

# local
from scrawl.image import ImageDisplay
from recipes.utils import duplicate_if_scalar

# relative
from .. import transforms as transform
from ..modelling import Model
from .utils import non_masked


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
        self._amplitudes = self._check_prop(amplitudes, 'amplitudes', (self.n,))

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
        # pre-factors so the individual gaussians integrate to 1 (probability)
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
        # default parameter values for evaluation
        return np.zeros(self.dof) if (p is None) or (p == ()) else p

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
        size = np.array(duplicate_if_scalar(size, self.n_dims))
        slices = map(slice, xyl.min(0), xyu.max(0), size * 1j)
        return np.moveaxis(np.mgrid[tuple(slices)], 0, -1)

    def plot(self, grid=None, size=100, show_xy=True, show_peak=True, **kws):
        """Image the model"""

        ndims = self.n_dims
        if ndims != 2:
            raise ValueError(f'Can only image 2D models. Model is {ndims}D.')

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
    therefore don't need to fit for the scale parameters. Furthermore, if the 
    frame to frame angle remains constant, set the `fit_angle` property to True.
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
        return transform.rigid if self.fit_angle else np.add

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
