from warnings import warn
from collections import Callable

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.measurements import center_of_mass as CoM

# from lmfit import minimize

from recipes.list import find_missing_numbers


# from magic.string import banner

# from myio import warn


# from decor.misc import cache_last_return

# from IPython import embed
# from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
# from decor.misc import unhookPyQt

# TODO:  use astropy.models!!!!??? // lmfit.models
# ****************************************************************************************************
def constant(p, grid):
    return p[0]


# ****************************************************************************************************
def circularGaussian(p, grid):
    """Circular Gaussian function for fitting star profiles."""
    _, _, z0, a, d = p
    yxm = grid - p[1::-1, np.newaxis, np.newaxis]
    r = (yxm * yxm).sum(0)  # squared radial distance from centre
    return z0 * np.exp(-a * r) + d


# ****************************************************************************************************
# def gaussian2D(p, grid):
# """Elliptical Gaussian function for fitting star profiles."""
# x0, y0, z0, a, b, c, d = p
# y, x = grid
# xm, ym = x-x0, y-y0
# return z0 * np.exp(-(a*xm**2 - 2*b*xm*ym + c*ym**2)) + d

def gaussian2D(p, grid):
    """Elliptical Gaussian function for fitting star profiles."""
    # recasting in terms of (a, b, c) parameters (instead of sigma)
    # yields faster computation times and better convergence results
    # at lower signal to noise ratios.
    _, _, z0, a, b, c, d = p
    yxm = grid - p[1::-1, np.newaxis, np.newaxis]
    yxm2 = yxm * yxm
    return z0 * np.exp(-(a * yxm2[1] - 2 * b * yxm[0] * yxm[1] + c * yxm2[0])) + d


# alias
ellipticalGaussian = gaussian2D

# from scipy.stats import multivariate_normal
# def gaussian2D(p, grid):
# """Elliptical Gaussian function for fitting star profiles."""
# _, _, z0, a, b, c, d = p
# detP = (a*c + b*b) * 4
##inverse of precision matrix
# covm = (1 / detP) * np.array([[c,   b],
# [b,   a]]) * 2

# return z0 * multivariate_normal.pdf(grid.T, p[1::-1], covm)

# def gaussian2D(p, grid):
# """Elliptical Gaussian function for fitting star profiles.""" #more optimized?
# _, _, z0, a, b, c, d = p
# P = np.array([[a,   -b],
# [-b,  c ]])
# yxm = grid - p[1::-1, np.newaxis]
# return z0 * np.exp(-(np.einsum('...i,...i->...', yxm.T.dot(P), yxm.T))) + d

# from scipy.linalg import blas
# def gaussian2D(p, grid):
# """Elliptical Gaussian function for fitting star profiles.""" #more optimized?
# _, _, z0, a, b, c, d = p
# P = np.array([[a,   -b],
# [-b,  c ]])
# gs = grid.shape
# yxm = grid.reshape(2,-1) - p[1::-1, np.newaxis]
# return z0 * np.exp(-(np.einsum('...i,...i->...',
# blas.dgemm(1., yxm.T, P), yxm.T))
# ).reshape(gs[1:]) + d


def no_nan(p):
    return ~np.isnan(p).any()


# ****************************************************************************************************
class PSF(object):
    """Class that implements the point source function."""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, F, default_params, to_cache=None):
        """
        Initialize a PSF model function.
        Parameters
        ----------
        F       :       callable
            Function which evaluates PSF at given location(s).
            Call sequence is F(p,grid) where:
                p - sequence of parameters
                grid -  the (2,...) grid positions to evaluate for. #TODO: infer if not given??
        to_cache:       sequence of ints or slice
            The index positions of parameters that will be cached by the StarFit class.
        default_params:  array-like
            default values of parameters used to build the cache.

        Attributes
        ----------

        """
        if not isinstance(F, Callable):
            raise ValueError('Please specify callable PSF function')

        self.F = F
        self.objective = self.fwrs  # use flattened weighted square residuals as objective
        self.npar = npar = len(default_params)
        self.default_params = defpar = np.asarray(default_params, float)
        if to_cache is None:
            ix = np.arange(len(defpar))  # cache all by default
        elif isinstance(to_cache, slice):
            ix = np.arange(*to_cache.indices(npar))  # convert to array of int
        else:
            ix = np.asarray(to_cache).astype(int)

        self.to_cache = ix
        self.to_guess = np.array(find_missing_numbers(np.r_[-1, ix, npar]))  # indeces of uncached params
        # self.default_cache = #defpar[ix]

        self.validations = [no_nan]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, p, grid):  # TODO: infer grid if not given??
        return self.F(p, grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __repr__(self):
        return self.F.__name__  # NOTE: This may be confusing, but it makes the logs more readible

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, data, grid, data_stddev=None, **kws):
        """guess p0 and fit"""
        p0 = self.p0guess(data)
        result = leastsq(self.objective, p0, args=(data, grid, data_stddev), **kws)
        return result

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def param_hint(self, data):
        """Return a guess of the fitting parameters based on the data"""
        raise NotImplementedError('To be over-written by subclass')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def p0guess(self, data, p0=None):
        """
        Return best guess paramaters based on param_hint
        if p0 is not given, the default parameters will be used as starting point
        if p0 is given, data in its `to_guess` index positions will be set
        """
        if p0 is None:
            p0 = self.default_params
        p0[self.to_guess] = self.param_hint(data)
        return p0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @cache_last_return
    def background_estimate(self, data, edgefraction=0.15):
        """background estimate using window edge pixels"""

        shape = data.shape
        # print(shape)
        bgix = np.multiply(shape, edgefraction).round().astype(int)  # use 10% edge pixels mean as bg estimate
        subset = np.r_[data[tuple(np.mgrid[:bgix[0], :shape[1]])].ravel(),
                       data[tuple(np.mgrid[:shape[0], :bgix[1]])].ravel()]
        #subset = subset[~np.isnan(subset)]
        return np.ma.median(subset), np.ma.std(subset)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def residuals(self, p, data, grid):
        """Difference between data and model"""
        return data - self(p, grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rs(self, p, data, grid):
        """squared residuals"""
        return np.square(self.residuals(p, data, grid))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def frs(self, p, data, grid):
        """squared residuals flattened"""
        return self.rs(p, data, grid).flatten()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rss(self, p, data, grid):
        """residual sum of squares"""
        return self.rs(p, data, grid).sum()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wrs(self, p, data, grid, data_stddev=None):
        """weighted squared residuals"""
        if data_stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / data_stddev

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fwrs(self, p, data, grid, data_stddev=None):
        """weighted squared residuals flattened"""
        return self.wrs(p, data, grid, data_stddev).flatten()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wrss(self, p, data, grid, data_stddev=None):
        """weighted residual sum of squars"""
        return self.wrs(p, data, grid, data_stddev).sum()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def validate(self, p, *args):
        """validate parameter values.  To be overwritten by sub-class"""
        return all([vf(p) for vf in self.validations])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_validation(self, func):
        if not isinstance(func, Callable):
            raise TypeError
        else:
            self.validations.append(func)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # def set_bounds(self):


# ****************************************************************************************************
class ConstantBG(PSF):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    name = 'bg'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        default_params = (0,)
        to_cache = []
        PSF.__init__(self, constant, default_params, to_cache)

        self.add_validation(lambda p: p[0] > 0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def param_hint(self, data):
        return data.mean()

    # def get_aperture_params

# TODO: class Gaussian parent class
# ****************************************************************************************************
class CircularGaussianPSF(PSF):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    name = 'circular'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        default_params = (0, 0, 1, 1, 0)
        to_cache = []
        PSF.__init__(self, circularGaussian, default_params, to_cache)

        self.add_validation(lambda p: np.greater(p, 0).all())

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def param_hint(self, data):
        """Return a guess of the fitting parameters based on the data"""
        # peak
        bg, bgsig = self.background_estimate(data)
        z0 = data.max() - bg

        # location
        y0, x0 = np.divide(data.shape, 2)  # assumes the peak is roughly in the center of the data array
        # y0, x0 = np.c_[np.where(data==data.max())][0]   #NOTE: in case of multiple maxima it only returns the first
        # y0, x0 = CoM(data - np.median(data))

        # estimate sigma
        hm = 0.5 * z0  # half max
        yx = np.where(np.ma.greater_equal(data, hm))
        delta = np.subtract(yx, ([y0], [x0]))
        fwhme = np.ptp(np.sqrt(np.square(delta).sum(0)))  # fwhm estimate
        a = 4 * np.log(2) / (fwhme * fwhme)

        return x0, y0, z0, a, bg  # cached parameters will be set elsewhere

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def integrate(self, p):
        _, _, z0, a, d = p
        return z0 * np.pi / a

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def integration_uncertainty(self, p, punc):
        """linearized propagation of uncertainty"""
        # NOTE:  Error estimates for non-linear functions are biased on account of using a truncated series expansion.
        _, _, z0, a, d = p
        _, _, z0v, av, dv = np.square(punc)
        return (np.pi / av) * np.sqrt((z0 / a) ** 2 * z0v + av)

    int_err = integration_uncertainty

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_fwhm(self, p):
        a = p[-2]
        return 2 * np.sqrt(np.log(2) / a)

    def get_sigma(self, p):
        a = p[-2]
        return 1 / (2 * a)

    # def get_sigma_xy(self, p):
    #     sigma = self.get_sigma(p)
    #     return sigma, sigma

    # def get_theta(self, p):
    #     return 0

    def get_coo(self, p):
        return p[1::-1]

    def get_aperture_params(self, p):
        fwhm = self.get_fwhm(p)
        return np.r_[self.get_coo(p), fwhm, fwhm, 0]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def validate(self, p, window):
        #     p = np.asarray(p)
        #     w2 = np.divide(window, 2)
        #     nans = np.isnan(p).any()                    # nans are bad
        #     negpars = np.any(p < 0)                     # negative parameters ==> bad fit!
        #     badcoo = np.any(np.abs(p[:2] - w2) >= w2)   # center coordinates outside of grid - unphysical
        #     return ~(badcoo | negpars | nans)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def reparameterize(self, p):
        #     #reparameterize to more physically meaningful quantities
        #     _, _, z0, a, _ = p
        #     sigma               = np.sqrt(2*a)
        #     fwhm                = self.get_fwhm(p)
        #     par_alt             =
        #
        #     return par_alt


    # def reparameterize(self, p):
    #     """
    #     Reparameterize to the more physically intuitive space of
    #     (sigma_x, sigma_y, theta) quantities.  This
    #     """
    #
    #     # covm = self.covariance_matrix(p)
    #     # sigx, sigy = np.sqrt(np.diagonal(covm))
    #     q = p.copy()
    #     q[3] = sigma
    #     sigma = self.get_sigma(p)
    #     p
    #
    #     return p[0], p[1], sigma, 0, 0, 0



# ****************************************************************************************************
class GaussianPSF(CircularGaussianPSF):
    # TODO: option to pass angle, semi-major, semi-minor; or covariance matrix; volume?
    """ 7 param 2D Elliptical Gaussian (+constant)"""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # used for validation
    _ix_not_neg = list(range(7))  # _GaussianPSF.npar
    _ix_not_neg.pop(4)  # parameter b is allowed to be negative

    name = 'elliptical'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self):
        default_params = (0, 0, 1, .2, 0, .2, 0)
        to_cache = [4]  # atm we're not guessing b from data
        PSF.__init__(self, gaussian2D, default_params, to_cache)

        self.add_validation(lambda p: ~np.any(np.array(p)[self._ix_not_neg] < 0))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def param_hint(self, data):
        """Return a guess of the fitting parameters based on the data"""
        x0, y0, z0, a, bg = super(GaussianPSF, self).param_hint(data)

        return x0, y0, z0, a, a, bg

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reparameterize(self, p):
        # reparameterize to more physically meaningful quantities
        covm = self.covariance_matrix(p)
        sigx, sigy = np.sqrt(np.diagonal(covm))
        cov = covm[0, 1]
        theta = self.get_theta(p)
        ratio = min(sigx, sigy) / max(sigx, sigy)
        ellipticity = np.sqrt(1 - ratio ** 2)
        fwhm = self.get_fwhm(p)
        par_alt = sigx, sigy, cov, theta, ellipticity, fwhm
        return par_alt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def integrate(self, p):
        """
        Analytical solution to the integral over R2,
        source: http://en.wikipedia.org/wiki/Gaussian_integral#n-dimensional_and_functional_generalization
        """
        _, _, z0, a, b, c, _ = p
        return z0 * np.pi / np.sqrt(a * c - b * b)
        # A = 0.5*np.array([[a, b],
        # [b, c]] )
        # detA = np.linalg.det(A)
        # return 2*z0*np.pi / np.sqrt(detA)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def integration_uncertainty(self, p, punc):
        """Uncertainty associated with integrated flux via linearlized propagation of uncertainty"""
        # NOTE: Error estimates for non-linear functions are biased on account of using a truncated series expansion.
        # SEE: https://en.wikipedia.org/wiki/Uncertainty_quantification
        _, _, z0, a, b, c, _ = p
        _, _, z0v, av, bv, cv, _ = np.square(punc)
        detr = a * c - b * b
        return np.pi * np.sqrt(z0 * z0 / detr ** 3 * (0.25 * (c * c * av + a * a * cv) + b * b * bv) + z0v / detr)

    int_err = integration_uncertainty

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_fwhma(self, p):
        """calculate fwhm from semi-major and semi-minor axes."""
        # NOTE: fwhm ~ 2.355 sigma
        # SEE: https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        a, c = p[3], p[5]
        log2 = np.log(2)
        fwhm_a = 2 * np.sqrt(log2 / a)  # FWHM along semi-major axis
        fwhm_c = 2 * np.sqrt(log2 / c)  # FWHM along semi-minor axis
        return fwhm_a, fwhm_c

    def get_fwhm(self, p):
        # geometric mean of the FWHMa along each axis
        fwhm_a, fwhm_c = self.get_fwhma(p)
        return np.sqrt(fwhm_a * fwhm_c)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_description(self, p, offset=(0, 0)):
        """
        Get a description of the fit based on the parameters.
        Optional offset gives the xy offset of the fit.
        """
        x, y, z, a, b, c, d = p

        fwhm = self.get_fwhm(p)
        counts = self.integrate(p)
        sigx, sigy = 1 / (2 * a), 1 / (2 * c)  # standard deviation along the semimajor and semiminor axes
        ratio = min(a, c) / max(a, c)  # Ratio of minor to major axis of Gaussian kernel
        theta = 0.5 * np.arctan2(-b, a - c)  # rotation angle of the major axis in sky plane
        ellipticity = np.sqrt(1 - ratio ** 2)
        coo = x + offset[0], y + offset[1]

        pdict = {'coo': coo,
                 'flux': counts,
                 'peak': z + d,
                 'sky_mean': d,
                 'fwhm': fwhm,
                 'sigma_xy': (sigx, sigy),
                 'theta': np.degrees(theta),
                 'ratio': ratio,
                 'ellipticity': ellipticity}
        return pdict

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def coeff(self, covariance_matrix):
        """
        Return a, b, c coefficents for the form:
        z0 * np.exp(-(a*(x-x0)**2 -2*b*(x-x0)*(y-y0) + c*(y-y0)**2 )) + d
        """
        sigx2, sigy2 = np.diagonal(covariance_matrix)
        cov = covariance_matrix[0, 1]
        cor2 = cov ** 2 / (sigx2 * sigy2)

        f = 0.5 / (1 - cor2)
        a = f / sigx2
        b = f * (cor2 / cov)
        c = f / sigy2

        return a, b, c

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def correlation(self, p):
        """Pearsons correlation coefficient """
        # covm = self.covariance_matrix(p)
        (sigx2, covar), (_, sigy2) = self.covariance_matrix(p)
        return covar / np.sqrt(sigx2 * sigy2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def covariance_matrix(self, p):
        _, _, _, a, b, c, _ = p
        # rho = self.correlation(p)
        # P = np.array([[a,   -b],
        #              [-b,  c ]]) * 2
        detP = (a * c + b * b) * 4
        # inverse of precision matrix
        return (1 / detP) * np.array([[c, b],
                                      [b, a]]) * 2
        # return np.linalg.inv(P)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def precision_matrix(self, p):
        _, _, _, a, b, c, _ = p
        P = np.array([[a, -b],
                      [-b, c]]) * 2
        return P

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_sigma_xy(self, p):
        covm = self.covariance_matrix(p)
        return np.sqrt(np.diagonal(covm))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_theta(self, p):
        _, _, _, a, b, c, _ = p
        return -0.5 * np.arctan2(-2 * b, a - c)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_aperture_params(self, p):
        return np.r_[self.get_coo(p), self.get_sigma_xy(p), self.get_theta(p)]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def validate(self, p, window):
        #     p = np.asarray(p)
        #     w2 = np.divide(window, 2)
        #     nans = np.isnan(p).any()                    # nans are bad
        #     negpars = np.any(p[self._ix_not_neg] < 0)   # negative parameters ==> bad fit!
        #     badcoo = np.any(np.abs(p[:2] - w2) >= w2)      # center coordinates outside of grid - unphysical
        #     return ~(badcoo | negpars | nans)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def jacobian(self, p, X, Y):
        # """ """
        # x0, y0, z0, a, b, c, d = p
        # Xm = X-x0
        # Ym = Y-y0

        # [(2*a*Xm +
        # -(X-x0)**2 * self(p, X, Y),
        # -2*(X-x0)*(Y-y0)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # @staticmethod
        # def radial(p, r):
        #     #WARNING: does this make sense????
        #     """
        #     Numerically integrate the psf with theta around (x0,y0) to determine mean radial
        #     profile
        #     """
        #     from scipy.integrate import quad
        #     from functools import partial
        #
        #     def integrand(psi, r, alpha, beta):
        #         return np.exp( -r*r*(alpha*np.cos(psi) + beta*np.sin(psi)) )
        #
        #     def mprof(p, r):
        #         x0, y0, z0, a, b, c, d = p
        #         alpha = (a-c)/2
        #         I = quad(integrand, 0, 4*np.pi, args=(r,alpha,b))
        #         return (0.25/np.pi)*z0*np.exp( -0.5*r*r*(a+c) ) * I[0]
        #
        #     R = partial(mprof, p)
        #     return np.vectorize(R)(r)

        # def param_hint(self, data):
        # """x0, y0, z0, a, b, c, d"""
        # return self.default_params


EllipticalGaussianPSF = GaussianPSF

# ====================================================================================================
import numpy.linalg as la
from scipy.special import erf


# TODO: as class
def sym2D(diag, cross):
    """return symmetric 2D array"""
    return np.eye(2) * diag + np.eye(2)[::-1] * cross


def _to_cov_matrix(var, cov):  # correlation=None
    """compose symmetric covariance tensor"""
    return sym2D(var, cov)


def _to_precision_matrix(var, cov):
    det = np.prod(var) - cov ** 2
    return (1. / det) * sym2D(var, -cov)


def _rot_mat(theta):
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin],
                     [sin, cos]])


def discretizedGaussian(amp, mu, cov, grid):
    """Convenience method for discretized Gaussian evaluation"""
    # eigenvalue decomposition of precision matrix
    P = la.inv(cov)  # precision matrix
    evl, M = la.eig(P)

    # assert np.allclose(np.diag(evl), iM.dot(cov).dot(M))
    # check if covariance is positive definite
    if np.any(evl < 0):
        raise ValueError('Covariance matrix should be positive definite')

    # make column vector for arithmetic
    mu = np.array(mu, ndmin=grid.ndim, dtype=float).T
    evl = np.array(evl, ndmin=grid.ndim, dtype=float).T

    xm = grid - mu  # (2,...) shape

    # return M, xm

    f = np.sqrt(2 / evl)
    pf = np.sqrt(np.pi / evl)
    td0 = np.tensordot(M, xm + 0.5, 1)
    td1 = np.tensordot(M, xm - 0.5, 1)
    w = pf * (erf(f * td0) - erf(f * td1))

    return amp * np.prod(w, axis=0)


# ****************************************************************************************************
# TODO: as Fittable2Dmodel! + contribute to photutils
class DiscretizedGaussian():  # GaussianPSF??
    """Gaussian Point Response Function """

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, p, grid):
        """ """
        # unpack parameters
        amp, mux, muy, w0, w1, theta = (p[n] for n in ('amp', 'mux', 'muy', 'w0', 'w1', 'theta'))

        # Make eigenvector matrix from rotation angle
        M = _rot_mat(float(theta))

        # make column vector for arithmetic
        mu = np.array([mux, muy], ndmin=grid.ndim, dtype=float).T
        evl = np.array([w0, w1], ndmin=grid.ndim, dtype=float).T

        # centered XY-grid
        xm = grid - mu  # (2,...) shape

        f = np.sqrt(2 / evl)
        pf = np.sqrt(np.pi / evl)
        td0 = np.tensordot(M, xm + 0.5, 1)
        td1 = np.tensordot(M, xm - 0.5, 1)
        w = pf * (erf(f * td0) - erf(f * td1))
        return amp * np.prod(w, axis=0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _eig2cov(w0, w1, theta):
        M = _rot_mat(float(theta))
        w0 * M[:, 0] + w1 * M[:, 1]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def residuals(self, p, data, grid):
        return np.square(data - self(p, grid))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rss2(self, p, data, grid):
        return self.residuals(p, data, grid).sum()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO: major merger below!
    def eigenvals(self, var, cov):
        """eigenvalues of precision matrix from variance-covariance"""
        varx, vary = var
        cov2 = np.square(cov)
        detSig = np.prod(var) - cov2

        k = 1. / detSig
        sq = np.sqrt(np.square(varx - vary) + 4 * cov2)

        return 1. / (2 * detSig) * (np.sum(var) + sq * np.array([1, -1]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eigenvals2(self, var, cor):
        """eigenvalues of precision matrix from variance & correlation"""
        varx, vary = var
        hvars = 1. / varx + 1. / vary
        t = 1 - cor * cor
        # print('t', t)
        sq = np.sqrt(np.square(hvars) - 4 * t)
        # print('sq', sq)
        return 1. / (2 * t) * (hvars + sq * np.array([1, -1]))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eigenvecs(self, var, cov, eigvals):
        """eigenvectorss of precision matrix from variance, covariance, eigenvalues"""
        # cor = cov / np.prod(var)
        e0, e1 = eigvals
        varx, vary = var
        cov2 = cov * cov

        m0 = np.sqrt(np.square(vary * vary - e0) / cov2 + 1)
        return m0
        m1 = np.sqrt(1 - m0 * m0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def eigenvecs2(self, var, cor, eigvals):
        """eigenvectorss of precision matrix from variance, correlation, eigenvalues"""
        cov = cor * np.prod(var)
        e0, e1 = eigvals
        varx, vary = var
        cov2 = cov * cov

        m0 = np.sqrt(np.square(vary * vary - e0) / cov2 + 1)
        return m0
        m1 = np.sqrt(1 - m0 * m0)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rss(self, p, data, grid):
        return np.square(self(p, grid) - data)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def Gaussian(p, x):
        # """Gaussian function for fitting radial star profiles"""
        # A, b, mx = p
        # return A*np.exp(-b*(x-mx)**2 )


# ====================================================================================================
def Moffat(p, x, y):
    x0, y0, z0, a, b, c = p
    return z0 * (1 + ((x - x0) ** 2 + (y - y0) ** 2) / (a * a)) ** -b + c


# ****************************************************************************************************
class StarFit(object):
    # TODO: Kernel Density Estimation... or is this too slow??
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, psf=None, algorithm=None, caching=True, hints=True,
                 _print=False, **kw):
        """
        Parameters
        ----------
        psf             :
            An instance of the PSF class
        algorithm       :       str
            Algorithm to use for fitting
        caching         :       bool
            Whether or not to save parameters of previous fits.  This may help to speed up subsequent fitting.
        hints           :       bool
            Guess uncached parameters based on data statistics?
        """
        # if not isinstance(psf, PSF):
        # raise ValueError( 'psf must be an instance of the PSF class.' )

        if algorithm is None:
            self.algorithm = leastsq
        else:
            raise NotImplementedError

        # set the psf function
        self.psf = psf or GaussianPSF()
        self.caching = caching
        # pre-allocate cache memory
        if caching:
            self.max_cache_size = n = kw.get('max_cache_size', 25)
            m = psf.npar
            self.cache = np.ma.array(np.zeros((n, m)),
                                     mask=np.ones((n, m), bool))

        self.hints = hints
        # initialise parameter cache with psf defaults
        # self.cache      = []                  #parameter cache

        self.call_count = 0  # keeps track of how many times the class has been called
        self._print = _print

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # @profile()
    def __call__(self, grid, data):  # TODO: make grid optional...
        """Fits the PSF model to the data on the grid given the input coordinates xy0"""
        psf = self.psf

        p0 = self.get_params_from_cache()  # just returns the default cache for the psf if caching is disabled
        if self.hints:
            p0[psf.to_guess] = psf.param_hint(data)[psf.to_guess]

        # print('\n\nINIT PARAMS' )
        # print( p0, '\n\n' )

        # pyqtRemoveInputHook()
        # embed()
        # pyqtRestoreInputHook()


        # Y, X = grid
        args = data, grid

        # minimize residual sum of squares
        plsq, _, info, msg, success = self.algorithm(psf.frs, p0, args=args, full_output=1)

        if success != 1:
            if self._print:
                warn('FIT DID NOT CONVERGE!')
                print(msg)
                print(info)
            self.call_count += 1  # NOTE: This could be a simple decorator??
            return
        else:
            if self._print:
                print('\nSuccessfully fit {} function to stellar profile.'.format(psf.F.__name__))
            if self.caching:
                # update cache with these parameters
                i = self.call_count % self.max_cache_size  # wrap!
                self.cache[i] = plsq  # NOTE: implicit unmasking!
                # print( '\n\nCACHE!', self.cache )
                # print( '\n\n' )

        self.call_count += 1  # NOTE: This could be a simple decorator??
        return plsq

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params_from_cache(self):
        """return the mean value of the cached parameters.  Useful as initial guess."""
        if self.caching and self.call_count:
            return self.cache.mean(0)
        else:
            return self.psf.default_params

    get_cached = get_params_from_cache
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
