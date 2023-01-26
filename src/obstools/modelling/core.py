# from pathlib import Path


# std
import numbers
import operator as op
from collections import OrderedDict, abc, defaultdict

# third-party
import numpy as np
from scipy.optimize import leastsq, minimize

# local
from recipes.oo import coerce
from recipes.lists import tally
from recipes.io import load_memmap
from recipes.logging import LoggingMixin

# relative
from .parameters import Parameters


# import multiprocessing as mp


LN2PI_2 = np.log(2 * np.pi) / 2


def int2tup(obj):
    return coerce(obj, tuple, numbers.Integral)
    


def _echo(*_):
    return _


def _sample_stat(data, statistic, sample_size, replace=True):
    return statistic(
        np.random.choice(np.ma.compressed(data), sample_size, replace)
    )


def nd_sampler(data, statistic, sample_size, axis=None, replace=True):
    # statistics on random samples from an ndarray
    if axis is None:
        return _sample_stat(data.ravel(), statistic, sample_size, replace)

    elif isinstance(axis, numbers.Integral):
        size_each = sample_size // data.shape[axis]
        return np.apply_along_axis(
            _sample_stat, axis, data, statistic, size_each, replace)
    else:
        raise ValueError('Invalid axis')


class UnconvergedOptimization(Exception):
    pass


# def nd_sampler(data, sample_size, statistic, axis=None):
#     # sample statistics on nd data  # can probably be sped up significantly
#     ndim = data.ndim
#     indices = np.empty((ndim, sample_size), int)
#
#     # print(ndim, sample_size)
#
#     for i, s in enumerate(data.shape):
#         indices[i] = np.random.randint(0, s, sample_size)
#
#     if axis is None:
#         return statistic(data[indices])
#
#     if isinstance(axis, numbers.Integral):
#         axes = list(range(ndim))
#         axes.pop(axis)
#
#         ixo = np.argsort(indices[axis])
#         six = indices[axis][ixo]
#         randix = np.split(ixo, np.nonzero(np.diff(six))[0])
#         rsz = data.shape[axis]
#         assert (len(randix) == rsz), 'Increase sample_size'
#         result = np.empty(rsz)
#
#         for i, rix in enumerate(randix):
#             ix = list(indices[axes][:, rix])
#             ix.insert(axis, np.full(len(rix), i, int))
#             result[i] = statistic(data[tuple(ix)])
#         return result
#     else:
#         raise ValueError('Invalid axis')


# class OptionallyNamed:
#     """
#     Implements optional mutable `name` for inherited classes
#     """
#     _name = None  # optional name.
#
#     @ClassProperty  # TODO: inherit these from LoggingMixin??
#     @classmethod
#     def name(cls):
#         # Will default to `cls.__name__` if class attribute `name` not
#         # over-written by inheritors
#         return cls._name or cls.__name__
#
#     @name.setter
#     def name(self, name):
#         # note - this will set the class attribute `name` from the instance
#         self.set_name(name)
#
#     @classmethod
#     def set_name(cls, name):
#         assert isinstance(name, str)
#         cls._name = name

class OptionallyNamed:
    """
    Implements optional, mutable name for inherited classes via `name` property
    """
    _name = None  # optional name.

    def get_name(self):
        # Will default to `cls.__name__` if class attribute `name` not
        # over-written by inheritors
        return self._name or self.__class__.__name__

    @property
    def name(self):
        return self.get_name()

    @name.setter
    def name(self, name):
        if not isinstance(name, str):
            raise ValueError('name must be a string')
        self._name = name


class Likelihood:
    pass


class IID(Likelihood):
    pass


class GaussianLikelihood(Likelihood):

    # def __init__(self, model):
    #     self.k = model.n_dims

    def __call__(self, p, data, grid=None, sigma=None):
        # assume uncorrelated gaussian uncertainties on data

        # TODO: allow sigma to be shaped
        #   (1)         same uncertainty for all data points and all dimensions
        #                  ---> IID
        #   (n),        n data points, uncertainty same along each dimension
        #   (n, k),     for n data points in k dimensions
        #   (n, k, k)   for n data points full covariance matrix for each data
        #               point
        #

        sigma_term = 0 if sigma is None else np.log(sigma).sum()
        
        return (- data.size * LN2PI_2
                # # TODO: einsum here for mahalanobis distance term
                - 0.5 * self.wrss(p, data, grid, stddev)
                - sigma_term)


class PoissonLikelihood(Likelihood):
    pass


class Lp:
    'todo'


class L1:
    pass


class L2:
    pass


# rv('μ') ~ Gaussian()
# MyModel(Model, GaussianLikelihood):


def echo(*args, **kws):
    return args


def ln_prior(priors, Θ):
    s = 0
    for prior, p in zip(priors.values(), Θ):
        pr = prior.pdf(p)
        if pr != 0:  # catch for 0 value probability
            s += np.log(pr)
        else:
            return -np.inf
    return s


class Model(OptionallyNamed, LoggingMixin):
    """Base class for fittable model"""

    # TODO: have a classmethod here that can turn on and off active view
    # castings so that we can work with nested parameters more easily

    # TODO: think of a way to easily fit for variance parameter(s)

    dof = None  # sub-class should set
    # TODO determine intrinsically from p?
    base_dtype = float  # FIXME - remove this here
    sum_axis = None
    # metric = L2               # TODO metric
    # minimizer = minimize

    # exception behaviour
    raise_on_failure = False  # if minimize reports failure, should I raise?
    # warnings for nans / inf
    do_checks = True
    # masked array handling
    compress_ma = True  # only if method == 'leastsq'

    # FIXME: Models with no parameters don't really fit this framework...

    def __call__(self, p, *args, **kws):
        """
        Evaluate the model at the parameter (vector) `p`

        Parameters
        ----------
        p
        grid
        args
        kws

        Returns
        -------

        """
        p, args = self._checks(p, *args, **kws)
        return self.eval(p, *args, **kws)

    def eval(self, p, *args, **kws):
        """
        This is the main compute method, while __call__ handles additional
        variable checks etc. Subclasses can overwrite both as needed, but must
        overwrite `eval`.  Important that `eval` has same signature as
        `__call__` since they will be dynamically swapped during `fit` for
        improving optimization performance.

        Parameters
        ----------
        p
        grid

        Returns
        -------

        """
        raise NotImplementedError

    def _checks(self, p, *args, **kws):
        return self._check_params(p), args

    def _check_params(self, p):
        if len(p) != self.dof:
            raise ValueError(
                f'Parameter vector size ({len(p)}) does not match '
                f'degrees of freedom ({self.dof}) for model {self!r}'
            )
        return p

    # def _check_grid(self, grid):
    #     return grid

    def p0guess(self, data, *args, **kws):
        raise NotImplementedError

    def get_dtype(self):
        # todo: use p0guess to determine the dtype in pre_fit?? / pre_fit
        # todo: eliminate this method

        if self.dof is None:
            raise TypeError('Subclass should set attribute `dof`')

        return [(self.get_name(), self.base_dtype, self.dof)]

    def residuals(self, p, data, *args):
        """
        Difference between data (observations) and model. a.k.a. deviation
        """
        return data - self(p, *args)

    def rs(self, p, data, *args):
        """squared residuals"""
        return np.square(self.residuals(p, data, *args))

    def frs(self, p, data, *args):
        """squared residuals flattened to a vector"""
        return self.rs(p, data, *args).flatten()

    def rss(self, p, data, *args):
        """residual sum of squares"""
        return self.rs(p, data, *args).sum(self.sum_axis)

    def wrs(self, p, data, *args, stddev=None):
        """
        weighted square residuals. aka sum of squares due to error (sic)
        """
        if stddev is None:
            return self.rs(p, data, *args)
        return self.rs(p, data, *args) / stddev / stddev

    def fwrs(self, p, data, *args, stddev=None):
        """weighted squared residuals vector"""
        return self.wrs(p, data, *args,  stddev=stddev).ravel()

    def wrss(self, p, data, *args, stddev=None):
        """weighted residual sum of squares. ie. The chi squared statistic χ²"""
        return self.wrs(p, data, *args, stddev=stddev).sum(self.sum_axis)

    # FIXME: alias not inherited if overwritten in subclass
    chisq = wrss  # chi2 # chiSquare # χ2

    def redchi(self, p, data, *args, stddev=None):  # chi2r
        """Reduced chi squared statistic χ²ᵣ"""
        #  aka. mean square weighted deviation (MSWD)
        return self.chisq(p, data, *args, stddev) / (data.size - self.dof)

    mswd = reduced_chi_squared = redchi  # χ2r

    def rsq(self, p, data, *args, **kws):
        """
        The coefficient of determination, denoted R2 or r2 and pronounced
        "R squared", is the proportion of the variance in the dependent
        variable that is predictable from the independent variable(s). It
        provides a measure of how well observed outcomes are replicated by
        the  model, based on the proportion of total variation of outcomes
        explained by the model.
        """
        # total sum of squares
        mu = data.mean()
        # fixme shouldn't recompute during fit
        tss = np.square(data - mu).sum()
        rss = self.rss(p, data, *args, **kws)
        return 1 - rss / tss

    # FIXME: alias not inherited if overwritten in subclass
    coefficient_of_determination = rsq

    def llh(self, p, data, *args, stddev=None):
        # assuming uncorrelated gaussian noise on data here
        # https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Continuous_distribution,_continuous_parameter_space

        # NOTE: optimizing with this objective is theoretically equivalent to
        #  least-squares

        # FIXME: more general metric here?
        nl = data.size * LN2PI_2 + 0.5 * self.wrss(
            p, data, *args, stddev=stddev)
        if stddev is not None:
            nl += np.log(stddev).sum()

        return -nl

    # def score(self, data, *args, **kws):
    # The score is the gradient (the vector of partial derivatives) of log⁡ L(θ)

    # FIXME: alias not inherited if overwritten in subclass
    logLikelihood = ln_likelihood = log_likelihood = llh

    def loss_mle(self, p, data, *args, **kws):
        """Objective for Maximum Likelihood Estimation"""
        # NOTE: will be a bit more efficient to skip adding the sigma term
        # return (data.size * LN2PI_2
        #         + 0.5 * self.wrss(p, data, *args, stddev=stddev, **kws))

        return -self.llh(p, data, *args,  **kws)

    def ln_posterior(self, p, data, *args, priors=None, prior_args=(), **kws):
        """
        Logarithm of posterior probability (up to a constant).

            logP = ln(Likelihood x prior)

        Parameters
        ----------
        p
        data
        grid
        stddev

        Returns
        -------

        """

        if priors:  # TODO: maybe manage through property ??
            log_prior = ln_prior(priors, p)  # logPrior(p, *prior_args)

            if not np.isfinite(log_prior):
                return -np.inf

            return self.llh(p, data, *args, **kws) + log_prior

        return self.llh(p, data, *args, **kws)

    def aic(self, p, data, *args, **kws):
        """
        Akaike information criterion. Assumes `p` is the parameter vector
        corresponding to the maximum likelihood.
        """
        k = len(p) + 2
        return 2 * (k - self.llh(p, data, *args, **kws))

    def aicc(self, p, data, *args, **kws):
        # "When the sample size is small, there is a substantial probability
        # that AIC will select models that have too many parameters...
        # AICc is essentially AIC with an extra penalty term for the number of
        #  parameters"
        #
        # Assuming that the model is univariate, is linear in its parameters,
        # and has normally-distributed residuals (conditional upon regressors),
        # then the formula for AICc is as follows.
        k = len(p)
        n = data.size
        return 2 * (k + (k**2 + k) / (n - k - 1) - self.llh(p, data, *args, **kws))
        # "If the assumption that the model is univariate and linear with normal
        # residuals does not hold, then the formula for AICc will generally be
        # different from the formula above. For some models, the precise formula
        # can be difficult to determine. For every model that has AICc
        # available, though, the formula for AICc is given by AIC plus terms
        # that includes both k and k2. In comparison, the formula for AIC
        # includes k but not k2. In other words, AIC is a first-order estimate
        # (of the information loss), whereas AICc is a second-order estimate."
        # -- from: https://en.wikipedia.org/wiki/Akaike_information_criterion

    def bic(self, p, data, *args, **kws):
        n = data.size
        k = len(p)
        return k * np.log(n) - 2 * self.llh(p, data, *args, **kws)

    def mle(self, data, p0=None, *args, **kws):
        """
        Maximum likelihood fit
        """
        return self.fit(data, p0, *args, loss=self.loss_mle, **kws)

    def fit(self, data, p0=None, *args, loss=None, **kws):
        """
        Minimize `loss` for `data` with uncertainties `stddev` on `grid` using
        `scipy.minimize` routine

        Parameters
        ----------
        p0
        data
        grid
        stddev
        args
        kws

        Returns
        -------

        """

        # set default loss / objective
        loss = loss or self.wrss  # fwrs for leastsq...

        # post-processing args
        post_args = kws.pop('post_args', ())
        post_kws = kws.pop('post_kws', {})

        # pre-processing
        p0, data, args, _ = self.pre_fit(loss, p0, data, *args, **kws)

        # TODO: move to HandleParameters Mixin??
        type_, dtype = type(p0), p0.dtype

        tmp = self._checks
        self._checks = _echo
        try:
            # minimization
            p = self._fit(loss, p0, data, *args, **kws)
        except Exception as err:
            raise err from None
        finally:
            # restore `_checks` before raising
            self._checks = tmp
            # del self.__call__  # deletes temp attribute on instance.

        # post-processing
        p = self.post_fit(p, *post_args, **post_kws)

        # TODO: move to HandleParameters Mixin??
        if p is not None:
            # get back structured view if required
            # for some strange reason `result.x.view(dtype, type_)` ends up
            # with higher dimensionality. weird. # TODO BUG REPORT??
            return p.view(dtype, type_)

    def pre_fit(self, loss, p0, data, *args, **kws):
        """This will be run prior to the minimization routine"""

        # Parameter checks
        # ----------------
        if p0 is None:
            p0 = self.p0guess(data, *args)
            self.logger.debug('p0 guess: {:s}', p0)
        else:
            p0 = np.asanyarray(p0)

        # nested parameters: flatten prior to minimize, re-structure post-fit
        # TODO: move to HandleParameters Mixin??
        if isinstance(p0, Parameters):
            p0 = p0.flattened
        else:
            # need to convert to float since we are using p0 dtype to type
            # cast the results vector for structured parameters and user might
            # have passed an array-like of integers
            p0 = p0.astype(float)

        # check that call works.  This check here so that we can identify
        # potential problems with the function call / arguments before entering
        # the optimization routine. Any potential errors that occur here will
        # yield a more readable traceback.  This is also done here so we can
        # check initially and then let the fitting run through `eval` instead of
        # `__call__` and skip all the checks during optimization for performance
        # gain.
        p0, args = self._checks(p0, *args, **kws)
        # loss(p0, data, *args, **kws)  # THIS IS ERROR PRONE!

        # Data checks
        # -----------
        # check for nan / inf
        if self.do_checks:
            #  this may be slow for large arrays.
            if np.isnan(data).any():
                self.logger.warning('Your data contains nans.')
            # check for inf
            if np.isinf(data).any():
                self.logger.warning('Your data contains non-finite elements.')

            # # check for masked data
            # if np.ma.is_masked(data):
            #     self.logger.warning('Your data has masked elements.')

        # Remove masked data
        if np.ma.is_masked(data):
            use = ~data.mask
            # grid = grid[..., use]  # FIXME: may still be None at this point
            # if stddev is not None:
            #     stddev = stddev[use]
            data = data.compressed()

        return p0, data, args, kws

    def _fit(self, loss, p0, data, *args, **kws):
        """Minimization worker"""

        # infix so we can easily wrap least squares optimization within the same
        # method
        args = (data, ) + args
        if kws.get('method') == 'leastsq':
            kws.pop('method')
            #
            p, cov_p, info, msg, flag = leastsq(self.fwrs, p0, args,
                                                full_output=True, **kws)
            success = (flag in [1, 2, 3, 4])
            # print('RESULT:', p)
        else:
            # minimization
            result = minimize(loss, p0, args, **kws)

            p = result.x
            success = result.success
            msg = result.message

        if success:
            if unchanged := np.allclose(p, p0):
                self.logger.warning('"Converged" parameter vector is '
                                    'identical to initial guess: {}', p0)
                msg = ''
            else:
                self.logger.debug('Successful fit {:s}', self.name)
                # TODO: self.logger.verbose() npars, niters, gof,
                #  ndata, etc
                return p

        # generate message for convergence failure
        from recipes import pprint

        # objective_repr =
        fail_msg = (f'{self.__class__.__name__} optimization with objective '
                    f'{pprint.caller(loss)!r} '
                    f'failed to converge: {msg}')

        # bork if needed
        if self.raise_on_failure:
            raise UnconvergedOptimization(fail_msg)

        # else emit warning for non-convergence
        self.logger.warning(fail_msg)

    def post_fit(self, p, *args, **kws):
        return p

    def run_mcmc(self, data, args, nsamples, nburn, nwalkers=None, threads=None,
                 p0=None, priors=None):
        """
        Draw posterior samples
        """

        # TODO: would be nice to have some kind of progress indicator here
        # could do this by wrapping the llh function with counter

        import emcee

        if nwalkers is None:
            nwalkers_per_dof = 4
            nwalkers = self.dof * nwalkers_per_dof
        nwalkers = int(nwalkers)

        if threads is None:
            import multiprocessing as mp
            threads = mp.cpu_count()
        threads = int(threads)

        # create sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, self.dof, self.ln_posterior,
            args=(data,) + args, kwargs=dict(priors=priors),
            threads=threads)

        # randomized initial guesses for parameters
        if p0 is None:
            if priors:
                # draw initial state values from priors
                # NOTE: order important!
                p0 = np.array([prior.rvs(nwalkers)
                               for prior in priors.values()]).T
            else:
                raise ValueError('Need either p0 or priors to be provided.')
        else:
            p0 = np.array(p0)
            if p0.shape != (nwalkers, self.dof):
                raise ValueError(
                    f'Please ensure p0 is of dimension ({nwalkers}, {self.dof})'
                    f' for sampler with {nwalkers} walkers and model with '
                    f'{self.dof} degrees of freedom.'
                )

        # burn in
        if nburn:
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()

        # sample posterior
        pos, prob, state = sampler.run_mcmc(p0, nsamples)  # should
        return sampler

    # TODO: Mixin class here??  SharedMemoryMixin
    def _init_mem(self, loc, shape, fill=np.nan, clobber=False):
        """Initialize shared memory for this model"""
        # final array shape is external `shape` + parameter shape `dof`
        if isinstance(shape, int):
            shape = shape,

        if isinstance(self.dof, int):
            shape += (self.dof,)

        # locPar = '%s.par' % self.name  # locStd = '%s.std' % self.name
        dtype = self.get_dtype()
        return load_memmap(loc, shape, dtype, fill, clobber)


class DataTransformBase(LoggingMixin):
    """
    Some models behave better with input data that have reasonably sane
    values. This mixin implements an affine transform to the data prior to
    fitting, and subsequently scaled the resultant parameters back to the
    original coordinates.  This acts like a quick and dirty re-parameterization.
    """

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, p):
        raise NotImplementedError

    def pre_process(self, p0, data, grid, stddev=None, *args, **kws):
        data = self.transform(data)
        return p0, data, grid, stddev

    def post_process(self, p, **kws):
        if p is not None:
            return self.inverse_transform(p, **kws)


class RescaleInternal(DataTransformBase):
    _yscale = None

    def get_scale(self, data, sample_size=100, axis=None):
        # Grab `sample size` elements randomly from the data array and
        # calculate the sample median.
        return nd_sampler(data, np.median, sample_size, axis)

    def transform(self, data):
        if self._yscale is None:
            self._yscale = self.get_scale(data)

        self.logger.debug('scale is {:s}', self._yscale)
        return data / self._yscale

    def inverse_transform(self, p, **kws):
        return p * self._yscale


class SummaryStatsMixin:
    """
    Mixin class that computes a summary statistic across one (or more) of
    the data axes before doing the fit.
    """

    center_func = np.ma.median  # for estimating central tendency of data
    disp_func = None  # for measuring dispersion of data

    def __init__(self, axis, ndim=2):
        self.axis = int(axis)
        self.ndim = int(ndim)
        axes = list(range(ndim))
        axes.pop(axis)
        self.other_axes = tuple(axes)

    def fit(self, data, grid, stddev=None, p0=None, **kws):
        y = self.center_func(data, self.other_axes)
        if self.disp_func:
            stddev = self.disp_func(data, self.other_axes)
        return super().fit(y, grid, stddev, p0, **kws)


# class Record(AttrReadItem, ListLike):
#     pass


def make_unique_names(names):
    # check for duplicate names
    unames = set(names)
    if len(unames) != len(names):
        # models have duplicate names
        new_names = []
        for name, indices in tally(names).items():
            fmt = '%s{:d}' % name
            new_names.extend(map(fmt.format, range(len(indices))))
        names = new_names
    return names


class ModelContainer(OrderedDict, LoggingMixin):
    """
    dict-like container for models
    """

    def __init__(self, models=(), **kws):
        """
        Create model container from sequence of models and or keyword,
        model pairs. Model names will be made a unique set by appending
        integers.
        """

        self._names = None

        mapping = ()
        if isinstance(models, abc.MutableMapping):
            mapping = models
        elif len(models):
            # ensure we have named models
            names = [getattr(m, 'name', None) for m in models]
            if None in names:
                raise ValueError('All models passed to container must be '
                                 'named. You can (re)name them implicitly by '
                                 'initializing %r via keyword arguments: eg:'
                                 '`%s(bg=model)`' %
                                 self.__class__.__name__)
            # ensure names are unique
            new_names = make_unique_names(names)
            mapping = zip(new_names, models)

        # load models into dict
        super().__init__(mapping)
        self.update(**kws)

        # note init with kws can mean we may loose order in python < 3.6

    def __setitem__(self, key, model):

        # HACK autoreload
        # if not isinstance(model, Model):
        #     raise ValueError('Components models must inherit from `Model`')
        # HACK autoreload

        # make sure the model has the same name that it is keyed on in the dict
        # if key != model.name:
        #     # set the model name (avoid duplicate names in dtype for models
        #     # of the same class)
        #     model.name = key

        return super().__setitem__(key, model)

    def __iter__(self):
        """Iterate over the models *not* the keys"""
        return iter(self.values())

    @property
    def names(self):
        """unique model names"""
        return self.attr_getter('name')

    def attr_getter(self, *attrs):
        getter = op.attrgetter(*attrs)
        return list(map(getter, self.values()))

    def invert(self, keys=all, one2one=False):
        """
        Mapping from models to list of keys.  Useful helper for constructing
        dtypes for compound models.  The default is to return a mapping from
        unique model to the list of corresponding key.  In the inverse
        mapping, models that apply to more than one key value will therefore
        map to a list that has multiple elements. A one-to-one inverted
        mapping can be obtained by passing `one2one=True`.

        Parameters
        ----------
        keys: sequence of keys, default all
            The list of keys to include in the inverted mapping
        one2one: bool, default False
            Whether the inverse mapping should be one to one

        Returns
        -------
        dict keyed on models containing lists of the keys (labels)
        corresponding to that model.
        """
        if keys is all:
            keys = self.keys()
        else:
            keys_valid = set(self.keys())
            keys_invalid = set(keys) - keys_valid  # maybe print these ???
            keys = set(keys) - keys_invalid

        if one2one:
            return dict(zip(map(self.get, keys), keys))

        inverse = defaultdict(list)
        for key in keys:
            inverse[self[key]].append(key)
        return inverse

    # def unique_names(self, default_name='model'):
    #     """
    #     Get mapping from labels to a set of unique model names.  Names are
    #     taken from component models where possible, substituting
    #     `default_name` for unnamed models. Numbers are underscore appended to
    #     ensure uniqueness of names. Useful for nested parameter construction.
    #
    #     Returns
    #     -------
    #     names: dict
    #         model names keyed on segment labels
    #     """
    #     assert isinstance(default_name, str)
    #     #
    #     names = [getattr(m, 'name', None) or default_name
    #              for m in self.values()]
    #
    #     # check for duplicate names
    #     return make_unique_names(names)
    #
    # def rename_models(self, names):
    #     for model, name in zip(self.values(), names):
    #         model.name = name


class CompoundModel(Model):

    def __init__(self, *args, **kws):
        self._models = ModelContainer(*args, **kws)

    def eval(self, p, grid):
        raise NotImplementedError

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, models):
        self.set_models(models)

    def set_models(self, models):
        self._models = ModelContainer(models)

    def add_model(self, model, keys):
        """
        Add a model to the compound model. If keys is array-like, this model
        will be added once for each key

        Parameters
        ----------
        model: Model instance

        keys: {str, int, array-like}
            key (or keys if array-like) for which this model will be used

        Returns
        -------

        """
        if not isinstance(keys, (list, tuple, np.ndarray)):
            keys = keys,

        # check if any models will be clobbered ?
        for key in keys:
            self.models[key] = model
            # one model may be used for many labels

    @property
    def dofs(self):
        """Number of free parameters for each of the component models"""
        return self.models.attr_getter('dof')

    @property
    def dof(self):
        """Total number of free parameters considering all constituent models"""
        return sum(self.dofs)

    # @property
    # def dtype(self):
    #     if self._dtype in None:
    #         self._dtype = self.get_dtype()
    #     else:
    #         return self._dtype

    def get_dtype(self, keys=all):
        """
        Build the structured np.dtype object for a particular model or
        set of models. default is to use the full set of models

        Parameters
        ----------
        keys: list of keys

        Returns
        -------

        """
        if keys is all:
            keys = self.models.keys()

        return [self._adapt_dtype(self.models[key], ()) for key in keys]

    def _adapt_dtype(self, model, out_shape):
        # adapt the dtype of a component model so that it can be used with
        # other dtypes in a (possibly nested) structured dtype. `out_shape`
        # allows for results (optimized parameter values) of models that are
        # used for more than one key (label) to be represented by a 2D array.

        # make sure size in a tuple
        out_shape = () if out_shape == 1 else int2tup(out_shape)
        dt = model.get_dtype()
        if len(dt) != 1: # simple model
            # structured dtype - nest!
            return model.name, dt, out_shape
        
        name, base, dof = dt[0]
        dof = int2tup(dof)
        # extend shape of dtype
        return model.name, base, out_shape + dof

    def _results_container(self, keys=all, dtype=None, fill=np.nan,
                           shape=(), type_=Parameters):
        """
        Create a container of class `type_` for the result of an optimization
        run on models associated with `keys`

        Parameters
        ----------
        keys
        dtype
        fill
        shape
        type_

        Returns
        -------

        """

        # build model dtype
        if dtype is None:
            # noinspection PyTypeChecker
            dtype = self.get_dtype(keys)

        # create array
        out = np.full(shape, fill, dtype)
        return type_(out)

    # def p0guess(self, data, grid=None, stddev=None, **kws):
    #     #
    #     if kws.get('nested', True):
    #         return Parameters({name: mdl.p0guess(data, grid, stddev, **kws)
    #                            for name, mdl in self.models.items()})
    #     else:
    #         return np.hstack([mdl.p0guess(data, grid, stddev, **kws)
    #                           for mdl in self.models])

    # def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
    #     #
    #     # if p0 is None:
    #     #     p0 = self.p0guess(data, grid, stddev, **kws)
    #
    #     # loop through models to fit
    #     p = np.empty_like(p0)
    #     for key, mdl in self.models.items():
    #         p[key] = mdl.fit(p0[key], *args, **kws)
    #
    #     return p

    def fit(self, data, grid=None, stddev=None, **kws):
        """
        Fit frame data by looping over segments and models

        Parameters
        ----------
        data
        stddev

        Returns
        -------

        """
        return self.fit_sequential(data, stddev, **kws)

    def fit_sequential(self, data, grid=None, stddev=None, keys=None,
                       reduce=False, **kws):
        """
        Fit data in the segments with labels.
        """

        full_output = kws.pop('full_output', False)
        # full results container is returned with nans where model (component)
        # was not fit / did not converged
        p0 = kws.pop('p0', None)

        if keys is None:
            keys = list(self.models.keys())

        # output
        results = self._results_container(None if full_output else keys)
        residuals = np.ma.getdata(data).copy() if reduce else None

        # optimize
        self.fit_worker(data, grid, stddev, keys, p0,
                        results, residuals, **kws)

        if reduce:
            return results, residuals

        return results

    def fit_worker(self, data, grid, stddev, keys, p0, result, residuals,
                   **kws):

        # todo: check keys against self.models.keys()

        # iterator for data segments
        itr_subs = self.iter_region_data(keys, data, grid, stddev, masked=True)

        reduce = residuals is not None
        for key, (reg, (sub, grd, std)) in zip(keys, itr_subs):
            model = self.models[key]

            # skip models with 0 free parameters
            if model.dof == 0:
                continue

            if p0 is not None:
                kws['p0'] = p0[model.name]

            # select data -- does the same job as `SegmentedImage.cutouts`
            # sub = np.ma.array(data[seg])
            # sub[..., self.seg.masks[label]] = np.ma.masked
            # std = None if (stddev is None) else stddev[..., slice_]

            # get coordinate grid
            # minimize
            # kws['jac'] = model.jacobian_wrss
            # kws['hess'] = model.hessian_wrss

            # note intentionally using internal grid for model since it may
            #  be transformed to numerically stable regime
            grd = getattr(model, 'grid', grd)
            r = model.fit(sub, grd, std, **kws)

            # if model.__class__.__name__ == 'MedianEstimator':
            #     from IPython import embed
            #     embed()

            if r is None:
                # TODO: optionally raise here based on cls.raise_on_failure
                #  can do this by catching above and raising from.
                #  raise_on_failure can also be function / Exception ??
                msg = (f'{self.models[key]!r} fit to model {key} '
                       f'failed to converge.')
                med = np.ma.median(sub)
                if np.abs(med - 1) > 0.3:
                    # TODO: remove this
                    msg += '\nMaybe try median rescale? data median is %f' % med
                raise UnconvergedOptimization(msg)
            else:
                # print(label, model.name, i)
                result[model.name] = r.squeeze()

                if reduce:
                    # resi = model.residuals(r, np.ma.getdata(sub), grid)
                    # print('reduce', residuals.shape, slice_, resi.shape)

                    residuals[reg] = model.residuals(r, np.ma.getdata(sub))

    def iter_region_data(self, keys, *data):
        """
        Iterator that yields (region, data) for each component model in keys
        """
        raise NotImplementedError


# class CompoundSequentialFitter(CompoundModel):
# FIXME: not always appropriate

# class SeqMod(CompoundModel):
#     def get_dtype(self):
#         return np.dtype([m.get_dtype() for m in self.models])
#
#     def p0guess(self, data, grid=None, stddev=None):
#         p0 = np.empty(1, self.get_dtype())
#         for mdl in enumerate(self.models):
#             p0[mdl.name] = mdl.p0guess(data, grid, stddev)
#         return p0


# TODO: SimultaneousCompound, SequentialCompound ??

# class _2dIndexEmulator():
#     def __getitem__(self, key):


class FixedGrid:
    """
    Mixin class that allows optional static grid to be set.  This makes the
    `grid` argument an optional is the model evaluation call and checks for
    the presence of the `grid` attribute which it falls back to if available.
    This allows for some convenience when fitting the same model
    repeatedly on the same grid for different data (of the same shape).
    Subclasses must implement the `set_grid` method will be used to set the
    default grid.

    It's important that this class comes before `Model` in order of
    inheritance so that the `__call__` and `fit` methods intercept that of
    `Model`. eg.:

    class MyModel(FixedGrid, Model):
        'whatever'
    """
    # based on the shape of the data upon the first call to `residuals`.

    # default value for static grid. Having this as a class variable avoids
    # having to initialize this class explicitly in inheritors
    _grid = None

    def __call__(self, p, grid=None, *args, **kws):
        # this just for allowing call without grid
        return super().__call__(p, grid, *args, **kws)

    def _checks(self, p, grid, *args, **kws):
        return self._check_params(p), self._check_grid(grid)

    # def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
    #     grid = self._check_grid(grid)  # set default grid before fit
    #     return super().fit(data, grid, stddev, p0, *args, **kws)

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self.set_grid(grid)

    def set_grid(self, grid):
        # TODO: probably check that it's an array etc
        self._grid = grid

    def _check_grid(self, grid):
        if grid is None:
            grid = self.grid
        if grid is None:
            raise ValueError(
                'Please specify the coordinate grid for evaluation, '
                'or use the `set_grid` method prior to the first call to '
                'assign a coordinate grid.')
        return grid

    # def residuals(self, p, data, grid=None):
    #     if grid is None and self.static_grid is None:
    #         self.set_grid(data)
    #         grid = self.static_grid  # TODO: emit an info message!!
    #
    #     grid = self._check_grid(grid)
    #     # can you set the method as super method here dynamically for speed?
    #     return super().residuals(p, data, grid)
