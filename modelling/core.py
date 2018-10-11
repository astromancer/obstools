# from pathlib import Path
import numbers
# import json
import operator
from collections import OrderedDict as odict

import numpy as np
from IPython import embed
from scipy.optimize import minimize

# from recipes.oop import ClassProperty
from recipes.logging import LoggingMixin
from recipes.dict import AttrReadItem, ListLike
from recipes.list import tally

from .utils import make_shared_mem, int2tup
from .parameters import Parameters


def _sample_stat(data, statistic, sample_size, replace=True):
    return statistic(np.random.choice(data, sample_size, replace))


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


# class OptionallyNamed(object):
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

class OptionallyNamed(object):
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


class Model(OptionallyNamed, LoggingMixin):
    """Base class for fittable model"""

    # TODO: have a classmethod here that can turn on and off active view
    # castings so that we can work with nested parameters more easily

    # TODO: think of a way to easily fit for variance parameter(s)

    dof = None  # sub-class should set  # todo determine intrinsically from p?
    base_dtype = float              # FIXME - remove this here
    objective = None

    # minimizer = minimize

    def __call__(self, p, grid=None):
        raise NotImplementedError

    def _check_params(self, p):
        if len(p) != self.dof:
            raise ValueError('Parameter vector size (%i) does not match '
                             'degrees of freedom (%i) for model %r' %
                             (len(p), self.dof, self))

    def p0guess(self, data, grid=None, stddev=None):
        raise NotImplementedError

    def get_name(self):
        # ensure lower case names
        return super().get_name().lower()

    def get_dtype(self):
        # todo: use p0guess to determine the dtype in pre_process?? / pre_fit
        # todo: eliminate this method

        if self.dof is None:
            raise TypeError('Subclass should set attribute `dof`')

        return [(self.get_name(), self.base_dtype, self.dof)]

    def residuals(self, p, data, grid=None):
        """Difference between data (observations) and model. a.k.a. deviation"""
        return data - self(p, grid)

    def rs(self, p, data, grid=None):
        """squared residuals"""
        return np.square(self.residuals(p, data, grid))

    def frs(self, p, data, grid=None):
        """squared residuals flattened"""
        return self.rs(p, data, grid).flatten()

    def rss(self, p, data, grid=None):
        """residual sum of squares"""
        return self.rs(p, data, grid).sum()

    def wrs(self, p, data, grid=None, stddev=None):
        """weighted squared residuals"""
        # aka  sum of squares due to error
        if stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / stddev

    def fwrs(self, p, data, grid=None, stddev=None):
        """flattened weighted squared residuals"""
        return self.wrs(p, data, grid, stddev).flatten()

    def wrss(self, p, data, grid=None, stddev=None):
        """weighted residual sum of squares. ie. The chi squared statistic χ²"""
        return self.wrs(p, data, grid, stddev).sum()

    # FIXME: alias not inherited if overwritten in subclass
    chisq = wrss  # chi2

    def redchi(self, p, data, grid=None, stddev=None):  # chi2r
        """Reduced chi squared statistic χ²ᵣ"""
        #  aka. mean square weighted deviation (MSWD)
        chi2 = self.chisq(p, data, grid, stddev)
        dof = data.size - len(p)
        return chi2 / dof

    def rsq(self, p, data, grid=None, stddev=None):
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
        tss = np.square(data - mu).sum()  # fixme shouldn't recompute during fit
        rss = self.rss(p, data, grid)
        return 1 - rss / tss

    coefficient_of_determination = rsq

    # FIXME: alias not inherited if overwritten in subclass

    def aic(self, p, data, grid, stddev):
        """
        Akaike information criterion. Assumes `p` is the parameter vector
        corresponding to the maximum likelihood.
        """
        k = len(p) + 2
        return 2 * (k - self.logLikelihood(p, data, grid, stddev))

    def aicc(self, p, data, grid, stddev):
        # "When the sample size is small, there is a substantial probability
        # that AIC will select models that have too many parameters...
        # AICc is essentially AIC with an extra penalty term for the number of
        #  parameters"
        #  - from:  https://en.wikipedia.org/wiki/Akaike_information_criterion
        # Assuming that the model is univariate, is linear in its parameters, and has normally-distributed residuals (conditional upon regressors), then the formula for AICc is as follows.
        k = len(p)
        n = data.size
        return 2 * (k + (k * k + k) / (n - k - 1) -
                    self.logLikelihood(p, data, grid, stddev))
        # If the assumption that the model is univariate and linear with normal residuals does not hold, then the formula for AICc will generally be different from the formula above. For some models, the precise formula can be difficult to determine. For every model that has AICc available, though, the formula for AICc is given by AIC plus terms that includes both k and k2. In comparison, the formula for AIC includes k but not k2. In other words, AIC is a first-order estimate (of the information loss), whereas AICc is a second-order estimate.

    def bic(self, p, data, grid, stddev):
        n = data.size
        k = len(p)
        return k * np.log(n) - 2 * self.logLikelihood(p, data, grid, stddev)

    def logLikelihood(self, p, data, grid=None, stddev=None):
        # assuming uncorrelated gaussian data here
        if stddev is None:
            sigma_term = 0
        else:
            sigma_term = np.log(stddev).sum()

        n = data.size
        return -n / 2 * np.log(2 * np.pi) - sigma_term \
               - 0.5 * self.wrss(p, data, grid, stddev)

    def objective_mle(self, p, data, grid=None, stddev=None):
        return -self.logLikelihood(p, data, grid, stddev)

    def logProb(self, p, data, grid=None, stddev=None, logPrior=None,
                prior_args=()):
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

        if logPrior:
            log_prior = logPrior(p, *prior_args)

            if not np.isfinite(log_prior):
                return -np.inf

            return self.logLikelihood(p, data, grid, stddev) + log_prior

        return self.logLikelihood(p, data, grid, stddev)

    def pre_process(self, p0, data, grid=None, stddev=None, *args, **kws):
        """This will be run prior to the minimization routine"""
        return p0, data, grid, stddev

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        """
        Minimize `objective` using `scipy.minimize`

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

        if self.objective is None:
            self.objective = self.wrss  # wrs for leastsq...

        if p0 is None:
            p0 = self.p0guess(data, grid)
            self.logger.debug('p0 guess: %s', p0)

        # nested parameters: flatten prior to minimize, re-structure post-fit
        # TODO: move to HandleParameters Mixin??
        type_, dtype = type(p0), p0.dtype
        if isinstance(p0, Parameters):
            p0 = p0.flattened

        # check for nans
        if np.isnan(data).any():
            self.logger.warning('Your data contains nans.')

        # # check for masked data
        # if np.ma.is_masked(data):
        #     self.logger.warning('Your data has masked elements.')

        # minimization
        p = self._fit(p0, data, grid, stddev=None, *args, **kws)

        if p is not None:
            # get back structured view if required
            return p.view(dtype, type_)  # .squeeze()
            # for some strange reason `result.x.view(dtype, type_)` ends up
            # with higher dimensionality. weird. #TODO BUG REPORT??

    def _fit(self, p0, data, grid, stddev=None, *args, **kws):
        """Minimization worker"""

        # pre-processing
        pre_args = kws.pop('pre_args', ())
        pre_kws = kws.pop('pre_kws', {})
        p0, data, grid, stddev = self.pre_process(p0, data, grid, stddev,
                                                  *pre_args, **pre_kws)

        # post-processing args
        post_args = kws.pop('post_args', ())
        post_kws = kws.pop('post_kws', {})

        # minimization
        result = minimize(self.objective, p0, (data, grid, stddev), *args,
                          **kws)

        if result.success:
            samesame = np.allclose(result.x, p0)
            if samesame:
                self.logger.warning('"Converged" parameter vector is identical '
                                    'to initial guess: %s', p0)
            # TODO: maybe also warn if any close ?
            else:
                self.logger.debug('Successful fit %s', self.name)
                # post-processing
                return self.post_process(result.x, *post_args, **post_kws)

        self.logger.warning('Fit did not converge! %s', result.message)

        # if not self._unconverged_None:
        #      return result.x

    def post_process(self, p, *args, **kws):
        return p

    def run_mcmc(self, data, nsamples, nburn, nwalkers=None, threads=None,
                 p0=None):
        """
        Draw posterior samples
        """

        # TODO: would be nice to have some kind of progress indicator here

        import emcee

        dof = self.dof
        if nwalkers is None:
            nwalkers_per_dof = 4
            nwalkers = dof * nwalkers_per_dof
        nwalkers = int(nwalkers)

        if threads is None:
            import multiprocessing as mp
            threads = mp.cpu_count()
        threads = int(threads)

        # create sampler
        sampler = emcee.EnsembleSampler(nwalkers, dof, self.logLikelihood,
                                        args=(data,), threads=threads)

        # randomized initial guesses for parameters
        if p0 is None:
            p0 = np.random.rand(nwalkers, dof)
            # FIXME: this should consider Priors
        else:
            raise NotImplementedError  # TODO

        # burn in
        if nburn:
            pos, prob, state = sampler.run_mcmc(p0, nburn)
            sampler.reset()

        # sample posterior
        pos, prob, state = sampler.run_mcmc(p0, nsamples)  # should
        return sampler

    # TODO: Mixin class here??  ConcurrentResultsContainer / SharedMemory
    def _init_mem(self, loc, shape, fill=np.nan, clobber=False):
        """Initialize shared memory for this model"""
        dof = self.dof
        # final array shape is external `shape` + parameter shape `dof`
        if isinstance(shape, int):
            shape = shape,

        if isinstance(self.dof, int):
            shape += (dof,)

        # locPar = '%s.par' % self.name  # locStd = '%s.std' % self.name
        dtype = self.get_dtype()
        return make_shared_mem(loc, shape, dtype, fill, clobber)


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

        self.logger.info('scale is %s', self._yscale)
        return data / self._yscale

    def inverse_transform(self, p, **kws):
        return p * self._yscale


class SummaryStatsMixin(object):
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


class CompoundModel(AttrReadItem, ListLike, Model):
    # base dict-like container for models

    def __init__(self, models=(), **kws):  #
        """
        Create model container from sequence of models and or keyword,
        model pairs. Model names will be made a unique set by appending
        integers.


        Parameters
        ----------
        models
        kws
        """

        # TODO: check if models is mapping!!!

        mapping = ()
        if len(models):
            models = tuple(filter(None, models))
            names = [m.name for m in models]
            if None in names:
                raise ValueError('All models passed to container must be '
                                 'named. You can name them implicitly by '
                                 'initializing %s via keyword arguments: '
                                 '`%s(bg=`' %
                                 self.__class__.__name__)
            # check for duplicate names
            unames = set(names)
            if len(unames) != len(names):
                # models have duplicate names
                self.logger.info('Renaming %i models', len(unames))
                new_names = []
                for name, indices in tally(names).items():
                    for i in range(len(indices)):
                        new_names.append('%s_%i' % (name, i))
                names = new_names
            #
            mapping = zip(names, models)

        # load models into dict
        super().__init__(mapping)
        self.update(**kws)

        # note init with kws can mean we loose order in python < 3.6

    # def __repr__(self):
    #     return object.__repr__(self)

    def __setitem__(self, key, value):
        # make sure the model has the same name that it is keyed on in the dict
        if key != value.name:
            # set the model name (avoid duplicate names in dtype for models
            # of the same class)
            value.name = key

        return super().__setitem__(key, value)

    @property
    def models(self):
        return self.values()

    @property
    def nmodels(self):
        return len(self)

    @property
    def names(self):
        return tuple(self.attrgetter('name'))

    @property
    def dofs(self):
        return self.attrgetter('dof')

    @property
    def dof(self):
        """Total number of free parameters considering all constituent models"""
        return sum(self.dofs)

    def attrgetter(self, *attrs):
        getter = operator.attrgetter(*attrs)
        return list(map(getter, self.values()))

    def p0guess(self, data, grid=None, stddev=None, **kws):
        #
        if kws.get('nested'):
            return Parameters({name: mdl.p0guess(data, grid, stddev, **kws)
                               for name, mdl in self.items()})
        else:
            return np.hstack([mdl.p0guess(data, grid, stddev, **kws)
                              for mdl in self.models])

    def fit(self, data, grid=None, stddev=None, p0=None, *args, **kws):
        #
        if p0 is None:
            p0 = self.p0guess(*args)

        # loop through models to fit
        p = np.empty_like(p0)
        for name, mdl in self.items():
            p[name] = mdl.fit(p0[name], *args, **kws)

        return p

    def get_dtype(self, models=None):
        # build the structured np.dtype object for a particular model or
        # group of models. default is to use the full set of models and all
        # groups
        if models is None:
            models = self.models

        dtype = []
        for i, mdl in enumerate(models):
            dt = self._adapt_dtype(mdl, 1)
            dtype.append(dt)
        return dtype

    def _adapt_dtype(self, model, out_shape):
        # adapt the dtype of a component model so that it can be used with
        # other dtypes in a structured dtype
        # make sure size in a tuple
        out_shape = int2tup(out_shape)
        dt = model.get_dtype()
        if len(dt) == 1:  # simple model
            name, base, dof = dt[0]
            dof = int2tup(dof)
            # extend shape of dtype
            return model.name, base, out_shape + dof
        else:  # compound model
            # structured dtype - nest!
            return model.name, dt, out_shape



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


class StaticGridMixin(object):
    """
    Mixin class that eliminated the need to pass a coordinate grid to every
    model evaluation call. This is convenience when fitting the same model
    repeatedly on the same grid for various data (of the same shape).
    Subclasses must implement the `set_grid` method will be used to construct
    an evaluation grid based on the shape of the data upon the first call to
    `residuals`.
    """

    # default value for static grid. Having this as a class variable avoids
    # having to initialize this class explicitly in inheritors
    static_grid = None

    def __call__(self, p, grid=None):
        grid = self._check_grid(grid)
        return super().__call__(p, grid)

    def residuals(self, p, data, grid=None):
        if grid is None and self.static_grid is None:
            self.set_grid(data)
            grid = self.static_grid     # TODO: emit an info message!!

        grid = self._check_grid(grid)
        # can you set the method as super method here dynamically for speed?
        return super().residuals(p, data, grid)

    def _check_grid(self, grid):        # todo rename get_static_grid
        if grid is None:
            grid = self.static_grid
        if grid is None:
            raise ValueError(
                    'Please specify the coordinate grid for evaluation, '
                    'or use the `set_grid` method prior to the first call to '
                    'assign a static coordinate grid.')
        return grid

    def set_grid(self, data):
        raise NotImplementedError(
                'Derived class should implement this method, or assign the '
                '`static_grid` attribute directly.')

    # def adapt_grid(self, grid):  # adapt_grid_segment
    #     return None

    # def
