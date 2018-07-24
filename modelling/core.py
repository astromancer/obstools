# from pathlib import Path
import numbers
# import json
from collections import OrderedDict as odict

import numpy as np
from IPython import embed
from scipy.optimize import minimize

# from recipes.oop import ClassProperty
from recipes.logging import LoggingMixin
from recipes.dict import AttrReadItem, ListLike, pformat as pformatDict

from obstools.modelling.utils import make_shared_mem, prod, assure_tuple


def _sample_stat(data, statistic, sample_size, replace=True):
    return statistic(np.random.choice(data, sample_size, replace))


def nd_sampler(data, statistic, sample_size, axis=None, replace=True):
    # statistics on samples from ndarray
    if axis is None:
        return _sample_stat(data.ravel(), statistic, sample_size, replace)

    elif isinstance(axis, numbers.Integral):
        size_each = sample_size // data.shape[axis]
        return np.apply_along_axis(
                _sample_stat, axis, data, statistic, size_each, replace)
    else:
        raise ValueError('Invalid axis')


class Priors():
    'todo'
    # p[:2].priors = Uniform(0, 1000)
    # p[0].prior = JeffreysPrior()
    # p.priors = ReferencePrior()


class Bounds():
    """layperson speak for uniform priors"""




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
    """Base class for model"""

    # TODO: have a classmethod here that can turn on and off active view
    # castings so that we can work with nested parametes more easily

    npar = None  # sub-class should set  #todo determine intrinsically from p?
    base_dtype = float  # FIXME - remove this here
    objective = None

    # minimizer = minimize
    # _unconverged_None = True

    def __call__(self, p, grid=None):
        raise NotImplementedError

    def p0guess(self, data, grid=None, stddev=None):
        raise NotImplementedError

    def get_dtype(self):  # todo: use p0guess to determine the dtype ?
        npar = self.npar
        if npar is None:
            raise TypeError('Subclass should set attribute `npar`')

        # if isinstance(self.npar, int):
        #     npar = self.npar,
        return [(self.get_name(), self.base_dtype, npar)]

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
        if stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / stddev

    def fwrs(self, p, data, grid=None, stddev=None):
        """flattened weighted squared residuals"""
        return self.wrs(p, data, grid, stddev).flatten()

    def wrss(self, p, data, grid=None, stddev=None):
        """weighted residual sum of squares. ie. The chi squared statistic χ²"""
        return self.wrs(p, data, grid, stddev).sum()

    chisq = wrss  # chi2

    def redchi(self, p, data, grid=None, stddev=None):  # chi2r
        """Reduced chi squared statistic χ²ᵣ"""
        chi2 = self.chisq(p, data, grid, stddev)
        dof = data.size - self.npar
        return chi2 / dof

    def rsq(self, p, data, grid=None):
        """
        The coefficient of determination, denoted R2 or r2 and pronounced
        "R squared", is the proportion of the variance in the dependent
        variable that is predictable from the independent variable(s). It
        provides a measure of how well observed outcomes are replicated by
        the  model, based on the proportion of total variation of outcomes
        explained by the model.
        """

        v = self(p, grid)
        mu = data.mean()
        # total sum of squares
        tss = np.square(data - mu).sum()  # fixme shouldn't recompute during fit
        rss = self.rss(p, data, grid)
        return 1 - rss / tss

    coefficient_of_determination = rsq

    # TODO: other GoF statistics

    def aic(self, p, data, grid, stddev):
        """Akaike information criterion"""
        # assuming p is the parameter vector corresponding to the maximum
        # likelihood
        return 2 * (len(p) - self.logLikelihood(p, data, grid, stddev))

    # def aicc(self, p):
    #     # "When the sample size is small, there is a substantial probability
    #     # that AIC will select models that have too many parameters...
    #     # AICc is essentially AIC with an extra penalty term for the number of
    #     #  parameter"
    #     #  - from:  https://en.wikipedia.org/wiki/Akaike_information_criterion

    def logLikelihood(self, p, data, grid=None, stddev=None):  # lnLh
        # assuming uncorrelated, gaussian distributed data here
        return -self.wrss(p, data, grid, stddev)

    def pre_process(self, p0, data, grid=None, stddev=None, *args, **kws):
        """This will be run prior to the minimization routine"""
        return p0, data, grid, stddev

    def fit(self, p0, data, grid=None, stddev=None, *args, **kws):
        # FIXME: since p0 is optional here, move down parameter list
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
        type_, dtype = type(p0), p0.dtype
        if isinstance(p0, Parameters):
            p0 = p0.unwrapped

        # check for nans
        if np.isnan(data).any():
            self.logger.warning('Your data contains nans.')

        # check for masked data
        if np.ma.is_masked(data):
            self.logger.warning('Your data has masked elements.')

        # minimization
        p = self._fit(p0, data, grid, stddev=None, *args, **kws)

        if p is not None:
            # get back structured view if required
            return p.view(dtype, type_).squeeze()
            # for some strange reason `result.x.view(dtype, type_)` ends up
            # with higher dimensionality. weird. # BUG REPORT??

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
                self.logger.warning('"Converged" parameters identical to'
                                    ' initial guess %s', p0)
            # TODO: also warn if any close ?
            else:
                self.logger.debug('Successful fit %s', self.name)
                # post-processing
                return self.post_process(result.x, *post_args, **post_kws)

        self.logger.warning('Fit did not converge! %s', result.message)

        # if not self._unconverged_None:
        #      return result.x

    def post_process(self, p, *args, **kws):
        return p

    def _init_mem(self, loc, shape, fill=np.nan, clobber=False):
        """Initialize shared memory for this model"""
        npar = self.npar
        # final array shape is external `shape` + parameter shape `npar`
        if isinstance(shape, int):
            shape = shape,

        if isinstance(self.npar, int):
            shape += (npar,)

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

    def fit(self, p0, data, grid, stddev=None, **kws):
        y = self.center_func(data, self.other_axes)
        if self.disp_func:
            stddev = self.disp_func(data, self.other_axes)
        return super().fit(p0, y, grid, stddev, **kws)


#
#
def echo(*_):
    return _


class _PHelper(object):
    def __init__(self, base_dtype=float):
        self.npar = 0
        self.base_dtype = base_dtype

    @staticmethod
    def get_npar(dtype):
        return sum(_walk_dtype_size(dtype))

    @staticmethod
    def get_size(data):
        return data.size

    def make_dtype_shape(self, name, shape):
        return name, self.base_dtype, shape

    def make_dtype(self, name, data):
        # print('_from_arrays got', name, array)
        shape = get_shape(data)
        self.npar += prod(shape)
        return name, self.base_dtype, shape

    def get_data(self, _, data):
        return data

    def _assure_tuple(self, name, obj):
        # turns integers to tuples, passes tuples, and borks on anything else
        return name, assure_tuple(obj)

    @staticmethod
    def walk(obj, call=echo, flat=False, with_keys=True,
             container_out=list, recurse_types=None):
        """
        recursive walker for dict-like objects. no safeguards, don't blow
        anything up!
        """

        if recurse_types is None:
            recurse_types = RECURSE_TYPES

        #  multiple dispatch item_getter for flexible obj construction
        for key, item in item_getter(obj):
            # print('key', key, 'item', type(item),
            #       isinstance(item, recurse_types))

            if not isinstance(key, str):
                raise ValueError('Not a valid name: %r' % key)

            if isinstance(item, recurse_types):
                # recurse
                gen = _PHelper.walk(item, call, flat, with_keys, container_out)
                if flat:
                    yield from gen
                else:
                    if with_keys:
                        yield key, container_out(gen)  # map(
                    else:
                        yield container_out(gen)
            else:
                yield call(key, item)
                # switch caller here to call(item) if with_keys is False ???

    # def walk(self, obj, do=None, lvl=0, container_out=list, flat=False,
    #          with_keys=True):
    #     # iterate a nested container (usually dict). Execute the function
    #     # `do` on each item
    #     if do is None:
    #         do = self._assure_tuple
    #
    #     if isinstance(obj, Parameters):
    #         # handle use case `Parameters(y=y, x=x)` where `x` and/or `y` are
    #         #  themselves `Parameter` objects
    #         obj = obj.to_dict()
    #
    #     if not isinstance(obj, dict):
    #         raise ValueError('dict please')
    #
    #     for k, v in obj.items():
    #         if not isinstance(k, str):
    #             raise ValueError('Not a valid name')
    #
    #         if isinstance(v, (dict,)):
    #             # recurse
    #             gen = self.walk(v, do, lvl + 1, container_out)
    #             if flat:
    #                 yield from gen
    #             else:
    #                 if with_keys:
    #                     yield k, container_out(gen)
    #                 else:
    #                     yield container_out(gen)
    #         else:
    #             # print('do', k, v, lvl)
    #             yield do(k, v)


class Parameters(np.recarray):
    """
    Array subclass that serves as a container for (nested) parameters
    """

    def __new__(cls, data=None, base_dtype=float, **kws):
        """

        Parameters
        ----------
        kws:
            (name, value) pairs for named parameters
            >>> Parameters(each=1, parameter=2, painstakingly=3, named=4)

            (name, sequence) pairs for sequences of named parameters
             >>> Parameters(coeff=[1,2,3,4],
                            hyper=42)

            Can be nested like so:
                >>> p = Parameters(Σ=dict(x=3,
                                          y=[7, 5, 3]),
                                   β=1)
                >>> p.β     # array(1.)


        """
        # primary objective of this class is to provide the ability to
        # initialize memory from (keyword, data) pairs
        if data is not None:
            if isinstance(data, dict):
                return cls.__new__(cls, None, base_dtype, **data)
            else:
                raise NotImplementedError
                # TODO: >>> Parameters([1, 2, 3, 4, 5])

        # first we have to construct the dtype.
        helper = _PHelper(base_dtype)
        dtype = list(helper.walk(kws, helper.make_dtype))
        # print(dtype)

        # now we construct the array
        obj = super(Parameters, cls).__new__(cls, (), dtype)
        # can give `titles` which are aliases for names

        # keep track of the total number of parameters so we can easily
        # switch between structured and unstructured views
        obj.npar = helper.npar
        obj.base_dtype = base_dtype

        # finally, populate array with data
        obj[...] = tuple(
                helper.walk(kws, helper.get_data, container_out=tuple,
                            with_keys=False))
        return obj

    def __array_finalize__(self, obj):
        # array subclass creation housekeeping
        # print('In array_finalize:')
        # print('   self type is %s' % type(self))
        # print('   obj type is %s' % type(obj))

        if obj is None:
            # explicit constructor eg:  NestedParameters(foo=1)
            return

        # if we get here object constructor is view casting or
        # new-from-template (slice)
        # set npar here since this method sees creation of  all `Parameters`
        # objects
        self.npar = sum(_walk_dtype_size(obj.dtype))
        self.base_dtype = getattr(obj, 'base_dtype', float)

    def __str__(self):
        if self.dtype.fields:
            s = pformatDict(self.to_dict())
            cls_name = self.__class__.__name__
            indent = ' ' * (len(cls_name) + 1)
            s = s.replace('\n', '\n' + indent)
            return '%s(%s)' % (cls_name, s)
        else:
            return super().__str__()

    def __repr__(self):
        return str(self)

    def to_dict(self):  # container=None, squeeze=False
        """Convert to (nested) dict of arrays"""
        return dict(_par_to_dict(self))

    @property  # doesn't work as a property
    def unwrapped(self):  # flat better ?
        if self.npar != self.size:  # check if already unwrapped
            return self.view((self.base_dtype, self.npar))
        return self

    @classmethod
    def from_shapes(cls, **shapes):
        return dict(_PHelper.walk(shapes, call=lambda k, sh: (k, np.empty(sh)),
                                  container_out=dict))

    # TODO
    # @property
    # def randomized(self):
    #     random parameter vector


from functools import singledispatch


@singledispatch  # probably overkill to use single dispatch, but hey!
def item_getter(obj):
    """default multiple dispatch func"""
    raise TypeError('No item getter for object of type %r' % type(obj))


@item_getter.register(dict)
def _(obj):
    return obj.items()


@item_getter.register(Parameters)
def _(p):
    return ((k, p[k]) for k in p.dtype.fields.keys())


@item_getter.register(np.dtype)
def _(obj):
    return obj.fields.items()


RECURSE_TYPES = tuple((set(item_getter.registry) - {object}))


# def item_getter(obj):
#     if isinstance(obj, Parameters):
#         return ((k, p[k]) for k in p.dtype.fields.keys())
#     elif isinstance(dict):
#         return obj.items()


def get_shape(data):
    if isinstance(data, Parameters):
        return data.npar
    else:
        return np.shape(data)


def _walk_dtype_size(obj):
    if not isinstance(obj, np.dtype):
        raise ValueError('dtype please')

    if obj.fields:
        for k, v in obj.fields.items():
            dtype, offset, *title = v
            yield from _walk_dtype_size(dtype)
    else:
        yield prod(obj.shape)


# from functools import singledispatch


# @singledispatch
# def dispatcher(_):
#     """default dispatch func"""
#     raise TypeError
#
# @dispatcher.register(dict)
# def _(obj):
#     for key, item in obj.items():
#
#
#
#
# @dispatcher.register(Parameters)
# def _(p):
#     for k in p.dtype.fields.keys():
#         yield k, p[k]
#
#
#


# def squeeze(key, val):
#     """convert to scalar if possible"""
#     return key, val.squeeze()

# def make_dtype(name, data):
#     # print('_from_arrays got', name, array)
#     a = np.asarray(data)
#     self.npar += a.size
#     return name, self.base_dtype, a.shape


#


# def _walk(obj, call=echo):
#     """recursive walker for Parameters"""
#     # if isinstance(obj, tuple(item_getter.registry)):
#
#     yield from dispatcher(obj)

# for key, item in item_getter(obj):
#     if isinstance(item, tuple(item_getter.registry)):
#         yield from _walk(item)
#     else:
#         yield call(item)


def _par_to_dict(p):
    """recursive walker for converting structured array to nested dict"""
    if p.dtype.fields:
        for k in p.dtype.fields.keys():
            subarray = p[k]
            if isinstance(subarray, np.recarray):
                yield k, dict(_par_to_dict(subarray))
            else:
                yield k, subarray


# def _par_to_dict(p):
#     """recursive walker for converting structured array to nested dict"""
#     if p.dtype.fields:
#         for k in p.dtype.fields.keys():
#             subarray = p[k]
#             if isinstance(subarray, np.recarray):
#                 yield k, dict(_par_to_dict(subarray))
#             else:
#                 yield k, subarray


class ModelContainer(AttrReadItem, ListLike):

    def __init__(self, models=(), **kws):

        mapping = ()
        if len(models):
            models = tuple(filter(None, models))
            names = [m.name for m in models]
            if None in names:
                raise ValueError('All models passed to container must be '
                                 'named. You can name them implicitly by '
                                 'initializing %s via keyword arguments' %
                                 self.__class__.__name__)
            mapping = zip(names, models)

        # load models into dict
        odict.__init__(self, mapping, **kws)
        # note init with kws can mean we loose order in python < 3.6

    def __setitem__(self, key, value):
        # make sure the model has the same name that it is keyed on in the dict
        if key != value.name:
            # set the model name (avoid duplicate names in dtype for models
            # of the same class)
            value.name = key

        return super().__setitem__(key, value)

    # @property
    # def models(self):
    #     return tuple(self.values())

    @property
    def nmodels(self):
        return len(self)

    @property
    def names(self):
        return tuple(mdl.name for mdl in self.values())


class CompoundModel(Model, ModelContainer):
    npar = ()  # compound model

    # use_record = True
    # def __init__(self, models=(), names=None, **kws):
    #     ModelContainer.__init__(self, models, names, **kws)

    # def __repr__(self):
    #     return object.__repr__(self)

    # FIXME: methods here operate on parameters

    @property
    def models(self):
        return self.values()

    def has_compound(self):
        for mdl in self.values():
            if len(mdl.get_dtype()) > 1:
                return True
        return False

    def get_dtype(self):  # TODO: don't need this method anymore!!!!!
        dtypes = [mdl.get_dtype() for mdl in self.values()]
        ncomponents = list(map(len, dtypes))
        has_compound = np.greater(ncomponents, 1).any()
        # if we have nested compound models, need to keep structured dtypes
        if has_compound:
            return list(zip(self.names, dtypes))

        # if only simple models, can just concatenate dtypes
        return list(next(zip(*dtypes)))

    def p0guess(self, data, grid=None, stddev=None, **kws):

        if kws.get('nested'):
            return Parameters({name: mdl.p0guess(data, grid, stddev, **kws)
                               for name, mdl in self.items()})

        else:
            return np.r_[[mdl.p0guess(data, grid, stddev, **kws)
                          for mdl in self.models]]

    def fit(self, p0=None, *args, **kws):
        if p0 is None:
            p0 = self.p0guess(*args)

        p = np.empty_like(p0)
        for name, mdl in self.items():
            p[name] = mdl.fit(p0[name], *args, **kws)
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


class StaticGridMixin(object):
    """
    class for convenience when fitting the same model
    repeatedly on the same grid for different data.
    `set_grid` method will be run on first call to `residuals`.
    """

    # default value for static grid. Having this as a class variable avoids
    # having to initialize this class explicitly in inheritors
    static_grid = None

    def set_grid(self, data):
        raise NotImplementedError(
                'Derived class should implement this method, or assign the '
                '`static_grid` attribute directly.')

    # not sure if this is really useful
    def residuals(self, p, data, grid=None):
        # grid argument ignored
        if grid is None and self.static_grid is None:
            self.set_grid(data)

        if grid is None:
            grid = self.static_grid

        # can you set the method as super method here dynamically for speed?
        return super().residuals(p, data, grid)

    def adapt_grid(self, grid):  # adapt_grid_segment
        return None

    # def


if __name__ == '__main__':
    # some tests for `Parameters`
    # todo: once you refactor to move `Parameters` to it's own module,
    # remove this block and make the `from .utils import` relative again

    p1 = Parameters(x=[1, 2, 3], y=1)
    p2 = Parameters(x=[1, 2, 3], y=1)

    p3 = Parameters(v=p1, w=p2)

    # embed()
