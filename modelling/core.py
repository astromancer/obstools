# from pathlib import Path
import numbers
# import json
from collections import OrderedDict as odict

import numpy as np
from scipy.optimize import minimize

# from recipes.oop import ClassProperty
from recipes.logging import LoggingMixin
from recipes.dict import AttrReadItem, ListLike, pformat as pformatDict

from .utils import make_shared_mem, prod, assure_tuple


def _sample_stat(data, statistic, sample_size, replace=True):
    return statistic(np.random.choice(data, sample_size, replace))


def nd_sampler(data, statistic, sample_size, axis=None, replace=True):
    # statistics on samples from ndarray
    if axis is None:
        return _sample_stat(data, statistic, sample_size, replace)

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
    """Base class for model"""

    npar = None  # sub-class should set  #fixme OR  0  cleaner??
    base_dtype = 'f'
    objective = None

    # minimizer = minimize
    # _unconverged_None = True

    def __call__(self, p, grid=None):
        raise NotImplementedError

    def p0guess(self, data, grid=None, stddev=None):
        raise NotImplementedError

    def get_dtype(self):
        npar = self.npar
        if npar is None:
            raise TypeError('Subclass should set attribute `npar`')

        # if isinstance(self.npar, int):
        #     npar = self.npar,
        return [(self.get_name(), self.base_dtype, npar)]

    def residuals(self, p, data, grid=None):
        """Difference between data (observations) and model. A.k.a. Deviation"""
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

    chisq = wrss

    def redchi(self, p, data, grid=None, stddev=None):
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

    # def logLikelihood(self): # lnLh

    def pre_process(self, p0, data, grid=None, stddev=None, *args, **kws):
        """This will be run prior to the minimization routine"""
        return p0, data, grid, stddev

    def fit(self, p0, data, grid=None, stddev=None, *args, **kws):
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

        # pre-processing
        pre_args = kws.pop('pre_args', ())
        pre_kws = kws.pop('pre_kws', {})
        post_args = kws.pop('post_args', ())
        post_kws = kws.pop('post_kws', {})
        p0, data, grid, stddev = self.pre_process(p0, data, grid, stddev,
                                                  *pre_args, **pre_kws)
        # minimization
        p = self._fit(p0, data, grid, stddev=None, *args, **kws)
        # post-processing
        return self.post_process(p, *post_args, **post_kws)

    def _fit(self, p0, data, grid, stddev=None, *args, **kws):
        """Minimization worker"""
        if self.objective is None:
            self.objective = self.wrss  # wrs for leastsq...

        if p0 is None:
            p0 = self.p0guess(data, grid)
            self.logger.debug('p0 guess: %s', p0)

        #
        # for nested parameters view as float
        # dt = p0.dtype
        # p0.view('f')p0b

        result = minimize(self.objective, p0, (data, grid, stddev), *args,
                          **kws)
        if result.success:
            samesame = np.allclose(result.x, p0)
            if samesame:
                self.logger.warning('"Converged" result parameters identical to'
                                    ' initial guess %s', p0)
            # TODO: also warn if any close ?
            else:
                self.logger.debug('Successful fit %s', self.name)
                return result.x  # .view(dt)

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

    def inverse(self, p):
        raise NotImplementedError

    def pre_process(self, p0, data, grid, stddev=None, *args, **kws):
        data = self.transform(data)
        return p0, data, grid, stddev

    def post_process(self, p, **kws):
        if p is not None:
            return self.inverse(p)


class RescaleInternal(DataTransformBase):
    def get_scale(self, data, sample_size=100, axis=None):
        return nd_sampler(data, np.median, sample_size, axis)

    def transform(self, data):
        self._yscale = self.get_scale(data)
        self.logger.info('scale is %s', self._yscale)
        return data / self._yscale

    def inverse(self, p):
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


class NestedParameters(object):  # could be an array subclass ??
    """
    Helper class for going to and from structured parameter arrays
    """

    use_record = True

    @classmethod
    def from_shapes(cls, base_dtype='f', **kws):
        # will be initialized empty
        return cls(cls._make_dtype_from_shapes, base_dtype, **kws)

    @classmethod
    def from_arrays(cls, base_dtype='f', **kws):  # fromkeys ???
        obj = cls(cls._make_dtype, base_dtype, **kws)

    def __array_finalize__(self):
        # object creation housekeeping


    def __init__(self, _make=None, base_dtype='f', **kws):
        """

        Parameters
        ----------
        kws:
            (name, size) pairs for parameter sets.
            Can be nested like so:
            (name1=[('sub1', 3), ('sub2', (7, 5, 3)))],
             name2=1)
        """

        self._init = kws
        self.npar = 0
        self._slc = []

        # self._shapes = odict()
        self.base_dtype = base_dtype
        from_arrays = (_make is None)
        if from_arrays:
            _make = self._make_dtype
        #     get_shapes = self._get_array_shapes
        else:
            #     get_shapes = None
            _make = self._assure_tuple
            # this is the default, explicit here for readability

        # create numpy dtype list
        self.dtype = list(self.walk(kws, _make))
        # set the data from input arrays and get nested shapes

        # self.shapes = odict(self.walk(kws, get_shapes,
        #                               container_out=odict))

        # finally set the data
        self.p = self.empty()
        # get unwrapped view
        p = self.p.view((self.base_dtype, self.npar))
        if from_arrays:
            for i, (k, data) in enumerate(
                    self.walk(kws, self._assure_array, flat=True)):
                p[self._slc[i]] = data

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = pformatDict(self._init)
        cls_name = self.__class__.__name__
        indent = ' ' * (len(cls_name) + 1)
        s = s.replace('\n', '\n' + indent)
        return '%s(%s)' % (cls_name, s)

    def __array__(self):
        return self.p  # HACK for a fake array subclass

    def _make_dtype(self, name, array):
        # print('_from_arrays got', name, array)
        array = np.asarray(array)
        size = array.size
        self._slc.append(slice(self.npar, self.npar + size))
        self.npar += size
        return name, self.base_dtype, array.shape

    def _make_dtype_from_shapes(self, name, shape):
        shape = assure_tuple(shape)
        size = prod(shape)
        self._slc.append(slice(self.npar, self.npar + size))
        self.npar += size
        return name, self.base_dtype, shape

    def _get_array_shapes(self, name, array):
        return name, np.shape(array)

    def _assure_array(self, name, array):
        return name, np.array(array)

    def _assure_tuple(self, name, shape):
        return name, assure_tuple(shape)

    def walk(self, d, do=None, lvl=0, container_out=list, flat=False):
        if do is None:
            do = self._assure_tuple

        if not isinstance(d, dict):
            raise ValueError('dict please')

        for k, v in d.items():
            if not isinstance(k, str):
                raise ValueError('Not a valid name')

            if isinstance(v, (dict,)):
                # recurse
                gen = self.walk(v, do, lvl + 1, container_out)
                if flat:
                    yield from gen
                else:
                    yield k, container_out(gen)
            else:
                print('do', k, v, lvl)
                yield do(k, v)
                # k, v = do(k, v)
                # yield k, v

    def empty(self, shape=()):
        out = np.empty(shape, self.dtype)
        if self.use_record:
            return np.rec.array(out)
        return out

    # def wrap(self, p):
    #     return p.view(self.dtype)

    @property
    def unwrapped(self):
        return self.p.view((self.base_dtype, self.npar))


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

    def get_dtype(self):
        dtypes = [mdl.get_dtype() for mdl in self.values()]
        ncomponents = list(map(len, dtypes))
        has_compound = np.greater(ncomponents, 1).any()
        # if we have nested compound models, need to keep structured dtypes
        if has_compound:
            return list(zip(self.names, dtypes))

        # if only simple models, can just concatenate dtypes
        return list(next(zip(*dtypes)))

    def p0guess(self, data, grid=None, stddev=None):
        p0 = np.empty((), self.get_dtype())
        for name, mdl in self.items():
            # p00 = mdl.p0guess(data, grid, stddev)
            # print('cmpd p0', data.shape, grid, p0)
            p0[name] = mdl.p0guess(data, grid, stddev)
        return p0

    def fit(self, p0=None, *args, **kws):
        if p0 is None:
            p0 = self.p0guess(*args)

        p = np.empty_like(p0)
        for name, mdl in self.items():
            p[name] = mdl.fit(p0[name], *args, **kws)


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
    # TODO: better to have models not require grid by default
    """
    static grid mixin class for convenience when fitting the same model
    repeatedly on the same grid for different data.
    `set_grid` method will be run on first call to `residuals`.
    """
    static_grid = None

    def set_grid(self, data):
        raise NotImplementedError(
                'Derived class should implement this method, or assign the '
                '`grid` attribute directly.')

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
