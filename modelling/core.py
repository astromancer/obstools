# from pathlib import Path

from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

# from recipes.oop import ClassProperty
from recipes.logging import LoggingMixin
from recipes.dict import Indexable, AttrReadItem

from .utils import make_shared_mem


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
    Implements optional mutable `name` for inherited classes
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

    def __call__(self, p, grid):
        raise NotImplementedError

    def residuals(self, p, data, grid):
        """Difference between data and model"""
        return data - self(p, grid)

    def rs(self, p, data, grid):
        """squared residuals"""
        return np.square(self.residuals(p, data, grid))

    def frs(self, p, data, grid):
        """squared residuals flattened"""
        return self.rs(p, data, grid).flatten()

    def rss(self, p, data, grid):
        """residual sum of squares"""
        return self.rs(p, data, grid).sum()

    def wrs(self, p, data, grid, stddev=None):
        """weighted squared residuals"""
        if stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / stddev

    def fwrs(self, p, data, grid, stddev=None):
        """flattened weighted squared residuals"""
        return self.wrs(p, data, grid, stddev).flatten()

    def wrss(self, p, data, grid, stddev=None):
        """weighted residual sum of squares"""
        return self.wrs(p, data, grid, stddev).sum()

    # def logLikelihood(self):

    def p0guess(self, data, grid, stddev=None):
        raise NotImplementedError

    def fit(self, p0, data, grid, stddev=None, *args, **kws):
        """
        Minimize `objective` using `minimizer` algorithm

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
                return result.x

        self.logger.warning('Fit did not converge! %s', result.message)

        # if not self._unconverged_None:
        #      return result.x

    # def validate(self, p, *args):
    #     """validate parameter values.  To be overwritten by sub-class"""
    #     return all([vf(p) for vf in self.validations])

    def _init_mem(self, loc, shape, fill=np.nan, clobber=False):
        """Initialize shared memory for this model"""
        npar = self.npar
        # final array shape is external `shape` + internal parameter shape `npar`
        if isinstance(shape, int):
            shape = shape,

        if isinstance(self.npar, int):
            shape += (npar,)

        # locPar = '%s.par' % self.name  # locStd = '%s.std' % self.name
        dtype = self.get_dtype()
        return make_shared_mem(loc, shape, dtype, fill, clobber)

    def get_dtype(self):
        npar = self.npar
        if npar is None:
            raise TypeError('Subclass should set attribute `npar`')

        if isinstance(self.npar, int):
            npar = self.npar,
        return [(self.get_name(), self.base_dtype, npar)]


class ModelContainer(AttrReadItem, Indexable, OrderedDict):

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
        OrderedDict.__init__(self, mapping, **kws)
        # note init with kws can mean we loose order in python < 3.6

    def __setitem__(self, key, value):
        # make sure the model has the same name that it is keyed on in the dict
        if key != value.name:
            # set the model name (avoid duplicate names in dtype for models
            # of the same class)
            value.name = key

        return super().__setitem__(key, value)

    # def make_repr(self):
    #     # ensure we format nicely considering we might have nested models
    #     clsname = self.__class__.__name__
    #     p = len(clsname) + 1
    #     names = self.names
    #     fmt = '{0[0]: >%is}: {0[1]: <%is}' % (p, max(map(len, names)))
    #     return fmt
    #     # % return ',\n'.join(map(fmt.format, self.items()))
    #     return clsname + '{' + \
    #            ',\n'.join(map(fmt.format, self.items())) \
    #            + '}'

    # def __getnewargs__(self):
    #     print('ping')
    #     return tuple(self.models)

    # @property
    # def models(self):
    #     return tuple(self.values())

    @property
    def nmodels(self):
        return len(self)

    @property
    def names(self):
        return tuple(mdl.name for mdl in self.values())


# class ModelContainer(object):
#     # TODO: just inherit from namedtuple / OrderedAttrDict
#
#     @property
#     def nmodels(self):
#         return len(self.models)
#
#
#
#     def __init__(self, models=(), names=None, **kws):
#         # TODO: just inherit from namedtuple?
#         if names is None:
#             names = [m.name for m in models]
#         else:
#             assert len(names) == len(models)
#
#         # load models into dict
#         kws.update(zip(names, models))
#         # check if anything given
#         assert len(kws), 'No models given.'
#
#         # update model names
#         for name, mdl in kws.items():
#             # set model names if given (avoid duplicate names in dtype)
#             # if name != mdl.name:
#             mdl.name = name  # TODO: log
#
#         # namedtuple will check valid field names etc
#         self.model_names = list(kws.keys())
#         ModelContainer = namedtuple('ModelContainer', self.model_names)
#         self.models = ModelContainer(**kws)
#
#     def __getstate__(self):
#         # capture what is normally pickled
#         state = self.__dict__.copy()
#         # replace namedtuple ModelContainer with a dict
#         state['models'] = self.models._asdict()
#         return state
#
#     def __setstate__(self, state):
#         model_dict = state['models']
#         ModelContainer = namedtuple('ModelContainer', model_dict.keys())
#         state['models'] = ModelContainer(**model_dict)
#         # re-instate our __dict__ state from the pickled state
#         self.__dict__.update(state)

# def __call__(self, p, grid):
#     # assert self.nmodels
#     result = np.empty_li
#     for mdl in self.models:
#         r[mdl.name] = mdl(p[mdl.name], grid)
#     return p


class CompoundModel(Model, ModelContainer):
    npar = ()  # compound model

    # def __init__(self, models=(), names=None, **kws):
    #     ModelContainer.__init__(self, models, names, **kws)

    # def __repr__(self):
    #     return object.__repr__(self)

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
        # if we have nested  compound models, need to keep structured dtypes
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


# TODO: SimultaneousCompound, SequentialCompound


class StaticGridMixin(object):
    """
    static grid mixin class for convenience when fitting the same model
    repeatedly on the same grid for different data. set_grid method will be
    run on first call to residuals.
    """
    grid = None  # todo. property?

    def set_grid(self, data):
        raise NotImplementedError(
                'Derived class should implement this method, or assign the '
                '`grid` attribute directly.')

    def residuals(self, p, data, grid=None):
        # grid argument ignored
        if grid is None and self.grid is None:
            self.set_grid(data)

        return super().residuals(p, data, self.grid)

    def adapt_grid(self, grid):  # adapt_grid_segment
        return None
