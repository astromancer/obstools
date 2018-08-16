"""
Classes for implementing prior probabilities in the context of Bayesian
modelling and inference.
"""
import functools

import numpy as np
from IPython import embed
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from obstools.modelling.utils import prod
from recipes.dict import pformat as pformat_dict


#
def echo(*_):
    return _


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


# TODO: _ParamRecurse???

def _par_walk(obj, call=echo, flat=False, with_keys=True,
             container_out=list, recurse_types=None):
    """
    Recursive walker for converting structured array to nested objects
    """
    #
    # if recurse_types is None:
    #     recurse_types = RECURSE_TYPES

    #  multiple dispatch item_getter for flexible obj construction
    for key, item in item_getter(obj):
        # make sure we have valid field names (keys)
        if not isinstance(key, str):
            raise ValueError('Not a valid name: %r' % key)

        if isinstance(item, recurse_types):
            # recurse
            gen = _RecurseHelper.walk(item, call, flat, with_keys,
                                      container_out)
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

def _par_to_dict(p):
    """recursive walker for converting structured array to nested dict"""
    if p.dtype.fields:
        for k in p.dtype.fields.keys():
            subarray = p[k]
            if isinstance(subarray, np.recarray):
                yield k, dict(_par_to_dict(subarray))
            else:
                if subarray.size == 1:
                    yield k, np.asscalar(subarray)
                else:
                    yield k, subarray
    #else???


def _par_to_tup(p, flat=False):
    """recursive walker for converting structured array to nested dict"""
    if p.dtype.fields:
        for k in p.dtype.fields.keys():
            subarray = p[k]
            if isinstance(subarray, np.recarray):
                gen =_par_to_tup(subarray)
                if flat:
                    yield from gen
                else:
                    yield tuple(gen)
            else:
                if subarray.size == 1:
                    yield np.asscalar(subarray)
                else:
                    yield subarray


class _RecurseHelper(object):
    """
    Helper class for initializing array subclasses by walking arbitrarily nested
    object definitions.
    """

    def __init__(self, allow_types=any):
        self.obj_count = 0
        self.allow_types = allow_types

    def make_dtype(self, kws, base_dtype):
        call = functools.partial(self._make_dtype, base_dtype=base_dtype)
        dtype = list(self.walk(kws, call))
        # reset object count so we can reuse this method
        size = self.obj_count
        self.obj_count = 0
        return dtype, size

    def _make_dtype(self, name, data, base_dtype):
        shape = get_shape(data)
        self.obj_count += prod(shape)
        return name, base_dtype, shape

    def get_data(self, kws, allow_types=any):  # to_tuple / to_dict etc...
        # get data as nested tuple
        call = functools.partial(self._get_data, allow_types=allow_types)
        return tuple(self.walk(kws, call, container_out=tuple, with_keys=False))

    def _get_data(self, name, data, allow_types=any):
        self.type_assertion(data, allow_types)
        return data

    @staticmethod
    def get_npar(dtype):
        return sum(_walk_dtype_size(dtype))

    @staticmethod
    def get_size(data):
        return data.size

    @staticmethod
    def type_assertion(obj, allow_types=any):
        if allow_types is any:
            return obj

        if not isinstance(obj, allow_types):
            raise TypeError('%s type objects are not supported' % type(obj))

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
            # make sure we have valid field names (keys)
            if not isinstance(key, str):
                raise ValueError('Not a valid name: %r' % key)

            if isinstance(item, recurse_types):
                # recurse
                gen = _RecurseHelper.walk(item, call, flat, with_keys,
                                          container_out)
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


# default helper singleton
_par_help = _RecurseHelper()


# _prior_help = _RecurseHelper()

# TODO: add Constant parameter to do stuff like p.x[:2] = Constant
# can you achieve this with masked arrays ??


class Parameters(np.recarray):
    """
    Array subclass that serves as a base container for (nested) parameters.
    Provides natural construction routine for `numpy.recarray` from
    hierarchically typed data.
    """

    # object type restrictions
    _allow_types = any  # any means no type checking will be done upon
    # initialization

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

            (name, dict) pairs for nested (hierarchical) parameter structures
                >>> p = Parameters(Σ=dict(x=3,
                                          y=[7, 5, 3]),
                                   β=1)
                >>> p.β     # array(1.)
                >>> p.Σ.x   # array(3.)

        """
        # the primary objective of this class is to provide the ability to
        # initialize memory from (keyword, data) pairs
        if data is not None:
            if isinstance(data, dict):
                return cls.__new__(cls, None, base_dtype, **data)
            else:
                # use case: Parameters([1, 2, 3, 4, 5])
                # use the `numpy.rec.array` to allow for construction from a
                # wide variety of compatible objects
                obj = np.rec.array(data)
                return obj.view(cls)  # view as Parameters object

        # first we have to construct the dtype by walking the (possibly nested)
        # kws that define the data structure.
        dtype, size = _par_help.make_dtype(kws, base_dtype)

        # construct the array
        obj = super(Parameters, cls).__new__(cls, (), dtype)
        # can give `titles` which are aliases for names

        # keep track of the total number of parameters so we can easily
        # switch between structured and unstructured views
        obj.npar = size
        obj.base_dtype = base_dtype

        # finally, populate array with data (nested tuple)
        obj[...] = _par_help.get_data(kws, cls._allow_types)
        return obj

    def __array_finalize__(self, obj):
        #
        if obj is None:
            # explicit constructor eg:  NestedParameters(foo=1)
            return

        # if we get here object constructor is view casting or new-from-template
        # (slice). Set `npar` here since this method sees creation of all
        # `Parameters` objects
        self.npar = sum(_walk_dtype_size(obj.dtype))
        self.base_dtype = getattr(obj, 'base_dtype', float)

    def __getattribute__(self, key):
        item = super().__getattribute__(key)
        # hack so we don't end up with unsized array containing single object
        if isinstance(item, np.ndarray) and not isinstance(item, Parameters):
            return np.asscalar(item)
        return item

    def __str__(self):
        cls_name = self.__class__.__name__
        if self.dtype.fields:
            s = pformat_dict(self.to_dict())
            indent = ' ' * (len(cls_name) + 1)
            s = s.replace('\n', '\n' + indent)
            return '%s(%s)' % (cls_name, s)
        else:
            return '%s(%s)' % (cls_name, super().__str__())

    def __repr__(self):
        return self.__str__()

    def to_dict(self):  # container=None, squeeze=False
        """Convert to (nested) dict of arrays"""
        return _par_help.walk(self, container_out=dict)

    @property
    def flattened(self):
        return self.denest()

    def denest(self):  # OR overwrite flatten ???????
        # if self.npar != self.size:  # check if already flattened
        return self.view((self.base_dtype, self.npar))
        # return self

    @classmethod
    def from_shapes(cls, **shapes):
        """Construct empty from shape"""
        return dict(
                _RecurseHelper.walk(shapes,
                                    call=lambda k, sh: (k, np.empty(sh)),
                                    container_out=dict))


class Priors(Parameters):
    """
    Define a joint prior distribution on your `Parameters`.

    Interfaces with `Parameters` through the `priors` attribute
    """

    # object type restrictions
    _allow_types = any  # TODO: only accept `Prior` objects

    # Record array containing functions ??!  Unusual, but effective
    def __new__(cls, distrs=None, **kws):
        return super().__new__(cls, distrs, 'O', **kws)

    # def denest(self):

    def random_sample(self, size=1):
        """
        Draw a random sample from the joint prior on the parameter vector.
        Return an instance of the `Parameters` class
        """

        samples = np.empty(size, self.dtype)
        for j, dist in enumerate(self.flattened):
            samples[:, j] = dist.rvs()


class MCMCParams(Parameters):
    def __new__(cls, priors=None, **kws):

        #
        if priors is not None:
            # allow direct construction
            'loop flattened tuple, assert isinstance(_, Prior) ???'

        obj = super().__new__(cls, priors, 'O', **kws)

        # here provide an interface for easily specifying priors for parameters
        obj._priors = Priors()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            # explicit constructor eg:  Parameters(foo=1)
            return

        super().__array_finalize__(obj)

        # priors
        self._priors = getattr(obj, '_priors', Priors())

    @property
    def priors(self):
        return self._priors

    # @priors.setter
    # def priors(self, values):
    #     assert len(values) == self.npar
    #     pass


class _Prior(object):  # Mixin
    def freeze(self, *args, **kwds):
        return Prior(self, *args, **kwds)


class Prior(rv_frozen):  # DistRepr
    """Base class for prior distributions.  These """

    #  Essentially a higher level wrapper for
    # `scipy.stats._distn_infrastructure.rv_frozen`

    # def random_sample(self, size):
    #     """(psuedo-)Random number generator"""

    def __repr__(self):
        return self.dist.name.title() + 'Prior' + str(self.args)

    def __str__(self):
        # here look at `dist` attribute to determine symbol
        symbol = '𝓤'
        args = self.args
        if len(args) == 0:
            ab = (0, 1)
        return symbol + str(args)

    def random_sample(self, size=None, random_state=None):
        return self.rvs(size, random_state)


# TODO: automate this inheritance? or MonkeyPatch?

class _Uniform(_Prior, stats.distributions.uniform_gen):
    """Uniform Prior"""

    # maximum entropy probability distribution for a random variate X under no
    # constraint other than that it is contained in the distribution's support.


Uniform = _Uniform(a=0.0, b=1.0, name='uniform')

# def freeze(self, *args, **kwds):
#     return Prior(self, *args, **kwds)


# class UniformPositive(Uniform):


# class UniformNegative(Uniform):
#     def __init__(self, infimum=-None):
#         ''


# TODO:
# p[-1].prior = UniformNegative
# p[:2].priors = Uniform(0, 1000)
# p[0].prior = JeffreysPrior()
# p.priors = ReferencePrior()

# LogUniform, Normal(μ, σ), etc....

# p.priors.random_sample(3)


# class Priors(ParameterBase):  # JointPrior
#
#
#     def __new__(cls, data=None, base_dtype='O', **kws):
#         return ParameterBase.__new__(cls, data, base_dtype, **kws)
#
#     # def __setitem__(self, key, values):
#     #     print(values)
#     #     for dist in values:
#     #         if not isinstance(dist, rv_frozen):
#     #             raise TypeError('Priors must be instances of '
#     #                             '`scipy.stats.rv_continuous. Received: %s.'
#     #                             % type(dist))
#     #     #
#     #     ParameterBase.__setitem__(self, key, values)


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


# @item_getter.register(np.dtype)
# def _(obj):
#     return obj.fields.items()


RECURSE_TYPES = tuple((set(item_getter.registry) - {object}))

# def item_getter(obj):
#     if isinstance(obj, Parameters):
#         return ((k, p[k]) for k in p.dtype.fields.keys())
#     elif isinstance(dict):
#         return obj.items()


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


# def _par_to_dict(p):
#     """recursive walker for converting structured array to nested dict"""
#     if p.dtype.fields:
#         for k in p.dtype.fields.keys():
#             subarray = p[k]
#             if isinstance(subarray, np.recarray):
#                 yield k, dict(_par_to_dict(subarray))
#             else:
#                 yield k, subarray


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
