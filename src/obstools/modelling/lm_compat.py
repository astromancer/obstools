"""
Library that dynamically converts models to be usable by lmfit.  All classes
are picklably and can therefore be used in parallelized applications via
multiprocessing
"""



# std libs
import inspect
import functools
import itertools as itt

# third-party libs
import lmfit as lm
import numpy as np

# local libs
from recipes.pprint import decimal_repr




# from recipes.logging import LoggingMixin


def plist(params):  # can import from lm_compat
    """convert from lm.Parameters to ordered list of float values"""
    if isinstance(params, lm.Parameters):
        params = list(params.valuesdict().values())
    return np.asarray(params)


def convert_params(func):
    """decorator to convert lm.Parameters to a list on the fly"""

    @functools.wraps(func)
    def wrapper(*args, **kws):
        obj, p, *rest = args
        return func(obj, plist(p), *rest, **kws)

    return wrapper


def make_params(pnames, values=()):
    # create Parameters object from list of names, values
    params = lm.Parameters()
    for pn, val in itt.zip_longest(pnames, values, fillvalue=1):
        params.add(pn, value=val)
    return params


class lmMixin():
    """
    Mixin class that implements the fit method for use by lmfit.
    """

    metrics = ('aic', 'bic', 'redchi')  # goodness of fit metrics to return

    def fit(self, p0, data, grid, data_stddev=None, **kws):

        self.logger.debug('Guessed: (%s)' % ', '.join(map(decimal_repr, p0)))
        params = self._set_param_values(p0)
        params = self._constrain_params(params, z0=(0, np.inf))

        kws.setdefault('maxfev', 2000)

        # Fit PSF here
        # TODO: optimize: psf.wrs is probably expensive since it calls many python funcs
        # TODO: check if scipy.optimize.leastsq is faster with jacobian + col_deriv

        result = lm.minimize(self.objective, params, 'leastsq',
                             args=(data, grid, data_stddev), **kws)

        if result.success and self.validate(result.params):
            plsq = result.params
            p, punc = np.transpose([(p.value, p.stderr)
                                    for p in plsq.values()])
            bad = np.allclose(p, p0)
            if bad:  # model "converged" to the initial values
                self.logger.warning('%s fit did not converge!', self)
                self.logger.debug('input parameters identical to output')

            self.logger.debug(
                'Successfully fit %s function to stellar profile.', self)
            # gof = {m: getattr(result, m) for m in self.metrics}
            gof = [getattr(result, m) for m in self.metrics]
            return p, punc, gof
        else:
            self.logger.warning('Fit did not converge!')
            self.logger.debug(result.message)
            return

    def _set_param_values(self, vals):
        pc = self.params.copy()
        for (pn, p), v in zip(pc.items(), vals):
            p.set(v)
        return pc

    def _constrain_params(self, params, **bounds):
        # pc = self.params.copy()
        for pn, interval in bounds.items():
            if pn in params:
                params[pn].min, params[pn].max = interval
        return params


        # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def _set_param_bounds(self, par0, data):
        #     """set parameter bounds based on data"""
        #     x0, y0, z0, a, b, c, d = self.plist(par0)
        #     #x0, y0 bounds
        #     #Let x, y only vary across half of window frame
        #     #sh = np.array(data.shape)
        #     #(xbl, xbu), (ybl, ybu) = (x0, y0) + (sh/2)[None].T * [1, -1]
        #     #par0['x0'].set(min=xbl, max=xbu)
        #     #par0['y0'].set(min=ybl, max=ybu)
        #
        #     #z0 - (0 to 3 times frame max value)
        #     #zbl, zbu = (0, z0*3)    #NOTE: hope your initial guess is robust
        #     #par0['z0'].set(min=zbl, max=zbu)
        #
        #     ##d - sky background
        #     #dbl, dbu = (0, d*5)
        #     #par0['d'].set(min=dbl, max=dbu)
        #
        #     for pn in self._pnames_ordered:
        #         par0[pn].set(min=0)
        #
        #     #NOTE: not sure how to constrain a,b,c params...
        #     return par0


def lmModelFactory(base_, method_names, param_names):
    """
    Construct the lmfit compatible class by inheriting from the *base_* class
    and decorating the methods in *method_names* to convert parameters on
    the fly with *convert_params*. Also creates the lm.Parameters instance
    from list of parameter names *param_names* and attaches it as a class
    attribute.

    Parameters
    ----------
    base_
    method_names
    param_names

    Returns
    -------

    """

    class lmConvertMeta(type):
        """Constructor that creates the converted class"""

        def __new__(meta, name, bases, namespace):
            for base in bases:
                for mn, method in inspect.getmembers(base, inspect.isfunction):
                    if (mn in method_names):
                        # decorate the method
                        namespace[mn] = convert_params(method)
            return type.__new__(meta, name, bases, namespace)

    class lmCompatModel(lmMixin, base_, metaclass=lmConvertMeta):
        params = make_params(param_names)

        def __reduce__(self):
            # used to recreate the class for (un-)pickling
            return (_InitializeParameterized(),
                    (base_, method_names, param_names),
                    self.__dict__)

    return lmCompatModel


class _InitializeParameterized:
    """
    When called with the param value as the only argument, returns an
    un-initialized instance of the parameterized class. Subsequent __setstate__
    will be called by pickle.
    """

    def __call__(self, base, method_names, param_names):
        obj = _InitializeParameterized()
        obj.__class__ = lmModelFactory(base, method_names, param_names)
        return obj
