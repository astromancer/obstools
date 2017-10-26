import inspect
import functools
import itertools as itt

import numpy as np
import lmfit as lm          #TODO: is this slow??

from recipes.string import minfloatformat
from recipes.logging import LoggingMixin
from obstools.psf.psf import ConstantBG, CircularGaussianPSF, EllipticalGaussianPSF

# from IPython import embed

#NOTE: CALCULATE THE PARAMETER UNCERTAINTY, AIC, BIC YOURSELF TO FORGO ALL THIS CRAP

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def make_params(pnames_ordered, values=()):
#     # create Parameters object from list of names, values
#     params = lm.Parameters()
#     for pn, val in itt.zip_longest(pnames_ordered, values, fillvalue=1):
#         params.add(pn, value=val)
#     return params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plist(params, pnames_ordered):
    # FIXME: dont need pnames_ordered since lm.Parameters is OrderedDict
    """convert from lm.Parameters to ordered list of float values"""
    if isinstance(params, lm.Parameters):
        pv = params.valuesdict()
        params = [pv[pn] for pn in pnames_ordered]
    return np.array(params)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def convert_params(pnames_ordered):
    """decorator to convert lm.Parameters to parameter list on the fly"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kws):
            obj, p, *rest = args
            p = plist(p, pnames_ordered)
            return func(obj, p, *rest, **kws)
        return wrapper
    return decorator


class lmMixin(LoggingMixin):
    # TODO: give me the string rep of my parent class??
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # pnames_ordered = pnames_ordered  # add parameter names as class attr
    # params = make_params(pnames_ordered)  #
    metrics = ('aic', 'bic', 'redchi')  # goodness of fit metrics to return

    @classmethod
    def make_params(cls, pnames_ordered, values=()):
        # create Parameters object from list of names, values
        params = lm.Parameters()
        for pn, val in itt.zip_longest(pnames_ordered, values, fillvalue=1):
            params.add(pn, value=val)
        cls.params = params  # add parameters to model namespace
        cls.pnames_ordered = pnames_ordered

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fit(self, p0, data, grid, data_stddev=None, **kws):

        # p0 = self.p0guess(data)

        self.logger.debug('Guessed: (%s)' % ', '.join(map(minfloatformat, p0)))
        params = self._set_param_values(p0)
        params = self._constrain_params(params, z0=(0, np.inf))

        kws.setdefault('maxfev', 2000)

        # from IPython import embed
        # embed()

        # Fit PSF here
        # TODO: optimize: psf.wrs is probably expensive since it calls many python funcs
        # TODO: check if scipy.optimize.leastsq is faster with jacobian + col_deriv

        # print('p0, data, grid, data_stddev\n', p0, data, grid, data_stddev)

        result = lm.minimize(self.objective, params, 'leastsq',
                             args=(data, grid, data_stddev), **kws)
        if result.success and self.validate(result.params):
            plsq = result.params
            p, punc = np.transpose([(plsq[pn].value, plsq[pn].stderr)
                                    for pn in self.pnames_ordered])
            fuckedup = np.allclose(p, p0)
            if fuckedup:
                self.logger.warning('Fit did not converge!')
                self.logger.debug('input parameters identical to output')

            self.logger.debug('Successfully fit %s function to stellar profile.',
                              self)
            #gof = {m: getattr(result, m) for m in self.metrics}
            gof = [getattr(result, m) for m in self.metrics]
            return p, punc, gof
        else:
            self.logger.warning('Fit did not converge!')
            self.logger.debug(result.message)
            return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_param_values(self, p):
        pc = self.params.copy()
        for pn, v in zip(self.pnames_ordered, p):
            pc[pn].set(v)
        return pc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


#===============================================================================
def lmModelConversionFactory(base, methods_names, pnames_ordered):
    """ """
    converter = convert_params(pnames_ordered)

    class lmConvertMeta(type):
        """metaclass that decorates the requested methods with the parameter conversion decorator"""
        def __new__(meta, name, bases, namespace):
            for base in bases:
                for mn, method in inspect.getmembers(base, inspect.isfunction):
                    if (mn in methods_names):# and (mn not in new_namespace):
                        namespace[mn] = converter(method)      # patch (decorate) the method
            return super().__new__(meta, name, bases, namespace)

    class lmModel(lmMixin, base, metaclass=lmConvertMeta):
        pass

    lmModel.make_params(pnames_ordered)
    return lmModel

#ConstantBG
convert = ('__call__', 'validate')
pnames_ordered = ['bg']
ConstantBG = lmModelConversionFactory(ConstantBG, convert, pnames_ordered)

#CircularGaussianPSF
convert = ('__call__', 'validate')
pnames_ordered = 'x0, y0, z0, a, d'.split(', ')
CircularGaussianPSF = lmModelConversionFactory(CircularGaussianPSF, convert, pnames_ordered)

#EllipticalGaussianPSF
convert = ('__call__', 'reparameterize', 'integrate', 'integration_uncertainty', 'int_err',
           'get_fwhm', 'get_description', 'correlation', 'covariance_matrix', 'precision_matrix',
           'get_sigma_xy', 'get_theta', 'validate')
pnames_ordered = 'x0, y0, z0, a, b, c, d'.split(', ')
EllipticalGaussianPSF = lmModelConversionFactory(EllipticalGaussianPSF, convert, pnames_ordered)








