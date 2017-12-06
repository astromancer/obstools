import logging

import numpy as np
import lmfit as lm
from scipy.optimize import leastsq

from obstools.psf.models import (lmGaussianPSF as GaussianPSF,
                                 ConstantBG)
from obstools.psf.psf import StarFit
from lll import SynchedCounter, LoggingMixin
from recipes.string import minfloatformat



#****************************************************************************************************
def fit_worker(model, data, grid, data_stddev=None):
    # fit the model
    result = model.fit(data, grid, data_stddev)
    # assess success
    if result.success and model.validate(result.params, data.shape):
        self.logger.debug('Successfully fit {} function to stellar profile.'
                          ''.format(self))
        plsq = result.params
        p, punc = np.transpose([(plsq[pn].value, plsq[pn].stderr)
                                for pn in model.pnames_ordered])
        stats = [getattr(result, m) for m in self.metrics]
        return p, punc, stats
    else:
        self.logger.warning('Fit did not converge!')
        self.logger.debug(result.message)
        return




#****************************************************************************************************
#TODO
class StarFieldFitter():
    pass

#****************************************************************************************************
class FitLM(LoggingMixin):
    #FIXME: parameter caching not working in parallel implementation
    #TODO: masked SynchedArray
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, psf, algorithm=None, caching=False, hints=True, **kw):
        '''
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
        '''
        #if not isinstance(psf, PSF):
            #raise ValueError( 'psf must be an instance of the PSF class.' )

        if algorithm is None:
            self.algorithm = leastsq
        else:
            raise NotImplementedError

        #set the psf function
        self.psf        = psf
        self.caching    = caching
        self.hints      = hints
        self.call_count = SynchedCounter(0)   #keeps track of how many times the class has been called

        #pre-allocate cache memory
        self.max_cache_size = n = kw.get('max_cache_size', 25 if caching else 0)
        m = len(psf.to_cache)
        self.cache = np.ma.array(np.zeros((n, m)),
                                 mask=np.ones((n, m), bool))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, grid, data, data_stddev=None):

        psf = self.psf
        params = self.p0guess(data)
        self.logger.debug('Guessed: (%s)' % ', '.join(map(minfloatformat, psf.plist(params))))

        #Fit PSF here
        #TODO: optimize: psf.wrs is probably expensive since it calls many python funcs
        #TODO: check if scipy.optimize.leastsq is faster with jacobian + col_deriv
        result = lm.minimize(psf.wrs, params, 'leastsq', args=(data, grid, data_stddev),
                             maxfev=2000)
        plsq = result.params

        if result.success and psf.validate(plsq, self.window):
            self.logger.debug('Successfully fit {} function to stellar profile.'
                             ''.format(psf))

            #if self.caching: #TODO
                ##update cache with these parameters
                #i = self.call_count % self.max_cache_size #wrap!
                #self.cache[i] = plsq        #NOTE: implicit unmasking!
                ##print( '\n\nCACHE!', self.cache )
                ##print( '\n\n' )

            self.call_count += 1    #NOTE: This could be a simple decorator??
            return plsq
                # p, punc, aics

        else:
            self.logger.info('FIT DID NOT CONVERGE!')
            self.logger.debug(result.message)
            #self.logger.debug(result.info)
            self.call_count += 1    #NOTE: This could be a simple decorator??
            #return
            return None#, None, None



# class ModelComparison(LoggingMixin):
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     def __init__(self, models, *metrics):
#         assert len(models)
#         if not len(metrics):
#             metrics = ('aic', 'bic', 'redchi')
#             #TODO: assertions
#         self.metrics = metrics
#         self.models = models
#
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     def __call__(self, data, grid, data_stddev=None):
#         for model in self.models:
#             self.fitter(model, data, grid, data_stddev)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





# bg = ConstantBG()
#
# # TODO: Circular Gaussian, Moffat, King, CoG, Lorentzian, shapelets???, PRF
#
# # Fit pure bg here
# pbg = self.bg.params
# pbg['bg'].set(value=plsq['d'].value)
# Rbg = lm.minimize(self.bg.wrs, pbg, 'leastsq', args=(data, data_stddev),
#                   maxfev=2000)
# # TODO: Polynomial
#
# # compare models by aic
# aics = result.aic, Rbg.aic

# compare models by aic
# if result.success and Rbg.success:
# aics = np.array(result.aic, Rbg.aic)
# aicmin = aics.min()
# aicmax = aics.max()
# rP = np.exp((aicmin - aicmax)/2)     #rP times as probable as the other model to minimize the information loss
# self.logger.info()
# print('preffered model (AIC):', [psf, ConstantBG][aics.argmin()])
# else: