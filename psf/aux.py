import logging
import numpy as np
import lmfit as lm
from scipy.optimize import leastsq

from obstools.psf.models import GaussianPSF, ConstantBG
from obstools.psf.psf import StarFit

from lll import SynchedCounter


#TODO: logging

class StarFieldFitter():
    pass

#****************************************************************************************************
class StarFitLM(StarFit):
    #FIXME: parameter caching not working in parallel implementation
        #TODO: masked SynchedArray
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bg = ConstantBG()
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, psf=None, algorithm=None, caching=False, hints=True, 
                 _print=False, **kw):       #TODO: logging!!!!
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
        self.psf        = psf           or      GaussianPSF()
        self.caching    = caching
        #pre-allocate cache memory
        if caching:
            self.max_cache_size = n = kw.get('max_cache_size', 25)
            m = len(psf.to_cache)
            self.cache = np.ma.array(np.zeros((n, m)),
                                     mask=np.ones((n, m), bool))
            
        self.hints      = hints
        #initialise parameter cache with psf defaults
        #self.cache      = []                  #parameter cache
        
        self.call_count = 0                    #keeps track of how many times the class has been called
        self.logger = logging.getLogger('phot.aux.StarFit')
        
        self._print = _print
        
        
        #StarFit.__init__(self, psf, algorithm, caching, hints, _print, **kw)
        
        self.call_count = SynchedCounter(0)
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, grid, data, data_stddev=None):
        
        Y, X = grid
        params = self.p0guess(data)
        #params = self.psf._set_param_bounds(params, data)
        #print( list(self.psf.params.valuesdict().values()) )
        
        #Fit PSF here
        psf = self.psf
        #res = lm.minimize(psf.rss, params, 'leastsq', args=(data, grid))
        result = lm.minimize(psf.wrs, params, 'leastsq', args=(data, grid, data_stddev),
                             maxfev=2000)
        plsq = result.params
        #TODO: Moffat, King, CoG, Lorentzian, shapelets???, PRF
        
        #Fit pure bg here
        pbg = self.bg.params
        pbg['bg'].set(value=plsq['d'].value)
        Rbg = lm.minimize(self.bg.wrs, pbg, 'leastsq', args=(data, data_stddev),
                          maxfev=2000)
        #TODO: Polynomial
        
        #compare models by aic
        aics = result.aic, Rbg.aic
        
        #compare models by aic
        #if result.success and Rbg.success:
            #aics = np.array(result.aic, Rbg.aic)
            #aicmin = aics.min()
            #aicmax = aics.max()
            #rP = np.exp((aicmin - aicmax)/2)     #rP times as probable as the other model to minimize the information loss
            #self.logger.info()
                #print('preffered model (AIC):', [psf, ConstantBG][aics.argmin()])
        #else:
        
        if result.success and psf.validate(plsq, self.window):
            self.logger.info('Successfully fit {} function to stellar profile.'
                             ''.format(psf))
            
            #if self.caching:
                ##update cache with these parameters
                #i = self.call_count % self.max_cache_size #wrap!
                #self.cache[i] = plsq        #NOTE: implicit unmasking!
                ##print( '\n\nCACHE!', self.cache )
                ##print( '\n\n' )
            self.call_count += 1    #NOTE: This could be a simple decorator??        
            
            p, punc = np.transpose([(plsq[pn].value, plsq[pn].stderr) 
                                        for pn in psf._pnames_ordered])
            return p, punc, aics
            
        else:    
            self.logger.info('FIT DID NOT CONVERGE!')         
            self.logger.debug(result.message)
            #self.logger.debug(result.info)
            self.call_count += 1    #NOTE: This could be a simple decorator??
            #return
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def p0guess(self, data):
        '''lmfit parameters from data hints'''
        #TODO: if cache is empty and hint is False, trigger warning?
        psf = self.psf
        p0 = psf.p0guess(data)
        
        if self.caching and self.call_count:
            p0[psf.to_cache] = self.cache.mean(0)
        
        params = self._set_param_values(p0)
        
        return params
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_param_values(self, p):
        pc = self.psf.params.copy()
        for pn, v in zip(self.psf._pnames_ordered, p):
            pc[pn].set(v)
        return pc
