import numpy as np
import lmfit as lm

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
                 _print=False, **kw):
        
        StarFit.__init__(self, psf, algorithm, caching, hints, _print, **kw)
        
        self.call_count = SynchedCounter(0)
            
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, grid, data, data_stddev=None):
        
        Y, X = grid
        params = self.get_param0(data)
        #params = self.psf._set_param_bounds(params, data)
        #print( list(self.psf.params.valuesdict().values()) )
        
        #Fit PSF here
        psf = self.psf
        #res = lm.minimize(psf.rss, params, 'leastsq', args=(data, grid))
        Rpsf = lm.minimize(psf.wrs, params, 'leastsq', args=(data, grid, data_stddev))
        #TODO: Moffat, King, CoG, Lorentzian, shapelets???, PRF
        
        #Fit pure bg here
        pbg = self.bg.params
        pbg['bg'].set(value=Rpsf.params['d'].value)
        Rbg = lm.minimize(self.bg.wrs, pbg, 'leastsq', args=(data, data_stddev))
        #TODO: Polynomial
        
        
        #compare models by aic
        aics = Rpsf.aic, Rbg.aic
        
        #compare models by aic
        #if Rpsf.success and Rbg.success:
            #aics = np.array(Rpsf.aic, Rbg.aic)
            #aicmin = aics.min()
            #aicmax = aics.max()
            #rP = np.exp((aicmin - aicmax)/2)     #rP times as probable as the other model to minimize the information loss
            #if self._print:
                #print('preffered model (AIC):', [psf, ConstantBG][aics.argmin()])
        #else:
        
        if Rpsf.success and psf.validate(Rpsf.params, self.window):
            if self._print:
                print( '\nSuccessfully fit {} function to stellar profile.'.format(psf.F.__name__) )
            #if self.caching:
                ##update cache with these parameters
                #i = self.call_count % self.max_cache_size #wrap!
                #self.cache[i] = plsq        #NOTE: implicit unmasking!
                ##print( '\n\nCACHE!', self.cache )
                ##print( '\n\n' )
            self.call_count += 1    #NOTE: This could be a simple decorator??        
            
            p, punc = zip(*((Rpsf.params[pn].value, Rpsf.params[pn].stderr)
                                for pn in psf._pnames_ordered))
            return p, punc, aics
            
        else:    
            if self._print:
                warnings.warn( 'FIT DID NOT CONVERGE!' )
                print( Rpsf.msg )
                print( Rpsf.info )
            self.call_count += 1    #NOTE: This could be a simple decorator??
            return
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_param0(self, data):
        '''lmfit parameters from data hints'''
        psf = self.psf
        p0 = self.get_params_from_cache()       #just returns the default cache for the psf if caching is disabled
        if self.hints:
            p0[psf.no_cache] = psf.param_hint(data)[psf.no_cache]
            
        #Y, X = grid
        params = self._set_param_values(p0)
        
        return params
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_param_values(self, p):
        pc = self.psf.params.copy()
        for pn, v in zip(self.psf._pnames_ordered, p):
            pc[pn].set(v)
        return pc
