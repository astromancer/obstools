import numpy as np
import lmfit as lm          #TODO: is this slow??

from scipy.ndimage.measurements import center_of_mass as CoM

from obstools.psf.psf import GaussianPSF as _GaussianPSF
                             

#****************************************************************************************************
class GaussianPSF(_GaussianPSF):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _pnames_ordered = 'x0, y0, z0, a, b, c, d'.split(', ')
    params = lm.Parameters()
    
    #limiting parameters seems to thwart uncertainty estimates of the parameters... so we HACK!
    _ix_not_neg = list(range(7))        #_GaussianPSF.Npar
    _ix_not_neg.pop(_pnames_ordered.index('b'))       #parameter b is allowed to be negative
    
    for pn in _pnames_ordered:
        params.add(pn, value=1)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def convert_params(self, p):
        #reparameterize to more physically meaningful quantities
        covm                = self.covariance_matrix(p)
        sigx, sigy          = np.sqrt(np.diagonal(covm))
        cov                 = covm[0,1]
        theta               = self.theta(p)
        ratio               = min(sigx, sigy) / max(sigx, sigy)
        ellipticity         = np.sqrt(1 - ratio**2)
        fwhm                = self.get_fwhm(p)
        par_alt             = sigx, sigy, cov, theta, ellipticity, fwhm
        
        return par_alt
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plist(self, params):
        '''convert from lm.Parameters to ordered list of float values'''
        if isinstance(params, lm.Parameters):
            pv = params.valuesdict()
            params = [pv[pn] for pn in self._pnames_ordered]
        return np.array(params)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, params, grid):
        p = self.plist(params)
        return _GaussianPSF.__call__(self, p, grid)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def coeff(self, covariance_matrix):
        '''
        Return a, b, c coefficents for the form:
        z0*np.exp(-(a*(x-x0)**2 -2*b*(x-x0)*(y-y0) + c*(y-y0)**2 )) + d
        '''
        sigx2, sigy2 = np.diagonal(covariance_matrix)
        cov = covariance_matrix[0,1]
        cor2 = cov**2 / (sigx2*sigy2)
        
        f = 0.5 / (1 - cor2)
        a = f / sigx2
        b = f * (cor2/cov)
        c = f / sigy2
        
        return a, b, c
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def correlation(self, params):
        '''Pearsons correlation coefficient '''
        #covm = self.covariance_matrix(params)
        (sigx2, covar), (_, sigy2) = self.covariance_matrix(params)
        return covar / np.sqrt(sigx2*sigy2)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def covariance_matrix(self, params):
        _, _, _, a, b, c, _ = self.plist(params)
        #rho = self.correlation(params)
        #P = np.array([[a,   -b],
        #              [-b,  c ]]) * 2
        detP = (a*c + b*b) * 4
        #inverse of precision matrix
        return (1 / detP) * np.array([[c,   b],
                                      [b,   a]]) * 2
        #return np.linalg.inv(P)                              
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def precision_matrix(self, params):
        _, _, _, a, b, c, _ = self.plist(params)
        P = np.array([[a,   -b],
                      [-b,  c ]]) * 2
        return P
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def theta(self, params):
        _, _, _, a, b, c, _ = self.plist(params)
        return -0.5*np.arctan2( -2*b, a-c )
        
    
        #fwhm = self.get_fwhm(p)
        #counts = self.integrate(p)
        #sigx, sigy = 1/(2*a), 1/(2*c)           #FIXME #standard deviation along the semimajor and semiminor axes
        #ratio = min(a,c)/max(a,c)               #Ratio of minor to major axis of Gaussian kernel
        #theta = 0.5*np.arctan2( -b, a-c )       #rotation angle of the major axis in sky plane
        #ellipticity = np.sqrt(1-ratio**2)
        #coo = x+offset[0], y+offset[1]
    
    def fwhm(self, params):
        return self.get_fwhm(self.plist(params))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def param_hint(self, data):
        '''Return a guess of the fitting parameters based on the data'''
        
        #location
        #y0, x0 = np.c_[np.where(data==data.max())][0]              #center_of_mass( data ) #NOTE: in case of multiple maxima it only returns the first
        
        bg = np.median(data)
        z0 = data.max() - bg    #use core area only????
        
        #location
        y0, x0 = np.divide(data.shape, 2)
        #y0, x0 = CoM(data)
        #y0, x0 = np.c_[np.where(data==data.max())][0]

        #p = np.empty(self.Npar)
        #p[self.to_hint] = x0, y0, z0, bg
        #cached parameters will be set by StarFit
        return x0, y0, z0, bg
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def p0guess(self, data, p0=None):
        '''
        Return best guess paramaters based on param_hint
        if p0 is not given, the default parameters will be used as starting point
        if p0 is given, data in its `to_hint` index positions will be set
        '''
        if p0 is None:
            p0 = self.default_params
        p0[self.to_hint] = self.param_hint(data)
        return p0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_param_bounds(self, par0, data):
         #HACK-ish! better to explore full posterior
        '''set parameter bounds based on data'''
        x0, y0, z0, a, b, c, d = self.plist(par0)
        #x0, y0 bounds
        #Let x, y only vary across half of window frame
        #sh = np.array(data.shape)
        #(xbl, xbu), (ybl, ybu) = (x0, y0) + (sh/2)[None].T * [1, -1]  
        #par0['x0'].set(min=xbl, max=xbu)
        #par0['y0'].set(min=ybl, max=ybu)
        
        #z0 - (0 to 3 times frame max value)
        #zbl, zbu = (0, z0*3)    #NOTE: hope your initial guess is robust
        #par0['z0'].set(min=zbl, max=zbu)
        
        ##d - sky background
        #dbl, dbu = (0, d*5)
        #par0['d'].set(min=dbl, max=dbu)
        
        for pn in self._pnames_ordered:
            par0[pn].set(min=0)
        
        #NOTE: not sure how to constrain a,b,c params...
        return par0
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def validate(self, p, window):
        p = self.plist(p)
        nans = np.isnan(p).any()
        negpars = any(p[self._ix_not_neg] < 0)
        badcoo = any(abs(p[:2] - window/2) >= window/2)
        
        return ~(badcoo | negpars | nans)
        
   
#****************************************************************************************************
class ConstantBG():
    
    params = lm.Parameters()
    params.add('bg', min=0)
    
    def rs(self, p, data):
        return np.square(data - p['bg'].value)

    def wrs(self, p, data, stddev):
        if stddev is None:
            return self.rs(p, data)
        return self.rs(p, data) / stddev
    
    def fwrs(self, p, data, stddev):
        return self.wrs(p, data, stddev)



