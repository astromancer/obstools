import numpy as np
import pylab as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D

from scipy.optimize import leastsq
from scipy.ndimage.measurements import center_of_mass
from collections import Callable
import itertools as itt


from misc import get_missing_numbers, shape2grid
from misc import make_ipshell
ipshell = make_ipshell()
#ipshell()

#===============================================================================================       
#def Gaussian(p, x):
    #'''Gaussian function for fitting radial star profiles'''
    #A, b, mx = p
    #return A*np.exp(-b*(x-mx)**2 )

#===============================================================================================    
def Gaussian2D(p, x, y):
    #TODO: option to pass angle, semi-major, semi-monor; or covariance matrix
    '''Elliptical Gaussian function for fitting star profiles.'''
    x0, y0, z0, a, b, c, d = p
    return z0*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 )) + d

#===============================================================================================    
def Moffat(p, x, y):
    x0, y0, z0, a, b, c  = p
    return z0*(1 + ((x-x0)**2 + (y-y0)**2) / (a*a))**-b + c


######################################################################################################    
class PSF(object):
    '''Class that implements the point source function.'''
    #===============================================================================================
    def __init__(self, F, default_params, to_cache=None):
        '''
        Initialize a PSF model function.
        Parameters
        ----------
        F       :       callable
            Function which evaluates PSF at given location(s).  
            Call sequence is F(p,X,Y) where:
                p - sequence of parameters
                X,Y the grid positions to evaluate for.
        to_cache:       sequence of ints or slice
            The index positions of parameters that will be cached by the StellarFit class.
        default_params:  array-like
            default values of parameters used to build the cache.
            
        Attributes
        ----------
        
        '''
        if not isinstance(F, Callable):
            raise ValueError( 'Please specify callable PSF function' )
        
        self.F = F
        self.Npar = Npar = len(default_params)
        defpar = np.asarray(default_params)
        if to_cache is None:
            ix = np.arange( len(defpar) )
        elif isinstance(to_cache, slice):
           ix = np.arange( *to_cache.indices(Npar) )
        else:
            ix = np.asarray( to_cache )
        
        self.to_cache = ix
        self.no_cache = np.array( get_missing_numbers( np.r_[-1, ix, Npar] ) )  #indeces of uncached params
        self.default_cache = defpar[ix]
    
    #===============================================================================================
    def __call__(self, p, X, Y):
        return self.F(p, X, Y)
    
    #===============================================================================================
    def __repr__(self):
        return self.F.__name__  
    
    #===============================================================================================
    def param_hint(self, data):
        '''Return a guess of the fitting parameters based on the data'''
        
        #location
        y0, x0 = center_of_mass( data )
        z0 = data.max()
       
        #background
        edgefraction = 5/100
        bgix = np.multiply(data.shape, edgefraction).round().astype(int)  #use 5% edge pixels mean as bg estimate
        indices = shape2grid( data.shape-bgix, data.shape+bgix )
        edgepix = data.take( indices, mode='wrap' )
        bg = edgepix.mean()
        
        p = np.empty( self.Npar )
        p[self.no_cache] = x0, y0, z0, bg
        #cached parameters will be set by StellarFit
        return p
    
    #===============================================================================================
    def residuals(self, p, data, X, Y ):
        '''Difference between data and model'''
        return data - self(p, X, Y)

    #===============================================================================================
    def err(self, p, data, X, Y ):
        return abs( self.residuals(p, data, X, Y).flatten() )

        
class GaussianPSF(PSF):
    def __init__(self):
        defpar = (0, 0, 1, .2,0, .2,0)
        PSF.__init__(self, Gaussian2D, defpar, slice(3,6))
        
    #def param_hint(self, data):
        #self.to_cache
        
        
        
######################################################################################################    
class StellarFit(object):
    #TODO: KDE...
    #===============================================================================================
    def __init__(self, psf, algorithm=None):
        '''
        Parameters
        ----------
        psf             :       An instance of the PSF class
        algorithm       :       Algorithm to use for fitting
        '''
        #if not isinstance(psf, PSF):
            #raise ValueError( 'psf must be an instance of the PSF class.' )
        
        if algorithm is None:
            self.algorithm = leastsq
        else:
            raise NotImplementedError
        
        self.psf = psf
        #self.to_cache = psf.to_cache
        self.cache = [psf.default_cache]                  #parameter cache
    
    #===============================================================================================    
    def __call__(self, grid, data):
        '''Fits the PSF model to the data on the grid given the input coordinates xy0'''
        psf = self.psf
        p0 = psf.param_hint( data )
        p0[psf.to_cache] = self.get_params_from_cache()
        Y, X = grid
        args = data, X, Y
        
        plsq, _, info, msg, success = self.algorithm( psf.err, p0, args=args, full_output=1 )
        
        if success != 1:
            print( msg )
            print( info )
        else:
            print( '\nSuccessfully fit {} function to stellar profile.'.format(psf.F.__name__) )
            self.cache.append( plsq[self.psf.to_cache] )
        
        return plsq
    
    #===============================================================================================    
    def get_params_from_cache(self):
        '''return the mean value of the cached parameters.  Useful as initial guess.'''
        return tuple(np.mean( self.cache, axis=0 ))
    
    #===============================================================================================    

######################################################################################################
class PSFPlot(object):
    '''Class for plotting PSF models.'''
    #===============================================================================================
    def __init__(self, fig=None):
        '''
        Initialize grid of 2x3 subplots. Top 3 are 3D wireframe, bottom 3 are colour images of 
        data, fit, residual.
        '''
        ##### Plots for current fit #####
        
        self.fig = fig = fig or plt.figure( figsize=(12,9) )
        gs = gridspec.GridSpec(2, 3)
        fig.suptitle( 'PSF Fitting' )
        titles = ['Data', 'Fit', 'Residual']
        self.plots = []
        for i,g in enumerate(gs):
            if i<3:
                ax = fig.add_subplot(g, projection='3d')
                pl = ax.plot_wireframe( [],[],[] )
                plt.title( titles[i] )
                _,ty = ax.title.get_position( )
                ax.title.set_y(1.1*ty)
            else:
                ax = fig.add_subplot( g )
                pl = ax.imshow([[0]], origin='lower')
            self.plots.append( pl )
            #plt.show( block=False )
    
    #===============================================================================================
    @staticmethod
    def update_segments(X, Y, Z):
        '''Update segments of wireframe plots.'''
        lines = []
        for v in range( X.shape[0] ):
            lines.append( list(zip(X[v,:],Y[v,:],Z[v,:])) )
        for w in range( X.shape[1] ):
            lines.append( list(zip(X[:,w],Y[:,w],Z[:,w])) )
        return lines

    #===============================================================================================
    def update(self, X, Y, Z, data):
        '''update plots with new data.'''
        res = data - Z
        plots = self.plots
        plots[0].set_segments( self.update_segments(X,Y,data) )
        plots[1].set_segments( self.update_segments(X,Y,Z) )
        plots[2].set_segments( self.update_segments(X,Y,res) )
        plots[3].set_data( data )
        plots[4].set_data( Z )
        plots[5].set_data( res )
        
        zlims = [np.min(Z), np.max(Z)]
        for q, pl in enumerate( plots ):
            ax = pl.axes
            ax.set_xlim( [X[0,0],X[0,-1]] ) 
            ax.set_ylim( [Y[0,0],Y[-1,0]] )
            if q<3:
                ax.set_zlim( zlims )
            else:
                pl.set_clim( zlims )
                pl.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
                #plt.colorbar()
        
        self.fig.canvas.draw()
        #TODO: SAVE FIGURES.................

if __name__ == '__main__':
    import pyfits
    
    psf = GaussianPSF()
    fitter = StellarFit( psf )
    plotter = PSFPlot()
    
    filename = '/media/Oceanus/UCT/Observing/data/Feb_2015/0218/ReducedData/20150218.0010.0001.fits'
    Data = pyfits.getdata( filename )
    
    data = Data[40:80, 80:120]
    
    grid = Y, X = shape2grid( data.shape )
    plsq = fitter(grid, data)
    Z = psf( plsq, X, Y )
    plotter.update( X, Y, Z, data )
    
    plt.show()

