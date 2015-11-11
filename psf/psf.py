from collections import Callable

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.measurements import center_of_mass

from matplotlib import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import AxesGrid

from superplot.multitab import MplMultiTab

from magic.list import find_missing_numbers
from magic.dict import TransDict
#from magic.string import banner

from myio import warn

from decor import cache_last_return

from IPython import embed
#from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
from decor import unhookPyQt

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
            The index positions of parameters that will be cached by the StarFit class.
        default_params:  array-like
            default values of parameters used to build the cache.
            
        Attributes
        ----------
        
        '''
        if not isinstance(F, Callable):
            raise ValueError( 'Please specify callable PSF function' )
        
        self.F = F
        self.Npar = Npar = len(default_params)
        self.default_params = defpar = np.asarray(default_params)
        if to_cache is None:
            ix = np.arange( len(defpar) )
        elif isinstance(to_cache, slice):
           ix = np.arange( *to_cache.indices(Npar) )
        else:
            ix = np.asarray( to_cache )
        
        self.to_cache = ix
        self.no_cache = np.array( find_missing_numbers( np.r_[-1, ix, Npar] ) )  #indeces of uncached params
        #self.default_cache = #defpar[ix]
    
    #===============================================================================================
    def __call__(self, p, X, Y):
        return self.F(p, X, Y)
    
    #===============================================================================================
    def __repr__(self):
        return self.F.__name__  
    
    #===============================================================================================
    #@profile()
    def param_hint(self, data):
        '''Return a guess of the fitting parameters based on the data'''
        
        #location
        y0, x0 = np.r_[np.where(data==data.max())]              #center_of_mass( data )
        bg, bgsig = self.background_estimate( data )
        z0 = data.max() - bg

        p = np.empty( self.Npar )
        p[self.no_cache] = x0, y0, z0, bg
        #cached parameters will be set by StarFit
        return p
    
    #===============================================================================================
    @cache_last_return
    def background_estimate(self, data, edgefraction=0.1):
        '''background estimate using window edge pixels'''
        
        shape = data.shape
        bgix = np.multiply(shape, edgefraction).round().astype(int)  #use 10% edge pixels mean as bg estimate
        subset = np.r_[ data[tuple(np.mgrid[:bgix[0], :shape[1]])].ravel(),
                        data[tuple(np.mgrid[:shape[0], :bgix[1]])].ravel() ]
        
        return np.mean(subset), np.std(subset)
        
    #===============================================================================================
    #TODO: memoize!!!!!!!!
    def residuals(self, p, data, X, Y ):
        '''Difference between data and model'''
        return data - self(p, X, Y)

    #===============================================================================================
    def err(self, p, data, X, Y ):
        return abs( self.residuals(p, data, X, Y).flatten() )



######################################################################################################    
def Gaussian2D(p, x, y):
    #TODO: option to pass angle, semi-major, semi-minor; or covariance matrix
    '''Elliptical Gaussian function for fitting star profiles.'''
    x0, y0, z0, a, b, c, d = p
    return z0*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 )) + d


######################################################################################################    
class GaussianPSF( PSF ):
    ''' 7 param 2D Gaussian '''
    #===============================================================================================
    def __init__(self):
        default_params = (0, 0, 1, .2, 0, .2, 0)
        to_cache = slice(3,6)
        PSF.__init__(self, Gaussian2D, default_params, to_cache)
    
    #===============================================================================================
    def integrate(self, p):
        '''
        Analytical solution to the integral over R2,
        source: http://en.wikipedia.org/wiki/Gaussian_integral#n-dimensional_and_functional_generalization
        '''
        _, _, z0, a, b, c, _ = p
        A = 2*np.array( [[a, b], [b, c]] )
        detA = np.linalg.det( A )
        return 2*z0*np.pi / np.sqrt(detA)
    
    #===============================================================================================
    def get_fwhm(self, p):
        '''calculate fwhm from semi-major and semi-minor axes.'''
        a, c = p[3], p[5]
        fwhm_a = 2*np.sqrt(np.log(2)/a)         #FWHM along semi-major axis
        fwhm_c = 2*np.sqrt(np.log(2)/c)         #FWHM along semi-minor axis
        return np.sqrt( fwhm_a*fwhm_c )         #geometric mean of the FWHMa along each axis
        
    #===============================================================================================
    def get_description(self, p, offset=(0,0)):
        '''
        Get a description of the fit based on the parameters.
        Optional offset gives the xy offset of the fit.
        '''
        x, y, z, a, b, c, d = p
        
        fwhm = self.get_fwhm(p)
        counts = self.integrate(p)
        sigx, sigy = 1/(2*a), 1/(2*c)           #standard deviation along the semimajor and semiminor axes
        ratio = min(a,c)/max(a,c)               #Ratio of minor to major axis of Gaussian kernel
        theta = 0.5*np.arctan2( -b, a-c )       #rotation angle of the major axis in sky plane
        ellipticity = np.sqrt(1-ratio**2)
        coo = x+offset[0], y+offset[1]
        
        pdict = {       'coo'           :       coo,
                        'flux'          :       counts,
                        'peak'          :       z+d,
                        'sky_mean'      :       d,
                        'fwhm'          :       fwhm,
                        'sigma_xy'      :       (sigx, sigy),
                        'theta'         :       np.degrees(theta),
                        'ratio'         :       ratio,
                        'ellipticity'   :       ellipticity            }
            
        return pdict
    
    #===============================================================================================
    @staticmethod
    def radial(p, r):
        '''
        Numerically integrate the psf with theta around (x0,y0) to determine mean radial
        profile
        '''
        from scipy.integrate import quad
        from functools import partial
        
        def integrand(psi, r, alpha, beta):
            return np.exp( -r*r*(alpha*np.cos(psi) + beta*np.sin(psi)) )
 
        def mprof(p, r):
            x0, y0, z0, a, b, c, d = p
            alpha = (a-c)/2
            I = quad(integrand, 0, 4*np.pi, args=(r,alpha,b))
            return (0.25/np.pi)*z0*np.exp( -0.5*r*r*(a+c) ) * I[0]
        
        R = partial(mprof, p)
        return np.vectorize(R)(r)
        
    #def param_hint(self, data):
        #'''x0, y0, z0, a, b, c, d'''
        #return self.default_params


#===============================================================================================       
#def Gaussian(p, x):
    #'''Gaussian function for fitting radial star profiles'''
    #A, b, mx = p
    #return A*np.exp(-b*(x-mx)**2 )



#===============================================================================================    
def Moffat(p, x, y):
    x0, y0, z0, a, b, c  = p
    return z0*(1 + ((x-x0)**2 + (y-y0)**2) / (a*a))**-b + c
        
        
        
        
######################################################################################################    
class StarFit(object):
    #TODO: Kernel Density Estimation... or is this too slow??
    #TODO:  cache decor ?????????????
    #===============================================================================================
    def __init__(self, psf=None, algorithm=None, caching=True, hints=True):
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
        self.hints      = hints
        #initialise parameter cache with psf defaults
        self.cache      = []                  #parameter cache
    
    #===============================================================================================    
    #@profile()
    def __call__(self, grid, data):
        '''Fits the PSF model to the data on the grid given the input coordinates xy0'''
        psf = self.psf
        
        p0 = self.get_params_from_cache()       #just returns the default cache for the psf if caching is disabled
        if self.hints:
            p0[psf.no_cache] = psf.param_hint( data )[psf.no_cache]
        
        print('\n\nINIT PARAMS' )
        print( p0, '\n\n' )
        
        #pyqtRemoveInputHook()
        #embed()
        #pyqtRestoreInputHook()
        
        
        Y, X = grid
        args = data, X, Y
        
        print( 'p0', p0 )
        plsq, _, info, msg, success = self.algorithm( psf.err, p0, args=args, full_output=1 )
        
        if success != 1:
            warn( 'FIT DID NOT CONVERGE!' )
            print( msg )
            print( info )
            return
        else:
            print( '\nSuccessfully fit {} function to stellar profile.'.format(psf.F.__name__) )
            if self.caching:
                if len(self.cache):
                    self.cache = np.r_['0,2', self.cache, plsq]     #update cache with these parameters
                    
                else:
                    self.cache = np.r_['0,2', plsq]
                print( '\n\nCACHE!', self.cache )
                print( '\n\n' )
        return plsq
    
    #===============================================================================================    
    def get_params_from_cache(self):
        '''return the mean value of the cached parameters.  Useful as initial guess.'''
        if len(self.cache):
            return np.mean( self.cache, axis=0 )  #The fitting function expects a tuple???
        else:
            return self.psf.default_params
    
    #===============================================================================================    

######################################################################################################
class NullPSFPlot( object ):
    def update(self, *args):
        pass

######################################################################################################
class PSFPlot( object ):
    #TODO: buttons for switching back and forth??
    #MODE = 'update'
    '''Class for plotting / updating PSF models.'''
    #===============================================================================================
    #@profile()
    def __init__(self, fig=None):
        
        self.setup_figure(fig)
        
    #===============================================================================================
    def setup_figure(self, fig=None):
        #TODO: Option for colorbars
        '''
        Initialize grid of 2x3 subplots. Top 3 are 3D wireframe, bottom 3 are colour images of 
        data, fit, residual.
        '''
        ##### Plots for current fit #####
        self.fig = fig = fig or plt.figure( figsize=(12,9), tight_layout=True )
        self.plots, self.images = [], []
        #TODO:  Include info as text in figure??????
        
        #Create the plot grid for the 3D plots
        self.grid_3D = AxesGrid( fig, 211, # similar to subplot(211)
                            nrows_ncols = (1, 3),
                            axes_pad = -0.2,
                            label_mode = None,          #This is necessary to avoid AxesGrid._tick_only throwing
                            share_all = True,
                            axes_class=(Axes3D,{}) )
        
        #Create the plot grid for the images
        self.grid_images = AxesGrid( fig, 212, # similar to subplot(212)
                            nrows_ncols = (1, 3),
                            axes_pad = 0.1,
                            label_mode = "L",           #THIS DOESN'T FUCKING WORK!
                            #share_all = True,
                            cbar_location="right",
                            cbar_mode="each",
                            cbar_size="5%",
                            cbar_pad="0%"  )

        titles = ['Data', 'Fit', 'Residual']
        for ax, title in zip(self.grid_3D, titles):
            pl = ax.plot_wireframe( [],[],[] )
            #set title to display above axes
            title = ax.set_title( title, {'fontweight':'bold'} )
            x,y = title.get_position()
            title.set_position( (x, 1.0) )
            ax.set_axis_bgcolor( 'None' )
            #ax.patch.set_linewidth( 1 )
            #ax.patch.set_edgecolor( 'k' )
            self.plots.append( pl )
        
        
        for i, (ax, cax) in enumerate(zip(self.grid_images, self.grid_images.cbar_axes)):
            im = ax.imshow( np.zeros((1,1)), origin='lower' )
            cbar = cax.colorbar(im)
            #make the colorbar ticks look nice
            cax.axes.tick_params(axis='y', pad=-15, labelcolor='w', labelsize='small')
            [t.set_weight( 'bold' ) for t in cax.axes.yaxis.get_ticklabels()]
            #if i>1:
                #ax.set_yticklabels( [] )       #FIXME:  This kills all ticklabels
            self.images.append( im )
        
        fig.set_tight_layout(True)
        #fig.suptitle( 'PSF Fitting' )                   #TODO:  Does not display correctlt with tight layout
    
    #===============================================================================================
    @staticmethod
    def make_segments(X, Y, Z):
        '''Update segments of wireframe plots.'''
        xlines = np.r_['-1,3,0', X, Y, Z]
        ylines = xlines.transpose(1,0,2)        #swap x-y axes
        return np.r_[xlines, ylines]

    #===============================================================================================
    def update(self, X, Y, Z, data):
        '''update plots with new data.'''
        res = data - Z
        plots, images = self.plots, self.images
        
        plots[0].set_segments( self.make_segments(X,Y,data) )
        plots[1].set_segments( self.make_segments(X,Y,Z) )
        plots[2].set_segments( self.make_segments(X,Y,res) )
        images[0].set_data( data )
        images[1].set_data( Z )
        images[2].set_data( res )
        
        zlims = [Z.min(), Z.max()]
        rlims = [res.min(), res.max()]
        #plims = 0.25, 99.75                             #percentiles
        #clims = np.percentile( data, plims )            #colour limits for data
        #rlims = np.percentile( res, plims )             #colour limits for residuals
        for i, pl in enumerate( plots ):
            ax = pl.axes
            ax.set_zlim( zlims if (i+1)%3 else rlims )
        ax.set_xlim( [X[0,0],X[0,-1]] ) 
        ax.set_ylim( [Y[0,0],Y[-1,0]] )
        
        for i,im in enumerate(images):
            ax = im.axes
            im.set_clim( zlims if (i+1)%3 else rlims )
            #artificially set axes limits --> applies to all since share_all=True in constuctor
            im.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
            
        self.fig.canvas.draw()
        #TODO: SAVE FIGURES.................
        
        
######################################################################################################        
class MultiPSFPlot( MplMultiTab ):
    '''Append each new fit plot as a figure in tab.'''
    #WARNING:  This is very slow!
    #TODO:  ui.show on first successful fit.
    #===============================================================================================
    def update(self, X, Y, Z, data):
        plotter = PSFPlot()
        plotter.update(X, Y, Z, data)
        
        self.add_tab( plotter.fig )
    
######################################################################################################        
class PSFPlotFactory():
    MODES = TransDict({ None       :      NullPSFPlot,
                        'update'   :      PSFPlot,
                        'append'   :      MultiPSFPlot } )
    MODES.add_translations( {False : None, True : 'update'} )

    def __call__(self, mode):
        #if not mode in MODES.allkeys():
            #raise ValueError
            
        c = self.MODES.get(mode, NullPSFPlot)
        return c

psfPlotFactory = PSFPlotFactory()