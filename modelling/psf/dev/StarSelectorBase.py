print( 'Importing modules' )
#Basic
import os
import itertools
import functools
from copy import copy
from string import Template

#Numerical
import numpy as np
import numpy.linalg as la
from scipy.optimize import leastsq

#Plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec, rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D

from imagine import AxesSliders

from ApertureCollections import *
ApertureCollection.WARN = False
SkyApertures.WARN = False

from zscale import zrange

#Parallelism
import multiprocessing as mpc
import ctypes

#Astro stuff
import pyfits
from astropy.wcs import WCS

#Convenience
from misc import *

#Debugging
from misc import make_ipshell
ipshell = make_ipshell()
from embedded_qtconsole import *

#lookup = copy(locals())
#lookup.update( globals() )
#qtshell( lookup )



######################################################################################################    
# Decorators
######################################################################################################    
#needed to make ipython / terminal input prompts work with pyQt
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
def unhookPyQt( func ):
    '''Decorator that removes the PyQt input hook during the execution of the decorated function.
    Used for functions that need ipython / terminal input prompts to work with pyQt.'''
    def unhooked_func(*args, **kwargs):
        pyqtRemoveInputHook()
        out = func(*args, **kwargs)
        pyqtRestoreInputHook()
        return out
    
    return unhooked_func
 
######################################################################################################
# Misc Functions
######################################################################################################    

def Gaussian2D(p, x, y):
    '''Elliptical Gaussian function for fitting star profiles'''
    A, a, b, c, x0, y0 = p
    return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))

def residuals( p, data, X, Y ):
    '''Difference between data and model'''
    return data - Gaussian2D(p, X, Y)

def err( p, data, X, Y ):
    return abs(residuals(p, data, X, Y).flatten() )

#===============================================================================================
def glorify(shared_arr, indeces):
    '''retrieve 2D image array from shared memory given indeces'''
    i, j, k = indeces
    return np.frombuffer(shared_arr.get_obj())[i:i+j*k].reshape((j,k))

#===============================================================================================
def starfucker( args ):
    
    p0, grid, star_im, sky_mean, sky_sigma, win_idx, apiidx, = args
    x0, y0 = p0[-2:]
    Y, X = grid
    args = star_im, X, Y
    
    plsq, _, info, msg, success = leastsq( err, p0, args=args, full_output=1 )
    
    if success!=1:
        print( 'FAIL!!', shidxs[0], win_idx )
        print( msg )
        print( info )
        print( )
        return
    else:
        #print( '\nSuccessfully fit {func} function to stellar profile.'.format(func=func) )
        pdict = StellarFit.get_fit_par( plsq )
        
        skydict = {     'sky_mean'      :       sky_mean,
                        'sky_sigma'     :       sky_sigma,
                        'star_im'       :       star_im,
                        '_window_idx'   :       win_idx          }
        pdict.update( skydict )
        star = Star( **pdict )
        
        il,iu, jl,ju = win_idx
        i = shared_arr_start_idx[apiidx]
        j,k = shared_arr_shapes[apiidx]
        arr = glorify( shared_arr, (i,j,k) )
        arr[il:iu, jl:ju] -= Gaussian2D(plsq, X, Y)


        return star


class StellarFit(object):
    #Currently only implemented for Gaussian2D!
    #===============================================================================================
    FUNCS = ['Gaussian', 'Gaussian2D', 'Moffat']
    #===============================================================================================
    def __init__(self, func='Gaussian2D', alg=None):
        ''' '''
        if alg is None:
            self.algo = leastsq
        else:
            raise NotImplementedError
        
        if func in self.FUNCS:
            self.F = getattr(self, func)
            if func=='Moffat':
                raise NotImplementedError( 'Currently only implemented for Gaussian2D' )
                self.to_cache = slice(1,3)
            elif func=='Gaussian2D':
                self.to_cache = slice(1,4)
        else:
            raise ValueError( 'func must be chosen from amongst {}'.format(self.FUNCS) )
        
        self.cache = []         #parameter cache
    
    #===============================================================================================    
    def __call__(self, xy0, grid, data):
        
        p0 = self.param_hint( xy0, data )
        Y, X = grid
        args = data, X, Y
        
        plsq, _, info, msg, success = self.algo( self.err, p0, args=args, full_output=1 )
        
        if success!=1:
            print( msg )
            print( info )
        else:
            print( '\nSuccessfully fit {func} function to stellar profile.'.format(func=self.F.__name__) )
            self.cache.append( plsq[self.to_cache] )
        
        return plsq
    
    #===============================================================================================    
    #@staticmethod
    def param_hint(self, xy0, data):
        '''initial parameter guess'''
        z0 = np.max(data)
        x0, y0 = xy0
        func = self.F.__name__
        
        if len(self.cache):
            cached_params = self.get_params_from_cache()
            
        elif func=='Moffat':
            cached_params = 1., 2.
        
        elif func=='Gaussian2D':    
            cached_params = 0.2, 0, 0.2
            
        return (z0,) +cached_params+ (x0, y0)
    
    #===============================================================================================    
    def get_params_from_cache(self):
        '''return the mean value of the cached parameters.  Useful as initial guess.'''
        return tuple(np.mean( self.cache, axis=0 ))
    
    #===============================================================================================    
    @staticmethod
    def get_fit_par( plsq ):
        A, a, b, c, x, y = plsq
        
        sigx, sigy = 1/(2*a), 1/(2*c)           #standard deviation along the semimajor and semiminor axes
        fwhm_a = 2*np.sqrt(np.log(2)/a)         #FWHM along semi-major axis
        fwhm_c = 2*np.sqrt(np.log(2)/c)         #FWHM along semi-minor axis
        fwhm = np.sqrt( fwhm_a*fwhm_c )         #geometric mean of the FWHMa along each axis
        
        ratio = min(a,c)/max(a,c)              #Ratio of minor to major axis of Gaussian kernel
        theta = 0.5*np.arctan2( -b, a-c )       #rotation angle of the major axis in sky plane
        ellipticity = np.sqrt(1-ratio**2)
        
        coo = x, y
        
        pdict = {'coo'          :       coo,
                'peak'          :       A,
                'fwhm'          :       fwhm,
                'sigma_xy'      :       (sigx, sigy),
                'theta'         :       np.degrees(theta),
                'ratio'         :       ratio,
                'ellipticity'   :       ellipticity,
                'cache'         :       (a, b, c)                        }
        
        return pdict
        
    #===============================================================================================    
    @staticmethod
    def print_params( pdict ):
        
        print( ('Stellar fit parameters: \nCOO = {coo[0]:3.2f}, {coo[1]:3.2f}'
                                        '\nPEAK = {peak:3.1f}'
                                        '\nFWHM = {fwhm:3.2f}'
                                        '\nTHETA = {theta:3.2f}'
                                        '\nRATIO = {ratio:3.2f}\n'
                                        
                'Sky parameters:         \nMEAN = {sky_mean:3.2f}'
                                        '\nSIGMA = {sky_sigma:3.2f}'
                                        
                ).format( **pdict ) )
    
    #===============================================================================================    
    @staticmethod
    def Gaussian(p, x):
        '''Gaussian function for fitting radial star profiles'''
        A, b, mx = p
        return A*np.exp(-b*(x-mx)**2 )
    
    @staticmethod
    def Gaussian2D(p, x, y):
        '''Elliptical Gaussian function for fitting star profiles'''
        A, a, b, c, x0, y0 = p
        return A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2 ))
    
    @staticmethod
    def Moffat(p, x, y):
        A, a, b, mx, my = p
        return A*(1 + ((x-mx)**2 + (y-my)**2) / (a*a))**-b
    
    
    def residuals(self, p, data, X, Y ):
        '''Difference between data and model'''
        return data - self.F(p, X, Y)

    def err(self, p, data, X, Y ):
        return abs( self.residuals(p, data, X, Y).flatten() )
    

    
#def get_func_name(func):
    #return func.__name__
    
######################################################################################################
# Class definitions
######################################################################################################

####################################################################################################

class AccessRadii(object):
    ATTRS = ['SKY_RADII', 'R_IN', 'R_OUT', 'R_OUT_UPLIM']
    def __getattr__(self, attr):
        if attr in AccessRadii.ATTRS:
            if attr == 'SKY_RADII':
                return SKY_RADII[self.idx]
            if attr == 'R_IN':
                return SKY_RADII[self.idx,0]
            if attr == 'R_OUT':
                return SKY_RADII[self.idx,1]
            if attr == 'R_OUT_UPLIM':
                return np.ceil(SKY_RADII[self.idx,1])
        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, vals):
        if attr in AccessRadii.ATTRS:
            if attr == 'SKY_RADII':
                SKY_RADII[self.idx] = vals
            if attr == 'R_IN':
                SKY_RADII[self.idx,0] = vals
            if attr == 'R_OUT':
                SKY_RADII[self.idx,1] = vals
            if attr == 'R_OUT_UPLIM':
                print( 'NO!' )
        else:
            return object.__setattr__(self, attr, vals)


#################################################################################################################################################################################################################          
    
class Star( object ):   #ApertureCollection
    '''
    '''
    ATTRS = ['coo', 'peak', 'flux', 'fwhm', 'sigma_xy', 'ratio', 'ellipticity', 
                'sky_mean', 'sky_sigma', 'star_im', 'id', '_window_idx', 'cache']
    #===============================================================================================
    def __init__(self, **params):
        
        #check = params.pop('check') if 'check' in params else 1
        #apertures = params.pop('apertures') if 'apertures' in params else None
        
        for key in self.ATTRS:
            setattr( self, key, params[key] if key in params else None )           #set the star attributes given in the params dictionary
        
        #apertures = apertures if apertures!=None else [1.5*self.fwhm if self.fwhm else None]
        #super( Star, self ).__init__( coords=self.coo, radii=apertures, check=check )
        
        #drp = params.pop('rad_prof') if 'rad_prof' in params else 1     #default is to compute the radial profile
        #drp = 0 if self.star_im is None else drp                        #if no stellar image is supplied don't try compute the radial profile
        
        if not self.star_im is None:
            self.flux = self.star_im.sum()
            self.rad_prof = self.radial_profile( )                       #containers for radial profiles of fit, data, cumulative data
    
    #===============================================================================================
    def __str__(self):
        info = Table( vars(self), 
                        title='Star No. {}'.format(self.id), title_props={'bg':'blue'},
                        col_headers='Fit params', 
                        ignore_keys=('star_im', '_window_idx', 'rad_prof', 'cache') ) 
        return str(info)
    
    #===============================================================================================
    def get_params(self):
        raise NotImplementedError
    
    #===============================================================================================
    def set_params(self, dic):
        raise NotImplementedError
    
    #===============================================================================================
    def radial_profile( self ):
        
        #im_shape = np.array( self.star_im.shape, ndmin=3 ).T
        il,iu, jl,ju = self._window_idx                           #position of pixel window in main image
        pix_cen = np.mgrid[il:iu, jl:ju]  + 0.5                   #the index grid for pixel centroids
        staridx = tuple(reversed(self.coo))                     #star position in pixel coordinates
        rfc = np.sqrt(np.sum(np.square(pix_cen.T-staridx), -1))    #radial distance of pixel centroid from star centoid
        rmax = SkyApertures.R_OUT_UPLIM + 1                       #maximal radial distance of pixel from image centroid
        
        return np.array([ np.mean( self.star_im[(rfc>=r-1)&(rfc<r)] ) for r in range(1, rmax) ])


def w2b(artist):
    ca = artist.get_colors()
    w = np.all(ca == colorConverter.to_rgba_array('w'), 1 )
    ca[w] = colorConverter.to_rgba_array('k')
    artist.set_color( ca )

    
    
    
#######################################################################################################    
class MeanStar( Star, AccessRadii ):   #ApertureCollection
    #===============================================================================================     
    def __init__( self, idx, window ):
        self.idx = idx
        self.window = window
        fakecoo = np.tile( window, (4,2))
        
        Star.__init__( self, coo=fakecoo, check=0 )      #won't calculate rad_prof
        self.apertures = ApertureCollection( radii=np.zeros(4), coords=fakecoo, 
                                                colours=['k','w','g','g'], 
                                                ls=['dotted','dotted','solid','solid'])
        
        self.rad_prof = [[], [], []]               #containers for radial profiles of fit, data, cumulative data
        self.has_plot = False
        #self.init_plots
        
    #=============================================================================================== 
    #@profile()
    def init_plots( self, fig ):
        #print('!'*88, '\ninitializing meanstar plots\n', '!'*88, )
        
        ##### Plots for mean profile #####
        self.mainfig = fig
        _, self.ax_zoom, self.ax_prof = filter(lambda i: isinstance(i, mpl.axes.Subplot), fig.axes)
  
        self.pl_zoom = self.ax_zoom.imshow([[0]], origin='lower', cmap = 'gist_heat')  #, animated=True #vmin=self.zlims[0], vmax=self.zlims[1]
        self.colour_bar = fig.colorbar( self.pl_zoom, ax=self.ax_zoom )
        
        ##### stellar profile + psf model #####
        labels = ['', 'Cumulative', 'Model']
        self.pl_prof = self.ax_prof.plot( 0,0,'g-', 0,0,'r.', 0,0, 'bs' )       #, animated=True
        [pl.set_label(label) for pl,label in zip( self.pl_prof, labels )]
        
        ##### apertures #####
        lines = self.apertures.convert_to_lines( self.ax_prof )
        w2b(lines)      #convert white lines to black for displaying on white canvas
        #lines.set_animated( True )
        
        ##### sky fill #####
        from matplotlib.patches import Rectangle
        
        trans = self.apertures.line_transform
        self.sky_fill = Rectangle( (0,0), width=0, height=1, transform=trans, color='b', alpha=0.3)     #, animated=True
        self.ax_prof.add_artist( self.sky_fill )

        self.ax_prof.add_collection( lines )
        self.apertures.axadd( self.ax_zoom )
        self.has_plot = True
        
        #set all the artist invisible to start with - we'll create a background from this for blitting
        #self.set_visible( False )
        
    #===============================================================================================
    @unhookPyQt
    def update( self, cached_params, data_profs ):
        
        print( 'meanstar update' )
        
        window = self.window
        Y, X = np.mgrid[:2*window, :2*window] + 0.5
        p = (1,) +cached_params+ (window, window)               #update with mean of cached_params
        
        pdict = StellarFit.get_fit_par( p )                     #TODO: set_params
        
        self.fwhm = fwhm = pdict['fwhm']
        b = 4*np.log(2) / fwhm / fwhm                           #derive the kernel width parameter for the 1D Gaussian radial profile from the geometric mean of the FWHMa of the 2D Gaussian
        
        self.star_im = StellarFit.Gaussian2D( p, X, Y )
        self.rad_prof[0] = StellarFit.Gaussian( (1., b, 0.), np.linspace(0, window) )
        self.rad_prof[1:] = data_profs
        
        self.apertures.radii = [0.5*fwhm, 1.5*fwhm, self.R_IN, self.R_OUT]
        
        #print( '^'*88 )
        #print( self.rad_prof )
        #print( list(map(len, self.rad_prof)) )
        #print( self.apertures )
        #print( self.apertures.coords )
        
        
        
    #===============================================================================================
    @unhookPyQt
    def update_plots(self):
        
        Z = self.star_im
        self.pl_zoom.set_data( Z )
        self.pl_zoom.set_clim( 0, 1 )
        self.pl_zoom.set_extent( [0, 2.*self.window, 0, 2.*self.window] )
        
        self.ax_prof.set_xlim( 0, self.R_OUT_UPLIM + 0.5 )
        
        rpx = np.arange( 0, self.window )
        rpxd = np.linspace( 0, self.window )
        
        for pl, x, y in zip( self.pl_prof, [rpxd, rpx, rpx], self.rad_prof ):
            pl.set_data( x, y )
        
        self.update_aplines()
        
    #===============================================================================================
    def update_aplines(self):
        ##### set aperture line position + properties #####
        self.apertures.update_aplines()
        
        #print( '*'*88 )
        #print( self.apertures )
        #print( self.apertures.aplines )
        
        ##### Shade sky region #####
        sky_width = np.ptp( self.SKY_RADII )
        self.sky_fill.set_x( self.R_IN )
        self.sky_fill.set_width( sky_width )
        
        ##### Update figure texts #####
        #plot_texts = self.plot_texts
        #bb = plot_texts[2].get_window_extent( renderer=self.mainfig.canvas.get_renderer() )
        #bb = bb.transformed( self.ax_prof.transAxes.inverted() )
        #sky_txt_offset = 0.5*sky_width - bb.width
        #offsets = 0.1, 0.1, sky_txt_offset, 0
        #xposit = np.add( radii, offsets )
        #y = 1.0
        #for txt, x in zip( plot_texts, xposit ):
            #txt.set_position( (x, y) )
    
    #===============================================================================================
    def set_visible(self, state):
        self.apertures.set_visible( state )
        self.apertures.aplines.set_visible( state )
        self.sky_fill.set_visible( state )
        for l in self.pl_prof:
            l.set_visible( state )
    
    #===============================================================================================
    def draw(self):
        axz, axp = self.ax_zoom, self.ax_prof
        axz.draw_artist( self.pl_zoom )
        axz.draw_artist( self.apertures )
        
        axp.draw_artist( self.sky_fill )
        axp.draw_artist( self.apertures.aplines )
        for l in self.pl_prof:
            axp.draw_artist( l )
    
#######################################################################################################    
                
class Stars( AccessRadii ):
    '''
    Class to contain measured values for selected stars in image.
    '''
    #===============================================================================================
    def __init__(self, idx,  **kw):
        
        self.idx = idx
        
        fwhm = kw.pop('fwhm')           if 'fwhm' in kw         else []
        coords = kw.pop('coords')       if 'coords' in kw       else []
        self.window = kw.pop('window')  if 'window' in kw       else SkyApertures.R_OUT_UPLIM
        
        self.stars = [Star(coo=coo, fwhm=fwhm) 
                        for (coo,fwhm) in itertools.zip_longest(coords, as_iter(fwhm)) ]
        self.star_count = len(self.stars)
        
        self.meanstar = MeanStar( idx, self.window )
        self.plots = []
        self.annotations = []
        
        if 'psfradii' in kw:
            psfradii = kw.pop('psfradii')
        else:
            psfradii = np.tile( fwhm, (2,1)).T * [0.5, 1.5]
        
        if 'skyradii' in kw:
            skyradii = kw.pop('skyradii')
        else:
            skyradii = np.tile( self.SKY_RADII, (len(fwhm),1) )
        
        apcoo = np.array( interleave( coords, coords ) )                #each stars has 2 skyaps and 2 psfaps!!!!
        self.psfaps = ApertureCollection( coords=apcoo, radii=psfradii, ls=':', gc=['k','w'], picker=False, **kw)
        self.photaps = ApertureCollection( gc='c' )
        self.skyaps = SkyApertures( radii=skyradii, coords=apcoo, **kw )
        
        self.has_plot = 0
        
    #===============================================================================================
    def __str__(self):
        attrs = copy(Star.ATTRS)
        ignore_keys = 'star_im', '_window_idx', 'rad_prof'              #Don't print these attributes
        [attrs.pop(attrs.index(key)) for key in ignore_keys if key in attrs]
        data = [self.pullattr(attr) for attr in attrs]
        table = Table( data, 
                        title='Stars', title_props={'text':'bold', 'bg':'light green'},
                        row_headers=attrs )
        
        #apdesc = 'PSF', 'PHOT', 'SKY'
        #rep = '\n'.join( '\t{} Apertures: {}'.format( which, ap.radii ) 
                    #for which, radii in zip(apdesc, self.get_apertures()) )
        return str(table)
    
    #===============================================================================================
    #def __repr__(self):
        #return str(self)
    
    #===============================================================================================    
    def __len__(self):
        return len(self.stars)
    
    #===============================================================================================    
    def __getitem__(self, key):
        return self.stars[key]
    
    #===============================================================================================    
    def __getattr__(self, attr):
        if attr in ('fwhm', 'sky_sigma', 'sky_mean'):
            return self.get_mean_val( attr )
        else:
            return super(Stars, self).__getattr__(attr)
    
    #===============================================================================================
    def pullattr(self, attr):
        return [getattr(star, attr) for star in self]
    
    #===============================================================================================
    def get_apertures(self):
        return [self.psfaps, self.photaps, self.skyaps]
    
    #===============================================================================================
    def append(self, star=None, **star_params):
        
        if star_params:
            star_params['id'] = self.star_count
            star = Star( **star_params )             #star parameters dictionary passed not star object
        
        if star.id is None:
            star.id = self.star_count
            
        self.stars.append( star )
        #print()
        #print('UPDATING PSFAPS')
        coo, fwhm = star.coo, star.fwhm
        
        if not np.size(self.psfaps.coords):
            coo = [coo]
        
        self.psfaps.append( coords = coo, radii = [0.5*fwhm, 1.5*fwhm], picker=False, ls=':', gc=['k','w'] )
        #print()
        #print('UPDATING SKYAPS')
        self.skyaps.append( coords = coo, radii = self.SKY_RADII )
        #self.photaps.append( something )
        
        ax = self.psfaps.axes
        txt_offset = 1.5*fwhm
        coo = np.squeeze(coo)
        anno = ax.annotate( str(self.star_count), coo, coo+txt_offset, color='w',size='small') #transform=self.ax.transData
        self.annotations.append( anno )
        
        self.star_count += 1
        
        return star
    
    #===============================================================================================
    def get_mean_val(self, key):
        if not len(self):
            #raise ValueError( 'The {} instance is empty!'.format(type(self)) )
            return None
        if not hasattr( self[0], key ):
            raise KeyError( 'Invalid key: {}'.format(key) )
        vals = [getattr(star,key) for star in self]
        if not None in vals:
            return np.mean( vals, axis=0 )
        
    #===============================================================================================
    def get_unique_coords(self):
        '''get unique coordinates'''
        nq_coords = self.psfaps.coords          #non-unique coordinates (possibly contains duplicates)
        _, idx = np.unique( nq_coords[:,0], return_index=1 )
        _, coords = sorter( list(idx), nq_coords[idx] )
        return np.array(coords)                   #filtered for duplicates
    
    coords = property(get_unique_coords)
    
    #===============================================================================================
    def axadd( self, ax ):
        #self.skyaps.axadd( ax )
        for aps in self.get_apertures():
            aps.axadd( ax )
    
    #===============================================================================================
    #def draw(self):
        #ax.draw_artist( stars.psfaps )
        #ax.draw_artist( stars.skyaps )
        #ax.draw_artist( stars.annotations[-1] )
    
    #===============================================================================================
    def remove(self, idx):
        
        self.psfaps.remove( idx )
        self.skyaps.remove( idx )
        #self.photaps.remove( idx )
        
        idx = idx[0][0]
        self.stars.pop( idx )
        rtxt = self.annotations.pop(idx)        
        rtxt.set_visible( False )
        n = int(rtxt.get_text())
        rtxt.set_text(None)
        
        #reidentify current stars
        for i, star in enumerate(self[idx:], idx):      #enumerate, but start at idx
            star.id = i
            self.annotations[i].set_text( str(i) )
        
        self.star_count -= 1
    
    #===============================================================================================
    def remove_all( self ):
        print( 'Attempting remove all on', self )
        while len(self):        
            print( self )
            self.remove([[0]])
    
    #===============================================================================================
    #@profile()
    def init_plots(self, mainfig):
        '''
        '''
        plot('!'*88, '\ninitializing stars plots\n', '!'*88, )
        ##### Plots for current fit #####
        fig = self.fitfig = plt.figure( figsize=(12,9) )
        gs = gridspec.GridSpec(2, 3)
        fig.suptitle('PSF Fitting')
        titles = ['Data', 'Fit', 'Residual']
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
            plt.show( block=False )
        
        self.has_plot = 1
        
        self.meanstar.init_plots( mainfig )
    
    #===============================================================================================
    @staticmethod
    def update_segments(X, Y, Z):
        lines = []
        for v in range( len(X) ):
            lines.append( list(zip(X[v,:],Y[v,:],Z[v,:])) )
        for w in range( len(X) ):
            lines.append( list(zip(X[:,w],Y[:,w],Z[:,w])) )
        return lines
    
    #===============================================================================================
    def update_plots(self, X, Y, Z, data):
        res = data - Z
        self.plots[0].set_segments( self.update_segments(X,Y,data) )
        self.plots[1].set_segments( self.update_segments(X,Y,Z) )
        self.plots[2].set_segments( self.update_segments(X,Y,res) )
        self.plots[3].set_data( data )
        self.plots[4].set_data( Z )
        self.plots[5].set_data( res )
        
        zlims = [np.min(Z), np.max(Z)]
        for q, pl in enumerate( self.plots ):
            ax = pl.axes
            ax.set_xlim( [X[0,0],X[0,-1]] ) 
            ax.set_ylim( [Y[0,0],Y[-1,0]] )
            if q<3:
                ax.set_zlim( zlims )
            else:
                pl.set_clim( zlims )
                pl.set_extent( [X[0,0], X[0,-1], Y[0,0], Y[-1,0]] )
                #plt.colorbar()
        self.fitfig.canvas.draw()
        #SAVE FIGURES.................
        
    #=============================================================================================== 
    def mean_radial_profile( self ):
        
        rp = [star.rad_prof for star in self]                    #radial profile of mean stellar image\
        rpml = max(map(len, rp))
        rpa = np.array( [np.pad(r, (0,rpml-len(r)), 'constant', constant_values=-1) for r in rp] )
        rpma = np.ma.masked_array( rpa, rpa==-1 )
        
        rpm = rpma.mean( axis=0 );      rpm /= np.max(rpm)       #normalized mean radial data profile
        cpm = np.cumsum( rpm )   ;      cpm /= np.max(cpm)       #normalized cumulative
        
        return rpm, cpm
        
def setup_figure_geometry(self):
    ''' Set up axes geometry '''
    fig = Figure( figsize=(8,10) )     #fig.set_size_inches( 12, 12, forward=1 )

    gs1 = gridspec.GridSpec(2, 1, height_ratios=(3,1))
    ax = fig.add_subplot( gs1[0,0], aspect='equal' )
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes('right', size=0.2, pad=0.05)
    sax = divider.append_axes('right', size=0.2, pad=0.2)
    #cax = fig.add_subplot( gs1[0,1] )
    #sax = fig.add_subplot( gs1[0,2] )

    gs2 = gridspec.GridSpec(2, 2, width_ratios=(1,2), height_ratios=(3,1) )
    ax_zoom = fig.add_subplot( gs2[1,0], aspect='equal' )
    ax_zoom.set_title( 'PSF Model' )

    ax_prof = fig.add_subplot( gs2[1,1] )
    ax_prof.set_title( 'Mean Radial Profile' )
    ax_prof.set_ylim( 0, 1.1 )
    
    fig.tight_layout()
    #gs1.tight_layout(fig)
    #gs2.tight_layout(fig)

    return fig 
    

####################################################################################################    
class ApertureInteraction( AccessRadii ):
    
    SAMESIZE = True
    msg = ( 'Please select a few stars (by clicking on them) for measuring psf and sky brightness.\n ' 
        'Right click to resize apertures.  Middle mouse to restart selection.  Close the figure when you are done.\n\n' )
    
    #===============================================================================================
    #@profile()
    def __init__(self, idx, fig=None, **kwargs):
        
        self.idx = idx
        self.figure = fig
        self.ax = fig.axes[0] if fig else None
       
        if 'window' not in kwargs:      window = self.R_OUT_UPLIM
        self.window = window
        self.cbox = cbox                if 'cbox' in kwargs          else 12
        
        self.stars = Stars( idx, window=window )
        self.zoom_pix = np.mgrid[:2*window, :2*window]                         #the pixel index grid
            
        self.selection = None                   # allows artist resizing only when an artist has been picked
        self.status = 0                         #ready for photometry??
        
        self._write_par = 1                     #should parameter files be written

        self.cid = []                           #connection ids
        
        self.fit = StellarFit()
    
    #===============================================================================================
    def __len__(self):
        return len(self.stars)
    
    #===============================================================================================
    #@profile()
    def load_image(self, filename=None, data=None, use_wcs=False):
        '''
        Load the image for display, applying IRAF's zscale algorithm for colour limits.
        '''
        print( 'Loading image {}'.format(filename) )
        
        ##### Load the image #####
        if filename and data is None:
            self.filename = filename
            data, header = pyfits.getdata( filename, header=1 )
            self.image_data, self.image_header = data, header
            
        elif not data is None:
            self.image_data = data
            self.image_header = None
            
        else:
            raise ValueError( 'Please provide either a FITS filename, or the actual data' )
        
        #if WCS:            data = np.fliplr( data )
        self.image_data_cleaned = copy(data)
        r,c =  self.image_shape = data.shape
        self.pixels = np.mgrid[:r, :c]                      #the pixel grid

        ##### Plot star field #####
        self.zlims = zlims = (0,1)#zrange(data, contrast=1./99, sigma_clip=5., maxiter=5 )
        
        fig, ax = self.figure, self.ax
        cax, sax = filter(lambda i: not isinstance(i, mpl.axes.Subplot), fig.axes)
        self.image = ax.imshow( data, origin='lower', cmap = 'spectral', vmin=zlims[0], vmax=zlims[1])                  #gist_earth, hot
        self.colour_bar = fig.colorbar( self.image, cax )
        self.CreateSliders( sax )
        
        #ax.set_title( 'PSF Measure' )                                                                                    #PRINT TARGET NAME AND COORDINATES ON IMAGE!! DATE, TIME, ETC.....
        #self.stars.zlims = zlims
        
        ##### Add ApertureCollection to axes #####
        self.stars.axadd( ax )
        
        #create the canvas backgrounds for blitting
        #canvas = FigureCanvas(self.figure)
        #self.backgrounds = [canvas.copy_from_bbox( ax.bbox ) for ax in fig.axes]
        
        #initialize meanstar plots
        self.stars.meanstar.init_plots( fig )
    
    #===============================================================================================
    def CreateSliders(self, sax):
        
        imin, imax = self.image_data.min(), self.image_data.max()
        sax.set_ylim((imin, imax))
        #sax.set_yscale( self.sscale )
        sax.patch.set_visible(0)
        sax.yaxis.set_tick_params(direction='inout', left='off', labelright=1, labelleft=0)
        [spine.set_visible(0) for where, spine in sax.spines.items() if where!='right']
        sax.set_xticks([])
        
        self.sliders = AxesSliders( sax, *self.zlims, slide_on='y' )
        
        def set_clim(self, clims):
            self.image.set_clim(clims)
            self.figure.canvas.draw() 
        
        self.sliders.on_changed( set_clim )
    
    #===============================================================================================
    def plotArrows(self, ax):
        '''plot NE arrows'''
        apos, alen = 0.02, 0.15                     #arrow position and length as fraction of image dimension
        x, y = (1-apos)*r, apos*c                   #arrow coordinates
        aclr = 'g'
        
        ax.arrow( x, y, -alen*r, 0, color=aclr, width=0.1 )
        ax.text( x-alen*r, 2*apos*c, 'E', color=aclr, weight='semibold' )
        ax.arrow( x, y, 0, alen*c, color=aclr, width=0.1 )
        ax.text( (1-3*apos)*r, y+alen*c, 'N', color=aclr, weight='semibold' )
    
    #===============================================================================================    
    def connect(self, msg='', **kwargs):
        '''
        Connect to the figure canvas for mouse event handeling.
        '''
        print(msg)
        #self.mode = kwargs['mode']          if 'mode' in kwargs         else 'fit'
        self.cid.append( self.figure.canvas.mpl_connect('button_press_event', self.select_stars) )
        #self.cid.append( self.figure.canvas.mpl_connect('close_event', self._on_close) )
        
        self.cid.append( self.figure.canvas.mpl_connect('pick_event', self._on_pick) )
        self.cid.append( self.figure.canvas.mpl_connect('button_release_event', self._on_release) )
        self.cid.append( self.figure.canvas.mpl_connect('motion_notify_event', self._on_motion) )
        
        self.cid.append( self.figure.canvas.mpl_connect('key_press_event', self._on_key) )             #IF MODE==SELECT --> A=SELECT ALL
        
        #self.sliders.connect()
        
        #self.apertures.connect( self.figure )    
    
    #===============================================================================================    
    def disconnect(self):
        '''
        Disconnect from figure canvas.
        '''
        for cid in self.cid:
            self.figure.canvas.mpl_disconnect( cid )
        print('Disconnected from figure {}'.format(self.figure) )
        
    #====================================================================================================
    def load_coo( self, fn ):
        '''load star cordinates from file and plot on image.'''
        
        #print( 'Loading coordinates from {}.'.format(fn) )
        
        coords, ids = get_coords_from_file( fn )
        coords = self.map_coo( coords )
        
        return coords
        #ipshell()
        ##print( coords )
        
        ##eliminate coordinate duplication (don't replot apertures that are on image already)
        ##also, daofind sometimes gives erroneous double stars
        #if len(self.stars.coords):
            #ldup = undup_coo( coords, self.stars.coords, fwhm )
            #coords = coords[~ldup]
            #ids = ids[~ldup]
        
        #txt_offset = 1.5*fwhm
        #for *coo, idd in zip(coords, ids):
            #if idd==1:          coo = [coo]     #ApertureCollection needs to know to anticipate extending horizontally
            
            #star = self.stars.append( coo=coo[:], fwhm=fwhm, id=idd )         #plots the star apertures on the image assuming the previously measured fwhm for all stars (average)
            ##NOTE: you should still do the fitting in the background????
            
            ##coo = np.squeeze(coo)
            ##anno = self.ax.annotate(str(int(idd)), coo, coo+txt_offset, color='w',size='small') #transform=self.ax.transData
            ##self.stars.annotations.append( anno )
            ##always add the figure text for id!
        
        ##Aperture checks + autocolour
        #self.auto_colour()
        
        ##(Re)plot apertures
        #psfaps, _, skyaps = self.stars.get_apertures()
        #psfaps.axadd( self.ax )
        #skyaps.axadd( self.ax )
        #self.update_legend( )
       
        #self.figure.canvas.draw()
        
        #self.coo_loaded = 1
    
    #===============================================================================================
    def auto_colour(self):
        psfaps, _, skyaps = self.stars.get_apertures()
        skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
        
    #====================================================================================================
    def map_coo( self, coords=None, use_wcs=1 ):
        ''' Apply coordinate mapping to render in coordinates with N up and E left. 
            This is also the inverse mapping'''
        if coords is None:
            coords = self.stars.coords
        else:
            coords = np.array( coords, ndmin=2 )
        if use_wcs:
            #print( 'WM'*100 )
            #print( coords )
            coords[:,0] = self.image_data.shape[1] - coords[:,0]
            #print( 'WM'*100 )
            #print( coords )
        return np.squeeze( coords )
    
    #===============================================================================================
    def gen_phot_ap_radii( self, rmax=None, naperts=None, string=1 ):
        '''
        Generate list of parabolically increasing aperture radii.
        '''
        rmax = self.R_OUT_UPLIM if rmax is None else rmax
        
        naperts = naperts if naperts else Phot.NAPERTS  
        aps = np.linspace(1, rmax, naperts)
        #aps =  np.polyval( (0.085,0.15,2), range(naperts+1) )            #apertures generated by parabola
        if string:
            aps = ', '.join( map(str, np.round(aps,2)) )
        return aps    
    
    
    #===============================================================================================
    def resize_selection( self, mouseposition ):
        apertures, idx = self.selection.artist, self.selection.idx
        rm = apertures.edge_proximity( mouseposition, idx )       #distance moved by mouse
        
        if self.SAMESIZE:
            if len(idx)>1:
                idx = ..., idx[1]      #indexes all apertures of this size 
        
        apertures.resize( rm, idx )
        
    #===============================================================================================
    def _on_release(self, event):
        '''
        Resize and deselct artist on mouse button release.
        '''
        
        if self.selection:
            #print( '**\n'*3, 'Releasing', self.selection.artist.radii )
            self.resize_selection( (event.xdata, event.ydata) )
            
            self.figure.canvas.draw()
        
        self.selection = None # the artist is deselected upon button released
        
    #===============================================================================================    
    def _on_pick(self, event):
        '''
        Select an artist (aperture)
        '''
        self.selection = event
    
    #===============================================================================================    
    def _on_motion(self, event):
        '''
        Resize aperture on motion.  Check for validity and colourise accordingly.
        '''
        if event.inaxes!=self.image.axes:
            return
        
        if self.selection:
            self.resize_selection( (event.xdata, event.ydata) )
            
            apertures, idx = self.selection.artist, self.selection.idx
            psfaps, photaps, skyaps = self.stars.get_apertures()
            skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
            #skyaps.auto_colour()
            
            if self.SAMESIZE:
                idx = ... if len(self.stars)==1 else 0
                self.SKY_RADII = skyaps.radii[idx].ravel()

            #### mean star apertures #####
            meanstar = self.stars.meanstar
            meanstar.apertures.radii[-2:] = self.SKY_RADII
            #self.stars.meanstar.radii = radii
            meanstar.apertures.auto_colour( check='sky' )
            #meanstar.apertures.auto_colour()
            meanstar.update_aplines( )
            
            #self.update_legend()
            self.figure.canvas.draw()
    
    #=============================================================================================== 
    def _on_key(self, event):
        print( 'ONKEY!!' )
        print( repr(event.key) )
        
        if event.key.lower()=='d':
            
            #print( 'psfaps.radii', self.stars.psfaps.radii )
            prox = self.stars.psfaps.center_proximity( (event.xdata, event.ydata) )
            
            #print( 'prox', prox )
            idx = np.where( prox==np.min(prox) )
            
            #print( 'where', np.where( prox==np.min(prox) ) )
            
            star = self.stars[idx[0][0]]
            
            try:
                il,iu, jl,ju = star._window_idx
                im = star.star_im
            except TypeError:
                #if the stars where found by daofind, the _window_idx would not have been determined!
                #Consider doing the fits in a MULTIPROCESSING queue and joining here...
                print( 'star.coo', star.coo )
                x, y = np.squeeze(star.coo)
                j,i = int(round(x)), int(round(y))
                il,iu ,jl,ju, im = self.zoom(i,j)
                
            self.image_data_cleaned[il:iu, jl:ju] = im                     #replace the cleaned image data section with the original star image
            
            self.stars.remove( idx )
            if len(self.stars):
                self.stars.skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=self.stars.psfaps )
            
            self.figure.canvas.draw()
    
    #===============================================================================================
    def update_legend(self):
        
        markers = []; texts = []
        #print( 'WTF~??!! '*8 )
        #print( self.stars[0].apertures )
        for ap in self.stars[0].apertures:           #CURRENT SELECTION?????????????
            mec = ap.get_ec()
            #print( 'mec' )
            #print( mec )
            mew = ap.get_linewidth()
            marker = Line2D(range(1), range(1), ls='', marker='o', mec=mec, mfc='none', mew=mew, ms=15)
            text = str( ap )
            markers.append( marker )
            texts.append( text )
            
        leg = self.figure.legend( markers, texts, loc='lower center', 
                fancybox=True, shadow=True, ncol=3, numpoints=1  )
    
    #bbox_to_anchor=(0.5,-0.25),
           
            
    #===============================================================================================    
    def snap2peak(self, x, y, offset_tolerance=10):
        '''
        Snap to the peak value within the window.
        Parameters
        ----------
        x, y :                  input coordinates (mouse position)
        offset_tolerance :      the size of the image region to consider. i.e offset tolerance
        
        Returns
        -------
        xp, yp, zp :            pixel coordinates and value of peak
        '''
        j,i = int(round(x)), int(round(y))
        il,_ ,jl,_, im = self.zoom(i,j, offset_tolerance)
        
        zp = np.max(im)
        ip, jp = np.where(im==zp)                                    #this is a rudamnetary snap function NOTE: MAY NOT BE SINGULAR.
        yp = ip[0]+il; xp = jp[0]+jl                                      #indeces of peak within sub-image
        
        return xp, yp, zp
    
    #===============================================================================================
    def snap(self, x, y, peak=1, threshold=5., edge_cutoff=5., offset_tolerance=10, pr=0):
            
        r,c = self.image_shape
        
        if peak:
            xp, yp, zp = self.snap2peak(x, y, offset_tolerance)
        else:
            xp, yp, zp = x, y, self.image_data[x,y]
            
        j, i = xp, yp
        
        #check threshold
        if not threshold is None:
            sky_mean, sky_sigma = self.measure_sky(i,j, pr=pr)
            if zp - sky_mean < threshold*sky_sigma:
                if pr:          print('Probably not a star!')
                return None, None
        
        #check edge proximity
        if not edge_cutoff is None:
            if any(np.abs([xp, xp-c, yp, yp-r]) < edge_cutoff):
                if pr:          print('Too close to image edge!')
                return None, None
        
        #check if known
        if not offset_tolerance is None:
            xs, ys = self.snap2known(xp, yp, offset_tolerance=offset_tolerance)
            if xs and ys:
                if pr:          print('Duplicate selection!')
                return None, None
            
        return xp, yp
            
    #===============================================================================================
    def snap2known(self, x, y, known_coo=None, offset_tolerance=10):
        
        if not known_coo:                   known_coo = self.stars.coords
        if not len(known_coo):
            return None, None

        if not offset_tolerance:            offset_tolerance = self.window
        
        known_coo = np.array(known_coo)
        rs = list( map(la.norm, known_coo - (x,y)) )                                              #distance from user input and known stars
        l_r = np.array(rs) < offset_tolerance

        if any(l_r):
            ind = np.argmin( rs )                                                                     #fin star that has minimal radial distance from selected point
            return known_coo[ind]
        else:
            print('No known star found within %i pixel radius' %offset_tolerance)
            return None, None
    
    #===============================================================================================
    #TODO:  REMOVE PIXELS FOR ANY STARS WITHIN THE SKY REGION
    def get_sky(self, i, j):                            #COMBINE THIS ALGORITHM WITH THE RAD PROF ALG FOR EFFICIENCY
        Yp, Xp = self.pixels + 0.5                      #pixel centroids
        Rp = np.sqrt( (Xp-j)**2 + (Yp-i)**2 )

        sky_inner, sky_outer = self.SKY_RADII
        l_sky = np.all( [Rp > sky_inner, Rp < sky_outer], 0 )
        return l_sky
    
    #===============================================================================================
    def measure_sky(self, i, j, pr=1):                           #WARN IF STARS IN SKY ANNULUS!!!!!
        '''Measure sky mean and deviation.'''
        if pr:
            print( 'Doing sky measurement...' )
        
        l_sky = self.get_sky(i, j)
        sky_im = self.image_data_cleaned[l_sky]
       
        sky_mean = sky_im.mean()
        sky_sigma = sky_im.std()

        return sky_mean, sky_sigma
    
    #===============================================================================================
    def zoom(self, i, j, window=None, use_clean=1):
        '''limits indeces to fall within image and returns zoomed section'''
        r,c = self.image_shape
        window = window if window else self.window
        il,iu = i-window, i+window
        jl,ju = j-window, j+window
        if il<0: il=0; iu = 2*window
        if iu>r: iu=r; il = r-2*window
        if jl<0: jl=0; ju = 2*window
        if ju>c: ju=c; jl = c-2*window
        
        il,iu, jl,ju = np.round((il,iu, jl,ju))
        
        if use_clean:
            star_im = self.image_data_cleaned[il:iu, jl:ju]
        else:
            star_im = self.image_data[il:iu, jl:ju]
        return il,iu, jl,ju, star_im
    
    #===============================================================================================
    #def radial_profile( self, coo, rmax=self.window ):
        #'''
        #rmax : maximal radial distance of pixel from stellar centroid
        #'''
        
        #coo = np.array( coo, ndmin=3 ).T
        #pix_cen = self.pixels + 0.5                                  #pixel centroids
       
        #rfc = np.linalg.norm(pix_cen - coo , axis=0 )                #radial distance of pixels from stellar centroid
        
        #return np.array([ np.sum( self.image_data[(rfc>=r-1)&(rfc<r)] ) for r in range(1, rmax) ])
    
    #===============================================================================================
    def pre_fit(self, xy, **snapkw):
        '''Get all the variables necessary to do the fitting'''
        if not 'threshold' in snapkw:           snapkw['threshold'] = None 
        if not 'edge_cutoff' in snapkw:         snapkw['edge_cutoff'] = None
        
        xs, ys = self.snap( *xy, pr=0, **snapkw )
        if not (xs and ys):
            return None

        j, i = xs, ys
        sky_mean, sky_sigma = self.measure_sky(i,j, pr=0)     #NOTE:  YOU CAN ELIMINATE THE NEED FOR THIS IF YOU ADD A BACKGROUND PARAM TO THE FITS
        il,iu, jl,ju, star_im = self.zoom(i,j)
        star_im = star_im - sky_mean
        Y, X = grid = np.mgrid[il:iu,jl:ju]
        
        p0 = self.fit.param_hint( (xs,ys), star_im )
        
        return p0, grid, star_im, sky_mean, sky_sigma, (il,iu, jl,ju)
            
    #===============================================================================================
    def fit_psf(self, xs, ys, plot=0):          #TODO: DISPLAY MEASUREMENTS ON FIGURE?????
        '''
        Fit 2D Gaussian distribution to stellar image
        Parameters
        ----------
        xs, ys : Initial coordinates for fit (maximal value in window)
        
        Returns
        ------
        coo : Star coordinates from fit
        fwhm : Full width half maximum from fit
        ellipticity : from fit
        sky_mean : measured mean sky counts
        sky_sigma : measured standard deviation of sky
        star_im : sky subtracted image of star
        '''
        j, i = xs, ys
        sky_mean, sky_sigma = self.measure_sky(i,j)     #NOTE:  YOU CAN ELIMINATE THE NEED FOR THIS IF YOU ADD A BACKGROUND PARAM TO THE FIT
        il,iu, jl,ju, star_im = self.zoom(i,j)
        star_im = star_im - sky_mean
        Y, X = grid = np.mgrid[il:iu,jl:ju]
        
        skydict = {     'sky_mean'      :       sky_mean,
                        'sky_sigma'     :       sky_sigma,
                        'star_im'       :       star_im,
                        '_window_idx'   :       (il,iu, jl,ju)          }
                        #'flux'          :       star_im.sum()
        
        ##### Fit function {self.fit.F.__name__} to determine fwhm psf ###
        plsq = self.fit( (xs,ys), grid, star_im )
        pdict = self.fit.get_fit_par( plsq )
        pdict.update( skydict )                                 #extend the fit parameters dictionary with the sky paramaters
        
        Z = self.fit.F(plsq, X, Y)
        self.image_data_cleaned[il:iu, jl:ju] -= Z              #remove star from image by subtracting Moffat fit
        
        if plot:
            #print( '!'*88, 'plotting!', plot )
            if self.stars.has_plot:
                self.stars.update_plots( X, Y, Z, star_im )
                #star_im = self.image_data[il:iu, jl:ju] - sky_mean
            else: 
                self.stars.init_plots( self.figure )   
        return pdict
    
    #===============================================================================================
    #@profile()
    def select_stars(self, event):
        
        if event.inaxes!=self.image.axes:
            return
        
        if event.button==3:
            #print( '!'*888 )
            #print( 'BLLUUUAAAAAAAGGGGGHHHHHHHHX!!!!!!!!!!' )
            return
        
        if event.button==2:
            self.restart()
            return
        
        x,y = event.xdata, event.ydata
        if None in (x,y):
            return
        
        else:
            xs, ys = self.snap(x,y)                                           #coordinates of maximum intensity within window / known star
            if not (xs and ys):
                return
            else:
                stars = self.stars
                fig, ax = self.figure, self.ax
                
                #Fitting
                params = self.fit_psf( xs,ys, plot=0 )
                
                #Add star to collection
                star = stars.append( **params )
                print( star )
                
                #Updating plots
                stars.skyaps.auto_colour( check='all', cross=stars.psfaps, edge=self.image_data.shape )
                stars.meanstar.update( self.fit.get_params_from_cache(), stars.mean_radial_profile() )
                if stars.meanstar.has_plot:
                    stars.meanstar.update_plots()
                
                #Drawing and blitting
                #background = fig.canvas.copy_from_bbox( fig.bbox )
                #[fig.canvas.restore_region(bg) for bg in self.backgrounds]
                
                #ax.draw_artist( stars.psfaps )
                #ax.draw_artist( stars.skyaps )
                #ax.draw_artist( stars.annotations[-1] )
                
                #ms = stars.meanstar
                #ms.set_visible( True )
                #ms.draw()
                
                #TODO button checks
                
                #pyqtRemoveInputHook()
                ##lookup = copy(locals())
                ##lookup.update( globals() )
                ##qtshell( lookup )
                #ipshell()
                #pyqtRestoreInputHook()
                
                #fig.canvas.blit(fig.bbox)
                self.figure.canvas.draw()               #BLITTING????
                
####################################################################################################    

#====================================================================================================
def samesize( imdata, interp='cubic' ):
    '''resize a bunch of images to the same resolution for display purposes.'''
    from scipy.misc import imresize
    
    shapes = np.array([d.shape for d in imdata])
    resize_to = shapes.max(0)
    scales = shapes / resize_to
    
    new = np.array( [imresize(im, resize_to, interp, mode='F') for im in imdata] )
    return new, scales

#====================================================================================================
def get_pixel_offsets( imdata ):
    '''Determine the shifts between images from the indices of the maximal pixel in each image.
    NOTE: This is not a robust algorithm!
    '''
    maxi = np.squeeze( [np.where(d==d.max()) for d in imdata] )
    return maxi - maxi[0]
    
#====================================================================================================
def blend_prep(data, offsets):
    '''shift and zero pad the data in preperation for blending.'''
    omax, omin = offsets.max(0), offsets.min(0)
    newshape = data.shape[:1] + tuple(data.shape[1:] + offsets.ptp(0))
    nd = np.empty(newshape)
    for i, (dat, padu, padl) in enumerate(zip(data, omax-offsets, offsets-omin)):
        padw = tuple(zip(padu,padl))
        nd[i] = np.pad( dat, padw, mode='constant', constant_values=0 )
    return nd      
                
#====================================================================================================
def get_coords_from_file( fn ):
        '''load star cordinates from file and return as array.'''
   
        if os.path.isfile( fn ):
            
            print( 'Loading coordinate data from {}.'.format( fn ) )
            data = np.loadtxt(fn, unpack=0)
            
            if data.shape[-1]==7:                         #IRAF generated coordinate files. TODO: files generated by this script should match this format
                coords, ids = data[:,:2], data[:,6]
                coords -= 1    #Iraf uses 1-base indexing??
                return coords, ids
            
            elif data.shape[-1]==2:
                ids = range(data.shape[0])
                return data, ids
            
            elif 0 in data.shape:
                warn( 'Coordinate file "{}" contains no data!'.format(fn) )
                return
            else:
                warn( 'Coordinate file "{}" format not understood.'.format(fn) )
                return
        else:
            print( 'Coordinate file {} does not exists!'.format(fn) )
            return
    
        
        
#====================================================================================================
def undup_coo( file_coo, known_coo, tolerance ):
    '''elliminate duplicate coordinates within radial tolerance.'''
    l_dup = np.array([ np.linalg.norm(known_coo - coo, axis=1)[0] < tolerance
                        for coo in file_coo ])
    return l_dup.ravel()

    
####################################################################################################    
from mplMultiTab import *
class StarSelectorBase(MplMultiTab):
    '''
    class which contains methods for selecting stars from image and doing basic psf fitting and sky measurements
    '''
    
    use_wcs = False                                  #Flag for whether to display the image as it would appear on sky
    #====================================================================================================
    #@profile()
    def __init__(self, imagefiles, coordfiles=None, offsets=None, show_blend=0):
        global SKY_RADII
        N = len(imagefiles)
        SKY_RADII = np.tile( DEFAULT_SKY_RADII, (N,1) )
        
        self.filenames = FileAccess()
        #self.status = 0
        #self._write_pars = 1
        
        #Check for existing coordinate files
        self.has_found = not coordfiles in [None, []]            #user defined coordinate files known to exist
        #if not self.has_found:
            #self.has_found, cfg = self.filenames.look_for('coo')
        #if self.has_found:
            #self.filenames.coo = cfg
        
        self.has_phot, _ = None, None #self.filenames.look_for('mag')
        
        #load the initialise apis and load figures TODO: MPC
        self.apis = []
        labels = []
        
        
        #Create figures & axes for apis
        pool = mpc.Pool()
        self.figures = pool.map( setup_figure_geometry, range(len(imagefiles)+show_blend) )
        
        for i, filename in enumerate(imagefiles):
            
            api = ApertureInteraction( i, self.figures[i] )
            api.load_image( filename )
            #bgs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in fig.axes]
            
            #self.figures.append( api.figure )
            self.apis.append( api )
            labels.append( 'Tab %i' %i )
        
        #self.imshapes = np.array([api.image_data.shape for api in self.apis])
        self.has_blend = bool(show_blend)
        
        #Determine image pixel offsets and blend
        if offsets is 'auto':
            data = np.array( [api.image_data for api in self.apis] )    #remember that this is flipped left to right if use_wcs
            data, self.scales = samesize(data)
            
            m = np.array([np.median(_) for _ in data], ndmin=3).T           #median of each image
            data = data / m                                             #normalised

            print( 'Determining image shifts...' )
            self.pixoff = get_pixel_offsets( data )                     #offsets in pixel i,j coordinates
            
            #Do image blend
            if show_blend:
                #SKY_RADII = np.vstack( (SKY_RADII, DEFAULT_SKY_RADII) )
                print( 'Doing data blend...' )
                data = blend_prep(data, self.pixoff)
                weights = np.ones(data.shape[0])
                self.blend = np.average(data, 0, weights)
                
                api_blend = ApertureInteraction( 0, self.figures[i+1] )
                api_blend.load_image( data=self.blend, use_wcs=self.use_wcs )
                self.apis.append( api_blend )
                #self.figures.append( api_blend.figure )
                labels.append( 'Blend' )
                
                #blend_offset = self.pixoff.max(0)
                #self.pixoff = np.vstack( (self.pixoff, blend_offset) )
                #self.scales = np.vstack( (self.scales, (1,1)) )
        elif offsets is None:
            self.offsets = np.zeros( (len(self),2) )
            self.scales = np.ones_like( self.offsets )
        #translate to on-sky coordinates
        
        #self.offsets = np.fliplr( self.pixoff )                         #offsets in x,y coordinates
        #if self.use_wcs:
            #self.offsets[:,0] = -self.offsets[:,0]
        
        #setup shared memory for fitting
        self.shapes = shapes = np.array([api.image_data.shape for api in self])             #image dimensions for each api
        shprod = np.append(0, np.prod(shapes,1) )                               #total number of pixels per image --> indexing 
        totpix = int(np.sum(shprod))                                            #total number of pixels in shared memory
        self.arr_st_idx =  np.cumsum( shprod )                                       #starting indeces for each image frame
        self.shared_arr = mpc.Array(ctypes.c_double, totpix )      #shared, can be used from multiple processes
        
        #initialize the shared memory with the data values
        self.shared_arr.get_obj()[:] = np.concatenate([api.image_data.ravel() for api in self])
        
        #ipshell()
        
        ##create the canvas backgrounds for blitting
        #for api in self:
            #FigureCanvas(api.figure)
            #api.background = api.figure.canvas.copy_from_bbox( api.figure.bbox )
        
        #load the figure manager widget
        MplMultiTab.__init__(self, figures=self.figures, labels=labels )
        
        #connect the callback for tab changes
        self.tabWidget.currentChanged.connect( self.tabChange )
        
        #Connect to the canvases for aperture interactions
        for api in iter(self):             api.connect()
        
        #set the focus on the first figure
        self.figures[0].canvas.setFocus()
        
    #====================================================================================================
    def __iter__(self):
        idx = slice(0,-1) if self.has_blend else ...
        for api in self[idx]:
            yield api
    
    #====================================================================================================
    def __len__(self):
        return len(self.apis)
    
    #====================================================================================================
    def __getitem__(self, key):
        if isinstance(key, str):
            if key.lower().startswith('c'):     #return current api from tabWidget
                return self.apis[self.cidx]
        
        if isinstance( key, (int, np.int0) ):
            if key >= len( self ) :
                raise IndexError( "The index (%d) is out of range." %key )
            return self.apis[key]
        
        if isinstance(key, slice):
            return self.apis[key]
            
        if isinstance(key, type(Ellipsis)):
            return self.apis
            
        if isinstance(key, (list, np.ndarray)):
            if isinstance(key[0], (bool, np.bool_)):
                assert len(key)==len(self)
                return [ self.apis[i] for i in np.where(key)[0] ]
            if isinstance(key[0], (int, np.int0)):
                return [ self.apis[i] for i in key ]
        
        raise IndexError( 'Invalid index {}'.format(key) )
    
    #====================================================================================================
    #def pullattr(self, attr):
        #return [getattr(star, attr) for star in self]
    
    #====================================================================================================
    def has_stars(self):
        return first_true_idx( self, len )
    
    #====================================================================================================
    def get_current_index(self):
        return self.tabWidget.currentIndex()    #index of currently active tab
    cidx = property(get_current_index)
    
    #====================================================================================================
    def create_main_frame(self, figures, labels):
        
        MplMultiTab.create_main_frame( self, figures, labels )
        
        buttonBox = self.create_buttons()
        
        hbox = qt.QHBoxLayout()
        hbox.addWidget(self.tabWidget )
        #hbox.addWidget(buttonBox)
        hbox.addLayout(buttonBox)
        self.vbox.addLayout(hbox)
        
        
        self.main_frame.setLayout(self.vbox)
        self.setCentralWidget(self.main_frame)
    
    #====================================================================================================
    #@unhookPyQt
    #def keyPressEvent(self, e):
        #print( 'ONKEY!!' )
        #print( e.key() )
        #ipshell()
        #if e.key() == QtCore.Qt.Key_Delete:
            #self['current']
            
    
    #====================================================================================================
    def create_buttons(self):
        ''' Add axes Buttons '''
        
        buttonBox = qt.QVBoxLayout()
         
        self.buttons = {}
        labels = ['load coords', 'write_coo', 'daofind', 'phot', 'Ap. Corr.', 'Propagate', 'Restart']
        func_names = ['load_coo', 'write_coo', '_on_find_button', '_on_phot_button',  '_on_apcor_button', 'propagate', 'restart']
        
        colours = self.button_checks()
        
        for i, label in enumerate(labels):
            F = getattr( self,  func_names[i] )
            #F = self._on_click#lambda s : print(s)
            button = self.add_button(F, label, colours[label])
            self.buttons[label] = button
            buttonBox.addWidget( button )
        
        buttonBox.addItem( qt.QSpacerItem(20,200) )
        
        return buttonBox
    
    #====================================================================================================
    def button_checks(self):
        
        labels = ['load coords', 'write_coo', 'daofind', 'phot', 'Ap. Corr.', 'Propagate', 'Restart' ]
        hf, hs, hp = self.has_found, self.has_stars(), self.has_phot
        checks = [hf, hs, 'starfinder' in globals(), 'photify' in globals(), hp, hs, 1]
        gc, ic, bc = 'g', 'orange', 'r'
        colours = [gc if ch else ic for ch in checks]
        
        colour_dic = dict( zip(labels, colours) )
        colour_dic['daofind'] = gc if 'starfinder' in globals() else bc
        colour_dic['phot'] = gc if 'photify' in globals() else bc
        
        return colour_dic
    
    #====================================================================================================
    def add_button(self, func, label, colour):
            
        button = qt.QPushButton(label, self)
        
        self.set_button_colours( button, colour )
        palette = qt.QPalette( button.palette() ) # make a copy of the palette
        palette.setColor( qt.QPalette.ButtonText, qt.QColor('black') )
        button.setPalette( palette )
        
        button.clicked.connect( func )
        
        return button
    
    #====================================================================================================
    def set_button_colours(self, button, colour, hover_colour=None, press_colour=None):
        
        def colourtuple( colour, alpha=1, ncols=255 ):
            rgba01 = np.array(colorConverter.to_rgba( colour, alpha=alpha ))    #rgba array range [0,1]
            return tuple( (ncols*rgba01).astype(int) )
        
        bg_colour = colourtuple( colour )
        hover_colour = colourtuple( colour, alpha=.7 ) if hover_colour is None else hover_colour
        press_colour = colourtuple( "blue", alpha=.7 ) if press_colour is None else press_colour
        
        style = Template("""
            QPushButton
            { 
                background-color: rgba$bg_colour;
                border-style: outset;
                border-width: 1px;
                border-radius: 3px;
                border-color: black;
                font: bold 14px;
                min-width: 10em;
                padding: 6px;
            }
            QPushButton:hover { background-color: rgba$hover_colour }
            QPushButton:pressed { background-color: rgba$press_colour }
            """).substitute( bg_colour=bg_colour, hover_colour=hover_colour,  press_colour=press_colour )
        
        button.setStyleSheet( style ) 
        
    #====================================================================================================
    #def _on_click(self):
        #sender = self.sender()
        #self.statusBar().showMessage(sender.text() + ' was pressed')  
    
    #===============================================================================================
    @unhookPyQt
    def tabChange(self, i):
        '''When a new tab is clicked, 
            assume apertures already added to axis
            initialise the meanstar plot (if needed)
            redraw the canvas.
        '''
        api = self[i]
        fig = api.figure
            
        api.stars.axadd( api.ax )
        
        fig.canvas.setFocus()
        if not api.stars.meanstar.has_plot:
            print( 'HERE! '*8 )
            api.stars.meanstar.init_plots( fig )
        fig.canvas.draw()
        #ipshell()
        
    #===============================================================================================
    @unhookPyQt
    #@profile()
    def propagate(self, TF=None, whence='current', from_coo=None, narcisistic=False, _pr=1, **snapkw): #NEEDS api index ---> prop from this one
        ''' 
        Propagate the stars to other apis
        Parameters
        ----------
        TF              :       not sure!? <-- pyQT thing
        whence          :       index of the api to propagate from / to
        from_coo        :       Optional coordinates to propagate the fits from.
                                If not given, the stars in the api indicated by 'whence' will be used to propagate the fits
        narcisistic     :       If True, only do the fitting for the api indicated by 'whence'
                                else do fitting for all apis except 'whence'
        _pr             :       whether to print the resulting stars
        
        '''
        #TODO:  FIT FOR GAUSSIAN SHAPE ONLY???????????
        #===============================================================================================
        def init(shared_arr_, start_idxs_, _shapes):
            '''function used to initialize the multiprocessing pool with shared image array as backdrop for ftting'''
            global shared_arr, shared_arr_start_idx, shared_arr_shapes
            shared_arr = shared_arr_ # must be inhereted, not passed as an argument
            shared_arr_start_idx = start_idxs_
            shared_arr_shapes = _shapes
        #===============================================================================================
        def warn_for_unconvergent(cluster, nidx, nargs):
              #Partition the fits into convergent and unconvergent parts
            ok, unconvergent = partitionmore( lambda i : i is None, cluster, nidx, nargs )
            _, unidx, unarg = map(list, unconvergent)
            #goodfits, nidx, _ = map(list, ok)

            for _, unidx, unarg in zip(*unconvergent):
                p0, _, star_im, sky_mean, sky_sigma, _, _ = unarg
                badcoo = coords[unidx]
                msg = ("""Star {unidx[1]} in api {unidx[0]} with 
                        [x,y] = {badcoo}\n
                        sky_mean = {sky_mean},\n
                        sky_sigma = {sky_sigma}\n
                        could not be fit with\n
                        p0 = {p0},\n"""
                        ).format( unidx=unidx, badcoo=badcoo, p0=p0, sky_mean=sky_mean, sky_sigma=sky_sigma)
                warn(  msg )
        
        #===============================================================================================
        N = len(self) - int(self.has_blend)
        scales = self.scales[:,None]
        offsets = self.offsets[:,None]
        whence = self.cidx      if whence is 'current'          else whence
        if from_coo is None:
            stars = self[whence].stars
            R_OUT = stars.R_OUT
            #skyradii = stars.SKY_RADII
            Nstars = len(stars)
            #Indexing array for stars shape: ( (N-1)*Nstars, 2 )
            idx = np.mgrid[:N,:Nstars].T.reshape(-1,2)
            compcoo = stars.coords
            #reshape, shift + rescale the star coordinates to match the images
            coords = np.tile( stars.coords, (N,1,1) )
            coords = (coords + offsets) * scales
            def get_coords(idx):
                return coords[idx]
        else:
            R_OUT = DEFAULT_SKY_RADII[1]
            idx = [list(zip(itt.repeat(i), range(l))) for (i,l) in enumerate(map(len, from_coo))]
            idx = sum(idx,[])
            coords = from_coo           #assuming the input coordinates are 3D
            compcoo = from_coo[0]
            def get_coords(idx):
                return coords[idx[0]][idx[1]]
            
        if narcisistic:         #fit the stars in the current api only  <--- this is needed if current stars where input from coordinate file and not from mouse clicks
            filterfunc = lambda ix: ix[0] == whence
        else:                   #fit the stars NOT in the current api
            filterfunc = lambda ix: ix[0] != whence
        idx = list( map(tuple, filter(filterfunc, idx)) )      #index tuples of each star in api

        #Retrieve arguments for fitting
        args = [self[ij[0]].pre_fit( get_coords(ij), **snapkw ) for ij in idx]                #arguments for fitting
        nargs, nidx = map(list, filtermore( lambda i : not i is None, args, idx ))         #filter out bad stars (duplicates, edges, threshold)
        
        #sort by star brightness.  Necessary for fitting.
        ims = nth(zip(*nargs), 2)                                               #the star image data
        _, nidx, nargs = map(list, sortmore(ims, nidx, nargs, key=np.max, order='descend'))                               #sort by maximum
        apiidxs = np.array(nidx)[:,0]
        nqidx = np.unique(apiidxs)
        
        #zip-append the shared memory indeces (so we can reconstruct the image array after the function call)
        nargs = zipapp(nargs, apiidxs )
        
        #Fit the stellar profiles (in multiprocessing pool)
        print( 'Fitting Gaussian2D to %d stars.' %len(nargs) )
        pool = mpc.Pool( initializer=init, initargs=(self.shared_arr, self.arr_st_idx, self.shapes) )       #pool of fitting tasks
        cluster = pool.map( starfucker, nargs )
        
        #Group the output by frame
        fitcoo = [s.coo if s else (-1e6,-1e6) for s in cluster]                      #star coordinates from fitting (false coordinates inserted to yield contiguous lists)
        gix, grp = zip( *groupmore( lambda x:x[0], nidx, cluster, fitcoo ) )          #groups the indeces/stars/coordinates according to api idx
        gidx, gstars, fitcoo = zip(*grp)
        gstars = lists(gstars);         fitcoo = lists( fitcoo )
        
        #Filter duplicate coordinates   (this is needed when fitting for position --> sometimes input coordinates that are close to each other converge to the same fit)
        dix = map(where_close_array, fitcoo)                      #indeces of duplicates
        dix, fcix = filtermore(None, dix, itt.count())          #indeces of frames where duplicates are (filter empty results where no duplicates are)
        dix = itt.starmap(fct.partial(nthzip,1), dix)
        for i, dx in zip(fcix, dix):
            for j in dx:
                gstars[i][j] = None
                fitcoo[i][j] = (-1e6,-1e6)                      #flag duplicates
            #gidx[i], gstars[i], fitcoo[i] = filtermore( lambda j: j in dx, gidx[i], gstars[i], fitcoo[i] )       #remove duplicates
            
        #Sort by corrispondence
        if narcisistic:
            gstars = sorter(*gstars, key=lambda x: x.coo[::-1] if x else (), order='reverse'  )               #sort the stars by descending y coord
            SI = [range(len(gstars[0]))]                                                                                #sorted indeces
        else:                                   #identify corresponding stars between apis. i.e. preserve star id --> position mapping between frames
            #sequentially append ficticious coordinates to yield contiguous shapes in fit coordinate lists (so we can use array opperations)
            fitcoo = [ list(fc) + [(-1e6,-1e6)]*(len(compcoo)-len(fc)) for fc in fitcoo ]
            fitcoo = (fitcoo/scales[nqidx]) - offsets[nqidx]           #coordinate array --> use the inter-image scale and offset to renormalise the coordinates for comparison
            SI = self.cross_coord_sort(fitcoo, compcoo)
        
        #Set the sky radii for the other frames. 
        #Inner is 3 times the mean hwhm; 
        #Outer is current frame outer scaled by relative image dimension.
        skyradii = np.array(DEFAULT_SKY_RADII[::-1]) if whence is None else SKY_RADII[whence,::-1]
        SKY_AREA = np.subtract( *np.pi*skyradii**2 )
        for ix, stargroup, si in zip(gix, gstars, SI): #zip(lbl,finmap):#
            sg, = sort_by_index( stargroup, index=si[si<len(stargroup)] )                        #sort by correspondence (position across frames)
            sg = list(filter(None, sg))                                                         #filter bad fits / duplicate convergence
            
            fwhm = np.mean([star.fwhm for star in sg])
            
            SKY_RADII[ix,0] = 1.5*fwhm + 1
            ROUT_MIN = np.sqrt( SKY_AREA/np.pi - SKY_RADII[ix,0]**2 )
            SKY_RADII[:,1] = np.max( [ROUT_MIN, R_OUT*scales[ix,0,0]], 0 )
            
            #append the stars to the apis 
            api = self[ix]
            for star in sg:     api.stars.append( star )        #SLOOOOWWWWWWW!!!
        
        #update plots and print summary
        for api in self[ nqidx ]:
            api.auto_colour()
            stars = api.stars
            meanstar = stars.meanstar
            cache = np.mean([star.cache for star in api.stars], 0)
            meanstar.update( tuple(cache), stars.mean_radial_profile() )
            if not meanstar.has_plot:
                meanstar.init_plots( api.figure )

            meanstar.update_plots()
            
            if _pr:
                print( api.filename )
                print( api.stars )
                print()
                
        self.summarise()
    
    #===============================================================================================
    @staticmethod
    def cross_coord_sort(fitcoo, compcoo):
        #===============================================================================================
        def coosrt(coo, threshold=3):
            ''' find the closest mathching coordinates (indeces) in the coordinate array to given coordinates 
                and flag them if they are farther than the threshold value'''
            d = np.linalg.norm( fitcoo - coo, axis=-1)
            mins, amins = d.min(1), d.argmin(1)
            amins[abs(mins)>threshold] = -1
            return amins
        #===============================================================================================
        
        #generate sorting indeces wrt compcoo
        SI = list(map( coosrt, compcoo ))            #get the sort indeces 
        SI, = sorter( SI, key=lambda x: -1 in x )                   #sort by inter-frame corrispondence
        SI = np.array(SI).T
        for si in SI:
            if -1 in si:      #find the missing star indeces (those that where further away than the threshold for all search coordinates)
                mn = get_missing_numbers( list(filter(lambda x:x>0, si))+[len(si)] )
                si[si == -1] = mn
        return SI
        
    #===============================================================================================
    def load_coo(self, *args):
        #===============================================================================================
        #def load_and_fit(idx):
            #fn = self.filenames.coo[idx]
            #api = self.apis[idx]#.load_coo(fn)
            #coords = api.load_coo(fn)

            ##background = fig.canvas.copy_from_bbox(fig.bbox)
            ##fig.canvas.restore_region(background)
            
            #self.propagate( from_coo=coords, whence=idx, narcisistic=1, offset_tolerance=3. ) #tol=self.fwhm
        #===============================================================================================
        
        if not self.has_found:
            warn( 'Nothing to load' )
            return
        
        #NOTE:  YOU CAN AVOID DOING THIS FILTERING EACH TIME, BY KEEPING THE EXISTING FNS AND GENERATED FNS SEPARATE....
        fns = self.filenames.coo
        existing, eidx = map(list, filtermore( os.path.exists, fns, range(len(self))) ) #existing coordinate files
        
        if len(eidx)==1:
            ix0 = eidx[0]
        
            fn = self.filenames.coo[ix0]
            api = self.apis[ix0]#.load_coo(fn)
            coords = api.load_coo(fn)
            #background = fig.canvas.copy_from_bbox(fig.bbox)
            #fig.canvas.restore_region(background)
            
            self.propagate( from_coo=[coords], whence=ix0, narcisistic=1, offset_tolerance=3. ) #tol=self.fwhm
            
            #api.ax.draw_artist( api.stars.psfaps )
            #api.ax.draw_artist( api.stars.skyaps )
            #fig.canvas.blit(fig.bbox)
            self['c'].figure.canvas.draw()
            
            self.propagate( whence=idx, offset_tolerance=3., edge_cutoff=3 )
        
        else:
            coords = [api.load_coo(fn) for (api,fn) in zip(self, fns)]
            self.propagate( from_coo=coords, whence=None, peak=None )
            self['c'].figure.canvas.draw()
        
    #===============================================================================================
    @unhookPyQt
    def write_coo( self, TF=0 ):
        
        if self.use_wcs:
            coords = [api.map_coo() for api in self]
        else:
            coords = [api.stars.coords for api in self]
        
        yn, _ = self.clobber_check( self.filenames.coo )
        
        if not yn:
            print( 'Nothing written.' )
            return
        
        #if os.path.exists( self.filenames.found_coo ):
            #data, header = partition( lambda s: s.startswith('#'), open(fn,'r') )
            ##eliminate duplicates in the coordinates found by daofind to within tolerance
            #within = 5.                             #tolerance for star proximity in coordinate file (daofind)
            #self.undup_coo( filecoo, knowncoo, within )
        #else:
            
        header=''
        for coo, outfn in zip(coords, self.filenames.coo):
            print( 'Writing coordinates of selected stars to file: {}'.format(outfn) )
            np.savetxt(outfn, coo, fmt='%.3f', header=header)
            
    
    
    #===============================================================================================
    @unhookPyQt
    def _on_find_button( self, tf ):
        '''run daofind on image and plot the found stars.'''
        
        #TODO: SHOW THRESHOLD ON COLOUR BAR???????????
        
        print( 'find button!!' )
        if self.has_found:
            _, clobber = self.clobber_check( self.filenames.coo )
            for fn in clobber:          os.remove(fn)

        #fwhma = [api.stars.fwhmpsf for api in self]
        #sky_sigma = [api.stars.sky_sigma for api in self]
        
        api = self[0]
        #if len(api.stars):
            #api.fwhmpsf
            #api.sky_sigma
        #else:
            #self.fwhmpsf = 3.0
            #self.sky_sigma = 50.
            #msg = ("No data available - assuming default paramater values:\n"
                    #"FWHM PSF = {}\nSKY SIGMA = {}\nDon't expect miracles...\n" ).format(self.fwhmpsf, self.sky_sigma)
            #warn( msg )
        
        
        for i, api in enumerate(self):
            if not api.stars.fwhm is None:
                starfinder.set_params( api )
                starfinder( api.filename, self.filenames.coo[i]  ) 
        
        #self.stars.skyaps.remove()
        #self.stars.psfaps.remove()
        
        self.has_found = 1
        
        self.load_coo( )
        self._write_par = 0                     #not necessary to write parameter file
        
        #self.figure.canvas.draw()
        
    #===============================================================================================
    @unhookPyQt
    def _on_phot_button( self, event ):
        
        if not self.status:
            self.finalise( )
            
            for api in iter(self):     api.gen_phot_aps()
        
        #save the parameters (fwhm, sky etc..) to file
        fns = self.filenames
        
        phot = iraf.noao.digiphot.daophot.phot
        iraf.unlearn( phot )
        
        for i, api in enumerate(self):
            photify.set_params( api )
            d,c,f,p = nthzip(i, fns.datapars, fns.centerpars, fns.skypars, fns.photpars )

            phot.datapars.saveParList( d )      #filename=d?
            phot.centerpars.saveParList( c )
            phot.fitskypars.saveParList( f )
            phot.photpars.saveParList( p )

        fns = self.filenames
        os.remove( self.clobber_check(fns.mags)[1] )
        
        args = zip(fns.split, fns.coo, fns.datapars, fns.centerpars, fns.skypars, fns.photpars, fns.mags)
        
        pyqtRemoveInputHook()
        ipshell()
        pyqtRestoreInputHook()
        
        pool = mpc.Pool()       #pool of photomerty tasks
        pool.map(do_phot, args )
    
            
            #ipshell()
            
            
            #photify( imlist, output )
        
    #===============================================================================================
    def _on_apcor_button( self ):
        
        apcorrs = ApCorr( self )                #Initialise Iraf task wrapper class
        apcorrs( )
    
    #===============================================================================================
    def restart(self, event):
        print('Restarting...')
        for api in self:
            api.remove_all()
        
    #===============================================================================================
    #@unhookPyQt
    def write_pars( self ):
        '''
        write sky & psf parameters to files.
        '''
        #functions = [photify.phot.fitskypars, starfinder.daofind.datapars]
        #fns = ['fitskypars.par', 'datapars.par']
        #for fn, func in zip(fns, functions):
            #if os.path.isfile( path+fn ):
                #yn = input( 'Overwrite {} ([y]/n)?'.format(fn) )
            #else:
                #yn = 'y'
            
            #if yn in ['y', 'Y', '']:
                #print( 'Writing parameter file: {}'.format( fn ) )
                #func.saveParList( path+fn )
        pass
    
    #===============================================================================================
    @staticmethod
    @unhookPyQt
    def clobber_check(filenames, ask=0):
        clobber = list(filter(os.path.exists, filenames))
        yn = True
        if any(clobber):
            warn( 'The following files are being clobbered:\n{}'.format('\n'.join(clobber)) )
            if ask:
                yn = input( 'Continue ([y]/n)?' ) in ['y', 'Y', '']
        return yn, clobber
    
    #===============================================================================================
    def savefigs(self): 
        
        print( '\n\nSaving figures!')
        path = self.filenames.path
        fns = self.filenames.gen( path=path,  extension='aps.png' )

        self.clobber_check( fns )
        
        #save images with aps
        for fig, fn in zip(self.figures, fns):
            fig.savefig( fn )
            
        #TODO: SAVE AS PICKLED OBJECT!
    
    #===============================================================================================
    @unhookPyQt
    def summarise(self):
        keys = 'fwhm', 'sky_mean', 'sky_sigma'
        data = [ [getattr(api.stars, key) for key in keys]
                    + [api.ron, api.saturation]
                    + [tuple(api.SKY_RADII)]               for api in iter(self) ]
        tbl = Table( data, 
                     title='Image properties', title_props={'bg' : 'magenta', 'text' : 'bold'},
                     row_headers=self.filenames.naked, 
                     col_headers=keys+('readout noise', 'saturation', 'SKY RADII') )
        print( tbl )
    
    #===============================================================================================
    def finalise( self ):
        '''
        Write parameter and coordinate files.
        '''
        _, nostars = filtermorefalse(len, self, self.filenames.naked)     #filenames of apis with no stars
        if any(nostars):
            warn( 'These images contain no stars:\n'+'\n'.join(nostars)+'\nNo photometry will be done on them!' )
        
        #change button colour
        colour = 'g'
        self.set_button_colours( self.buttons['phot'], colour )
        
        path = self.filenames.path
        self.filenames.split = self.filenames.gen( path=path, extension='split.txt' )
        self.filenames.datapars = selector.filenames.gen(path=path, extension='dat.par')
        self.filenames.skypars = selector.filenames.gen(path=path, extension='sky.par')
        self.filenames.centerpars = selector.filenames.gen(path=path, extension='cen.par')
        self.filenames.photpars = selector.filenames.gen(path=path, extension='phot.par')
        self.filenames.mags = self.filenames.gen( path=path, extension='mag' )
        self.status = 1
        
        #if self._write_par:
            #self.write_pars()
        self.write_coo()
        
        #self.savefigs()
    
    #===============================================================================================
    def _on_close(self, event):
        '''
        When figure is closed disconnect event listening.
        '''
        self.disconnect()
        self.savefigs()
        #self.finalise()
        
        plt.close(self.figure)
        print('Closed')


    #===============================================================================================
    @unhookPyQt
    def remove_all(self):                                                                                    #NAMING
        
        for artist in self.apertures.get_all():
            artist.remove()
        
        self.apertures.__init__()                                         #re-initialise Aperture class
        self.stars.__init__()                                             #re-initialise stars
        self.image_data_cleaned = copy(self.image_data)                   #reload image data for fitting

        self.figure.canvas.draw()
            
    #===============================================================================================
    def sky_map(self):
        '''produce logical array which serves as sky map'''
        r,c = self.image_shape
        Xp, Yp = self.pixels                                #the pixel grid
        sbuffer, swidth = self.sbuffer, r/2                         #NOTE: making the swidth paramater large (~half image width) will lead to skymap of entire image without the detected stars
        L_stars, L_sky = [], []
        for x,y in self.stars.found_coo:
            Rp = np.sqrt((Xp-x)**2+(Yp-y)**2)
            l_star = Rp < sbuffer
            l_sky = np.all([~l_star, Rp < sbuffer+swidth],0)          #sky annulus
            L_stars.append(l_star)
            L_sky.append(l_sky)

        L_all_stars = np.any(L_stars,0)
        L_all_sky = np.any(L_sky,0)
        L = np.all([L_all_sky,~L_all_stars],0)

        return L


    #===============================================================================================
    def get_fn(self):
        return self.filenames


        


def val_float(num_str):
    try:
        return isinstance(float(num_str), float)
    except:
        return False

        
def par_from_file( filename ):
    name, typ, mode, default, mn, mx, prompt = np.genfromtxt(filename, dtype=str, delimiter=',', skip_footer=1, unpack=1)
    value_dict = dict( zip(name, default) )
    return value_dict

#################################################################################################################################################

#if __name__=='__main__':
RUN_DIR = os.getcwd()

#Parse command line arguments
import argparse, glob
parser = argparse.ArgumentParser(description='')
#parser.add_argument('-d', '--dir', dest='dir', help='The data directory. Defaults to current working directory.')
parser.add_argument('-o', '--destination', help='The output data directory. Defaults to input directory pointed to by the images argument.')
parser.add_argument('-x', '--coords', dest='coo', help='File containing star coordinates.')
parser.add_argument('-i', '--images', nargs='+', help='Fits images to be processed.  Requires at least one argument.  Argument can be explicit list of files, a glob expression that evaluates to a file list, the name of a txt file containing a list of filenames, or a directory in which case all the fits files in that directory will be used.')

args = parser.parse_args()

from myio import *
imagefiles = parselist( args.images )
if args.destination:
    outpath = iocheck(args.destination, os.path.exists, raise_error=1)

#Read input coordinates
if args.coo:
    #args.coo = os.path.join( path, args.coo )
    args.coo = validity.test_ls(args.coo, os.path.exists, raise_error=1)

DEFAULT_SKY_RADII = SkyApertures.RADII
    
    
class FileAccess(object):
    #====================================================================================================
    #basenames           =       basenames
    images              =       imagefiles
    naked               =       [os.path.split(fn)[1] for fn in images]
    coo                 =       args.coo
    #path                =       path
    #outpath             =       outpath
    
    #====================================================================================================
    def gen(self, **kw):
        '''Generator of filenames of unpacked cube.'''
        path = os.path.abspath(kw['path'])      if 'path' in kw         else ''
        sep = kw.setdefault('sep', '.')
        extension = kw.setdefault('extension', 'fits')
        
        basenames = nthzip(0, *map(os.path.splitext, self.images))       #filenames with path, but without extension
        
        return ['{}{}{}'.format( os.path.join(path, base), sep, extension ) for base in basenames]

    #====================================================================================================
    def look_for(self, ext):
        gf = self.gen( path=self.path, extension=ext )
        gfe = list(filter( os.path.isfile, gf ))        #those generated filenames that exist
        found = 0
        if len(gfe):
            print( 'Files detected:\n', '\n'.join(gfe) )
            #self.coo = cfg
            found = 1
        return found, gf


#################################################################################################################################################
#Set global plotting parameters
#rc( 'savefig', directory=outpath )
#rc( 'figure.subplot', left=0.065, right=0.95, bottom=0.075, top=0.95 )    # the left, right, bottom, top of the subplots of the figure
#################################################################################################################################################






#Check for pre-existing psf measurements
#try:
    #fitskypars = par_from_file( path+'fitskypars.par' )
    #sky_inner = float( fitskypars['annulus'] )
    #sky_d = float( fitskypars['dannulus'] )
    #sky_outer = sky_inner +  sky_d
    
    #datapars = par_from_file( path+'datapars.par' )
    #fwhmpsf = float( datapars['fwhmpsf'] )
    #sky_sigma = float( datapars['sigma'] )
    #ron = float( datapars['readnoise'] )
    #saturation = datapars['datamax']
    
    #msg = ('Use the following psf + sky values from fitskypars.par and datapars.par? ' 
                #'\nFWHM PSF: {:5.3f}'
                #'\nSKY SIGMA: {:5.3f}'
                #'\nSKY_INNER: {:5.3f}'
                #'\nSKY_OUTER: {:5.3f}\n'
                #'\nREADOUT NOISE: {:5.3f}'
                #'\nSATURATION: {}'.format( fwhmpsf, sky_sigma, sky_inner, sky_outer, ron, saturation ) )
    #print( msg )
    #yn = input('([y]/n):')
#except IOError:
    #yn = 'n'
    #mode = 'fit'

    
#Initialise GUI
app = qt.QApplication(sys.argv)
selector = StarSelectorBase( imagefiles, args.coo )
    
    
#if yn in ['y','Y','']:
    #selector.fwhmpsf, selector.sky_mean, selector.sky_sigma, selector.ron = fwhmpsf, sky_mean, sky_sigma, ron
    #mode = 'select'


#THIS NEEDS TO HAPPEN IN A QUEUE!!!!!!!!!!!! 
#ipshell()

selector.show()
sys.exit( app.exec_() )

print( 'AccessRadii.SKYRADII', AccessRadii.SKYRADII )

def goodbye():
    os.chdir(RUN_DIR)
    print('Adios!')

import atexit
atexit.register( goodbye )
