#TODO: TESTS!

import itertools as itt
import functools

import numpy as np

from matplotlib import pylab as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.axes_grid1 import AxesGrid

from superfits import fetch_first_frame, quickheader
from ApertureCollections import ApertureCollection

from magic.array import grid_like
from magic.meta import flaggerFactory

from myio import warn

#module level imports
from .psf import *
from .stars import *
from .snappers import ImageSnapper, CleanerMixin, DoubleZoomMixin

from decor import print_args

from IPython import embed
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook
from decor import unhookPyQt, profile

from pySHOC.readnoise import ReadNoiseTable
RNT = ReadNoiseTable()

#pyqtRemoveInputHook()
#embed()
#pyqtRestoreInputHook()


######################################################################################################
ConnectionManager, connect = flaggerFactory( collection='_connections' )

class ConnectionMixin( ConnectionManager ):
    '''Mixin for connecting the decorated methods to the figure canvas'''    
    #=============================================================================================== 
    def add_connection(self, name, method):
        self.connections[name] = self.figure.canvas.mpl_connect(name, method)
    
    #=============================================================================================== 
    def remove_connection(self, name):
        self.figure.canvas.mpl_disconnect( self.connections[name] )
        self.connections.pop( name )
    
    #=============================================================================================== 
    def connect(self):
        '''connect the flagged methods to the canvas'''
        self.connections = {}                   #connection ids
        for (name,), method in self._connections.items():  #TODO: map??
            self.add_connection( name, method )
                
    #=============================================================================================== 
    def disconnect(self):
        '''
        Disconnect from figure canvas.
        '''
        for name, cid in self.connections.items():
            self.figure.canvas.mpl_disconnect( cid )
        print('Disconnected from figure {}'.format(self.figure) )


######################################################################################################
class Snapper(CleanerMixin, DoubleZoomMixin, ImageSnapper):
        pass


######################################################################################################
class ApertureInteraction( ConnectionMixin ):          #TODO: INHERIT FROM AXES??
    ''' '''
    #===============================================================================================
    DEFAULT_WINDOW = 30.
    #DEFAULT_SKYRADII = 15, 20
    
    msg = ( 'Please select a few stars (by clicking on them) for measuring psf and sky brightness.\n ' 
        'Right click to resize apertures.  Middle mouse to restart selection.  Close the figure when you are done.\n\n' )
    
    #===============================================================================================
    def __init__(self, idx, fig=None, **kwargs):
        
        super().__init__()
        
        self.idx        = idx
        self.figure     = fig
        self.ax         = fig.axes[0]           if fig          else None
        
        self.window     = kwargs.get( 'window', self.DEFAULT_WINDOW )
        self.cbox       = kwargs.get( 'cbox', 12 )
        self.plot_fits  = kwargs.get( 'plot_fits', 'update' )
        
        #self.stars = Stars( idx, window=self.window )
            
        self.selection = None                   # allows artist resizing only when an artist has been picked
        self.status = 0                         #ready for photometry??
        
        self._write_par = 1                     #should parameter files be written
        
    #===============================================================================================
    def __len__(self):
        return len(self.stars)
    
    #===============================================================================================
    @property
    def rsky(self):
        return np.atleast_2d( self.stars.apertures.sky.radii )[0]       #FIXME: SMELLY!
    skyradii = rsky
   
    @property
    def r_in(self):
        return self.skyradii[0]
    
    @property
    def rphotmax(self):
        if 0 in self.stars.apertures.phot.radii.shape:
            return np.array([self.DEFAULT_WINDOW / 2])
        return [self.stars.apertures.phot.radii.max()]   #FIXME: SMELLY!
    
    #===============================================================================================
    @property
    def rphot(self):
        return np.atleast_2d( self.stars.apertures.phot.radii )[0]   #FIXME: SMELLY!
    
    #np.ceil(SKY_RADII[self.idx,1])
    
    #===============================================================================================
    def build_wcs(self):
        sc = SkyCoord( ra=header['ra'], dec=header['dec'], unit=('h', 'deg') )
        w.wcs.crpix = api.stars.coords[0]
        w.wcs.crval = sc.ra.deg, sc.dec.deg
        w.wcs.cdelt = np.array([-0.076, 0.076]) *8/3600
        w.wcs.ctype = ["RA", "DEC"]

        w.to_header()
    
    #===============================================================================================
    #@profile()
    @unhookPyQt
    def load_image(self, filename=None, data=None, **kw):
        '''
        Load the image for display, applying IRAF's zscale algorithm for colour limits.
        '''
        
        print( 'Loading image {}'.format(filename) )
        
        ##### Load the image #####
        if filename and data is None:
            self.filename = filename
            self.image_data = data = fetch_first_frame( filename )
            self.image_header = quickheader( filename )
            
            #get readout noise and saturation if available
            self.ron = self.image_header.get('ron')
            try:        #Only works for SHOC data
                self.saturation = RNT.get_saturation(filename)                                             #NOTE: THIS MEANS RNT NEEDS TO BE INSTANTIATED BEFORE StarSelector!!!!!!!!
            except Exception as err:
                warn( 'Error in retrieving saturation value:\n' + str(err) )
                self.saturation = 'INDEF'
                    
        elif not data is None:
            self.image_data = data
            self.ron = self.saturation = None
        else:
            raise ValueError( 'Please provide either a FITS filename, or the actual data' )
        
        if kw.get( 'fliplr' , False ):
            data = np.fliplr( data )
            
        #if WCS:            data = np.fliplr( data )
        self.image_data_cleaned = data.copy()
        r,c =  self.image_shape = data.shape
        self.pixels = np.mgrid[:r, :c]                      #the pixel grid

        
        ##### Plot star field #####
        #zilms = zrange(data, sigma_clip=5., maxiter=5, contrast=1./99)
        zlims = np.percentile( data, (0.25, 99.75) )
        
        fig, ax = self.figure, self.ax
        self.canvas = self.figure.canvas
        self.image = ax.imshow( data, origin='lower', cmap = 'gist_heat', vmin=zlims[0], vmax=zlims[1])                  #gist_earth, hot
        #ax.set_title( 'PSF Measure' ) 
        
        #TODO: #PRINT TARGET NAME AND COORDINATES ON IMAGE!! DATE, TIME, ETC.....
        #TODO: Axes Sliders
        #TODO: Tight colourbar
        self.colour_bar = fig.colorbar( self.image, ax=ax )
        #self.stars.zlims = zlims
        #box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,
                #box.width, box.height * 0.9])
        
        

        #create the canvas backgrounds for blitting
        #canvas = FigureCanvas(self.figure)
        #self.backgrounds = [canvas.copy_from_bbox( ax.bbox ) for ax in fig.axes]
        
        self.snap = Snapper( self.image_data_cleaned, 
                                  window=self.window, 
                                  snap='peak', 
                                  edge='clip',
                                  noise_threshold=3,
                                  offset_tolerance=3 )
        
        self.psf = GaussianPSF()
        self.fit = StarFit( self.psf )
        self.fitplotter = psfPlotFactory( self.plot_fits )()
        
        ##### Add ApertureCollection to axes #####
        self.stars = Stars( self.idx, window=self.window )
        self.stars.axadd( ax )
        
        #initialize model plots
        self.stars.model.init_plots( fig )
    
    #=============================================================================================== 
    #@print_args()
    @connect( 'axes_enter_event' )
    def _enter_axes(self, event):
        '''save the canvas background for blitting'''
        if (event.inaxes == self.ax) and not hasattr(self.canvas, 'background'): 
            #First entry triggers background save and connections
            self.canvas.background = self.canvas.copy_from_bbox( self.figure.bbox )
    
    #===============================================================================================    
    @connect( 'pick_event' )
    def _on_pick(self, event):
        '''
        Select an artist (aperture)
        '''
        #if event.artist is self:
        #print( 'PICK!'*10 )
        #print( self )
        #print( vars(event) )
        #print()
        
        self.selection = event
        
        #self.canvas.restore_region( self.canvas.background )
        #event.artist.set_visible( False )
        #event.artist.draw( self.figure._cachedRenderer )
        
        #self.canvas.blit( self.figure.bbox )        
        #self.canvas.background = self.canvas.copy_from_bbox( self.figure.bbox )
        
        #event.artist.set_visible( True )
        #event.artist.draw( self.figure._cachedRenderer )
        #self.canvas.blit( self.figure.bbox )
    
        
        self.add_connection( 'motion_notify_event',
                              self._on_motion )
        
    #===============================================================================================
    #@profile()
    @connect('button_press_event' )
    def _on_click(self, event):
        
        print( vars(event) )
        
        if event.inaxes != self.image.axes:
            return
        
        if event.button==3:
            return
        
        #Middle mouse restarts the selection process
        if event.button==2:
            self.restart()
            return
        
        #Double click to fit stars
        if not event.dblclick:
            return
        
        x,y = event.xdata, event.ydata
        xs, ys = self.snap(x,y)                                           #coordinates of maximum intensity within window / known star
        #print( '!'*100, xs, ys )
        if not (xs and ys):
            return
        else:
            print( '\n\nSNAPPED',  xs, ys, '\n\n' )
            
            #TODO: NEW METHOD HERE??
            
            #Fitting
            data, index = self.snap.zoom.cache
            grid = Y, X = grid_like( data )
            params = self.fit( grid, data )
            Z = self.psf( params, X, Y )
            
            if params is None:
                return
            
            #update fit plots
            self.fitplotter.update( X, Y, Z, data )
            
            #get dict of fit descriptors from params
            info = self.psf.get_description( params, index[::-1] )
            _, sky_sigma = self.psf.background_estimate.cache
            wslice = tuple(map( slice, index, index+data.shape))        #2D window slice
            
            #Add star to collection
            stars = self.stars
            star = stars.append( sky_sigma      =       sky_sigma,
                                 slice          =       wslice,
                                 image          =       data,
                                 #rmax           =       self.rphotmax,
                                 **info                                 )
            
            
            
            #remove star from image by subtracting model
            bg = info['sky_mean']     #background value from fit
            self.snap.clean( Z-bg )
            
            #Check apertures and colourise
            #TODO:  ONLY trigger checks on second draw...........
            psfaps, photaps, skyaps = self.stars.apertures
            skyaps.auto_colour( check='all', 
                                        cross=psfaps, 
                                        edge=self.image_data.shape )
            photaps.auto_colour( check='all',
                                        cross=psfaps,
                                        edge=self.image_data.shape )
            
            #pyqtRemoveInputHook()
            #embed()
            #pyqtRestoreInputHook()
            
            
            stars.model.update( self.fit.get_params_from_cache(), 
                                stars.mean_radial_profile(),
                                self.rsky,
                                self.rphotmax )
            
            #stars.model.apertures.phot.radii = [self.rphotmax]
            
            #stars.model.update_plots()
            
            
            #print( stars.photaps.get_ec() )
            #print( 'x'*100 )
            
            
            #Redraw & blit
            #self.canvas.restore_region( self.canvas.background )
            #stars.draw_all()
            #stars.model.draw()
            #self.canvas.blit( self.figure.bbox )
            
            self.canvas.draw()
    
    #=============================================================================================== 
    @unhookPyQt
    def _on_motion(self, event ):
        #if (self.selection is None):
            #return
        mouseposition = event.xdata, event.ydata
        aps, index = self.selection.artist, self.selection.index
        
        #embed()
        
        rm = aps.edge_proximity( mouseposition, index )
        
        #embed()
        aps.resize( rm, index  )
        
        aps.auto_colour( check='all', 
                         cross=self.stars.apertures.psf, 
                         edge=self.image_data.shape )
        
         ##### model update #####
        model = self.stars.model
        #TODO:  mapping between the image apertures and the mean star apertures. 
        #TODO:  link their radii together apriori and you won't have to do this here.......
        #model.apertures.sky.radii = self.skyradii
        #model.apertures.phot.radii = [self.rphotmax]
        ##self.stars.model.radii = radii
        #model.apertures.auto_colour( check='sky' )
        ##model.apertures.auto_colour()
        model.update_aplines( self.skyradii )
        
        
        #self.canvas.restore_region( self.canvas.background )
        #NOTE: at the moment you have to redraw all the apertures because restoring the bg kills them
        #Another option would be to capture the bg (without selection -->set visible=False) upon selection
        #NOTE: Seeing as your redrawing all of them anyway, consider making them all part of the same collection??
        #self.stars.draw_all()
        #self.canvas.blit( self.figure.bbox )
        
        self.canvas.draw()
    
    #===============================================================================================
    @connect( 'button_release_event' )
    def _on_release(self, event):
        '''
        Resize and deselect artist on mouse button release.
        '''
        if self.selection:
            print( '**\n'*3, 'Releasing', self )
            #self.resize_selection( (event.xdata, event.ydata) )
            self.remove_connection( 'motion_notify_event' )
            #self.draw_blit()
        
        self.selection = None

    #===============================================================================================
    def draw_blit(self):
        self.canvas.restore_region( self.background )
        self.draw( self.figure._cachedRenderer )
        self.canvas.blit( self.figure.bbox )
    
    #=============================================================================================== 
    @connect( 'key_press_event' )
    def _on_key(self, event):
        print( 'ONKEY!!' )
        print( repr(event.key) )
        
        if event.key.lower().startswith('d'):   #either 'delete' key or simply 'd'
            
            #pyqtRemoveInputHook()
            #embed()
            #pyqtRestoreInputHook()
            
            #print( 'psfaps.radii', self.stars.psfaps.radii )
            prox = self.stars.apertures.psf.center_proximity( (event.xdata, event.ydata) )
            
            #print( 'prox', prox )
            idx = np.where( prox==np.min(prox) )
            
            #print( 'where', np.where( prox==np.min(prox) ) )
            
            star = self.stars[idx[0][0]]
            
            #try:
                #il,iu, jl,ju = star._window_idx
                #im = star.star_im
            #except TypeError:
                ##if the stars where found by daofind, the _window_idx would not have been determined!
                ##Consider doing the fits in a MULTIPROCESSING queue and joining here...
                #print( 'star.coo', star.coo )
                #x, y = np.squeeze(star.coo)
                #j,i = int(round(x)), int(round(y))
                #il,iu ,jl,ju, im = self.zoom(i,j)
            
            #pyqtRemoveInputHook()
            #embed()
            #pyqtRestoreInputHook()
            
            
            self.image_data_cleaned[star.slice] = star.image                     #replace the cleaned image data section with the original star image
            self.snap.known.pop( idx[0][0] )
            
            self.stars.remove( idx )
            if len(self.stars):      #TODO:  include this if in the class method
                #NOTE:  not really necessary to check all here.  can check intersections and update lc.
                #       but this will mean remembering lc (which you might want to do anyway if you want to improve the interactions)
                psfaps, photaps, skyaps = self.stars.apertures
                skyaps.auto_colour( check='all', 
                                    edge=self.image_data.shape, 
                                    cross=psfaps )
                photaps.auto_colour( check='all', 
                                        edge=self.image_data.shape, 
                                        cross=psfaps )
            
            #self.canvas.restore_region( self.canvas.background )
            #self.stars.draw_all()
            #self.canvas.blit( self.figure.bbox )
            
            self.canvas.draw()
            
            #self.canvas.background = self.canvas.copy_from_bbox( self.figure.bbox )
    
    
    #===============================================================================================    
    #def _on_motion(self, event):
        #'''
        #Resize aperture on motion.  Check for validity and colourise accordingly.
        #'''
        ##if event.inaxes!=self.image.axes:
            ###prevents resizing when image leaves axis.  This may not be desirable
            ##return
        
        #apertures, idx = self.selection.artist, self.selection.index
        #psfaps, photaps, skyaps = self.stars.get_apertures()
        #skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
        ##skyaps.auto_colour()
        
        #if self.SAMESIZE:
            #idx = ... if len(self.stars)==1 else 0
            #self.SKY_RADII = skyaps.radii[idx].ravel()

        ##### mean star apertures #####
        #model = self.stars.model
        #model.apertures.radii[-2:] = self.SKY_RADII
        ##self.stars.model.radii = radii
        #model.apertures.auto_colour( check='sky' )
        ##model.apertures.auto_colour()
        #model.update_aplines( )
        
        ##self.update_legend()
        #self.figure.canvas.draw()

    #===============================================================================================
    #def auto_colour(self):
        #psfaps, _, skyaps = self.stars.get_apertures()
        #skyaps.auto_colour( check='all', edge=self.image_data.shape, cross=psfaps )
    
    #===============================================================================================
    def restart(self):
        print('Restarting...')
        self.stars.remove_all()
        self.snap.known = []
        self.canvas.restore_region( self.canvas.background )
    
    #===============================================================================================
    ##@profile()
    #def select_stars(self, event):
        #t0 = time()
        #if event.inaxes != self.image.axes:
            #return
        
        #if event.button==3:
            #return
        
        #if event.button==2:
            #self.restart()              #Middle mouse restarts the selection process
            #return
        
        #x,y = event.xdata, event.ydata
        #if None in (x,y):               #Out of bounds on one of the axes
            #return
        
        #xs, ys = self.snap(x,y)                                           #coordinates of maximum intensity within window / known star
        #if not (xs and ys):
            #return
        #else:
            #stars = self.stars
            #fig, ax = self.figure, self.ax
            
            ##Fitting
            #params = self.fit_psf( xs,ys, plot=0 )
            
            ##Add star to collection
            #star = stars.append( **params )
            #print( star )

            ##Updating plots
            #stars.skyaps.auto_colour( check='all', cross=stars.psfaps, edge=self.image_data.shape )
            ##stars.model.update( self.fit.get_params_from_cache(), stars.mean_radial_profile() )
            ##if stars.model.has_plot:
                ##stars.model.update_plots()
            
            ##Drawing and blitting
            ##background = fig.canvas.copy_from_bbox( fig.bbox )
            ##[fig.canvas.restore_region(bg) for bg in self.backgrounds]
            
            ##ax.draw_artist( stars.psfaps )
            ##ax.draw_artist( stars.skyaps )
            ##ax.draw_artist( stars.annotations[-1] )
            
            ##ms = stars.meanstar
            ##ms.set_visible( True )
            ##ms.draw()
            
            ##TODO button checks
            
            ##pyqtRemoveInputHook()
            ###lookup = locals().copy()
            ###lookup.update( globals() )
            ###qtshell( lookup )
            ##embed()
            ##pyqtRestoreInputHook()
            
            ##fig.canvas.blit(fig.bbox)
            #self.figure.canvas.draw()               #BLITTING????

    










if __name__ == '__main__':
    
    fig, ax = plt.subplots()
    api = ApertureInteraction( 0, fig, plot_fits=False )
    
    filename = '/media/Oceanus/UCT/Observing/data/June_2013/V895Cen/0615/202130615.0022.fits'
    api.load_image( filename )
    
    api.connect()
    #api.fitplotter.show()
    
    #Middle mouse restarts the selection process
    
    #'/media/Oceanus/UCT/Observing/data/Feb_2015/0218/ReducedData/20150218.0010.0001.fits'
    #filename = '/media/Oceanus/UCT/Observing/SALT/MASTER_J0614-2725/20150424/product/bxgpS201504240003.fits'
    #filename = '/media/Oceanus/UCT/Observing/data/June_2013/V895Cen/0615/202130615.0022.fits'
    #api.load_image( filename )
    
    
    #snap = Snapper(api.image_data, window=25, snap='peak', edge='edge')
    #data, xypeak = snap.zoom( *snap(47, 50) )
    
    #grid = Y, X = grid_like( data )
    #plsq = fitter(grid, data)
    #Z = psf( plsq, X, Y )
    #plotter.update( X, Y, Z, data )
    
    #info = psf.get_description( plsq )
    #print( info )
    
    plt.show()

