#TODO:  ILLIMINATE IRAF!!! aka. the cumbersome ancient beast
        
#TODO: MAKE MODULAR (mostly complete)

#TODO: re-work horrendous propagate method!!!!!!!!!


#TODO: check buttons for same size apertures

#TODO: Annotations for apertures --> informative data display

#TODO; Draggable lines on profile plot

#TODO: Iterate through frames
#TODO: Colorbar sliders

#FIXME: ap line in profile plot not updating

print( 'Importing modules' )
import os
import sys
import re
import ctypes

import numpy as np
import numpy.linalg as la
import pyfits

from time import time
from scipy.optimize import leastsq

import itertools as itt
import functools as ft
from string import Template
from collections import OrderedDict

from matplotlib import pyplot as plt
from matplotlib import gridspec, rc

import multiprocessing as mp

#import multiprocessing as mp
from zscale import zrange
from pySHOC.readnoise import ReadNoiseTable

from magic.meta import altflaggerFactory
from magic.list import find_missing_numbers, sortmore
from magic.iter import first_true_idx, filtermore, filtermorefalse, partitionmore, zipapp, nthzip


from myio import warn, linecounter
from superplot.multitab import MplMultiTab
from superstring import Table
from superfits import quickheader
from iraf_hacks import PhotStreamer


from obstools.psf.stars import *
from obstools.psf.gui import ApertureInteraction

from PyQt4 import QtGui as qt

from IPython import embed
#print( 'Done!')

from decor import unhookPyQt

#lookup = copy(locals())
#lookup.update( globals() )
#qtshell( lookup )

######################################################################################################    
# Decorators
######################################################################################################    
#needed to make ipython / terminal input prompts work with pyQt
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook

    
######################################################################################################
# Misc Functions
######################################################################################################    

R_IN, R_OUT = SKY_RADII = SkyApertures.RADII





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
        print( 'FAIL!!', win_idx, apiidx )
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



    
def get_func_name(func):
    return func.__name__
    
######################################################################################################
# Class definitions
######################################################################################################

def setup_figure_geometry():
    ''' Set up axes geometry '''
    fig = plt.figure( figsize=(8,16), tight_layout=1 )     #fig.set_size_inches( 12, 12, forward=1 )
    
    gs1 = gridspec.GridSpec(2, 1, height_ratios=(3,1))
    ax = fig.add_subplot( gs1[0] )
    ax.set_aspect( 'equal' )
    
    gs2 = gridspec.GridSpec(2, 2, width_ratios=(1,2), height_ratios=(3,1) )
    ax_zoom = fig.add_subplot( gs2[2], aspect='equal' )
    ax_zoom.set_title( 'PSF Model' )
    
    ax_prof = fig.add_subplot( gs2[3] )
    ax_prof.set_title( 'Mean Radial Profile' )
    ax_prof.set_ylim( 0, 1.1 )
    ax_prof.grid()
    
    return fig

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
    NOTE:  This is not a robust algorithm!
    '''
    maxi = np.squeeze( [np.where(d==d.max()) for d in imdata] )
    return maxi - maxi[0]
    
#====================================================================================================
def blend(data, offsets):
    '''Average a stack of images together '''
    
    def blend_prep(data, offsets):
        '''shift and zero pad the data in preperation for blending.'''
        omax, omin = offsets.max(0), offsets.min(0)
        newshape = data.shape[:1] + tuple(data.shape[1:] + offsets.ptp(0))
        nd = np.empty(newshape)
        for i, (dat, padu, padl) in enumerate(zip(data, omax-offsets, offsets-omin)):
            padw = tuple(zip(padu,padl))
            nd[i] = np.pad( dat, padw, mode='constant', constant_values=0 )
        return nd    
    
    print( 'Doing data blend...' )
    
    data = blend_prep(data, offsets)
    weights = np.ones(data.shape[0])
    return np.average(data, 0, weights)
                
#====================================================================================================
def get_coords_from_file( fn ):
    '''load star cordinates from file and return as array.'''

    if os.path.isfile( fn ):
        
        print( 'Loading coordinate data from {}.'.format( fn ) )
        data = np.atleast_2d( np.loadtxt(fn, unpack=0) )            #in case there is only one star in the coordinate list
        
        if data.shape[-1]==7:                         #IRAF generated coordinate files. TODO: files generated by this script should match this format
            coords, ids = data[:,:2], data[:,6]
            coords -= 1    #Iraf uses 1-base indexing??
            return coords, ids
        
        elif data.shape[-1]==2:
            ids = range(data.shape[0])
            return data, ids
        
        elif 0 in data.shape:
            warn( 'Coordinate file "{}" contains no data!'.format(fn) )
            return None, None
        else:
            warn( 'Coordinate file "{}" format not understood.'.format(fn) )
            return None, None 
    else:
        print( 'Coordinate file {} does not exists!'.format(fn) )
        return None, None 
    
        
        
#====================================================================================================
def undup_coo( file_coo, known_coo, tolerance ):
    '''elliminate duplicate coordinates within radial tolerance.'''
    l_dup = np.array([ np.linalg.norm(known_coo - coo, axis=1)[0] < tolerance
                        for coo in file_coo ])
    return l_dup.ravel()


#===============================================================================================
#@unhookPyQt
def gen_phot_ap_radii( rmax, naperts=None, string=1 ):
    '''
    Generate list of parabolically increasing aperture radii.
    '''
    naperts = naperts       or      Phot.NAPERTS 
    
    #embed()
    
    r = np.linspace(1, rmax, naperts)
    #aps =  np.polyval( (0.085,0.15,2), range(naperts+1) )            #apertures generated by parabola
    if string:
        r = ', '.join( map(str, np.round(r,2)) )
    return r


####################################################################################################    
ButtonsManager, button = altflaggerFactory( collection='_buttons' )   #TODO: auto-name, ordered

class ButtonsMixin( ButtonsManager ):
    
    #====================================================================================================
    @unhookPyQt
    def __init__(self):
        ''' Add GUI Buttons '''
        
        super().__init__()
        
        self.buttonBox = qt.QVBoxLayout()
         
        self.buttons = {}
        colours = self.button_checks()
        for method, (label,) in self._buttons.items():
            button = self.buttons[label] = self.add_button(method, label, colours[method])
            self.buttonBox.addWidget( button )
        
        self.buttonBox.addItem( qt.QSpacerItem(20,200) )
    
    #====================================================================================================
    def button_checks(self):
        
        hf, hs, hp = self.has_found, self.has_stars(), self.has_phot
        checks = [hf, hs, True, 'starfinder' in globals(), 'photify' in globals(), hp, hs, 1]
        gc, ic, bc = 'g', 'orange', 'r'
        colours = [gc if ch else ic for ch in checks]
        
        colour_dic = dict( zip(self._buttons.keys(), colours) )
        colour_dic['daofind'] = gc if 'starfinder' in globals() else bc
        colour_dic['phot'] = gc if 'photify' in globals() else bc
        
        return colour_dic
    
    #====================================================================================================
    def add_button(self, method, label, colour):
            
        button = qt.QPushButton(label, self)
        
        self.set_button_colours( button, colour )
        palette = qt.QPalette( button.palette() ) # make a copy of the palette
        palette.setColor( qt.QPalette.ButtonText, qt.QColor('black') )
        button.setPalette( palette )
        
        button.clicked.connect( method )
        
        return button
    
    #====================================================================================================
    def set_button_colours(self, button, colour, hover_colour=None, press_colour=None):
        
        def colourtuple( colour, alpha=1, ncols=255 ):
            rgba01 = np.array(colorConverter.to_rgba( colour, alpha=alpha ))    #rgba array range [0,1]
            return tuple( (ncols*rgba01).astype(int) )
        
        bg_colour = colourtuple( colour )
        hover_colour = hover_colour     or      colourtuple( colour, alpha=.7 )
        press_colour = press_colour     or      colourtuple( colour, alpha=.5 )
        
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


        
####################################################################################################    
class StarSelector(ButtonsMixin, MplMultiTab):
    '''
    class which contains methods for selecting stars from image and doing basic psf fitting and sky measurements
    '''
    
    WCS = False                                  #Flag for whether to display the image as it would appear on sky
    #====================================================================================================
    @unhookPyQt
    def __init__(self, filelist, coord_fns=None, offsets=None, show_blend=0):
        
        global SKY_RADII
        N = len(filelist)
        SKY_RADII = np.tile( DEFAULT_SKY_RADII, (N,1) )
        
        print( 'SKY_RADII', SKY_RADII )
        
        self.filenames = FileAccess()
        self.status = 0
        #self._write_pars = 1
        
        #Check for existing coordinate files
        self.has_found = not coord_fns in [None, []]            #user defined coordinate files known to exist
        if not self.has_found:
            self.has_found, self.filenames.coo = self.filenames.look_for('coo')
        #if self.has_found:
            #self.filenames.coo = cfg
        
        self.has_phot, self.filenames.mags = self.filenames.look_for('mag')
        
        #load the initialise apis and load figures TODO: MPC
        self.apis = []
        #self.figures = []
        labels = []
        
        
        #Create figures & axes for apis
        #pool = mp.Pool()
        #self.figures = pool.map( setup_figure_geometry, range(len(filelist)+int(show_blend)) )
        #pool.close()
        #pool.join()
        
        self.figures = [setup_figure_geometry() for _ in range(len(filelist)+int(show_blend))]
        
        for i, filename in enumerate(filelist):
            
            api = ApertureInteraction( i, self.figures[i], plot_fits=False )
            api.load_image( filename )
            #bgs = [fig.canvas.copy_from_bbox(ax.bbox) for ax in fig.axes]
            
            #self.figures.append( api.figure )
            self.apis.append( api )
            mo = re.search( '([\d.]+).\w+\Z', filename )           #search for filename number string
            label = mo.groups()[0] if mo else 'Tab %i' %i
            labels.append( label )
        
        #self.imshapes = np.array([api.image_data.shape for api in self.apis])
        self.has_blend = bool(show_blend)
        
        #Guess the default sky radii based on the image dimensions
        guessed_skyradii = np.array( [api.image_shape for api in self] ) / 5
        guessed_skyradii[:,0] -= 5
        
        #api.stars.DEFAULT_SKY_RADII = guessed_skyradii
        print( 'GUESSED', guessed_skyradii )

        
        #SKY_RADII = guessed_skyradii    #np.tile( DEFAULT_SKY_RADII, (N,1) )
        
        #Determine image pixel offsets and blend
        if len(self)==1:
            self.pixoff = np.zeros( (1,2) )
            self.scales = np.ones( (1,2) )
        #else:
            #if offsets is None:
                #data = np.array( [api.image_data for api in self.apis] )    #remember that this is flipped left to right if WCS
                #data, self.scales = samesize(data)
                
                #m = np.median( data, axis=(1,2), keepdims=True )            #median of each image
                #data /= m                                                   #normalised

                #print( 'Determining image shifts...' )
                #self.pixoff = get_pixel_offsets( data )                     #offsets in pixel i,j coordinates
                
                ##Do image blend
                #if show_blend:
                    #self.blend = blend( data, self.pixoff )
                        
                    #api_blend = ApertureInteraction( 0, self.figures[i+1] )
                    #api_blend.load_image( data=self.blend )
                    #self.apis.append( api_blend )
                    #labels.append( 'Blend' )
            
        #translate to on-sky coordinates
        #self.offsets = np.fliplr( np.atleast_2d(self.pixoff) )                         #offsets in x,y coordinates
        if self.WCS:
            self.offsets[:,0] = -self.offsets[:,0]

        #setup shared memory for fitting
        self.shapes = shapes = np.array([api.image_data.shape for api in self])             #image dimensions for each api
        shprod = np.append(0, np.prod(shapes,1) )                               #total number of pixels per image --> indexing 
        totpix = int(np.sum(shprod))                                            #total number of pixels in shared memory
        self.arr_st_idx =  np.cumsum( shprod )                                       #starting indeces for each image frame
        self.shared_arr = mp.Array(ctypes.c_double, totpix )      #shared, can be used from multiple processes
        
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
        
        #pyqtRemoveInputHook()
        #embed()
        #pyqtRestoreInputHook()
        
        #set the focus on the first figure
        self.figures[0].canvas.setFocus()
        
    #====================================================================================================
    def __iter__(self):
        endix= -1 if self.has_blend else len(self)
        idx = slice(0,endix)
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
        
        ButtonsMixin.__init__(self)
        
        hbox = qt.QHBoxLayout()
        hbox.addWidget(self.tabWidget )
        #hbox.addWidget(buttonBox)
        hbox.addLayout(self.buttonBox)
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
    @button( 'WCS' )
    def build_wcs(self):
        pass
    

        
    #====================================================================================================
    #def _on_click(self):
        #sender = self.sender()
        #self.statusBar().showMessage(sender.text() + ' was pressed')  
    
    #===============================================================================================
    @unhookPyQt
    def tabChange(self, i):
        '''When a new tab is clicked, 
            assume apertures already added to axis
            initialise the model star plot (if needed)
            redraw the canvas.
        '''
        api = self[i]
        fig = api.figure
            
        api.stars.axadd( api.ax )
        
        fig.canvas.setFocus()
        if not api.stars.model.has_plot:
            print( 'HERE! '*8 )
            api.stars.model.init_plots( fig )
        fig.canvas.draw()
        #ipshell()
        
    #===============================================================================================
    @button( 'Propagate' )
    #@unhookPyQt
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
            '''
            function used to initialize the multiprocessing pool with shared
            image array as backdrop for ftting.
            '''
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
        
        #embed()
        
        if from_coo is None:
            stars = self[whence].stars
            #R_OUT = stars.R_OUT
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
            R_OUT = SKY_RADII[:,1]
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
        #print( 'IN propagate' )
        #ipshell()
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
        pool = mp.Pool( initializer=init, initargs=(self.shared_arr, self.arr_st_idx, self.shapes) )       #pool of fitting tasks
        cluster = pool.map( starfucker, nargs )
        
        #Group the output by frame
        fitcoo = [s.coo if s else (-1e6,-1e6) for s in cluster]                      #star coordinates from fitting (false coordinates inserted to yield contiguous lists)
        gix, grp = zip( *groupmore( lambda x:x[0], nidx, cluster, fitcoo ) )          #groups the indeces/stars/coordinates according to api idx
        gidx, gstars, fitcoo = zip(*grp)
        gstars = lists(gstars);         fitcoo = lists( fitcoo )
        
        #Filter duplicate coordinates   (this is needed when fitting for position --> sometimes input coordinates that are close to each other converge to the same fit)
        dix = map(where_close_array, fitcoo)                      #indeces of duplicates
        dix, fcix = filtermore(None, dix, itt.count())          #indeces of frames where duplicates are (filter empty results where no duplicates are)
        dix = itt.starmap(ft.partial(nthzip,1), dix)
        for i, dx in zip(fcix, dix):
            for j in dx:
                gstars[i][j] = None
                fitcoo[i][j] = (-1e6,-1e6)                      #flag duplicates
            #gidx[i], gstars[i], fitcoo[i] = filtermore( lambda j: j in dx, gidx[i], gstars[i], fitcoo[i] )       #remove duplicates
            
        #Sort by correspondence
        if narcisistic:
            gstars = sorter(*gstars, key=lambda x: x.coo[::-1] if x else (), order='reverse'  )               #sort the stars by descending y coord
            SI = [np.arange(len(gstars[0]))]                                                                                #sorted indeces
        else:                                   #identify corresponding stars between apis. i.e. preserve star id --> position mapping between frames
            #sequentially append ficticious coordinates to yield contiguous shapes in fit coordinate lists (so we can use array opperations)
            fitcoo = [ list(fc) + [(-1e6,-1e6)]*(len(compcoo)-len(fc)) for fc in fitcoo ]
            fitcoo = (fitcoo/scales[nqidx]) - offsets[nqidx]           #coordinate array --> use the inter-image scale and offset to renormalise the coordinates for comparison
            SI = self.cross_coord_sort(fitcoo, compcoo)
        
        #Set the sky radii for the other frames. 
        #Inner is 3 times the mean hwhm; 
        #Outer is current frame outer scaled by relative image dimension.
        skyradii = np.array(SKY_RADII[::-1]) if whence is None else SKY_RADII[whence,::-1]
        SKY_AREA = np.abs( np.subtract( *np.pi*skyradii.T**2 ) )
        
        #ipshell()
        
        for ix, stargroup, si in zip(gix, gstars, SI): #zip(lbl,finmap):#
            sg, = sort_by_index( stargroup, index=si[si<len(stargroup)] )                        #sort by correspondence (position across frames)
            sg = list(filter(None, sg))                                                         #filter bad fits / duplicate convergence
            
            fwhm = np.mean([star.fwhm for star in sg])
            
            #ipshell()
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
            model = stars.model
            cache = np.mean([star.cache for star in api.stars], 0)
            model.update( tuple(cache), stars.mean_radial_profile() )
            if not model.has_plot:
                model.init_plots( api.figure )

            model.update_plots()
            
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
                mn = find_missing_numbers( list(filter(lambda x:x>0, si))+[len(si)] )
                si[si == -1] = mn
        return SI
        
    #===============================================================================================
    @button( 'load_coords' )
    #@unhookPyQt
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
        
        
        #ipshell()
        
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
            
            if len(self)>1:
                self.propagate( whence=idx, offset_tolerance=3., edge_cutoff=3 )
        
        else:
            coords = [api.load_coo(fn) for (api,fn) in zip(self, fns)]
            coords = list(filter( lambda a: not 0 in a.shape, coords ))
            self.propagate( from_coo=coords, whence=None, peak=None )
            self['c'].figure.canvas.draw()
        
    #===============================================================================================
    @button( 'write_coo' )
    @unhookPyQt
    def write_coo( self, TF=0 ):
        
        if self.WCS:
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

        header = ''
        for coo, outfn in zip(coords, self.filenames.coo):
            print( 'Writing coordinates of selected stars to file: {}'.format(outfn) )
            np.savetxt(outfn, coo, fmt='%.3f', header=header)
            
    
    
    #===============================================================================================
    @button( 'daofind' )
    #@unhookPyQt
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
    #@unhookPyQt
    @button( 'phot' )
    def _on_phot_button( self, event ):
        
        if not self.status:
            self.finalise( )
            
            #for api in iter(self):     api.gen_phot_aps()
        
        #save the parameters (fwhm, sky etc..) to file
        fns = self.filenames
        
        phot = iraf.noao.digiphot.daophot.phot
        iraf.unlearn( phot )
        
        image_setup_filenames = fns.gen( path=fns.path, extension='phot.setup.png' )
        
        for i, api in enumerate(self):
            photify.set_params( api )
            d,c,f,p = nthzip(i, fns.datapars, fns.centerpars, fns.skypars, fns.photpars )

            phot.datapars.saveParList( d )      #filename=d?
            phot.centerpars.saveParList( c )
            phot.fitskypars.saveParList( f )
            phot.photpars.saveParList( p )
            
            filename = image_setup_filenames[i]
            print( 'Saving figure as {}'.format( os.path.basename(filename) ) )
            api.figure.savefig( filename )

        fns = self.filenames
        _, existing = self.clobber_check(fns.mags)
        if len(existing):
            [os.remove(ex) for ex in existing]
        
        args = zip(fns.split, fns.coo, fns.datapars, fns.centerpars, fns.skypars, fns.photpars, fns.mags)
        
        #for a in args:
            #do_phot( a )
        
        #pyqtRemoveInputHook()
        #embed()
        #pyqtRestoreInputHook()
        
        
        pool = mp.Pool()                       #pool of photomerty tasks
        pool.map(do_phot, args )
        pool.close()
        pool.join()
        
        print( 'DONE!' )
        
    #===============================================================================================
    @button( 'CoG' )
    #@unhookPyQt
    def _on_cog_button( self, event ):
        
        print( '\n\nDoing aperture corrections' )
        
        mkapfile = iraf.noao.digiphot.photcal.mkapfile
        iraf.unlearn( mkapfile )
        
        fns = self.filenames
        fns.apcorpars   = path + 'apcor.par'
        fns.apcors      = fns.gen( path=path, extension='apcor.cor' )
        fns.bestmags    = fns.gen( path=path, extension='apcor.mag' )
        fns.logfile     = fns.gen( path=path, extension='apcor.log' )
        fns.obspar      = fns.gen( path=path, extension='obspar' )
        
        #for i, api in enumerate(self):
        apertif.set_params( None )
        mkapfile.saveParList( fns.apcorpars )
        
        for fs in [fns.apcors, fns.bestmags, fns.logfile]:
            _, existing = self.clobber_check( fs )
            lmap( os.remove, existing )
        
        args = zip(fns.mags, fns.apcors, fns.bestmags, fns.logfile, fns.obspar)
        
        pool = mp.Pool()                       #pool of photomerty tasks
        pool.map(do_apcorrs, args )
        pool.close()
        pool.join()
        
        print( 'DONE!' )
        print( 'Best magnitudes:\n', fns.bestmags )
        
        #ipshell()
        
        #apcorrs = ApCorr( self )                #Initialise Iraf task wrapper class
        #apcorrs( )
    
    #===============================================================================================
    @button( 'Restart' )
    def restart(self, event):
        print('Restarting...')
        for api in self:
            api.stars.remove_all()
        
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
    #TODO:  Consider a context manager such as astropy.utils.compat.ignored
    #from astropy.utils.compat import ignored

    def clobber_check(filenames, ask=0):
        if filenames is None:
            return True, []     
            
        clobber = list(filter(os.path.exists, filenames))
        yn = True
        if any(clobber):
            warn( 'The following files are being clobbered:\n{}'.format('\n'.join(clobber)) )
            if ask:
                yn = input( 'Continue ([y]/n)?' ) in ['y', 'Y', '']
        return yn, clobber
    
    #===============================================================================================
    def savefigs(self): 
        
        print( '\n\nSaving figures!' )
        path = self.filenames.path
        fns = self.filenames.gen( path=path,  extension='aps.png' )

        self.clobber_check( fns )
        
        #save images with aps
        for fig, fn in zip(self.figures, fns):
            fig.savefig( fn )
            
        #TODO: SAVE AS PICKLED OBJECT!
    
    #===============================================================================================
    #@unhookPyQt
    def summarise(self):
        keys = 'fwhm', 'sky_mean', 'sky_sigma'
        data = [ [getattr(api.stars, key) for key in keys]
                    + [api.ron, api.saturation]
                    + [tuple(api.skyradii)]               for api in iter(self) ]
        tbl = Table( data, 
                     title='Image properties', title_props={'bg' : 'magenta', 'text' : 'bold'},
                     row_headers=self.filenames.naked, 
                     col_headers=keys+('readout noise', 'saturation', 'SKY RADII') )
        print( tbl )
    
    #===============================================================================================
    def finalise( self ):
        '''
        Write parameter and coordinate files in preparation for photometry.
        '''
        _, nostars = filtermorefalse(len, self, self.filenames.naked)     #filenames of apis with no stars
        if any(nostars):
            warn( 'These images contain no stars:\n'+'\n'.join(nostars)+'\nNo photometry will be done on them!' )
        
        #change button colour
        colour = 'g'
        self.set_button_colours( self.buttons['phot'], colour )
        
        path = self.filenames.path
        self.filenames.split            = self.filenames.gen( path=path, extension='slice' )
        self.filenames.datapars         = self.filenames.gen(path=path, extension='dat.par')
        self.filenames.skypars          = self.filenames.gen(path=path, extension='sky.par')
        self.filenames.centerpars       = self.filenames.gen(path=path, extension='cen.par')
        self.filenames.photpars         = self.filenames.gen(path=path, extension='phot.par')
        self.filenames.mags             = self.filenames.gen( path=path, extension='mag' )
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
        self.image_data_cleaned = self.image_data.copy()                  #reload image data for fitting

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


        
    
        
        
#################################################################################################################################################
class DaoFind( object ):
    DEFAULT_THRESHOLD = 7.5
    #===============================================================================================
    def __init__(self, datapars=None, findpars=None):
        self.datapars = datapars        or      path + 'datapars.par'
        self.findpars = findpars        or      path +'findpars.par'
        
        iraf.noao( _doprint=0 )
        iraf.noao.digiphot( _doprint=0  )
        iraf.noao.digiphot.daophot( _doprint=0 )
        self.daofind = iraf.noao.digiphot.daophot.daofind
    
    #===============================================================================================    
    def __call__(self, image, output):
        print( 'Finding stars...' )
        #datapars, findpars = self.daofind.datapars, self.daofind.findpars
        self.daofind(   image=image,
                        output=output )
                        
                        #fwhmpsf=datapars.fwhmpsf,
                        #sigma=datapars.sigma,
                        #datamax=datapars.datamax,
   
    #===============================================================================================
    def set_params(self, api, threshold=None, **kwargs):
        
        datapars, findpars = self.daofind.datapars, self.daofind.findpars
        self.daofind.verify = 'no'
        
        #daofind.setParList( ParList='home/hannes/iraf-work/uparm/daofind.par' )
        #daofind.datapars.saveParList( path+'datapars.par' )                 #cannot assign parameter attributes without *.par file existing
        #daofind.datapars.unlearn()
        
        datapars.fwhmpsf =          api.stars.fwhm                                 #ELSE DEFAULT
        datapars.sigma =            api.stars.sky_sigma
        datapars.datamax =          api.saturation
        #datapars.gain
        datapars.readnoise =        api.ron
        datapars.exposure =         'exposure'                      #Exposure time image header keyword
        datapars.airmass =          'airmass'                       #Airmass image header keyword
        datapars.filter =           'filter'                        #Filter image header keyword
        datapars.obstime =          'utc-obs'                       #Time of observation image header keyword
        
        findpars.threshold =        threshold  or self.DEFAULT_THRESHOLD                    #Threshold in sigma for feature detection
        findpars.nsigma =           1.5                             #Width of convolution kernel in sigma
        findpars.ratio  =           1.                              #Ratio of minor to major axis of Gaussian kernel
        findpars.theta  =           0.                              #Position angle of major axis of Gaussian kernel
        findpars.sharplo=           0.2                             #Lower bound on sharpness for feature detection
        findpars.sharphi=           1.                              #Upper bound on sharpness for  feature detection
        findpars.roundlo=           -1.                             #Lower bound on roundness for feature detection
        findpars.roundhi=           1.                              #Upper bound on roundness for feature detection
        findpars.mkdetec=           'no'                            #Mark detections on the image display ?
    
    #===============================================================================================    
    def save_params( self ):
        datapars, findpars = self.daofind.datapars, self.daofind.findpars
        datapars.saveParList( self.datapars )
        findpars.saveParList( self.finpars )
        
        
#################################################################################################################################################
class Phot(object):
    NAPERTS = 12
    #===============================================================================================
    def __init__(self):
        
        #self.fitskypars = selector.filenames.gen(path=path, extension='sky.par')
        #self.centerpars = selector.filenames.gen(path=path, extension='cen.par')
        #self.photpars = selector.filenames.gen(path=path, extension='phot.par')
        
        iraf.noao()
        iraf.noao.digiphot()
        iraf.noao.digiphot.daophot()
        self.phot = iraf.noao.digiphot.daophot.phot
    
    #===============================================================================================    
    #def __call__(self, image_list, output):
        #'''
        #image_list - txt list of images to do photometry on
        #output - output magnitude filename 
        #'''
        
        #if os.path.isfile( path+output ):
            #os.remove( path+output )

        #print('Doing photometry')
        #coords = api.filenames.coo[api.idx]
        #image = '@'+image_list
        
        #datapars, centerpars, fitskypars, photpars = self.phot.datapars, self.phot.centerpars, self.phot.fitskypars, self.phot.photpars
        #self.phot(      image=image,
                        #coords=coords,
                        #datapars=datapars,
                        #centerpars=centerpars,
                        #fitskypars=fitskypars,
                        #photpars=photpars,
                        #output=output)
    
    #===============================================================================================
    def set_params(self, api):
        
        datapars, centerpars, fitskypars, photpars = self.phot.datapars, self.phot.centerpars, self.phot.fitskypars, self.phot.photpars
        
        datapars.fwhmpsf = api.stars.fwhm
        datapars.sigma = api.stars.sky_sigma
        datapars.datamax = api.saturation
        #datapars.gain
        datapars.readnoise =           api.ron
        datapars.exposure =            'exposure'                      #Exposure time image header keyword
        datapars.airmass =             'airmass'                       #Airmass image header keyword
        datapars.filter =              'filter'                        #Filter image header keyword
        datapars.obstime =             'utc-obs'                       #Time of observation image header keyword
        datapars.mode = 'h'
        
        centerpars.calgorithm = "centroid"                             #Centering algorithm
        centerpars.cbox = api.cbox                                   #Centering box width in scale units
        centerpars.cthreshold = 3.                                     #Centering threshold in sigma above background
        centerpars.minsnratio = 1.                                     #Minimum signal-to-noise ratio for centering algorithim
        centerpars.cmaxiter = 10                                       #Maximum iterations for centering algorithm
        centerpars.maxshift = 15.                                       #Maximum center shift in scale units
        centerpars.clean = 'no'                                        #Symmetry clean before centering
        centerpars.rclean = 1.                                         #Cleaning radius in scale units
        centerpars.rclip = 2.                                          #Clipping radius in scale units
        centerpars.kclean = 3.                                         #K-sigma rejection criterion in skysigma
        centerpars.mkcenter = 'no'                                     #Mark the computed center
        centerpars.mode = 'h'
        
        #NOTE: The sky fitting  algorithm  to  be  employed.  The  sky  fitting
                #options are:

                #constant                                                                       <----- IF STARS IN SKY
                        #Use  a  user  supplied  constant value skyvalue for the sky.
                        #This  algorithm  is  useful  for  measuring  large  resolved
                        #objects on flat backgrounds such as galaxies or commets.

                #file                                                                           <----- IF PREVIOUSLY KNOWN????
                        #Read  sky values from a text file. This option is useful for
                        #importing user determined sky values into APPHOT.

                #mean                                                                           <----- IF LOW SKY COUNTS
                        #Compute  the  mean  of  the  sky  pixel  distribution.  This
                        #algorithm  is  useful  for  computing  sky values in regions
                        #with few background counts.

                #median                                                                         <----- IF RAPIDLY VARYING SKY / STARS IN SKY
                        #Compute the median  of  the  sky  pixel  distribution.  This
                        #algorithm  is  a  useful for computing sky values in regions
                        #with  rapidly  varying  sky  backgrounds  and  is   a   good
                        #alternative to "centroid".

                #mode                                                                           <----- IF CROWDED FIELD AND STABLE SKY
                        #Compute  the  mode  of  the sky pixel distribution using the
                        #computed  mean  and  median.   This   is   the   recommended
                        #algorithm  for  APPHOT  users  trying  to  measuring stellar
                        #objects in crowded stellar  fields.  Mode  may  not  perform
                        #well in regions with rapidly varying sky backgrounds.

                #centroid                                                                       <----- DEFAULT
                        #Compute  the  intensity-weighted mean or centroid of the sky
                        #pixel histogram.  This  is  the  algorithm  recommended  for
                        #most  APPHOT  users.  It  is  reasonably  robust  in rapidly
                        #varying and crowded regions.

                #gauss
                        #Fit a single Gaussian function to the  sky  pixel  histogram
                        #using non-linear least-squares techniques.

                #ofilter
                        #Compute  the sky using the optimal filtering algorithm and a
                        #triangular weighting function and the histogram of  the  sky
                        #pixels.
        
        swidth = abs(np.subtract( *api.skyradii ))
        
        fitskypars.salgorithm = "centroid"                             #Sky fitting algorithm
        fitskypars.annulus = api.r_in                                  #Inner radius of sky annulus in scale units
        fitskypars.dannulus = swidth                                   #Width of sky annulus in scale units
        fitskypars.skyvalue = 0.                                       #User sky value                                 #self.sky_mean
        fitskypars.smaxiter = 10                                       #Maximum number of sky fitting iterations
        fitskypars.sloclip = 3.                                        #Lower clipping factor in percent
        fitskypars.shiclip = 3.                                        #Upper clipping factor in percent
        fitskypars.snreject = 50                                       #Maximum number of sky fitting rejection iterations
        fitskypars.sloreject = 3.                                      #Lower K-sigma rejection limit in sky sigma
        fitskypars.shireject = 3.                                      #Upper K-sigma rejection limit in sky sigma
        fitskypars.khist = 3.                                          #Half width of histogram in sky sigma
        fitskypars.binsize = 0.1                                       #Binsize of histogram in sky sigma
        fitskypars.smooth = 'no'                                       #Boxcar smooth the histogram
        fitskypars.rgrow = 0.                                          #Region growing radius in scale units
        fitskypars.mksky = 'no'                                        #Mark sky annuli on the display
        fitskypars.mode = 'h'
        
        photpars.weighting = "constant"                                #Photometric weighting scheme
        photpars.apertures = gen_phot_ap_radii( api.rphotmax[0] )       #List of aperture radii in scale units
        photpars.zmag = 25.                                            #Zero point of magnitude scale
        photpars.mkapert = 'no'                                        #Draw apertures on the display
        photpars.mode = 'h'
        
    #===============================================================================================
    #def save_params(self, i):
        #phot = self.phot
        #c,f,p = nthzip(i, self.centerpars, self.fitskypars, self.photpars )
        #phot.centerpars.saveParList( c )
        #phot.fitskypars.saveParList( f )
        #phot.photpars.saveParList( p )
    

        
#################################################################################################################################################
class ApCorr( object ):
    
    #===============================================================================================
    def __init__(self):
        
        #self.mkappars = mkappars if mkappars else path+'phlmkapfe.par'
        #self.photfile =  selector.filenames.mags
        #self.naperts = selector.naperts
        
        iraf.noao()
        iraf.noao.digiphot()
        iraf.noao.digiphot.photcal()
        self.mkapfile = iraf.noao.digiphot.photcal.mkapfile
    
    #===============================================================================================    
    #@unhookPyQt
    #def __call__( self, photfile, apercors='all.mag.cor', bestmagfile='apcor.mag', logfile='apcor.log'):
        
        #apercors = path + apercors
        #bestmagfile = path + bestmagfile
        #logfile = path + logfile
        
        #for fn in [apercors, bestmagfile, logfile]:                             #OVERWRITE FUNCTION
            #if os.path.isfile(fn):       os.remove(fn)
        
        #print('Doing aperture corrections...')
        #naperts = self.mkapfile.naperts
        #self.mkapfile( photfile=photfile, 
                        #naperts=naperts, 
                        #apercors=apercors, 
                        #magfile=bestmagfile,
                        #logfile=logfile         )

        #selector.filenames.apercors = apercors
        #selector.filenames.magfile = bestmagfile
        #selector.filenames.logfile = logfile
        
    #===============================================================================================   
    def set_params( self, api ):
        
        mkapfile = self.mkapfile
        #mkapfile.photfiles = api.mags                                       #The input list of APPHOT/DAOPHOT databases
        mkapfile.naperts = Phot.NAPERTS                                     #The number of apertures to extract
        #mkapfile.apercors = apercors                                       #The output aperture corrections file
        mkapfile.smallap = 2                                                #The first aperture for the correction
        mkapfile.largeap = Phot.NAPERTS                                     #The last aperture for the correction
        #mkapfile.magfile = magfile                                         #The optional output best magnitudes file
        #mkapfile.logfile = logfile                                         #The optional output log file
        mkapfile.plotfile = ''                                              #The optional output plot file
        mkapfile.obsparams = ''                                             #The observing parameters file
        mkapfile.obscolumns = '2 3 4 5'                                     #The observing parameters file format
        mkapfile.append = 'no'                                              #Open log and plot files in append mode
        mkapfile.maglim = 0.1                                               #The maximum permitted magnitude error
        mkapfile.nparams = 3                                                #Number of cog model parameters to fit
        mkapfile.swings = 1.2                                               #The power law slope of the stellar wings
        mkapfile.pwings = 0.1                                               #The fraction of the total power in the stellar wings
        mkapfile.pgauss = 0.5                                               #The fraction of the core power in the gaussian core
        mkapfile.rgescale = 0.9                                             #The exponential / gaussian radial scales
        mkapfile.xwings = 0.                                                #The extinction coefficient
        mkapfile.interactive = 'no'                                         #Do the fit interactively ?
        mkapfile.verify = 'no'                                              #Verify interactive user input ?
        mkapfile.graphics = 'no'
        
    #===============================================================================================    
    def save_params( self ):
        self.mkapfile.saveParList( self.mkappars )
        
        
        
#################################################################################################################################################

class Cube(StarSelector):                                                #NEED TO FIGURE OUT HOW TO DO THIS WITH SUPER
    def __init__(self, stars):
        self.filenames = stars.get_fn()
        self.fwhmpsf = stars.fwhmpsf
        self.sky_mean = stars.sky_mean
        self.sky_sigma = stars.sky_sigma
        self.saturation = stars.saturation
        
    #===============================================================================================
    def get_ron(self):
        return float( quickheader( self.filenames.image ).get('ron') )
        #return float(pyfits.getval(self.filenames.image,'ron'))

    #===============================================================================================
    def sky_cube(self, im_list, stars, n):
        '''Determine the variation of the background sky across the image cube.'''
        '''n - number of images in cube to measure. '''
        print('Determining sky envelope from subsample of %i images...' %n)
        im_list = np.array(im_list)
        N = len(im_list)
        j = N/2                                                                     #use image in middle of cube for sky template
        im_data = pyfits.getdata( im_list[j] )
        sky_envelope = np.zeros( im_data.shape )
        L = stars.sky_map()                         #!!!!!!!!!!!!!!!!!!!!!                                  #sky template
        Bresenham = lambda m, n: np.array([i*n//m + n//(2*m) for i in range(m)])    #select m points from n integers using Bresenham's line algorithm
        ind = Bresenham(n,N)                                                        #sequence of images used in determining sky values
        sky_mean = []; sky_sigma = []
        for im in im_list[ind]:
            im_data = pyfits.getdata( im )
            sky_mean.append( np.mean( im_data[L] ))                                   #USE MEDIAN FOR ROBUSTNESS???? (THROUGHOUT??)
            sky_sigma.append( np.std( im_data[L] ))
            sky_envelope = np.max([sky_envelope,im_data], axis=0)
        #sky_envelope[~L] = 0
        self.sky_mean = np.mean(sky_mean)
        self.sky_sigma = np.mean(sky_sigma)
        return ind, sky_mean, sky_sigma, sky_envelope

#################################################################################################################################################

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

#TODO: if __name__ == '__main__':

#Initialise readout noise table
RNT = ReadNoiseTable()
RUN_DIR = os.getcwd()

#Parse command line arguments
import argparse, glob
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--dir', default=os.getcwd(), dest='dir', 
                    help = 'The data directory. Defaults to current working directory.')
parser.add_argument('-o', '--output-dir',
                    help = ('The data directory where the reduced data is to be placed.' 
                            'Defaults to input directory'))
parser.add_argument('-x', '--coords', dest='coo', 
                    help = 'File containing star coordinates.')
parser.add_argument('-c', '--cubes', nargs='+',
                    help = 'Science data cubes to be processed.  Requires at least one argument.  Argument can be explicit list of files, a glob expression, or a txt list.')
#parser.add_argument('-l', '--image-list', default='all.split.txt', dest='ims', help = 'File containing list of image fits files.')
parser.add_argument('-tel', '--telescope', nargs=1, default='1.9m',
                    help = 'Telescope on which data was taken.  Will determine pixel scale and FOV for WCS.')

args = parser.parse_args()

from pySHOC.pre import SHOC_Run
from myio import iocheck, parsetolist

if args.cubes is None:
    args.cubes = args.dir       #no cubes explicitly provided will use list of all files in input directory

path = iocheck(args.dir, os.path.exists, raise_error=1)
path = os.path.abspath(path) + os.sep
#fns = parsetolist( args.cubes )

print( args )

if args.output_dir:
    args.output_dir = iocheck(args.output_dir, os.path.exists, 1)
else:
    args.output_dir = path

#Read the cubes as SHOC_Run
#ipshell()
args.cubes = parsetolist( args.cubes, os.path.exists, path=path, raise_error=1 )
cubes = SHOC_Run( filenames=args.cubes, label='sci' )

#generate the unique filenames (as was done when splitting the cube)
#reduction_path = os.path.join( path, 'ReducedData' )
basenames = cubes.magic_filenames( args.output_dir )
#fns = cubes.genie(1)[0]



#Read input coordinates
if args.coo:
    args.coo = parsetolist(args.coo, os.path.exists,  path=path)

    
class FileAccess(object):
    #====================================================================================================
    basenames           =       basenames
    images              =       args.cubes#fns
    naked               =       [os.path.split(fn)[1] for fn in images]
    coo                 =       args.coo
    path                =       path
    reduction_path      =       args.output_dir
    
    #====================================================================================================
    def gen(self, **kw):
        '''Generator of filenames of unpacked cube.'''
        path = kw.get( 'path', '' )
        if path:
            path = os.path.abspath(path)
        sep = kw.get( 'sep', '.' )
        extension = kw.get('extension', 'fits' )
        
        return ['{}{}{}'.format( os.path.join(path, base), sep, extension ) for base in self.basenames]

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
rc( 'savefig', directory=path )
#rc( 'figure.subplot', left=0.065, right=0.95, bottom=0.075, top=0.95 )    # the left, right, bottom, top of the subplots of the figure
#################################################################################################################################################

DEFAULT_SKY_RADII = SkyApertures.RADII

pixscales = np.array([cube.get_pixel_scale( args.telescope ) for cube in cubes])

print( 'Importing IRAF...' )
t0 = time()
from pyraf import iraf
from stsci.tools import capable
print( 'Done! in ', time()-t0, 'sec')
capable.OF_GRAPHICS = False                     #disable pyraf graphics                                 
#iraf.prcacheOff()
#iraf.set(writepars=0)



#Initialise Iraf task wrapper classes
starfinder = DaoFind( )
photify = Phot( )
apertif = ApCorr( )


def do_phot(args):
    """ processing delegated to child process """
    #print("Running imstat on:"+fname+", with format = "+str(showformat))
    #iraf.imstat(fname, format=showformat, cache=False)
    #print()
    #print( 'WHOOP!' )
    #print()
    
    
    
    phot = iraf.noao.digiphot.daophot.phot
    image, coords, datapars, centerpars, fitskypars, photpars, output = args
    
    bar = PhotStreamer( infospace=10 )
    N = linecounter(image)
    bar.create( N )
    
    image = '@'+image
    
    phot.datapars.setParList( ParList=datapars )
    phot.centerpars.setParList(  ParList=centerpars )
    phot.fitskypars.setParList( ParList=fitskypars )
    phot.photpars.setParList( ParList=photpars )
    
    print( 'phot.datapars.fwhmpsf', phot.datapars.fwhmpsf )
    print( 'phot.datapars.sigma', phot.datapars.sigma )
    print( 'phot.fitskypars.annulus', phot.fitskypars.annulus )
    print( 'phot.photpars.apertures', phot.photpars.apertures )
    
    #return
    
    q = phot(   image=image,
                coords=coords,
                output=output, 
                Stderr=1)  #Stdout=bar.stream
    return q
            
def do_apcorrs(args):
    naperts = Phot.NAPERTS
    mkapfile = iraf.noao.digiphot.photcal.mkapfile
    
    photfile, corfile, bestmagfile, logfile, obspar = args
    mkapfile( photfile=photfile, 
                naperts=naperts, 
                apercors=corfile, 
                magfile=bestmagfile,
                obspar=obspar,
                logfile=logfile,
                interactive='no'         )
    
            
#Check for pre-existing psf measurements
try:
    fitskypars = par_from_file( path+'fitskypars.par' )
    sky_inner = float( fitskypars['annulus'] )
    sky_d = float( fitskypars['dannulus'] )
    sky_outer = sky_inner +  sky_d
    
    datapars = par_from_file( path+'datapars.par' )
    fwhmpsf = float( datapars['fwhmpsf'] )
    sky_sigma = float( datapars['sigma'] )
    ron = float( datapars['readnoise'] )
    saturation = datapars['datamax']
    
    msg = ('Use the following psf + sky values from fitskypars.par and datapars.par? ' 
                '\nFWHM PSF: {:5.3f}'
                '\nSKY SIGMA: {:5.3f}'
                '\nSKY_INNER: {:5.3f}'
                '\nSKY_OUTER: {:5.3f}\n'
                '\nREADOUT NOISE: {:5.3f}'
                '\nSATURATION: {}'.format( fwhmpsf, sky_sigma, sky_inner, sky_outer, ron, saturation ) )
    print( msg )
    yn = input('([y]/n):')
except IOError:
    yn = 'n'
    #mode = 'fit'

    
#Initialise GUI
app = qt.QApplication(sys.argv)
selector = StarSelector( args.cubes, args.coo, show_blend=0 )
    
    
if yn in ['y','Y','']:
    selector.fwhmpsf, selector.sky_mean, selector.sky_sigma, selector.ron = fwhmpsf, sky_mean, sky_sigma, ron
    #mode = 'select'


#THIS NEEDS TO HAPPEN IN A QUEUE!!!!!!!!!!!! 
#ipshell()

selector.show()
sys.exit( app.exec_() )

#daofind, phot = queue.get()





#L = selector.sky_map()
#sky_data = copy(selector.image_data)
#sky_mean = np.mean(sky_data[L])                                        #USE MEDIAN FOR ROBUSTNESS???? (THROUGHOUT??)
#sky_sigma = np.std(sky_data[L])
#sky_data[~L] = 0

#plt.figure(-1)
#plt.imshow(sky_data)


#Here the sky envelope + skybox
#stack = Cube(selector)
#inds, cube_mean, cube_sigma, sky_env = stack.sky_cube(im_list,selector,100)
#print 'Cube sky mean = ', np.mean(cube_mean)
#print 'Cube sky stdev = ', np.mean(cube_sigma)





#plt.figure(3)
#plt.errorbar(inds, cube_mean, yerr=cube_sigma, fmt='o')
#plt.title( 'Mean Sky Counts' )
#plt.ylabel( 'Counts / pixel' )

#plt.figure(4)
#zlims = plf.zscale_range(sky_env,contrast=1./99)
#plt.imshow(sky_env, origin='lower', vmin=zlims[0], vmax=zlims[1])
#plt.colorbar()
#plt.title('Sky envelope')



#selector.filenames.phot_coo = 'phot.coo'
#selector.filenames.mags = 'allmag'                                 #NOTE: YOU NEED TO DO THIS OUTSIDE THIS METHOD IF YOU WISH TO BE ABLE TO RUN phot AND ap_cor    METHODS INDEPENDANTLY........................



##stars.aperture = [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 14, 16]
#stack.naperts = 10


#stack.phot('all.split.txt', stars, 'allmag')
#stack.ap_cor()

def goodbye():
    os.chdir(RUN_DIR)
    print('Adios!')

import atexit
atexit.register( goodbye )
