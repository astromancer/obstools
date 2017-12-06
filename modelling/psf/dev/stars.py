import numpy as np

import itertools as itt
from magic.iter import as_iter, interleave, pairwise

from superstring import Table

from ApertureCollections import *
                                #(ApertureCollection, 
                                 #SkyApertures, 
                                 #InteractionMixin, 
                                 #SameSizeMixin)

from obstools.psf.psf import GaussianPSF

from IPython import embed
from PyQt4.QtCore import pyqtRemoveInputHook, pyqtRestoreInputHook

from decor import unhookPyQt, print_args
from magic.string import banner
    

from matplotlib.colors import colorConverter    #HACK!!!

######################################################################################################
# Class definitions
######################################################################################################

#**********************************************************************************************************************************************
ApertureCollection.WARN = False


class PhotApertures( SameSizeMixin, InteractionMixin, ApertureCollection ):
    SAME_SIZE_AXIS = 1
    
class SkyApertures( SameSizeMixin, InteractionMixin, SkyApertures ):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SAME_SIZE_AXIS = 1
    #WARN = False
    #_properties = SkyApertureProperties

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#**********************************************************************************************************************************************
class Star( object ):   #ApertureCollection
    ''' '''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ATTRS = ['coo', 'peak', 'flux', 'fwhm', 'sigma_xy', 'ratio', 'ellipticity', 
             'sky_mean', 'sky_sigma', 'image', 'id', 'slice']#'cache']
    IGNORE = ('image', 'slice', 'rprofile')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, **params):
        
        for key in self.ATTRS:
            #set the star attributes given in the params dictionary
            setattr( self, key, params.get(key) )
            
        if not self.image is None:      #if no stellar image is supplied don't try compute the radial profile
            #self.flux = self.image.sum()
            self.rprofile = self.radial_profile( )                       #containers for radial profiles of fit, data, cumulative data
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __str__(self):
        info = Table( vars(self), 
                        title='Star No. {}'.format(self.id), 
                        title_props={'bg':'blue'},
                        col_headers='Fit params', 
                        ignore_keys=self.IGNORE )  #'cache'
        return str(info)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_params(self):
        raise NotImplementedError
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_params(self, dic):
        raise NotImplementedError
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def radial_profile( self, bins=None ):
        '''Calculate radial profile for star'''
        #NOTE. Maximal radius for calculation depends on size of image.
        
        pix_cen = np.mgrid[self.slice]  + 0.5                           #the index grid for pixel centroid
        rfc = np.linalg.norm( pix_cen.T-self.coo[::-1], axis=-1 ).T     #radial distance of pixel centroid from star centoid
        
        image = self.image - self.sky_mean
        
        if bins is None:
            bins = range(0, min(image.shape)//2+1)
        bin_centres = np.fromiter(map(np.mean, pairwise(bins)), float)
        
        annuli = [image[(rfc>=bl)&(rfc<bu)] for bl,bu in pairwise(bins)]
        val = np.array( [a.mean() for a in annuli] )
        err = np.array( [a.std() for a in annuli] )
        
        return bin_centres, val, err


def w2b(artist):        #TODO:  this can be a decorator!
    ca = artist.get_colors()
    w = np.all(ca == colorConverter.to_rgba_array('w'), 1 )
    ca[w] = colorConverter.to_rgba_array('k')
    artist.set_color( ca )
        

#**********************************************************************************************************************************************
class StarApertures( list ):
    '''Container class for various apertures associated with the star'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    #@unhookPyQt
    def __init__(self, psf, phot, sky):
        '''Initialise from dictionaries of properties, or from ApertureCollection'''
        
        #banner( 'psf', psf , bg='yellow' )
        self.psf     = ApertureCollection( **psf )
       
        #banner( 'phot', phot, bg='cyan' )
        #embed()
        self.phot    = PhotApertures( **phot  )
        
        #banner( 'sky', sky, bg='green' )
        self.sky     = SkyApertures( **sky )
        
        super().__init__( [self.psf, self.phot, self.sky] )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def draw(self):
        
        renderer = self.sky.figure._cachedRenderer
        
        #draw all the apertures
        for aps in self:
            aps.draw( renderer )
       
        #Draw the annotations
        for ann in self.annotations:
            ann.draw( renderer )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def append( self, *props ):
        
        for aps, props in zip(self, props):
            
            #print()
            #print( '!'*100 )
            #print( aps )
            #print()
            aps.append( **props )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def get_radii(self):
        return {key : getattr(self, key).radii 
                    for  key in ['psf', 'sky', 'phot']}
    
    radii = property(get_radii)
    
#**********************************************************************************************************************************************
class RadialProfiles():
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, ax):
        self.ax = ax
        
        #plots
        eprops = dict(capsize=0, ls='', marker='o', ms=5, mec='none')
        null = [0]*4
        self.ax.errorbar( *null, color='g', label='Data', **eprops )
        self.ax.errorbar( *null, color='b', label='Cumulative', **eprops )
        self.ax.plot( 0, 0, color='k', label='Model' )
        
        #self.data = []
        #self.cumulative = []
        #self.model = []
        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @unhookPyQt
    def update(self, rsky):
        
        
        
        self.ax.set_xlim( 0, np.ceil(rsky.max()) + 0.5 )
        
        #rpx = np.arange( 0, self.window )
        #rpxd = np.linspace( 0, self.window )
        #embed()
        xerr = 0.5
        for handle, label in zip(*self.ax.get_legend_handles_labels()):
            
            
            #banner( handle, label, bg='magenta' )

            if label in ('Data', 'Cumulative'):
                x,y,yerr = getattr( self, label.lower() )
                points, caps, (xbars, ybars) = handle
                points.set_data(x,y)
                
                
                try:
                    #NOTE: Don't switch the order of the lines below - if you do, some utterly bizarre fuck-up occurs (with y)
                    ybars.set_segments( np.r_['-1,3,0', np.c_[x, x],
                                                        np.c_[y-yerr, y+yerr]] )

                    xbars.set_segments( np.r_['-1,3,0', np.c_[x-xerr, x+xerr], 
                                                        np.c_[y, y]] )
                except:
                    embed()
            else:
                x,y = getattr( self, label.lower() )
                handle.set_data(x, y)
        
        #for pl, x, y in zip( self.pl_prof, [rpxd, rpx, rpx], self.rprofile ):
            #x = np.arange( 0, len(y) )
            #pl.set_data( x, y )
            
    

#**********************************************************************************************************************************************
class ModelStar( Star ):   #StarApertures
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    psf = GaussianPSF()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    #@unhookPyQt
    def __init__(self, idx, window, resolution=None):
        
        banner( 'ModelStar.__init__', bg='green' )
        
        #TODO:  kill idx
        self.idx = idx
        self.window = window
        self.resolution = resolution or window
        
        #self.psf = psf
        coo = [( window/2, window/2 )]
        wslice = (slice(0, window),)*2
        
        super().__init__( coo=coo[0], 
                          slice=wslice,
                          sky_mean=0 )      #won't calculate rprofile as no image kw
        
        #self.apertures = ApertureCollection( radii=np.zeros(4), 
                                             #coords=fakecoo, 
                                             #colours=['k','w','g','g'], 
                                             #ls=['dotted','dotted','solid','solid'])
        
        orange = colorConverter.to_rgba_array('orangered')         #HACK!
        
        psfaps = dict( coords=[coo,coo], 
                        radii=np.zeros(2), 
                        ls=':', 
                        colours=['k','w'],
                        pickradius=0, )
                        #**kw)
        photaps = dict( coords=coo, 
                        radii=[0],
                        gc='c',
                        badcolour=orange)     #FIXME:  keyword translation
        skyaps = dict( radii=np.zeros(2), 
                        coords=[coo,coo], )
                                    #**kw )
                                    
        self.apertures = StarApertures( psfaps, photaps, skyaps )
        
        self.has_plot = False
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    #@profile()
    @unhookPyQt
    def init_plots(self, fig):
        #print('!'*88, '\ninitializing model plots\n', '!'*88, )
        
        ##### Plots for mean profile #####
        self.mainfig = fig
        _, self.ax_zoom, self.ax_prof, _ = fig.axes
        
        self.radial = RadialProfiles( self.ax_prof )
        
        #TODO:  AxesContainer
        
        self.pl_zoom = self.ax_zoom.imshow([[0]], origin='lower', cmap = 'gist_heat')  #, animated=True #vmin=self.zlims[0], vmax=self.zlims[1]
        self.colour_bar = fig.colorbar( self.pl_zoom, ax=self.ax_zoom )
        
        self.pl_zoom.set_clim( 0, 1 )
        self.pl_zoom.set_extent( [0, self.window, 0, self.window] )

        
        ##### stellar profile + psf model #####
       
        #self.pl_prof = 
        
        #embed()
        
        ##### apertures #####
        for aps in self.apertures:
            lines = aps.aplines = ApLineCollection( aps, self.ax_prof  )
            #lines.set_animated( True )
            self.ax_prof.add_collection( lines )
            #lines.axadd( self.ax_prof )
            aps.axadd( self.ax_zoom )
        
        #w2b(self.apertures.psf.aplines)      #convert white lines to black for displaying on white canvas
        
        ##### sky fill #####
        from matplotlib.patches import Rectangle
        
        trans = self.apertures.psf.aplines.get_transform()
        self.sky_fill = Rectangle( (0,0), width=0, height=1, transform=trans, color='b', alpha=0.3)     #, animated=True
        self.ax_prof.add_artist( self.sky_fill )
        
        self.ax_prof.set_xlim( 0, self.window/2 )
        
        self.has_plot = True
        lines.set_visible( False )
        
        #set all the artist invisible to start with - we'll create a background from this for blitting
        #self.set_visible( False )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @unhookPyQt
    def update(self, cached_params, data_profs, rsky, rphot, plot=True):
        
        print( 'model update' )
        #embed()
        
        window = self.window
        #res = complex(self.resolution)
        Y, X = np.mgrid[0:window, 0:window] + 0.5
        p = cached_params[:]                                    #update with mean of cached_params
        p[:2], p[2], p[-1] = self.coo, 1, 0                     #coordinates, peak, background
        fwhm = self.psf.get_fwhm(p)                             #retrieve FWHM
        
        self.image = self.psf( p, X, Y )
        
        #FIXME: YOU MAY WANT TO INCREASE THE RESOLUTION OF THE RADIAL PROFILE.....
        rmax = min(self.image.shape)
        r = np.linspace(0, rmax, self.resolution)
        self.radial.model = r, self.psf.radial(p, r)
        self.radial.data, self.radial.cumulative = data_profs
        
        print( 'self.radial.data', self.radial.data )
        print( 'self.radial.cumulative', self.radial.cumulative )
        
        self.apertures.psf.radii = 0.5*fwhm, 1.5*fwhm
        self.apertures.sky.radii = rsky
        self.apertures.phot.radii = rphot
        
        
        if plot:
            self.update_plots( rsky )
        
        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@unhookPyQt
    def update_plots(self, rsky):
        if self.has_plot:
            Z = self.image
            self.pl_zoom.set_data( Z )
            
            self.radial.update( rsky )
            
            self.update_aplines( rsky )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #@print_args()
    #@unhookPyQt
    def update_aplines(self, rsky):
        ##### set aperture line position + properties #####
        
        #embed()
        
        for apertures in self.apertures:
            #try:
            apertures.aplines.update_from( apertures )
            apertures.aplines.set_visible( True )      #NOTE:  ONLY HAVE TO DO THIS ON THE FIRST CALL
            #except:
                #pyqtRemoveInputHook()
                #embed()
                #pyqtRestoreInputHook()
            
        #print( '*'*88 )
        #print( self.apertures )
        #print( self.apertures.aplines )
        
        ##### Shade sky region #####
        #TODO: MAKE THIS A PROPERTY!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        sky_width = np.ptp( rsky )
        self.sky_fill.set_x( rsky[0] )
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
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO:  Move to StarApertures class???
    def set_visible(self, state):
        for apertures in self.apertures:
            apertures.set_visible( state )
            apertures.aplines.set_visible( state )
            sky_fill.set_visible( state )
        
        for l in self.pl_prof:
            l.set_visible( state )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def draw(self):
        
        #draw all the apertures
        self.apertures.draw() #for ax_zoom
        
        axz, axp = self.ax_zoom, self.ax_prof
        axz.draw_artist( self.pl_zoom )         #redraw the image
        
        #redraw the ap lines + sky fill
        axp.draw_artist( self.sky_fill )
        for apertures in self.apertures:
            #axz.draw_artist( apertures )
            axp.draw_artist( apertures.aplines )
        
        #redraw the profiles
        for l in self.pl_prof:
            axp.draw_artist( l )



#class LinkRadii


    
#**********************************************************************************************************************************************
class Stars( list ):
    '''
    Class to contain measured values for selected stars in image.
    '''
    
    DEFAUL_SKYRADII = (10., 20.)   #These are the skyradii the stars start with. 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, idx,  **kw):
        
        self.idx = idx
        
        fwhm = kw.pop( 'fwhm', [] )
        coords = kw.pop( 'coords', [] )
        self.window = kw.pop('window',  SkyApertures.R_OUT_UPLIM )  #NOTE: CONSIDER USING THE FRAME DIMENSIONS AS A GUESS?
        
        super().__init__( [Star(coo=coo, fwhm=fwhm) 
                                for (coo,fwhm) in itt.zip_longest(coords, as_iter(fwhm)) ] )
        #self.star_count = len(self.stars)
        resolution = 50
        self.model = ModelStar( idx, self.window, resolution )
        self.plots = []
        self.annotations = []           #TODO:  AnnotationMixin for ApertureCollection????
        
        psfradii = kw.pop( 'psfradii',   np.tile( fwhm, (2,1) ).T * [0.5, 1.5] )
        skyradii = kw.pop( 'skyradii',   np.tile( self.DEFAUL_SKYRADII, (len(fwhm),1) ) )            #SKY_RADII
        photradii =                      np.tile( fwhm, (1,1) ).T * 2.5
        
        apcoo = np.array( interleave( coords, coords ) )                #each stars has 2 apertures.sky and 2 apertures.psf!!!!
        
        #banner( 'Stars.__init__', bg='magenta' )
        
        orange = colorConverter.to_rgba_array('orangered')         #HACK!
        
        self.apertures = StarApertures( dict(coords=apcoo, 
                                                radii=psfradii, 
                                                ls=':', 
                                                colours=['k','w'],
                                                pickradius=0),
                                            
                                        dict(coords=apcoo, 
                                                radii=photradii,
                                                gc='c',
                                                badcolour=orange),    #FIXME:  keyword translation
                                            
                                        dict(coords=apcoo, 
                                                radii=skyradii,
                                                gc='g' ) )
                        
        
        #self.apertures.psf = ApertureCollection( , 
                                          #**kw)
        #self.photaps = PhotApertures(  )
        #self.apertures.sky = SkyApertures( radii=skyradii, 
                                    #coords=apcoo, 
                                    #**kw )
        
        self.has_plot = 0
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #TODO: A better way of doing this is this:  have a class LinkRadii that manages the linked properties???
    def get_skyradii(self):
        return self.apertures.sky.radii[0]
    
    def set_skyradii(self, radii):
        self.apertures.sky.radii = radii
        self.model.apertures.sky.radii = radii[0]
    
    skyradii = property(get_skyradii, set_skyradii)
    
    def get_photradii(self):
        return self.apertures.sky.radii[0]
    
    def set_photradii(self, radii):
        self.apertures.phot.radii = radii
        self.model.apertures.phot.radii = radii[0]
    
    photradii = property(get_skyradii, set_skyradii)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __str__(self):
        attrs = Star.ATTRS.copy()
        ignore_keys = 'image', 'slice', 'rprofile'              #Don't print these attributes
        [attrs.pop(attrs.index(key)) for key in ignore_keys if key in attrs]
        data = [self.pullattr(attr) for attr in attrs]
        table = Table(  data, 
                        title='Stars', title_props={'text':'bold', 'bg':'light green'},
                        row_headers=attrs )
        
        #apdesc = 'PSF', 'PHOT', 'SKY'
        #rep = '\n'.join( '\t{} Apertures: {}'.format( which, ap.radii ) 
                    #for which, radii in zip(apdesc, self.apertures) )
        return str(table)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def __repr__(self):
        #return str(self)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    def __getattr__(self, attr):
        if attr in ('fwhm', 'sky_sigma', 'sky_mean'):
            return self.get_mean_val( attr )
        else:
            return super().__getattribute__(attr)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    #def connect(self):
        #self.apertures.sky.connect()
        #self.apertures.sky.connect()
    

        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def pullattr(self, attr):
        return [getattr(star, attr) for star in self]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def append(self, star=None, **star_params):
        
        if star_params:
            #star_params['id'] = self.star_count
            star = Star( **star_params )             #star parameters dictionary passed not star object
        
        if star.id is None:
            star.id = len(self)
            
        super().append( star )                  #add the Star instance to the list
        #print()
        #print('UPDATING PSFAPS')
        coo, fwhm = star.coo, star.fwhm
        
        #if not np.size(self.apertures.psf.coords):
            #coo = [coo]
        
        self.apertures.append(  dict( coords = [coo, coo],
                                      radii = [0.5*fwhm, 1.5*fwhm] ), 
                                    #pickradius=0, 
                                    #ls=':', 
                                    #gc=['k','w'] ), 
        
                                dict( coords = coo, 
                                      radii = 5*fwhm ),
                                      #gc='c' ),
                                
                                dict( coords = [coo, coo], 
                                      radii = self.DEFAUL_SKYRADII ), )  #These are the skyradii the stars start with. They will be resized by InteractionMixin.append if needed
        
        
        ax = self.apertures.psf.axes
        txt_offset = 1.5*fwhm
        coo = np.squeeze(coo)
        anno = ax.annotate( str(len(self)), 
                            coo, 
                            coo+txt_offset, 
                            color='w',
                            size='small' ) #transform=self.ax.transData
        self.annotations.append( anno )
        
        return star
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_mean_val(self, key):
        if not len(self):
            #raise ValueError( 'The {} instance is empty!'.format(type(self)) )
            return
        if not hasattr( self[0], key ):
            raise KeyError( 'Invalid key: {}'.format(key) )
        vals = [getattr(star,key) for star in self]
        if not None in vals:
            return np.mean( vals, axis=0 )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_unique_coords(self):
        #'''get unique coordinates'''
        #nq_coords = self.apertures.psf.coords          #non-unique coordinates (possibly contains duplicates)
        #_, idx = np.unique( nq_coords[:,0], return_index=1 )
        #_, coords = sorter( list(idx), nq_coords[idx] )
        #return np.array(coords)                   #filtered for duplicates
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_coords(self):
        return np.array([star.coo for star in self])
    
    coords = property(get_coords)        #get_unique_coords
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def axadd( self, ax ):
        #self.apertures.sky.axadd( ax )
        for aps in self.apertures:
            aps.axadd( ax )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def draw(self):
        #ax.draw_artist( stars.psfaps )
        #ax.draw_artist( stars.apertures.sky )
        #ax.draw_artist( stars.annotations[-1] )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def remove(self, idx):
        
        for aps in self.apertures:
            aps.remove( idx )
        
        idx = idx[0][0]
        self.pop( idx )
        rtxt = self.annotations.pop(idx)        
        rtxt.set_visible( False )
        rtxt.set_text(None)
        #n = int(rtxt.get_text())
        
        #FIXME;  is this really the best way??
        #reidentify current stars
        for i, star in enumerate(self[idx:], idx):      #enumerate, but start at idx
            star.id = i
            self.annotations[i].set_text( str(i) )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def remove_all( self ):
        #print( 'Attempting remove all on', self )
        while len(self):        
            #print( self )
            self.remove([[0]])

                ##plt.colorbar()
        #self.fitfig.canvas.draw()
        ##SAVE FIGURES.................
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    @unhookPyQt
    def mean_radial_profile( self ):
        #TODO: RadialProfile class
        
        x,y,e = zip(*[star.rprofile for star in self])                    #radial profile of mean stellar image\

        profiles = get_masked_profiles(y)
        errors = get_masked_profiles(e)
        
        rpm = profiles.mean( axis=0 )
        mx = np.max(rpm)
        rpm /= mx                               #normalized mean radial data profile
        
        ep = np.linalg.norm( errors, axis=0 )    #add errors in quadrature
        ep /= mx
        
        cpm = np.cumsum( rpm )
        cpm /= np.max(cpm)                      #normalized cumulative
        
        return (x[0], rpm, ep), (x[0], cpm, ep)       #FIXME: NEED ERROR CALCULATION FOR CUMULATIVE SUM
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    
def get_masked_profiles(data):
    ml = max(map(len, data))
    
    #ix = np.digitize( tfr.frq[l], bins )
    #apowb = np.array([apower[:,ix==i].mean(1) for i in np.unique(ix)])
    
    profiles = np.array( [np.pad( dat, 
                                (0, ml-len(dat)), 
                                'constant',
                                constant_values=-1 ) 
                        for dat in data] )
    return np.ma.masked_array( profiles, profiles==-1 )