

from matplotlib.patches import Circle
from matplotlib.collections import Collection
from matplotlib import cbook
from matplotlib.colors import colorConverter

class Aperture( Circle ):
    '''
    Class defines an Aperture as subclass of matplotlib Circle
    '''
    #===============================================================================================
    RADIUS = 5.
    BAD_COLOUR = 'y'
    GOOD_COLOUR = 'g'
    #===============================================================================================
    def __init__(self, coo, radius=None, **kw):
        '''
        Create aperture instance at coordinates coo, with radius.
        Checks for image edge overlap if check is True.
        '''
        check = kw.pop('check')                 if 'check' in kw                else 0
        picker = kw['picker']                   if 'picker' in kw               else 0
        self.good_colour = kw['good_colour']    if 'good_colour' in kw          else self.GOOD_COLOUR
        self.good_colour = kw['edgecolor']      if 'edgecolor' in kw            else self.GOOD_COLOUR
        self.bad_colour = kw['bad_colour']      if 'bad_colour' in kw           else self.BAD_COLOUR

        super(Aperture, self).__init__(coo, radius, **kw)

        if picker:
            self.set_picker( self.pick_handler )

        self.state = self.check_edge()  if check                else 0
        c = self.bad_colour             if self.state           else self.good_colour
        self.set_ec( c )


    #===============================================================================================
    def __repr__(self):
        xc, yc = self.center
        return 'Aperture( ({:3.2f},{:3.2f}), r={:3.2f} )'.format(xc, yc, self.radius)

    #===============================================================================================
    def __str__(self):
        return 'Aperture r={:3.2f}'.format( self.radius )

    #===============================================================================================
    def center_proximity(self, mouse_position):
        x, y = mouse_position
        if not( x and y ):
            return

        xc, yc = self.center

        r = np.sqrt((x-xc)**2 + (y-yc)**2)          #radial cursor position relative to circle center
        return r

    #===============================================================================================
    def edge_proximity(self, mouse_position):
        '''
        Calculate the proximity of a given position to the aperture edge. Used for event control.
        Parameters
        ----------
        mouse_position : tuple with x, y coordinates

        Returns
        -------
        rp : radial distance in pixels from aperture edge.
        '''
        x, y = mouse_position
        if not( x and y ):
            return

        rc = self.center_proximity( mouse_position )
        r = self.radius

        rp = rc - r                                   #proximity of event position to circle edge
        return rp

    #===============================================================================================
    @staticmethod
    def pick_handler(artist, event):
        '''
        The picker function
        '''
        print( 'Aperture pick handler' )
        tolerance = 1.0
        mouse_position = (event.xdata, event.ydata)
        if event.xdata and event.ydata:
            rp = artist.edge_proximity( mouse_position )
            hit = abs(rp) < tolerance
            props = {}
        else:
            hit, props = False, {}

        return hit, props

    #===============================================================================================
    def check_edge(self, pr=0):
        '''
        Checks whether the aperture crosses the image edge.

        Returns
        -------
        l : boolean flag
        '''
        try:
            coo = np.array( self.center )
            q1 = coo + self.radius >= selector.image_shape
            q2 = coo - self.radius <= (0., 0.)
            return np.any( q1 | q2 )
        except:
            if pr:
                print( 'WARNING: Aperture image edge check failed for {}'.format(repr(self)) )
            return False

    #===============================================================================================    
    def plot(self, ax=None):
        '''
        Add the aperture to the figure and redraw.
        '''
        ax = ax if ax else self.ax
        ax.add_artist( self )
        ax.draw_artist( self )


class Apertures( Collection ):
    '''
    Class that handles methods common to image apertures collection.
    '''
    #===============================================================================================
    RADIUS = Aperture.RADIUS
    BAD_COLOUR = Aperture.BAD_COLOUR
    GOOD_COLOUR = Aperture.GOOD_COLOUR

    #map the Apertures attributes to the Aperture attribute
    prop_dic = {'radii': 'radius',
                'ec' :  'edgecolor',
                'colour' : 'edgecolor' ,
                'colours' : 'edgecolor',
                'fc' : 'facecolor',
                'facecolour' : 'facecolor',
                'facecolours' : 'facecolor',
                'facecolors' : 'facecolor'
                }
    #===============================================================================================
    @classmethod
    def _prop_map( cls, attr ):
        '''map the Apertures attributes to the Aperture attribute.'''
        if attr in cls.prop_dic:
            return cls.prop_dic[attr]
        else:
            return attr

    #===============================================================================================
    @classmethod
    def _prepdict( cls, dic, ndmin=0 ):
        '''
        Iterates through dictionary key, val pairs fixing keys according to mappings in prop_dic, and
        cycling through the items that are of type list such that all new lists cast at the same length
        as the longest list, or length given by ndmin.  If unzip is True, returns generator object that
        creates n dic with the same (fixed) keywords and a single cycled value per keyword.  Else return
        the fixed dictionary with n cycled values per keyword.
        '''
        newdic = {}
        def lenotype(obj, types=(list,) ):
            '''returns the length of an object if object has type in types'''
            return len(obj) if isinstance( obj, types ) else 0

        ndmx0 = max(map(lenotype, dic.values()))  #maximal object length amongs dic items
        if ndmx0 < ndmin:
            ndmx0 = ndmin

        for key, obj in dic.items():
            key = cls._prop_map(key)            #replace property keywords from prop_dic
            if 'color' in key:
                obj = list( map( colorConverter.to_rgba, obj ) )

            newdic[ key ] = self._prep_vals( obj, ndmx0 )

        return newdic

    #===============================================================================================
    @staticmethod
    def _zipdic( dic ):
        '''unzips a dictionary with fields each of len n and yields n dictionaries with the same keywords 
        Used to propagate the artist properties downwards.'''
        for vals in zip(*dic.values()):
            yield dict( zip(dic.keys(), vals) )

    #===============================================================================================
    @classmethod
    def cycle( cls, obj, mx ):
        obj = makelist( obj )
        cyc = itertools.cycle( obj )
        for i in range(mx):
            yield next(cyc)

    #===============================================================================================
    @classmethod
    def _prep_vals( cls, obj, ndmax=1 ):
        '''cast vals into list of same length as radii.'''
        if not isinstance( obj, list ):
            obj = [obj]                     #cast all values into lists first
        if len(obj)==ndmax:
            return obj                      #return the list if it has the right size

        warning = "WARNING: Keyword '{}' contains list of {} items ({} ndmax = {}). Values will be {}."
        if len(obj)<ndmax:
            gtlt = '<'; msg = 'cycled.'
        elif len(obj)>ndmax:
            gtlt = '>'; msg = 'truncated'
        print( warning.format(key, len(obj), gtlt, ndmax, msg) )

        return list( cls.cycle(obj, ndmax) )

    #===============================================================================================
    def __init__(self, **kw ):

        if not ('radii' in kw or 'radius' in kw):
            raise ValueError( "Please specify the radii. The shape of radii determines the shape of Apertures, unless keyword 'shape' is given." )
        else:
            self.radii = kw.pop('radii')

        self.shape =  kw['shape'] if 'shape' in kw else np.shape( self.radii )
        self.ndims = len(self.shape)
        self.size = np.prod( self.shape )

        np.reshape( self.radii, self.shape ) #Error catch

        kw = self._prepdict( kw, self.size )
        print( 'Whoop! '*3, '\n', kw )
        ap_kws = self._zipdic( kw )

        self.coords = kw.pop['coords']

        self.apertures = []
        for coo, r_ap, kw in zip(self.coords, self.radii, ap_kws):
            self.apertures.append( Aperture(coo, r_ap, **kw) )

        check = kw['check']                 if 'check' in kw                else 0
        #picker = kw['picker']                   if 'picker' in kw               else 0

        #Collection.__init__( self, **kw )
        #self.set_picker( self.picker )

        #dynamic
        self.state = self.check_edge( selector.image_shape ) if check else self._prep_vals( False )
        #self.radii = self.get('radii')
        self.colours = kw['edgecolor']

        #static
        self.bad_colours = kw['bad_colour']

        #self.set( 'state', state )
        #self.set( 'colours', colours )
        #self.set( 'bad_colours', bad_colours )

    #===============================================================================================
    def __str__(self):
        return 'ApertureCollection of shape {}'.format(self.shape)

    #===============================================================================================
    def __len__(self):
        return len(self.apertures)

    #===============================================================================================
    def __getitem__(self, key):         #NOTE: this effectively creates a copy of the aperture list...

        if isinstance( key, int ):
            if key >= len( self ) :
                raise IndexError( "The index (%d) is out of range." %key )
            return self.apertures[key]
        if isinstance( key, tuple ):
            assert len(key)==self.ndims
            #print( key )
            #print( self.apertures )
            return self.apertures[key]

        elif isinstance(key, slice):
            return [ self.apertures[i] for i in range(*key.indices(len(self))) ]
        elif isinstance(key, (list, np.ndarray)):
            assert isinstance(key[0], (bool, np.bool_))
            assert len(key)==len(self)
            return  [ self.apertures[i] for i in np.where(key)[0] ]
        else:
            raise KeyError('barf!')


    #===============================================================================================
    def __contains__(self, ap):
        return ap in self.apertures

    #===============================================================================================
    def set(self, attr, vals, ndmin=1, propagate=1):
        '''Set artist properties and propagate downstream.'''
        #NEED UPDATE METHOD?????
        vals = self._prep_vals( vals, ndmin=ndmin )
        setattr( self, attr, vals )

        if propagate:
            attr = self._prop_map( attr )
            ss = 'set_' + attr
            for ap, val in zip(self.apertures, vals):
                if hasattr(ap, ss):    #use the property setter method if it exists
                    setter = getattr(ap, ss)
                    setter( val )
                else:
                    setattr( ap, attr, val )

    #===============================================================================================    
    def get(self, attr):
        attr = self._prop_map( attr )
        gtr = 'get_' + attr

        if not len(self):
            return []

        elif hasattr(self, gtr):
            return [getattr(ap, gtr)() for ap in self]

        return [ getattr( ap, attr ) for ap in self]


    #===============================================================================================    
    #def insert(self, idx, ap):
        #if not isinstance(ap, Aperture):
            #raise TypeError( 'Invalid type {}'.format(type(ap)) )
        #else:
            #self.apertures.insert( idx, aps )

    #===============================================================================================
    #def index(self, ap):
        #return self.apertures.index( ap )

    #def join( self, other ):
        #assert len(self)==len(other)
        #tmp = np.vstack( self.apertures, other.apertures )
        #print( tmp )
    #===============================================================================================
    def get_update(self):
        for attr in zip('radii', 'colours', 'state'):
            setattr(self, attr, self.get(attr))

    #===============================================================================================    
    def append(self, ap ):

        if isinstance( ap, tuple ):
            ap = Aperture( ap, ec='y', fc='none', lw=2, ls='dotted')
        if isinstance(ap, Aperture): 
            print( 'attempting aperture append' )
            #combine_method = np.hstack
        elif isinstance(ap, Apertures):
            print( 'YEEEEAAAAAAAAYYYYYYY' )

        else:
            raise TypeError( 'Invalid type {}'.format(type(ap)) )

        self.append( ap )
        self.get_update( )

        #self.radii.append( ap.radius )
        #self.coords.append( ap.center )
        #self.colours.append( ap.get_ec() )
        #self.bad_colours.append( ap.bad_colour )
        #self.set_colours( self.colours )
        #self.state.append( ap.state )



        ap.collection = self            # DO YOU NEED THIS, OR CAN YOU WRITE A CONTAINS METHOD?

    #===============================================================================================
    def pop( self, idx ):

        self.apertures.pop( idx )
        self.radii.pop( idx )
        self.coords.pop( idx )
        self.colours.pop( idx )
        self.bad_colours.pop( idx )
        self.set_colours( idx )
        self.state.pop( idx )

    #===============================================================================================
    def within_allowed(self, r):
        return True

    #===============================================================================================    
    def resize(self, relative_motion):
        if relative_motion:
            rnew = self.radii[0] + relative_motion               #GETTER??
            if not self.within_allowed( rnew ):
                return

            self.set('radii', rnew, propagate=1)
            type(self).RADIUS = rnew            #Set the class radius - all new apertures of this class will assume this radius
            type(self).RADIUS_UPLIM = int(np.ceil(rnew))

    #===============================================================================================
    def plot(self, ax=None):
        ax = ax if ax else self.ax
        for ap in self:
            ap.plot( ax )
        #ax.draw_artist( self )         #else canvas only redrawn once window focus changes

    #===============================================================================================
    def get_ec(self):
        return self.get_edgecolor()

    #===============================================================================================
    def set_colours(self, colours):
        self.set_edgecolor( colours )

    #===============================================================================================
    def get_colours(self):
        return self.get_edgecolor()

    #===============================================================================================
    def cross(self, aps):
        ''' 
        Checks whether one set of apertures crosses another set.
        '''

        assert len(self)==len(aps)              #THIS DOES NOT NEED TO BE THE CASE!!!!!!!!!!!
        coords = np.array(self.coords, ndmin=2)
        dist_mat = cdist( coords, coords )                     #distance matrix for aperture centres (in pixels)

        ls = dist_mat - self.radii - aps.radii < 0                      #which apertures intersect (cross) one another (including self intersection)  
        np.fill_diagonal(ls, 0)                                         #self intersection ignored

        if len(self)>1:
            if aps[0].axes:
                same_ax = np.array([ap in aps[0].axes.artists for ap in aps])     #check if all apertures are in the same axes as the first
                #print( 'same_ax', same_ax )
                ls[:, ~same_ax] = 0;         ls[~same_ax,:] = 0             #only consider those within the same axis

        l = np.any(ls, 1)                           #which stars have apertures that cross another
        if any(l):
            print( 'WARNING: Sky aperture falls within 3 sigma of the psf fwhm.' )
        return l

    #===============================================================================================
    def center_proximity(self, mouse_position ):
        '''
        Calculate radial distances between aperture centres and mouse_position.
        '''
        x, y = mouse_position
        if not( x and y ):
            return
        xyprox = np.array(self.coords) - mouse_position
        return np.linalg.norm(xyprox , axis=1 )

    #===============================================================================================
    def check_edge(self, edges):
        '''
        Checks which apertures cross the image edge.

        Returns
        -------
        l : boolean array
        '''
        r = np.array(self.radii, ndmin=2).T          #reshapes r for arithmetic
        l1 = self.coords + r >= edges
        l2 = self.coords - r <= (0, 0)
        return np.any( l1 | l2, axis=1 )