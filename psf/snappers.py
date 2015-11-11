import numpy as np
from astropy.stats import sigma_clipped_stats

from magic.array import neighbours

from decor import cache_last_return# unhookPyQt

#from IPython import embed

######################################################################################################    
class Snapper( object ):
    '''Various snap-to-pixel methods for image GUI.'''
    def __init__(self, data, window=None, snap='centroid', edge='edge'):
        '''
        Parameters
        ----------
        data            :       array-like:
            The image data
        window          :      int
            size of the sub-image region to consider. i.e offset tolerance for snapping
        edge            :       str
            How to deal with the edges. Options are:
            *padding:
                'constant', 'maximum', 'minimum', 'mean', 'median', 
                'reflect', 'symmetric', 'edge', 'linear_ramp', 'wrap', 
            *'shift', 
            *'clip'
            See neighbours function for more info
        '''
        #TODO:  Transdict:      'com'
        methods = {'centroid'   :       self.snap2com,
                   'peak'       :       self.snap2peak }

        self.image_data = np.asarray( data )
        self.edge = edge

        #choose window size as fraction of image size if not given 
        if window is None:
            window_frac = 0.05          #use this fraction of data for the window
            minwin = (5,5)              #minimal window size
            window = np.multiply(data.shape, window_frac).astype(int)
            self.window = np.max( [window, minwin], 0 )
        else:
            window        = np.atleast_1d(window)
            if len(window)==1:
                self.window = np.r_[window,window]
                
        if snap.lower() in methods:
            self._snapper = methods[snap.lower()]
        else:
            raise ValueError

    #===============================================================================================
    def __call__(self, x,y):
        return self._snapper(x,y)

    #===============================================================================================
    def zoom(self, *coo):
        x,y = coo
        ij = int(round(y)), int(round(x))
        win, il = neighbours( self.image_data, ij, self.window, pad=self.edge, return_index=1 )
        #self.zoomed = win
        return win, il

    #===============================================================================================
    def snap2com(self, x, y):
        '''snap to center of mass'''
        zoomed, il = self.zoom(x,y)
        coo = il + center_of_mass( zoomed )     #ij indexing
        xy = coo[::-1]                          #xy indexing
        return tuple(xy) + zoomed.max(),

    #===============================================================================================
    def snap2peak(self, x, y):
        '''
        Snap to the peak value within the window.  
        Note: If peak value is non-singular, function returns the first peak by index.

        Parameters
        ----------
        x, y :                  input coordinates (mouse position)

        Returns
        -------
        xp, yp :                pixel coordinates and value of peak
        '''

        zoomed, (il,jl) = self.zoom(x,y)

        zp = np.max(zoomed)
        ip, jp = np.where( zoomed==zp )                #this is a rudamnetary snap function NOTE: MAY NOT BE SINGULAR.
        yp = ip[0]+il; xp = jp[0]+jl                   #indeces of peak within sub-image

        return xp, yp, zp

######################################################################################################    
class ImageSnapper( Snapper ):
    '''Does a few sanity checks after doing snapping'''
    _print = True
    _mem = True
    #===============================================================================================
    def __init__(self, data, window=None, snap='centroid', edge='edge', **kw):
        
        self.noise_threshold    = kw.pop( 'noise_threshold',    None )
        self.edge_cutoff        = kw.pop( 'edge_cutoff',        None )
        self.offset_tolerance   = kw.pop( 'offset_tolerance',   None )
        
        mean, median, std = sigma_clipped_stats(data, sigma=3)
        self.sky_mean = mean
        self.threshold = self.sky_mean + std*self.noise_threshold
        
        self.known = []
        
        super().__init__(data, window, snap, edge)
    
    #===============================================================================================
    def __call__(self, x, y):
        
        xp, yp, zp = self._snapper(x, y)
        j, i = xp, yp
        
        #check noise threshold
        if not self.noise_threshold is None:
            if zp < self.threshold:
                if self._print:
                    print( 'Probably not a star! '
                           'Pixel value {:.1f} < {:.1f} (threshold)'.format(zp, self.threshold ) )
                return None, None
        
        #check edge proximity
        if not self.edge_cutoff is None:
            r,c = self.image_data.shape
            if any(np.abs([xp, xp-c, yp, yp-r]) < edge_cutoff):
                if self._print:          print('Too close to image edge!')
                return None, None
        
        #check if known
        if len(self.known):             #are any coords known?
            xs, ys = self.snap2known(xp, yp, offset_tolerance=self.offset_tolerance)
            if xs and ys:
                if self._print:          print('Duplicate selection!')
                return None, None
        
        if self._mem:             #Not a knomn coord. remember it?
            self.known = list(set(self.known + [(xp, yp)]))
        
        return xp, yp
            
    #===============================================================================================
    def snap2known(self, x, y, offset_tolerance=None):
        
        offset_tolerance = offset_tolerance or min(self.window)
        
        rs = np.linalg.norm( np.subtract(self.known, (x,y)), axis=1 )
        #l_r = rs  < offset_tolerance                              #distance between point and known stars within offset tolerance?
        
        if any(rs  < offset_tolerance):                          #find star with minimal radial distance from selected point
            return self.known[np.argmin(rs)]
        else:
            if self._print:
                print('No known star found within %i pixel radius' %offset_tolerance)
            return None, None


######################################################################################################    
class DoubleZoomMixin():
    #===============================================================================================
    def __call__(self, x, y):
        xs, ys = super().__call__(x, y)
        self.zoom( xs, ys )     #zoom with snapped pixel as center & cache (save) image
        return xs, ys 
    
    #===============================================================================================
    @cache_last_return
    def zoom(self, *coo):
        return super().zoom( *coo )
    
            
######################################################################################################    
class CleanerMixin():
    
    ##===============================================================================================
    #def __init__(self, data, window=None, snap='centroid', edge='edge', **kw):
        #super().__init__(data, window, snap, edge, **kw)
        
        ##self.image_data_cleaned = self.image_data.copy()
    
    #===============================================================================================
    def clean(self, model):
        zoomed, (il,jl) = self.zoom.cache
        iu, ju = np.add( (il,jl), self.window )
        self.image_data[il:iu, jl:ju] -= model
        