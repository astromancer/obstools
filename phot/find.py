import numpy as np
from scipy.ndimage.measurements import center_of_mass


#****************************************************************************************************
class SourceFinder():

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, snr=3, npixels=7, **kw):
        ''' '''
        self.snr = snr
        self.npixels = npixels

        #remove sources that are too close to the edge
        self.edge_cutoff = kw.get('edge_cutoff')
        self.max_shift = kw.get('max_shift', 20)
        self.Rcoo = kw.get('Rcoo')
        self.fallback = kw.get('fallback', 'prev')
        self.window = kw.get('window', 20) #case fallback == 'peak'

        #self._coords = None

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(self, data):
        ''' '''
        threshold = detect_threshold(data, self.snr)
        self.im = im = detect_sources(data, threshold, self.npixels)

        if self.edge_cutoff:
            im.remove_border_labels(self.edge_cutoff)

        found = np.array(center_of_mass(data, im.data, im.labels)) #NOTE: ij coords
        
        if self.Rcoo is None:
            return found
        else:
            #if one of the stars are not found during this call (e.g. eclipse or clouds etc.)
            #fall back to one of the following:
            #WARNING: this conditional may be a hot-spot
            if self.fallback == 'prev': #use the previous location for the source
                new = self._prev_coords[:]
            elif self.fallback == 'mask':
                shape = self.Rcoo.shape
                new = np.ma.array(np.empty(shape),
                                  mask=np.ones(shape, bool))
            elif self.fallback == 'ref':
                new = self.Rcoo
            elif self.fallback == 'peak':
                w = self.window
                hw = w/2
                new = np.empty(self.Rcoo.shape)
                for i, (j,k) in enumerate(Rcoo):
                    #TODO: use neighbours here for robustness
                    sub = data[j-hw:j+hw, k-hw,k+hw]
                    new[i] = np.add((j,k), divmod(sub.argmax(), w)) + 0.5
            else:
                new = found
            
            #if maxshift is set, cut out the detections that are further away from the Rcoo than this value
            #i.e. remove unwanted detections
            if self.max_shift:
                d = found[:,None] - self.Rcoo
                x, y = d[...,0], d[...,1]
                r = np.sqrt(x*x + y*y)
                l = r < self.max_shift
                #set the new coordinates where known 
                new[l.any(0)] = found[l.any(1)]
            
        self._prev_coords = new
        return new 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~