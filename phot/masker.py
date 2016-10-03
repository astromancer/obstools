import numpy as np

#====================================================================================================
class Masker():     #TODO: optimize - combs of coords, r that yield the same masks...
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, grid, coords): # shrink=True
        ''' '''
        self.update(grid, coords)
        #self._shrink = lambda _ : _
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, grid, coords):
        self.d = self.get_pixel_distance(grid, coords)  #TODO: update simply with relative shift!!!
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_pixel_distance(self, grid, coords):
        _ix = (slice(None),) + (None,) * (grid.ndim-1)
        cxx = coords[_ix].T                       #cast for arithmetic 
        return np.sqrt(np.square(grid[...,None] - cxx).sum(0))     #pixel distances from star centre
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def masked_within_radii(self, r):
        '''Mask all pixels around coords within radius `r`'''
        return self.d - r < 0
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def masked_any_within_radii(self, r):
        return self.masked_within_radii(r).any(-1)         #all stars masked
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_between_radii(self, rsky):
        rin, rout = rsky
        return (rin < self.d) & (self.d < rout)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def single_all_others(self, r):
        starmasks = self.masked_within_radii(r)
        allmasked = starmasks.any(-1)         #all stars masked
        others = ~starmasks & allmasked[...,None]   #all stars masked except indexed one
        return starmasks, allmasked, others
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def photmasks(self, rstar, rclose, shrink=True):
        '''as with starmask, except that only masked pixels of other stars 
        that are within radius `rclose` from coords are masked'''
        _, _, others = self.single_all_others(rstar)
        m = others & self.masked_within_radii(rclose)     #all nearby stars that may interfere with photometry
        if shrink:
            return self.shrink(m)
        return m
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def skymask(self, rstar, rsky, shrink=True):
        _, allmasked, others = self.single_all_others(rstar)
       
        skyannuli = self.mask_between_radii(rsky)
        bgmask = allmasked[...,None] & skyannuli
        if shrink:
            return self.shrink(bgmask)
        return bgmask
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_masks(self, rstar, rclose, rsky, shrink=True):
        
        _, allmasked, others = self.single_all_others(rstar)
       
        skyannuli = self.mask_between_radii(rsky)
        bgmask = allmasked[...,None] & skyannuli
        
        photmask = others & self.masked_within_radii(rclose)      #all nearby stars that may interfere with photometry
        
        if shrink:
            return self.shrink(photmask), self.shrink(bgmask)
        return photmask, bgmask
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def shrink(mask):
        # since the masks are expected to contain far tess True values than False, 
        # we pass down the indeces to save memory
        #n = masks[0].shape[-1]
        return [np.where(mask[...,i]) for i in range(mask.shape[-1])]
                
    #return np.moveaxis(photmasks, -1, 1), np.moveaxis(bgmasks, -1, 1) #photmasks, bgmask 

#class Masker2(Masker):
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def get_pixel_distance(self, grid, coords):
        #cxx = coords[:, None].T                       #cast for arithmetic 
        #return np.sqrt(np.square(grid[...,None] - cxx).sum(0))     #pixel distances from star centre
    