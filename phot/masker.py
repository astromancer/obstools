import numpy as np

#TODO: Optimize to deal with aperture cutouts

# ====================================================================================================
class MaskMachine():
    # FIXME: implement the functionality here as methods Aperture class?
    # Can you attach a mask to the aperture? that will allow these methods
    # to be implemented there like ap.mask_overlapping(ap2)

    def shrink_on(self):
        self.shrinker = self.shrink

    def shrink_off(self):
        self.shrinker = self.null_shrink

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def shrink(mask):
        # since we expect the masks to be sparse, pass down the indices to save RAM
        return np.where(mask)
        # TODO: check if this really boosts efficiency

    @staticmethod
    def null_shrink(mask):
        return mask

    @staticmethod
    def expand_mask(mask, shape):
        # expand mask indices to boolean array of given shape
        z = np.zeros(shape, dtype=bool)
        z[mask] = True
        return z

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, grid, coords=None, shrink=True):
        '''
        self.d has dimension (gy, gx, N) where gx, gy = grid.shape[:2] and N = len(coords)
        Most of the functions below return output with the same dimensionality.
        i.e. each star's mask is within a unique frame
        '''
        self._grid = grid[..., None]                # shape eg: (2, 36, 254, 1)
        self._cast = (slice(None),) + (None,) * (grid.ndim - 1)
        # used to cast coordinates for arithmetic -
        # eg. will add 2 axes dimensions to array of coords (2, 5) -->  (2, 1, 1, 5)
        # so can arith with grid

        if coords is not None:
            self.update(coords)

        if shrink:
            self.shrink_on()
        else:
            # print('HEY')
            self.shrink_off()
        # if not shrink:
        #    self.shrink = lambda _: _

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, coords):
        """
        (Re)calculate the distance grids
        """
        # TODO: optimize: only update if coord shift > 1?
        self.d = self.get_pixel_distance(coords)
        # TODO: update simply with relative shift!!!?

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_pixel_distance(self, coords):
        """get a distance grid for each coord"""
        cxx = coords[self._cast].T  #TODO: shape?
        return np.sqrt(np.square(self._grid - cxx).sum(0))  # pixel distances from star centre

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_circles(self, r):
        '''Mask all pixels around coords within radius `r`
        r can be array of length N'''
        return self.d - r < 0       # (gx, gy, N)  #FIXME: shrink??

    masked_within_radii = mask_circles
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_all_circles(self, r):
        return self.mask_circles(r).any(-1)  # all stars masked. dim (gx, gy)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_annuli(self, rsky):
        rin, rout = rsky
        return (rin < self.d) & (self.d < rout)
    mask_between_radii = mask_annuli

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_all_others(self, r):
        """For each frame, all the other stars are masked (except the one indexed by that frame)"""
        starmasks = self.mask_circles(r)
        allmasked = starmasks.any(-1)  # single image with all stars masked
        others = ~starmasks & allmasked[..., None]  #N frames with all stars masked except indexed one
        return starmasks, allmasked, others

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def mask_close_others(self, rstar, rclose):
        '''as with *mask_all_others*, except that only masked pixels of other stars that are within
         radius `rclose` from  the frame star are included in the mask. Optimization to save RAM when
         shrink is True
        '''
        _, _, others = self.mask_all_others(rstar)
        m = others & self.mask_circles(rclose)  # all nearby stars that may interfere with photometry
        return self.shrink(m)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def skymask(self, rstar, rsky):
        """Mask all other stars within the sky annulus of frame star"""
        _, allmasked, others = self.mask_all_others(rstar)

        skyannuli = self.mask_annuli(rsky)
        skymask = allmasked & skyannuli.any(-1)  #[..., None]
        # all stars inside all sky annuli masked in single frame
        return self.shrink(skymask)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_masks(self, rstar, rclose, rsky):

        _, allmasked, others = self.mask_all_others(rstar)

        skyannuli = self.mask_annuli(rsky)
        skymask = allmasked[..., None] & skyannuli

        # all nearby stars that may interfere with photometry
        photmask = others & self.mask_circles(rclose)

        return self.shrink(photmask), self.shrink(skymask)
        #return photmask, skymask



        # class Masker2(Masker):
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def get_pixel_distance(self, grid, coords):
        # cxx = coords[:, None].T                       #cast for arithmetic
        # return np.sqrt(np.square(grid[...,None] - cxx).sum(0))     #pixel distances from star centre
