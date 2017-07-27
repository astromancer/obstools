
import itertools as itt

import numpy as np

from obstools.fastfits import FitsCube
from obstools.phot.find.utils import sourceFinder
from obstools.phot.masker import MaskMachine

from recipes.array import ndgrid

class Modeller(FitsCube):
    def __init__(self, filename):
        FitsCube.__init__(self, filename)

        self.grid = ndgrid.from_shape(self.ishape)
        self.ugrid = ndgrid.unitary(self.ishape)
        # self.ngrid = np.moveaxis(ngrid, 0, -1)

        # self.mask_helper = MaskMachine

    def pre_find(self, nframes, snr=2.5, npixels=7, edge_cutoff=3, deblend=False):
        # source finding params
        fmean = np.mean(self[:nframes], axis=0)
        fmean_bg = fmean - np.median(fmean)

        coords, flux, segm = sourceFinder(fmean_bg, snr, npixels, edge_cutoff, deblend)

        return coords, flux, fmean, segm  # , Nscale

    def isolate_stars(self, coords, nstars, rbig=15, rsmall=5):
        # Isolate stars with mask
        mm = MaskMachine(self.grid, coords[:nstars], shrink=False)
        mask1 = mm.mask_all_circles(rbig)
        mm.update(coords[nstars:])
        mask2 = mm.mask_all_circles(rsmall)
        mask = ~mask1 | (mask1 & mask2)
        return mask

    def gen_slices(self, coords, window):
        w2 = int(window // 2)
        for i, coo in enumerate(coords):
            coo = coo.round(0).astype(int)
            lwr = np.max([coo - w2, (0, 0)], axis=0)
            upr = np.min([coo + w2, self.ishape], axis=0)
            win = wy, wx = list(map(slice, lwr, upr))
            yield win

    def gen_subs(self, i, coords, window):
        data = self[i]
        for win in self.gen_slices(coords, window):
            yield data[win].copy()

    def gen_subs_slices(self, i, coords, window):
        data = self[i]
        for win in self.gen_slices(coords, window):
            yield data[win].copy(), win

    def gen_subs_others_masked(self, i, coords, window, segm):
        data = self[i]
        for j, win in enumerate(self.gen_slices(coords, window)):
            sub = data[win].copy()
            mask_other_stars = (segm.data[win] != j + 1) & (segm.data[win] != 0)
            yield np.ma.array(sub, mask=mask_other_stars)

    def gen_subs_targets_masked(self, i, coords, window, segm, dilate=None):
        for sub, win in self.gen_subs_slices(i, coords, window):
            bg = (segm.data[win] == 0)
            yield np.ma.array(sub, mask=~bg)

    def gen_subs_masks(self, i, coords, window, segm):
        data = self[i]
        for j, win in enumerate(self.gen_slices(coords, window)):
            try:
                sub = data[win].copy()
                bg = (segm.data[win] == 0)
            except Exception as e:

                from IPython import embed
                embed()
                raise e

            mask_other_stars = (segm.data[win] != j + 1) & ~bg
            bg &= ~mask_other_stars
            yield sub, mask_other_stars, bg

    def guess_fwhma(self, i, coords, window, segm):
        return list(itt.starmap(guess_fwhm, self.gen_subs_masks(i, coords, window, segm)))


def guess_fwhm(sub, mask_other_stars, bg):
    # estimate fwhm
    # subtract bg
    sub -= np.ma.median(sub[bg])
    hm = 0.5 * sub.max()                    #half max value

    masked = np.ma.array(sub, mask=mask_other_stars)
    core = np.ma.greater_equal(masked, hm)
    xy_core = np.where(core.filled(False))
    fwhm = np.ptp(np.sqrt(np.square(xy_core).sum(0)))
    return fwhm


def guess_Z(data, coords):
    mm = MaskMachine(grid, coords)
    masks = mm.mask_circles(2)
    return np.array([data[m].max() for m in np.rollaxis(masks, -1, 0)])


def gen_slices(coords, window, shape):
    w2 = int(window // 2)
    for i, coo in enumerate(coords):
        coo = coo.round(0).astype(int)
        lwr = np.max([coo - w2, (0, 0)], axis=0)
        upr = np.min([coo + w2, shape], axis=0)
        win = wy, wx = list(map(slice, lwr, upr))
    yield win


