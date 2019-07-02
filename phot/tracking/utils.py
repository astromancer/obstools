"""
Utilities for finding / tracking star positions in CCD data
"""

from functools import partial
import multiprocessing as mp

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.ndimage.measurements import center_of_mass as CoM

from photutils.detection import detect_threshold
from photutils.segmentation import detect_sources


def sourceFinder(data, snr=3., npixels=12, edge_cutoff=3, deblend=False,
                 flux_sort=True,                 return_index=False):
    # TODO: evolve a class that partitions the input frame spatially based on
    # window, and data value thresholds (upper and lower) as input for fitting
    # NOTE: Pretty damn slow.... can you generalize to higher dimension?
    # NOTE: Making npixels large avoids false flagging of cosmic rays

    threshold = detect_threshold(data, snr)
    segm = detect_sources(data, threshold, npixels)

    if edge_cutoff:
        segm.remove_border_labels(edge_cutoff, partial_overlap=False)

    if deblend:
        from photutils import deblend_sources
        segm = deblend_sources(data, segm, npixels)

    found = np.array(CoM(data, segm.data, segm.labels))
    # NOTE: yx coords (row, column)
    try:
        flux_est = detected_flux(data, segm)
    except:
        from IPython import embed
        embed()
        raise

    if flux_sort:
        found, flux_est, segm, ix = sort_flux(found, flux_est, segm)

    returns = found, flux_est, segm
    if return_index:
        returns += (ix, )

    return returns


def detected_flux(data, segm):
    # Crude flux estimate: sum detected pixels
    # Will not return flux for stars that have None as a slice - i.e. removed by previous compute
    data = data - np.median(data)       # bg subtract
    flux_est = np.empty(segm.nlabels)
    for i, sl in enumerate(filter(None, segm.slices)):
        flux_est[i] = data[sl][segm.data[sl].astype(bool)].sum()

    return flux_est / segm.areas[segm.areas != 0][1:]
    #return flux_est


def sort_flux(coords, flux, segm=None):
    """re-order segmented image labels and coordinates by decending flux"""

    # flux_thresh_psf = 1e4
    flux_est_sorted, ix, coords = zip(*sorted(zip(flux, range(segm.nlabels), coords), reverse=1))
    coords = np.array(coords)
    flux_est_sorted = np.array(flux_est_sorted)
    # Nscale = np.greater(flux_est_sorted, flux_thresh_psf).sum()

    if segm is None:
        return coords, flux, None, ix

    # re-order segmented image labels
    offset = 100
    for new, old in enumerate(ix):
        old += 1
        new += offset + 1
        segm.relabel(old, new)
    segm.data[segm.data != 0] -= offset

    return coords, flux_est_sorted, segm, ix


def _bulk_finder(snr, npixels, edge_cutoff, return_mask, dilate_mask, data):
    # NOTE: Pretty damn slow.... can you generalize to higher dimension?
    # Making npixels large avoids false flagging of cosmic rays

    found, im = sourceFinder(data, snr, npixels, edge_cutoff)

    if return_mask:
        m = im.data != 0
        if dilate_mask:
            m = binary_dilation(m, iterations=dilate_mask)
        return found, m
    return found


def bulk_find(data, snr=3., npixels=12, edge_cutoff=3, return_mask=True, dilate_mask=3):

    func = partial(_bulk_finder, snr, npixels, edge_cutoff, return_mask, dilate_mask)
    pool = mp.Pool()
    res = pool.map(func, data)
    pool.close()
    pool.join()

    if return_mask:
        found, masks = zip(*res)
        return np.array(found), np.array(masks)

    return np.array(res)



# def _bulk_mask(snr, npixels, edge_cutoff, dilate, data):
#
#     found, im = sourceFinder(data, snr, npixels, edge_cutoff)
#     m = im.data != 0
#     if dilate:
#         m = binary_dilation(m, iterations=3)
#     return m
#
#
# def bulk_mask(data, snr=3., npixels=12, edge_cutoff=3, dilate=True):
#
#     func = partial(_bulk_mask, snr, npixels, edge_cutoff, dilate)
#     pool = mp.Pool()
#     res = pool.map(func, data)
#     pool.close()
#     pool.join()
#     return np.array(res)






#****************************************************************************************************
# class SourceFinder():
#
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     def __init__(self, snr=3, npixels=7, **kw):
#         ''' '''
#         self.snr = snr
#         self.npixels = npixels
#
#         #remove sources that are too close to the edge
#         self.edge_cutoff = kw.get('edge_cutoff')
#         self.max_shift = kw.get('max_shift', 20)
#         self.Rcoo = kw.get('Rcoo')
#         self.fallback = kw.get('fallback', 'prev')
#         self.window = kw.get('window', 20) #case fallback == 'peak'
#
#         #self._coords = None
#
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     def __call__(self, data):
#         ''' '''
#         threshold = detect_threshold(data, self.snr)
#         self.im = im = detect_sources(data, threshold, self.npixels)
#
#         if self.edge_cutoff:
#             im.remove_border_labels(self.edge_cutoff)
#
#         found = np.array(CoM(data, im.data, im.labels)) # NOTE: ij coords
#
#         if self.Rcoo is None:
#             return found
#         else:
#             # if one of the stars are not found during this call (e.g. eclipse or clouds or whatever.)
#             # fall back to one of the following:
#             # NOTE: this conditional may be a hot-spot??
#             # NOTE: all this stuff only works sequentially...
#             if self.fallback == 'prev': #use the previous location for the source
#                 new = self._prev_coords[:]
#             elif self.fallback == 'mask':
#                 shape = self.Rcoo.shape
#                 new = np.ma.array(np.empty(shape),
#                                   mask=np.ones(shape, bool))
#             elif self.fallback == 'ref':
#                 new = self.Rcoo
#             elif self.fallback == 'peak':
#                 w = self.window
#                 hw = w/2
#                 new = np.empty(self.Rcoo.shape)
#                 for i, (j,k) in enumerate(Rcoo):
#                     #TODO: maybe use neighbours here for robustness
#                     sub = data[j-hw:j+hw, k-hw,k+hw]
#                     new[i] = np.add((j,k), divmod(sub.argmax(), w)) + 0.5
#             else:
#                 new = found
#
#             #if maxshift is set, cut out the detections that are further away from the Rcoo than this value
#             #i.e. remove unwanted detections
#             if self.max_shift:
#                 d = found[:,None] - self.Rcoo
#                 x, y = d[...,0], d[...,1]
#                 r = np.sqrt(x*x + y*y)
#                 l = r < self.max_shiftexit
#
#                 #set the new coordinates where known
#                 new[l.any(0)] = found[l.any(1)]
#
#         self._prev_coords = new
#         return new
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





if __name__ == '__main__':

    from pathlib import Path

    from obstools.fastfits import FitsCube
    from decor.profiler.timers import timer

    path = Path('/media/Oceanus/UCT/Observing/SALT/2016-2-DDT-006/0209/')
    ff = FitsCube(path/'s.fits')


    res = timer(bulk_find)(ff[:100])

    from IPython import embed
    embed()



