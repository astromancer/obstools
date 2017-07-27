import logging
import functools

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass as CoM

from obstools.phot.find import sourceFinder
from obstools.phot.utils import mad
from recipes.array.neighbours import neighbours


from IPython import embed



def cdist_tri(coo):
    """distance matirx with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)           # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    sdist = np.ma.masked_array(sdist)
    sdist[np.tril_indices(n)] = np.ma.masked
    return sdist



#NOTE: consider that StarTracker is also in some sense a model: CentroidModel.
#  if we view it as such we can pipe it through the same machinery as the
# psf models


class SourceFinder():  #FitsCube

    _snr_cut = 250
    _distance_cut = 10
    _saturation_cut = 1 # percentage (closeness to saturation)

    # @chrono.timer
    def __init__(self, image, snr=2.5, npixels=7, edge_cutoff=3,
                 deblend=False, flux_sort=True):

        image_bg = image - np.median(image)
        found, flux, segm = sourceFinder(
            image_bg, snr, npixels, edge_cutoff, deblend, flux_sort)

        self.image = image
        self.segm = segm
        self.found = found
        # pixel distance between stars
        self.sdist = cdist_tri(found)

        # crude snr estimate for stars
        self.flux = flux
        self.crude_snr = flux / image_bg[segm.data_masked.mask].std()

        # satlvl = get_saturation(fitsCube.master_header)
        # self.best_for_tracking(fmean, satlvl, snr_cut=self.crude_snr)

    # def

    def best_for_tracking(self, close_cut=None, snr_cut=_snr_cut, saturation=None):
        """
        Find stars that are best suited for centroid tracking based on the
        following criteria:
        """
        too_bright, too_close, too_faint = [], [], []
        msg = 'Stars: %s too %s for tracking'
        if saturation:
            too_bright = self.too_bright(self.image, saturation)
            if len(too_bright):
                logging.debug(msg, str(too_bright), 'bright')
        if close_cut:
            too_close = self.too_close(close_cut)
            if len(too_close):
                logging.debug(msg, str(too_close), 'close')
        if snr_cut:
            too_faint = self.too_faint(snr_cut)
            if len(too_faint):
                logging.debug(msg, str(too_faint), 'faint')

        ignore = functools.reduce(np.union1d, (too_bright, too_close, too_faint))
        ix = np.setdiff1d(np.arange(len(self.found)), ignore)
        if len(ix) == 0:
            logging.warning('No suitable stars found for tracking!')
        return ix

    # def auto_window(self):
    #     sdist_b = self.sdist[snr > self._snr_thresh]

    def too_faint(self, threshold=_snr_cut):
        crude_snr = self.flux / self.image[self.segm.data_masked.mask].std()
        return np.where(crude_snr < threshold)[0]

    def too_close(self, threshold=_distance_cut):
        # Check for potential interference problems from stars that are close together
        # threshold = threshold or self.window
        return np.unique(np.ma.where(self.sdist < threshold))
        # return np.intersect1d(self.ix_loc, too_close)

    def too_bright(self, data, saturation, threshold=_saturation_cut):
        # Check for saturated stars by flagging pixels withing 1% of saturation level
        # TODO: make exact
        lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        # TODO check if you can improve speed here - dont have to check entire array?
        satpix = np.where((lower < data) & (data < upper))
        b = np.any(np.abs(np.array(satpix)[:, None].T - self.found) < 3, 0)
        w, = np.where(np.all(b, 1))
        return w

class StarTracker():
    """Track stars in sub-regions of the image"""

    # TODO: let call update shared coords. Then move to external lib

    # cfunc = CoM                 # default for locating stars
    # bgfunc = np.median         # default for determining background level
    # findpars = dict(snr=2.5, npixels=7, edge_cutoff=3)

    mad_thresh = 10.
    _snr_cut = 250

    @classmethod
    def from_image(cls, image, window, max_stars=None, cfunc=None, bgfunc=None,
                   **findkws):

        finder = SourceFinder(image, **findkws)
        ixLoc = finder.best_for_tracking()
        #window = finder.sdist.min()

        instance = cls(finder.found, window, ixLoc, max_stars=max_stars,
                       cfunc=cfunc, bgfunc=bgfunc)
        return instance


    def __init__(self, Rcoo, window, ix_loc=None, ir=0, max_stars=None,
                 cfunc=None, bgfunc=None): #**kws
        """
        Parameters
        ----------
        cfunc
        """
        # _kws = self.findpars.copy()
        # _kws.update(kws)

        if np.size(Rcoo) < 2:
            raise ValueError('Bad reference coordinates.')

        if (ix_loc is None) or (len(ix_loc) == 0):
            ix_loc = [ir]
        if not len(ix_loc):
            raise ValueError('Invalid indices')

        self.ix_loc = np.asarray(ix_loc)
        self.ir = ir  # Reference star relative to which to calculate vectors
        self.window = window
        self.Rcoo = Rcoo
        self.rcoo = rcoo = Rcoo[ir]
        self.Rvec = Rvec = Rcoo - rcoo
        self.Rvec_int = Rvec_int = Rvec.round().astype(int)
        ixll = np.round(rcoo - window / 2).astype(int)
        self.Rllcs = ixll + Rvec_int
        self.max_stars = max_stars

        # self.sdist = cdist_tri(found)

        # funcs for main compute
        self.cfunc = cfunc
        self.bgfunc = bgfunc

        #
        # tc = self.too_close(window)

    @property
    def nloc(self):
        return len(self.ix_loc)

    def __call__(self, data, i=None):#, relative=False):
        # TODO: check snr dynamically as it may change....
        shift = self.guess_shift(i)
        coo = self.locate_all(data, shift)
        newShift = self.calculate_shift(coo)
        newCoo = self.rcoo + newShift
        # if relative:
        #     newCoo += self.Rvec
        return newCoo

    def guess_shift(self, i=None):
        if i is None:
            return 0

        previous = coords[i - 1]
        if np.isnan(previous).any():
            raise Exception('BORK! %s at %s' % (previous, i))

        return previous - self.rcoo

    def locate_all(self, data, shift, i=None):
        # TODO: check snr dynamically as it may change....
        ix_loc = self.ix_loc
        guessed = self.Rcoo[ix_loc] + shift
        coo = np.empty((self.nloc, 2))
        for jj, j in enumerate(ix_loc):
            coo[jj] = self.locate_star(j, guessed[jj], data, i)
        return coo

    def locate_star(self, j, coo, data, mask=None, i=None, **kws):
        """slice and calculate centre. return (y, x) pos in frame coordinates"""
        sub, ixll = self.get_sub_window(j, coo, data, **kws)
        coo = self.cfunc(sub - self.bgfunc(sub))
        if np.any((coo < 0) | (coo > data.shape)):
            logging.warning('Bad coordinates in frame %s: %s', i, coo)
        return ixll + coo

    # TODO: checks
    # w2 = np.divide(self.window, 2)
    # np.all(np.abs(coo - w2) < w2)

    def get_sub_window(self, j, coo, window, data, **kws): # mask=None, fill=np.inf,
        """
        Generate a sequence of data sub-sections for the stars located at cxx
        """
        # TODO: use photutils apertures BoundingBox??
        # if mask is not None:
            # this effectively masks the data for the fitting routine
            # data[mask] = fill

        # subwindow focussed on star
        # coo = self.Rcoo[j] + self.guess_shift(j)
        kws.setdefault('return_index', 1)
        try:
            sub, ixll = neighbours(data, coo, window, **kws)
        except Exception as e:
            print('FIX ' * 100, str(e))
            embed()
            raise
        # pad='constant', constant_values=fill,
                         # return_index=1, #slice
                         # **kws) # 1-2 ms per call #FIXME: speedup!!?
        # sub_stddev = neighbours(data_stddev, coo, self.window,
        #                         pad='constant', constant_values=np.inf)
        return sub, ixll #, self.regularization_hack(coo, sub_stddev)


    def calculate_shift(self, coo):
        """Calculate x,y shift of frame from reference by combining measured star positions"""
        l = ~self.is_outlier(coo)
        shift = np.mean(coo[l] - self.Rcoo[self.ix_loc[l]], 0) #TODO: median??
        return shift


    def is_outlier(self, coo):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """
        if len(coo) < 6:  # scatter large for small sample sizes
            return np.zeros(len(coo), bool)

        r = np.sqrt(np.square(self.Rcoo[self.ix_loc] - coo).sum(1))
        return r - np.median(r) > self.mad_thresh * mad(r)

    @staticmethod
    def save_coords(i, coo):
        """saves found coordinates to shared memory"""
        coords[i] = coo

    def centre_markers(self):
        ''

    # def to


class StarTrackerFixedWindow(StarTracker):
    """Centroid within a fixed reference window"""
    # NOTE: this is not such a robust location measure
    # ~10 times faster than neighbours!
    def __init__(self, Rcoo, window, ix_loc=None, ir=0, max_stars=None,):
        """
        Parameters
        ----------
        cfunc
        """
        StarTracker.__init__(self, Rcoo, window, ix_loc, ir, max_stars,
                 cfunc=None, bgfunc=None)

        self.fixed_slices = []
        for ixll in self.Rllcs:
            slices = list(map(slice, ixll, ixll + window)) # y, x slices
            self.fixed_slices.append(slices)
        self.rwin = self.fixed_slices[ir]


    def guess_shift(self, j):
        return 0

    def get_sub_window(self, j, coo, window, data, **kws):
        """
        Generate a sequence of data sub-sections for the stars located at cxx
        """
        # if mask is not None:
        #     data[mask] = fill

        slice_ = self.fixed_slices[j]
        sub = data[slice_]
        return sub, self.Rllcs[j] #, self.regularization_hack(coo, sub_stddev)


# class StarTrackerFit(StarTracker):
#     """saves fit coordinates to shared memory"""
#     @staticmethod
#     def save_coords(i, _, coo):
#         coords[i] = coo

    # def skymask(self, _, __):
    #     # This would otherwise just be a waste of time since we don't need the skymasks for fitting,
    #     # only for photometry
    #     return (None,) * Nstars

    # def get_sub_window(self, j, coo, data, **kws):
    #     """
    #     Generate a sequence of data sub-sections for the stars located at cxx
    #     """
    #     kws.setdefault('pad', 'constant')
    #     kws.setdefault('constant_values', np.nan)
    #     sub = super().get_sub_window(j, coo, data,
    #                                  **kws) # 1-2 ms per call #FIXME: speedup!!
    #     # sub_stddev = neighbours(data_stddev, coo, self.window,
    #     #                         pad='constant', constant_values=np.inf)
    #     return sub #, self.regularization_hack(coo, sub_stddev)


