import itertools as itt
import ctypes

import numpy as np

from recipes.parallel.synched import SyncedArray
from recipes.array import neighbours
from recipes.logging import LoggingMixin
from recipes.dict import AttrDict
import obstools.psf.lm_compat as modlib

from IPython import embed

# TODO: class that fits all stars simultaneously with MCMC. compare for speed / accuracy etc...

class ModelDb(LoggingMixin):
    """container for model data"""

    @property
    def gaussians(self):
        return [mod for mod in self.models if 'gauss' in str(mod).lower()]

    def __init__(self, model_names):

        self.model_names = model_names
        self.nmodels = len(model_names)
        self.build_models(model_names)
        self._indexer = {}

    def build_models(self, model_names):
        counter = itt.count()
        self.db = AttrDict()

        for name in model_names:
            cls = getattr(modlib, name)
            model = cls()  # initialize
            model.basename = self.basename  # logging!!
            self.db[model.name] = model
            self._indexer[model] = next(counter)

        self.models = list(self._indexer)

    def __getstate__(self):
        # FIXME: make your models picklable to avoid all this crap
        # capture what is normally pickled
        state = self.__dict__.copy()
        # since the models from lm_compat are not picklable, we replace their instances with
        # their class names in the following data containers

        for attr in ('_indexer', 'data', 'resData'):
            dic = state.get(attr)
            if dic:
                state[attr] = type(dic)((mod.__class__.__bases__[-1].__name__, val)
                                        for mod, val in dic.items())

        state.pop('db')
        state.pop('models')

        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, state):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(state)
        # rebuild the models
        # self.build_models(self.model_names)

        # from IPython import embed
        # print('\n' * 10, 'BARF!!', )
        # embed()

        # try:
        # for attr in ('data', 'resData'):
        #     dic = state.get(attr)
        #     for model in self.models:
        #         name = model.__class__.__bases__[-1].__name__
        #         dic[model] = dic.pop(name)
        #         # except Exception as err:
        #         #     print('\n' * 10, 'BARF!!', )
        #         #     embed()


class ModelData():
    def __init__(self, model, N, Nfit):
        """Shared, Synced Data containers for model"""
        npars = model.npar
        self.params = SyncedArray(shape=(N, Nfit, npars))  # fitting parameters
        self.params_std = SyncedArray(shape=(N, Nfit, npars))  # standard deviation (error?) on parameters

        # FIXME: you can avoid this conditional by null object pattern for 'integrate'
        if hasattr(model, 'integrate'):
            self.flux = SyncedArray(shape=(N, Nfit))
            self.flux_std = SyncedArray(shape=(N, Nfit))
            # NOTE: can also be computed post-facto

        if hasattr(model, 'reparameterize'):
            nparAlt = len(model.reparameterize(model.default_params))
            self.alt = SyncedArray(shape=(N, Nfit, nparAlt))



def cdist_tri(coo):
    """distance matirx with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)           # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    sdist = np.ma.masked_array(sdist)
    sdist[np.tril_indices(n)] = np.ma.masked
    return sdist




from collections import defaultdict, namedtuple
from obstools.phot.masker import MaskMachine
from recipes.array import ndgrid
from scipy.spatial.distance import cdist

from photutils.segmentation import SegmentationImage

class SegmentationHelper(SegmentationImage):

    @classmethod
    def from_image(cls, image, window, **findkws):

        finder = SourceFinder(image, **findkws)
        return cls(finder.found, window, image.shape)


    @classmethod
    def circles(cls, coords, sizes, ishape):
        data = np.zeros(ishape, int)
        grid = ndgrid.from_shape(ishape)
        for i, mask in enumerate(MaskMachine(grid, coords).mask_circles(sizes)):
            data[mask] = i

#





class ImageSegmenter():  #TODO: ImageSlicer?
    """
    Base class for modelling stars in sub-regions of the main frame.

    This class implements a divide and conquer strategy for frequentist modeling
    of the Point Source Function in a CCD frame.
    """

    # @classmethod
    # def from_image(cls):
    #     return cls
                                                      # size,
    def __init__(self, centres, window, ishape, use, handle_prox='mask'):

        w = self.window = int(window)
        x = self.centres = np.atleast_2d(centres)
        # self.sdist = cdist_tri(self.centres)

        llc, urc, seg = self.make_segments(x, w, ishape)
        self.corners = namedtuple('Corners', 'llc urc')(llc=llc, urc=urc)
        self.segments = seg
        self._overlaps = self.where_overlap(x, w)

        # for the overlapping segments: options here:
        # 1) mask
        # 2) fit simultaneously with coordinates fixed?

        self.grid = ndgrid.from_shape(ishape)
        self.subgrids = self.make_grids(self.segments)

        #self.masker = MaskMachine(grid, self.centres)


    @property
    def nseg(self):
        return len(self.centres)

    # def get_corners(self, centres, window, ishape):

    def make_segments(self, centres, window, ishape):
        yx = np.round(centres)
        w = window / 2
        l, u = yx - w, yx + w
        lyu, lxu = (u > ishape).T
        l[l < 0] = 0
        u[lyu, 0], u[lxu, 1] = ishape
        z = np.dstack([l, u]).astype(int)
        llc, urc = np.rollaxis(z, -1, 0) # corners
        return llc, urc, [list(map(slice, *zz.T)) for zz in z]

    def make_grids(self, segments):
        subgrids = []
        for sy, sx in segments:
            grid = self.grid[:, sy, sx]
            subgrids.append(grid)
        return subgrids


    def update(self, coords, window, ishape):
        """recenter"""
        # calculate shift from reference:
        shift = np.median(self.centres - coords, 0)
        ishift = np.round(shift)
        if (ishift > 1).any():
            self.segments = self.make_segments(coords, window, ishape)
            self.subgrids = self.make_grids(self.segments)

    def where_overlap(self, coords, window):
        xyd = np.rollaxis(coords[..., None] - coords.T, 1, 0)
        ir, ic = np.diag_indices_from((xyd[0]))
        #ir, ic = np.tril_indices_from(xyd[0])
        xyd[:, ir, ic] = np.inf
        loverlap = np.all(np.abs(xyd) < window, 0)
        ovr = np.array(np.where(loverlap)).T
        return ovr

    def get_masks(self, r):
        """Mask 'other' stars in the window"""
        ovr = self._overlaps
        masks = defaultdict(bool)
        for s, o in ovr:
            sly, slx = self.segments[o]
            grid = self.grid[:, sly, slx]
            x = self.centres[s, None, None].T
            d = np.sqrt(np.square(grid - x).sum(0))
            masks[s] |= (d < r)
        return masks

    # def merge_overlapped(self):
    # def remove_overlapped(self):
    # def update_masks(self, ):


    def iter_segments(self, data, std):
        for sl, g in zip(self.segments, self.subgrids):
            yield data[sl], std[sl], g



class ImageSegmenterArb(ImageSegmenter):

    def __init__(self, centres, window, ishape):







class ImageSegmentsModeller(ImageSegmenter):

    _metrics = ('aic', 'bic', 'redchi')

    def __init__(self, centres, window, ishape, use, models, metrics=_metrics,
                 handle_prox='mask'):
        ImageSegmenter.__init__(self, centres, window, ishape, use, handle_prox)

        # if not len(models):
        self.models = list(models)
        self.metrics = list(metrics)
        # TODO: assign metrics to model?

    @property
    def nmodels(self):
        return len(self.models)

    def fit(self, data, std=None, mask=None, models=None):
        """
        Fit frame data by looping over segments

        Parameters
        ----------
        data
        std
        mask
        models
        i

        Returns
        -------

        """
        if std is None:
            std = np.ones_like(data)

        if mask is not None:
            std[mask] = np.inf  # FIXME: copy?
            # this effectively masks the data point for the fitting routine

        # use the init models if none explicitly provided
        models = models or self.models

        # loop over models and fit
        gof = np.full((len(models), self.nseg, len(self.metrics)), np.nan)
        pars, paru = [], []
        for m, model in enumerate(models):
            p, pu, gof[m] = self._fit_model(model, data, std)
            pars.append(p)
            paru.append(pu)

        return pars, paru, gof


    def _fit_model(self, model, data, std):
        """
        Fit this model for all segments
        """
        # this method is for production
        p, pu = np.full((2, self.nseg, model.npar), np.nan)
        gof = np.full((self.nseg, len(self.metrics)), np.nan)
        for j, (sub, sub_std, grid) in enumerate(self.iter_segments(data, std)):
            p0 = model.p0guess(sub)
            p[j], pu[j], gof[j] = model.fit(p0, sub, grid, sub_std)

        return p, pu, gof


    def _fit_segment(self, sub, sub_std, grid, models=None, i=None):
        """
        Fit various models for single segment
        """
        # this method is for cenvenience
        models = models or self.models
        gof = np.full((len(models), len(self.metrics)), np.nan)
        par, paru = [], []      # parameter values and uncertainty
        for m, r in enumerate(
                self._fit_models_gen(sub, sub_std, grid, models)):
            if r is None:
                continue

            # aggregate results
            p, pu, gof[m] = r
            par.append(p)
            paru.append(pu)

            # convert here??
            # if i:
            #     p, _, gof[m] = r
            #     model = models[m]
                # self.save_params(i, j, model, r, sub)

        # choose best fitting model and save those fluxes
        #ix_bm, best_model = self.model_selection(i, j, gof, models)


        # if i:
        #     # save best fit flux
        #     self.best.ix[i, j] = ix_bm
        #     self.best.flux[i, j] = self.data[best_model].flux[i, j]
        #     self.best.flux_std[i, j] = self.data[best_model].flux_std[i, j]

        return par, paru, gof

    def _fit_models_gen(self,sub, sub_stddev, models=None):
        models = models or self.models
        for model in models:
            p0 = model.p0guess(sub)
            # p0[1::-1] = coo
            yield model.fit(p0, sub, self.grid, sub_stddev)

    def model_selection(self, pars, paru, gof):
        """
        Do model selection based on goodness of fit metric(s)
        """
        # loop over models for each segment
        for p, u, g in zip(pars, paru, gof):
            return



        if np.isnan(gof).all():
            self.logger.warning(
                'No model converged for Frame %i, Star %i', i, self.ix_fit[j])
            return None, None

        # the model indices at the minimal gof metric value
        best = np.nanargmin(gofs, 0)
        ub, ib, cb = np.unique(best, return_index=True, return_counts=True)
        # check cross metric consistency
        if len(ub) > 1:
            self.logger.warning(
                'Model selection metrics do not agree. Frame %i, Star %i. '
                'Selecting on AIC.',  i, self.ix_fit[j])
            # NOTE: This probably means none of the models are an especially good fit

        # choose model that most metrics agree is best
        ix_bf = ub[cb.argmax()]
        best_model = models[ix_bf]
        self.logger.info('Best model Frame %i, Star %i: %s' % (i, self.ix_fit[j], best_model))
        # return index of best fitting model
        if best_model is self.db.bg:
            "logging.warning('Best model is BG')"
            "flux is upper limit?"

        return ix_bf, best_model


    def combine_results(self, fit_results):
        return fit_results

    def get_apertures(self):
        ''


    # def get_segments(self, j, coo, data, std, mask):
    #     """
    #     Generate a sequence of data sub-sections for the stars located at coo
    #     """
    #
    #
    #
    #
    #     # get fit window from best coordinates (finder)
    #     sub, slice_ = neighbours(data, coo, self.window, pad='constant',
    #                              constant_values=np.nan, return_index=slice)
    #
    #     # mask out of frame data / nearby stars
    #     out_of_frame = np.isnan(sub)
    #     sub_std = std[slice_]
    #     # NOTE: the way the mask is applied here is convenient for fitting but not photometry
    #     sub_std[mask] = np.inf
    #     sub_std[out_of_frame] = np.inf
    #     ixll = list(s.start for s in slice_)
    #     sub_std = self.regularization_hack(sub_std, coo - ixll)
    #
    #     return sub, sub_std, ixll

#     def get_segments(self, j, coo, window, data, **kws): # mask=None, fill=np.inf,
#         """
#         Generate a sequence of data sub-sections for the stars located at cxx
#         """
#         try:
#             sub, ixll = neighbours(data, coo, window, **kws)
#             return sub, ixll
#
#         except Exception as e:
#             print('FIX ' * 100, str(e))
#             embed()
#             raise
#         # pad='constant', constant_values=fill,
#                          # return_index=1, #slice
#                          # **kws) # 1-2 ms per call #FIXME: speedup!!?
#         # sub_stddev = neighbours(data_stddev, coo, self.window,
#         #                         pad='constant', constant_values=np.inf)
#         return sub, ixll #, self.regularization_hack(coo, sub_stddev)

from scipy.ndimage import center_of_mass as CoM

class CentroidModel():
    """
    If we think of the centroid operation as a model, we can use the FrameModeller
    machinery to compute and agregate
    """

    # Would be cool to have an interactive aperture / window that displays
    # the centroid of the underlying data

    npar = 2

    def __init__(self, cfunc, bgfunc):
        # funcs for main compute
        self.cfunc = cfunc
        self.bgfunc = bgfunc

    def p0guess(self, *args):
        return

    # def __call__(self):

    def fit(self, p0, data, grid, std):
        coo = np.array(self.cfunc(data - self.bgfunc(data)))
        if np.any((coo < 0) | (coo > data.shape)):
            coo = None
        return coo, None, None


from obstools.phot.utils import mad

class StarTracker(ImageSegmentsModeller):
    # @classmethod
    # def from_image(cls, image, window, max_stars=None, cfunc=None, bgfunc=None,
    #                **findkws):
    #
    #     finder = SourceFinder(image, **findkws)
    #     ixLoc = finder.best_for_tracking()
    #     #window = finder.sdist.min()
    #
    #     instance = cls(finder.found, window, image.shape, ixLoc, )
    #     return instance
    #

    mad_thresh = 10.

    def __init__(self, centres, window, ishape, use, handle_prox='mask'):

        model = CentroidModel(CoM, np.median)
        ImageSegmentsModeller.__init__(self, centres, window, ishape, use,
                                       (model,),                # models
                                       (None, ),                # metrics
                                       handle_prox='mask')

    def (self, make_grids):

    def combine_results(self, com):
        yxshift = np.nanmedian(com + self.corners.llc - self.centres, 0)
        newCoords = self.centres + yxshift
        #return newCoords


    def calculate_shift(self, coo):
        """Calculate x,y shift of frame from reference by combining measured star positions"""
        l = ~self.is_outlier(coo)
        shift = np.mean(coo[l] - self.centres[l], 0)
        return shift

    def is_outlier(self, coo, mad_thresh=mad_thresh):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """
        # anything larger than 6 is a sample
        if len(coo) < 6:  # expect large scatter for small sample sizes - cannot flag
            return np.zeros(len(coo), bool)

        r = np.sqrt(np.square(self.centres - coo).sum(1))
        return r - np.median(r) > mad_thresh * mad(r)






class FrameModeller(ImageSegmentsModeller, LoggingMixin):
    """
    Fitting methods and containers for model data
    Interface between modelling and apertures
    """

    # TODO: option to fit coordinates or not
    # TODO: simultaneous as option

    _metrics = ('aic', 'bic', 'redchi')

    def __init__(self, models, window, ix_fit=None, track_residuals=True,
                 metrics=_metrics, fallback=None):
        # if tracker given, use for positions
        # TODO: self.bg
        self.models = models
        self.nmodels = len(models)
        self.metrics = metrics

        self.fallback = fallback

        # HACK:
        self._ix_bg = [i for i, m in enumerate(models) if 'bg' in m.name]

        if (ix_fit is None):
            ix_fit = []

        self.ix_fit = np.asarray(ix_fit)
        self.nfit = len(self.ix_fit)
        self.window = window = int(window)   # this may be different from the tracker window
        self.grid = np.mgrid[:window, :window]  # grid for sub window

        # Data containers
        self.coords = None    # reference star coordinates across frames
        self.data = {}
        self.metricData = AttrDict()
        self.best = AttrDict()
        self.resData = {}
        self.trackRes = track_residuals

    def __call__(self, data, std=None, masks=None, models=None, coords=None):
        """
        convenience wrapper around main `fit` method

        Parameters
        ----------
        data
        std
        masks
        models: the models to fit
        coords: if not provided and object has a tracker, use tracker to guess
                initial coordinates

        Returns
        -------

        """
        if std is None:
            std = np.ones_like(data)
        if masks is None:
            masks = np.empty((3, 0, len(coords)))
        if (coords is None) and (self.tracker is not None):
            coords = self.tracker(data)
            coords += self.tracker.Rvec # now relative to frame
        if coords is None:
            raise ValueError('Need coords')

        return self.fit(coords, data, std, masks, models)

    # def fit(self, *args):
    #     # null fitting loop
    #     return None, None, None

    def init_mem(self, n):
        # initialize syncronized data containers for parallel processing
        nfit = self.nfit

        for model in self.models:
            self.data[model] = ModelData(model, n, nfit)
            if self.trackRes:
                w = self.window
                self.resData[model] = SyncedArray(shape=(nfit, w, w), fill_value=0)

        for metric in self.metrics:
            self.metricData[metric] = SyncedArray(shape=(n, nfit, self.nmodels))

        # syncronized Data containers for best fit flux
        self.best.ix = SyncedArray(shape=(n, nfit), fill_value=-99, dtype=ctypes.c_int)

        # syncronized Data containers for best fit flux
        # TODO merge containers below?
        self.best.flux = SyncedArray(shape=(n, nfit))
        self.best.flux_std = SyncedArray(shape=(n, nfit))


    def guess_params(self, i, previous=5, cfunc=np.nanmedian):
        # Aperture location, scale, angle parameters taken as median across
        # best fitting model for each star in ix_fit

        slice_ = slice(i-previous-1, i-1)
        ix = self.best.ix[slice_]
        p = np.full((len(ix), 5), np.nan)
        for i, j in enumerate(ix):
            if (i is np.nan) or (j in self._ix_bg):
                continue
            # FIXME: keeping the STAR / BG models seperate will save this hack
            # but then again this whole class is essentially a frequentist hack
            # FIXME: or give the bg models a to_aperture method that returns nans
            p[i] = self.data[self.models[j]][slice_][i]

        if np.isnan(p).all():
            if self.fallback is None:
                raise ValueError('No fallback option')
            self.logger('Using fallback params')
            return self.fallback

        return np.split(cfunc(p, 0), [2, 4, 5], 1)[:-1]


    # def fit(self, coords, data, std, masks, models=None, i=None):
    #     # NOTE: Adding Poisson uncertainty to the data leads to parameters converging
    #     # less frequently when using least squares.
    #     # NOTE: since the infinitely uncertain pixels will effectively mask the out-of-frame data
    #     # for the fitting, we use constant pad value below (which is ~twice as fast as pad='mask')
    #     # TODO: pass in window slices since potentially already computed by finder?
    #
    #     best_model_ix, llcs, apPars = self.fit_stars(coords, data, std, masks, models, i)
    #     # Get parametes from fit if available else (None, None, None)
    #     coo_fit, sigma_xy, theta = np.split(apPars, [2, 4, 5], 1)[:-1]
    #     return coo_fit, sigma_xy, theta
    #
    #     # #
    #     # if coo_fit is None:
    #     #     return coo_fit, sigma_xy, theta
    #     #
    #     # if self.tracker:
    #     #     # calculate
    #     #     shifts = coo_fit + llcs - self.tracker.Rcoo[self.ix_fit]
    #     #     shift = np.nanmean(shifts, 0)
    #     #     coo = self.tracker.rcoo + shift
    #     # else:
    #     #     coo = coo_fit
    #     #
    #     # return coo, sigma_xy, theta
    #
    # # def
    #
    # def fit_stars(self, coords, data, std, masks, models=None, i=None):
    #
    #     best_models_ix = np.empty(self.nfit, int)  # dtype=int
    #     masks = np.array(masks)
    #     llcs = np.empty((2, self.nfit))
    #     apPars = np.full((self.nfit, 5), np.nan)
    #     for jj, j in enumerate(self.ix_fit):
    #         coo = coords[j]
    #         # expand compacted mask
    #         lm = (masks[-1] == jj)
    #         mask = tuple(masks[:2, lm])
    #         sub, sub_std, ixll = \
    #             self.get_segments(j, coo, self.window, data, std, mask)
    #         # do fit and get best model for each star
    #         best_models_ix[jj], best_model, apPars[jj] =\
    #             self._fit_compare(coo - ixll, sub, sub_std, models, i)
    #         llcs[jj] = ixll
    #
    #     return best_models_ix, llcs, apPars
    #
    #
    # # def fit_sub(self, coo, sub, sub_stddev, models=None):
    # #     return list(
    # #         self._fit_models_gen(coo, sub, sub_stddev, models)
    # #     )
    #
    # def _fit_models_gen(self, coo, sub, sub_stddev, models=None):
    #     models = models or self.models
    #     for model in models:
    #         p0 = model.p0guess(sub)
    #         p0[1::-1] = coo
    #         yield model.fit(p0, sub, self.grid, sub_stddev)
    #
    #
    # def _fit_compare(self, j, coo, sub, sub_stddev, models=None, i=None):
    #     """Compare various models for single star"""
    #     models = models or self.models
    #     gof = np.empty((len(models), len(self.metrics)))
    #     for m, r in enumerate(
    #             self._fit_models_gen(coo, sub, sub_stddev, models)):
    #         if r is None:
    #             continue
    #         # convert here??
    #         if i:
    #             p, _, gof[m] = r
    #             model = models[m]
    #             self.save_params(i, j, model, r, sub)
    #
    #     # choose best fitting model and save those fluxes
    #     ix_bm, best_model = self.model_selection(i, j, gof, models)
    #     apPar = best_model.get_aperture_params(p)
    #
    #     if i:
    #         # save best fit flux
    #         self.best.ix[i, j] = ix_bm
    #         self.best.flux[i, j] = self.data[best_model].flux[i, j]
    #         self.best.flux_std[i, j] = self.data[best_model].flux_std[i, j]
    #
    #     return ix_bm, best_model, apPar
    #
    #
    # def model_selection(self, i, j, gofs, models):
    #     """
    #     i - frame number
    #     j - star index relative to modelDb
    #     """
    #     if np.isnan(gofs).all():
    #         self.logger.warning(
    #             'No model converged for Frame %i, Star %i', i, self.ix_fit[j])
    #         return None, None
    #
    #     # the model indices at the minimal gof metric value
    #     best = np.nanargmin(gofs, 0)
    #     ub, ib, cb = np.unique(best, return_index=True, return_counts=True)
    #     # check cross metric consistency
    #     if len(ub) > 1:
    #         self.logger.warning(
    #             'Model selection metrics do not agree. Frame %i, Star %i. '
    #             'Selecting on AIC.',  i, self.ix_fit[j])
    #         # NOTE: This probably means none of the models are an especially good fit
    #
    #     # choose model that most metrics agree is best
    #     ix_bf = ub[cb.argmax()]
    #     best_model = models[ix_bf]
    #     self.logger.info('Best model Frame %i, Star %i: %s' % (i, self.ix_fit[j], best_model))
    #     # return index of best fitting model
    #     if best_model is self.db.bg:
    #         "logging.warning('Best model is BG')"
    #         "flux is upper limit?"
    #
    #     return ix_bf, best_model

    def regularization_hack(self, sub_stddev, coo):
        # TODO: BAYESIAN PLEASE GOD THIS IS FUCKING AWFUL

        dcom = np.sqrt(np.square(self.grid - coo[..., None, None]).sum(0))
        # data_stddev = np.sqrt(dcom / dcom.min())
        # noise weighted by inverse sqrt distance from CoM to assist convergence *I HATE THIS*
        try:
            # !!HACK!!                  because least squares is atrocious!!!
            sub_stddev[dcom > 7.5] *= 1  # up-weight stellar core for fitting
            # !!HACK!!
        except:
            self.logger.warning('Fuckup with regularization_hack')  # frame %i star %i' % (i, j))
            # print(coo)
            # print(sub_stddev)
            # print(dcom)
        return sub_stddev

    def save_params(self, i, j, model, results, sub):
        """save fitted / derived paramers for this model"""
        if i is None:
            return

        p, pu, gof = results

        # Set shared memory
        psfData = self.data[model]
        psfData.params[i, j] = p
        psfData.params_std[i, j] = pu

        # Goodness of fit statistics
        k = self.models.index(model)
        for i, metric in enumerate(self.metrics):
            self.metricData[metric][i, j, k] = gof[metric]

        if self.trackRes:
            # save total residual image
            self.resData[model][jj] += model.rs(p, sub, self.grid)

    # def show():
        # Y, X = grid
        # Z = model(p0, grid)
        #
        # from grafico.imagine import Compare3DImage
        # Compare3DImage(X, Y, Z, sub)
        #
        # embed()
        # # plt.show()
        # raise SystemExit


    # @chrono.timer
    # def fit(self, image, model, ix_fit, window):  # FIXME: redundant??
    #     """Run as a test for fitting stars"""
    #     # TODO: shift to initializer method
    #
    #     # sigma0 = np.empty(Nstars)
    #     # sigma0.fill(np.nan)
    #     params = np.empty((Nstars, model.npar))
    #     data = np.empty((Nstars, window, window))
    #     modelled = np.empty((Nstars, window, window))
    #
    #     for i, ii in enumerate(ix_fit):
    #         coo = Rcoo[ii]
    #         sub, ixll = neighbours(image, coo, window, pad='mask', return_index=True)
    #         data[i] = sub
    #         p0 = model.p0guess(sub)
    #         # print('Guessed', p0)
    #         res = model.fit(p0, sub, grid)
    #         if res is None:
    #             warnings.warn('Pre-fit for star %d did not converge' % ii)
    #             params[i] = None
    #         else:
    #             _p, _, _ = res
    #             _p[:2] += ixll
    #             params[i] = _p
    #             modelled[i] = model(_p, grid)
    #     return params, data, modelled


# from collections import namedtuple
# namedtuple('Data', model_names)
# namedtuple('MetricData', metrics)
# namedtuple('BestModel', ('flux', 'flux_std'))




    # def get_aperture_params(self, i, best_model_ix):
    #     """Parameters to create apertures around stars"""
    #
    #     best_model_ix = best_model_ix[~np.isnan(best_model_ix)]
    #     if not np.size(best_model_ix):
    #         # no convergence!!
    #         return None, None, None
    #
    #     modix, cnts = np.unique(best_model_ix, return_counts=True)
    #
    #     best_overall_ix = int(modix[cnts.argmax()])
    #     # best_model_frame = modelDb.models[best_overall_ix]
    #
    #     # if best model is a bg model, fall back to previous frame values
    #     # if best_model_frame is modelDb.db.bg:
    #     #     logging.warning('Best model overall is ConstantBG! Frame %i' % i)
    #     #     time.sleep(0.5)
    #     #     return get_aperture_params(i-1, modelDb.best.ix[i-1])
    #     # return
    #
    #     # Aperture location and scale parameters taken as median across best fitting models for each star in ix_fit
    #     appars = np.full((len(self.ix_fit), 5), np.nan)
    #     # bestpsf = best_model_ix[best_model_ix != modelDb._ix[modelDb.db.bg]]
    #     # if bestpsf.size:
    #     for j, b in enumerate(best_model_ix):  # bestpsf
    #         model = self.models[int(b)]
    #         p = self.data[model].params[i, j]
    #         appars[j] = model.get_aperture_params(p)
    #
    #     coo_fit, sigma_xy, theta = np.split(appars, [2, 4, 5], 1)[:-1]
    #     return coo_fit, sigma_xy, theta



        # @property
        # def bg(self):
        #     return [mod for (key, mod) in self.db.items() if 'bg' in key]
