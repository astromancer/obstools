import ctypes
from pathlib import Path

import numpy as np
from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from recipes.dict import AttrDict
# from recipes.array import neighbours
from recipes.logging import LoggingMixin

from collections import defaultdict
from ..phot.trackers import LabelUse


# import obstools.modelling.psf.lm_compat as modlib


def make_shared_mem(loc, shape, fill=0., dtype='f', clobber=False):
    """
    Pre-allocate a writeable shared memory map as a container for the
    results of parallel computation. If file already exists and clobber is False
    open in update mode and fill will be ignored. Data persistence ftw.
    """

    # Note: Objects created by this function have no synchronization primitives
    # in place. Having concurrent workers write on overlapping shared memory
    # data segments, for instance by using inplace operators and assignments on
    # a numpy.memmap instance, can lead to data corruption as numpy does not
    # offer atomic operations. Here we does not risk that issue as each task is
    # updating an exclusive segment of the shared result array.

    loc = Path(loc)
    if not loc.parent.exists():
        loc.parent.mkdir()

    if not loc.exists() or clobber:
        mm = np.memmap(loc, dtype, 'w+', shape=shape)
        if fill:
            mm[:] = fill
    else:  # update mode
        mm = np.memmap(loc, dtype, 'r+', shape=shape)

    return mm


# from IPython import embed

# TODO: class that fits all stars simultaneously with MCMC. compare for speed / accuracy etc...
#
# class ModelData():
#     def __init__(self, model, N, Nfit):
#         """Shared, Synced Data containers for model"""
#         npars = model.npar
#         self.params = SyncedArray(shape=(N, Nfit, npars))  # fitting parameters
#         self.params_std = SyncedArray(shape=(N, Nfit, npars))  # standard deviation (error?) on parameters
#
#         # FIXME: you can avoid this conditional by null object pattern for 'integrate'
#         if hasattr(model, 'integrate'):
#             self.flux = SyncedArray(shape=(N, Nfit))
#             self.flux_std = SyncedArray(shape=(N, Nfit))
#             # NOTE: can also be computed post-facto
#
#         if hasattr(model, 'reparameterize'):
#             nparAlt = len(model.reparameterize(model.default_params))
#             self.alt = SyncedArray(shape=(N, Nfit, nparAlt))


def cdist_tri(coo):
    """distance matrix with lower triangular region masked"""
    n = len(coo)
    sdist = cdist(coo, coo)  # pixel distance between stars
    # since the distance matrix is symmetric, ignore lower half
    sdist = np.ma.masked_array(sdist)
    sdist[np.tril_indices(n)] = np.ma.masked
    return sdist


# from collections import namedtuple
# ModelContainer = namedtuple('ModelContainer', ('psf', 'bg'))





class ImageSegmentsModeller(LabelUse, LoggingMixin):
    """
    Model fitting and comparison on segmented CCD frame
    """

    _metrics = ('aic', 'bic', 'redchi')

    # TODO: mask_policy.  filter / set error to inf
    # TODO: incorporate background fit
    # TODO: option to fit centres or use Centroids

    def __init__(self, segm, psf, bg=None, metrics=_metrics, use_labels=None,
                 fit_positions=True, track_residuals=False):
        """

        Parameters
        ----------
        segm : SegmentationHelper instance
        psf : list of psf models
        bg : list of bg models
        metrics : goodness of fit evaluation metrics
        """

        # ideas: detect stars that share windows and fit simultaneously

        self.segm = segm
        self.models = np.array(psf, 'O')
        self.bg = bg  # TODO: many models - combinations of psf + bg models
        self.metrics = list(metrics)
        self.grid = np.indices(self.segm.shape)

        LabelUse.__init__(self, use_labels)

        # if use_labels is None:
        #     use_labels = self.segm.labels
        # self.use_labels = use_labels

    @property
    def nmodels(self):
        return len(self.models)

    @property
    def nfit(self):
        return len(self.use_labels)

    def iter_segments(self, data, std, labels=None):
        """
        Yields rectangular sub-regions of the image and stddev image.
        Overlapping detections masked.

        Parameters
        ----------
        data
        std
        labels

        Returns
        -------

        """
        for d, (sy, sx) in self.segm.iter_segments(data, labels, True, True):
            yield d, std[sy, sx], self.grid[:, sy, sx]

    def fit(self, data, std=None, mask=None, models=None, labels=None):
        """
        Fit frame data by looping over segments

        Parameters
        ----------
        data
        std
        mask
        models
        labels

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

        # fit data for all labels if none specified
        if labels is None:
            labels = self.use_labels

        # loop over models and fit
        gof = np.full((len(models), len(labels), len(self.metrics)), np.nan)
        pars, paru = [], []
        for m, model in enumerate(models):
            p, pu, gof[m] = self._fit_model(model, data, std, labels)
            pars.append(p)
            paru.append(pu)

        return pars, paru, gof

    def _fit_model(self, model, data, std, labels):
        """
        Fit this model for all segments
        """
        # this method is for production
        p, pu = np.full((2, len(labels), model.npar), np.nan)
        gof = np.full((len(labels), len(self.metrics)), np.nan)
        for j, (sub, sub_std, grid) in enumerate(
                self.iter_segments(data, std, labels=labels)):
            p0 = model.p0guess(sub, grid)
            r = model.fit(p0, sub, grid, sub_std, nan_policy='omit')
            if r is not None:
                p[j], pu[j], gof[j] = r

        return p, pu, gof

    def _fit_segment(self, sub, sub_std, grid, models=None):
        """
        Fit various models for single segment
        """
        # this method is for convenience
        models = models or self.models
        gof = np.full((len(models), len(self.metrics)), np.nan)
        par, paru = [], []  # parameter values and associated uncertainty
        for m, r in enumerate(self._gen_fits(sub, sub_std, grid, models)):
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
        # ix_bm, best_model = self.model_selection(i, j, gof, models)

        # if i:
        #     # save best fit flux
        #     self.best.ix[i, j] = ix_bm
        #     self.best.flux[i, j] = self.data[best_model].flux[i, j]
        #     self.best.flux_std[i, j] = self.data[best_model].flux_std[i, j]

        return par, paru, gof

    def _gen_fits(self, sub, sub_stddev, grid, models=None):
        # TODO: use this task producer more efficiently
        models = models or self.models
        for model in models:
            # try:
            p0 = model.p0guess(sub, grid, sub_stddev)
            yield model.fit(p0, sub, grid, sub_stddev)
            # except Exception as err:
            #     from IPython import embed
            #     embed()
            #     raise SystemExit

    def model_selection(self, gof):
        """
        Select best model based on goodness of fit metric(s)
        """
        msg = None  # info message

        # no convergence
        if np.isnan(gof).all():
            return -99, None, 'No model converged.'

        # only one model
        if len(gof) == 1:
            return 0, self.models[0], msg

        # the model indices at the minimal gof metric value
        best = np.nanargmin(gof, 0)
        ub, ib, cb = np.unique(best, return_index=True, return_counts=True)

        # check cross metric consistency
        if len(ub) > 1:
            msg = 'GoF metrics inconsistent. Using %r.' % self.metrics[0]
            # NOTE: This probably means none of the models is an especially good fit

        # choose model that most metrics agree is best
        ix_bf = ub[cb.argmax()]
        best_model = self.models[ix_bf]

        return ix_bf, best_model, msg

    def background_subtract(self, image, mask=None, p0=None):

        # background subtraction
        imbg = self.segm.mask_detected(image)

        if mask is not None:
            imbg.mask |= mask

        # return imbg

        # fit background
        mdl = self.bg
        if mdl:
            # try:
            results = mdl.fit(imbg, p0)
            # background subtracted image
            im_bg = mdl.residuals(results, image)
            # except Exception as err:
            #     self.logger.exception('BG')

            return im_bg, results
        else:
            raise NotImplementedError

    def emperical_psf(self, image, centres, labels=None):

        from scipy.stats import binned_statistic_2d

        if labels is None:
            labels = self.use_labels

        # NOTE: this is pretty crude

        # bgstd = np.empty(len(labels))
        # for i, m in enumerate(self.segm.to_annuli(2, 5, labels)):
        #     bgstd[i] = image[m].std()

        # imbg = tracker.background(image)
        # resi, p_bg = mdlr.background_subtract(image, imbg.mask)

        x, y, z = [], [], []
        v = []
        for i, (thumb, (sly, slx)) in enumerate(
                self.segm.iter_segments(image, labels, True, True)):
            g = self.grid[:, sly, slx] - centres[i, None, None].T
            x.extend(g[0].ravel())
            y.extend(g[1].ravel())
            v.extend((thumb / thumb.max()).ravel())

        rng = np.floor(min(np.min(x), np.min(y))), np.ceil(max(np.max(x), np.max(y)))
        stat, xe, ye, bn = binned_statistic_2d(x, y, v,
                                               bins=np.ptp(rng),
                                               range=(rng, rng))
        return stat


class ModelData(LoggingMixin):
    def __init__(self, model, n, nfit, folder, clobber=False):
        """Shared data containers for model"""

        folder = Path(folder)
        if not folder.exists():
            self.logger.info('Creating folder: %s', str(folder))
            folder.mkdir(parents=True)

        self.loc = str(folder)
        npars = model.npar

        # fitting parameters
        shape = (n, nfit, npars)
        locPar = '%s.par' % model.name
        locStd = '%s.std' % model.name
        self.params = make_shared_mem(locPar, shape, np.nan, 'f', clobber)
        # standard deviation on parameters
        self.params_std = make_shared_mem(locStd, shape, np.nan, 'f', clobber)

        # self.
        #
        # if hasattr(model, 'integrate'):
        #     locFlx = folder / 'flx'
        #     locFlxStd = folder / 'flxStd'
        #     shape = (n, nfit)
        #     self.flux = make_shared_mem(locFlx, shape, np.nan)
        #     self.flux_std = make_shared_mem(locFlxStd, shape, np.nan)
        #     # NOTE: can also be computed post-facto

    def save_params(self, i, j, p, pu):
        self.params[i, j] = p
        self.params_std[i, j] = pu


class ModellingResultsMixin(object):
    def __init__(self, save_residual=False):

        # Data containers
        # self.coords = None    # reference star coordinates across frames
        self.data = {}
        #self.metricData = AttrDict()
        self.best = AttrDict()
        self.resData = defaultdict(list)
        self.saveRes = save_residual

    def init_mem(self, n, nfit, loc=None, clobber=False):
        # initialize sharable data containers for parallel processing

        if loc is None:
            import tempfile
            loc = Path(tempfile.mkdtemp())

        self.loc = str(loc)

        for k, model in enumerate(self.models):
            self.data[k] = ModelData(model, n, nfit, loc)
            # if self.saveRes
            #     sizes = self.segm.box_sizes(self.use_labels)
            #     for j, (sy, sx) in enumerate(sizes.astype(int)):
            #         r = SyncedArray(shape=(sy, sx), fill_value=0)
            #         self.resData[model].append(r)

        # global bg model
        self.data[-1] = ModelData(self.bg, n, 1, loc)

        if self.nmodels:
            locMetric = loc / 'GoF.dat'
            shape = (n, nfit, self.nmodels, len(self.metrics))
            self.metricData = make_shared_mem(
                    locMetric, shape, np.nan, 'f', clobber)

            # shared Data containers for best fit flux
            # columns:
            locBest = loc / 'bestIx.dat'
            self.best.ix = make_shared_mem(
                    locBest, (n, nfit), -99, ctypes.c_int, clobber)

            # Data containers for best fit flux
            locFlx = loc / 'bestFlx.dat'
            # locFlxStd = loc / 'bestFlxStd.dat'
            self.best.flux = make_shared_mem(
                    locFlx, (n, nfit, 2), np.nan, 'f', clobber)
            # self.best.flux_std = make_shared_mem(locFlxStd, (n, nfit), np.nan)

        return loc

    # def get_best_models(self, bestIx):
    #     # get the best fit models
    #     bestIx = np.asarray(bestIx)
    #     hasBest = (bestIx != -99)
    #     # init with None
    #     best_models = np.empty(len(bestIx), 'O')
    #     # fill best models
    #     best_models[hasBest] = self.models[list(bestIx[hasBest])]
    #     return best_models

    def save_best_flux(self, i, best):
        bestIx, bestModels, params, pstd = best

        self.best.ix[i] = bestIx  # per detected object
        for j, (m, p, pu) in enumerate(zip(bestModels, params, pstd)):
            if m is not None:
                self.best.flux[i, j] = m.flux(p), m.fluxStd(p, pu)
                # self.best.flux_std[i] =  m.fluxStd(p, pu)


    def save_params(self, i, results, best):

        p, pu, gof = results
        # loop over models # FIXME: eliminate this loop
        for k, m in enumerate(self.models):
            self.data[k].params[i] = p[k]       # (nstars, npars)
            self.data[k].params_std[i] = pu[k]

            # loop over stars
            # for j, r in enumerate(zip(p[k], pu[k], gof[k])):
            #     self._save_params(m, i, j, r)

        self.save_best_flux(i, best)


        # bestIx, params, pstd = best
        # best_models = mdlr.models[bestIx]  #
        # self.best.ix[i] = bestIx # per detected object
        # self.best.flux[i] =
        # self.best.flux_std[i] = pstd

    # def _save_params(self, model, i, j, results):  # sub=None, grid=None
    #     """save fitted / derived paramers for this model"""
    #     if i is None:
    #         return
    #
    #     p, pu, gof = results
    #
    #     # set shared memory
    #     psfData = self.data[model]
    #     psfData.params[i, j] = p
    #     psfData.params_std[i, j] = pu
    #
    #     print('saved', psfData.params)
    #
    #     Goodness of fit statistics
    #     if gof is not None:
    #         k = self.models.index(model)
    #         self.metricData[i, j] = gof.swapaxes(0, 1) # (nstars, nmodels, nmetrics)
    #         for m, metric in enumerate(self.metrics):
    #             self.metricData[i, j, k,] = gof[m]



class AperturesFromModel(LoggingMixin):
    """
    class that creates apertures based on modelling results
    """

    def __init__(self, r, rsky):
        self.r = np.asarray(r)
        self.rsky = np.asarray(rsky)

        self._pars = None

    # @property
    # def coo(self):
    #     return self._pars[:, :2]
    #
    # @property
    # def sigmaXY(self):
    #     return self._pars[:, 2:4]
    #
    # @property
    # def theta(self):
    #     return self._pars[:, -1]

    def init_mem(self, n, loc, clobber=False):
        self.coords = make_shared_mem(loc, (n, 2), np.nan, 'f', clobber)
        self.sigmaXY = make_shared_mem(loc, (n, 2), np.nan, 'f', clobber)
        self.theta = make_shared_mem(loc, (n, ), np.nan, 'f', clobber)


    def combine_results(self, models, p, pstd, cfunc=np.nanmedian, axis=(), ):
        # nan_policy=None):
        """
        Get aperture parameters from best model for each star

        Parameters
        ----------
        p, pstd
        r_sigma
        cfunc
        axis

        Returns
        -------

        """
        coo = np.full((len(p), 2), np.nan)
        sxy = np.full((len(p), 2), np.nan)
        th = np.full((len(p), 1), np.nan)
        for i, (mdl, q) in enumerate(zip(models, p)): # TODO: eliminate this loop
            if mdl is not None:
                coo[i] = q[1::-1]
                sxy[i] = mdl.get_sigma_xy(q)  # [None]
                th[i] = mdl.get_theta(q)  # [None]

        r = cfunc(sxy, axis)
        th = cfunc(th, axis)

        return coo, r, th

    def save(self, i, appars):

        coo, sxy, th = appars
        self.coords[i] = coo
        self.sigmaXY[i] = sxy
        self.theta[i] = th


    # def forced_apertures(self, coo, r, th):

    # def _fallback(self, a, fallback):
    #     """
    #     Used to substite aperture parameters for unconvergent models
    #     (i.e force photometry)
    #     """
    #
    #     nans = np.isnan(a).any(1)
    #     if nans.any() and (fallback is not None):
    #         sz = np.size(fallback)
    #         if isinstance(fallback, Callable):
    #             a[nans] = fallback(a)
    #         elif sz == 1:  # a number
    #             a[nans] = fallback
    #         elif sz > 1:  # presumably an array
    #             a[nans] = fallback[nans]
    #
    # for i, (a, f) in enumerate(zip(appars, fallbacks)):
    #     nans = np.isnan(a).any(1)
    #     a[nans] = f(a, nans)

    def create_apertures(self, appars, coords=None, sky=False, fallbacks=None):
        """
        Initialize apertures at com coordinates with radii determined from
        model dispersion (standard deviation (sigma) or fwhm) and the scaling
        attributes *r*, *rsky* (in units of sigma)

        Note: even though photutils supports multiple positions for apertures
        with the same radii, this is not preferred if we want to mask background
        stars for each star aperture. So we return a list of apertures
        """
        fcoords, sigma_xy, theta = appars
        rx, ry = sigma_xy * self.r

        if coords is None:
            coords = fcoords  # use fit coordinates for aperture positions

        if sky:
            sx, sy = sigma_xy
            rxsky = sx * self.rsky
            rysky = sy * self.rsky[1]
            return [EllipticalAnnulus(coo, *rxsky, rysky, theta)
                    for coo in coords[:, 1::-1]]

        else:
            return [EllipticalAperture(coo, rx, ry, theta)
                    for coo in coords[:, 1::-1]]

        # TODO: use_fit_coords
        # TODO: handle bright and faint seperately here
        # ixm = mdlr.segm.indices(mdlr.ignore_labels) # TODO: don't recompute every time

        # coo, r, th = appars

        # coords = np.empty_like(com)
        # rnew = np.ones_like(r)

        # self.modeller.use_labels
        # self.tracker.use_labels

        # coo, r, th = (self._fallback(a, f) for a, f in zip(appars, fallbacks))

    def check_aps_sky(self, i, rsky, rmax):
        rskyin, rskyout = rsky
        info = 'Frame {:d}, rin={:.1f}, rout={:.1f}'
        if np.isnan(rsky).any():
            self.logger.warning('Nans in sky apertures: ' + info, i, *rsky)
        if rskyin > rskyout:
            self.logger.warning('rskyin > rskyout: ' + info, i, *rsky)
        if rskyin > rmax:
            self.logger.warning('Large sky apertures: ' + info, i, *rsky)


class ImageModeller(ImageSegmentsModeller, ModellingResultsMixin):
    def __init__(self, segm, psf, bg, metrics=None, use_labels=None,
                 fit_positions=True, save_residual=True):
        ImageSegmentsModeller.__init__(self, segm, psf, bg, self._metrics, use_labels)
        ModellingResultsMixin.__init__(self, save_residual)
