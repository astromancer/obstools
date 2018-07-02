import ctypes
import logging


import numpy as np
import astropy.units as u

from recipes.parallel.synced import SyncedArray
from recipes.dict import AttrDict
from recipes.logging import LoggingMixin
# from recipes.list import flatten


def lm_extract_values_stderr(pars):
    return np.transpose([(p.value, p.stderr) for p in pars.values()])


class FrameProcessor(LoggingMixin):
    # @classmethod
    # def from_fits(self, filename, **options):
    #     ''

    def __init__(self, datacube, tracker=None, modeller=None, apmaker=None,
                 bad_pixel_mask=None):

        self.data = datacube
        self.tracker = tracker
        self.modeller = modeller
        self.maker = apmaker
        self.bad_pixel_mask = bad_pixel_mask

    def __call__(self, i):

        data = self.data[i]
        track = self.tracker
        mdlr = self.modeller
        mkr = self.maker
        apD = self.apData

        # prep background image
        imbg = track.background(data)

        # fit and subtract background
        residu, p_bg = mdlr.background_subtract(data, imbg.mask)
        dat = mdlr.data[mdlr.bg]
        p, pstd = lm_extract_values_stderr(p_bg)
        # try:
        dat.params[i] = p
        dat.params_std[i] = pstd
        # except Exception as err:
        #     print(p, pstd)
        #     print(dat.params[i]._shared)
        #     print(dat.params_std[i]._shared)

        # track stars
        com = track(residu)
        # save coordinates in shared data array.
        self.coords[i] = com[track.ir]

        # PSF photometry
        # Calculate the standard deviation of the data distribution of each pixel
        data_std = np.ones_like(data)  # FIXME:
        # fit models
        results = mdlr.fit(residu, data_std, self.bad_pixel_mask, )
        # save params
        mdlr.save_params(i, results)
        # model selection for each star
        best_models, params, pstd = self.model_selection(i, results)

        # PSF-guided aperture photometry
        # create scaled apertures from models
        appars = mkr.combine_results(best_models, params, axis=0)  # coo_fit, sigma_xy, theta
        aps = mkr.create_apertures(com, appars)
        apsky = mkr.create_apertures(com, appars, sky=True)

        # save appars
        apD.sigma_xy[i], apD.theta[i] = appars[1:]

        # do background subtracted aperture photometry
        flx, flxBG = self.aperture_photometry(residu, aps, apsky)
        apD.flux[i], apD.bg[i] = flx, flxBG

        # save coordinates in shared data array.
        # if
        # self.coords[i] = coo_fit
        # only overwrites coordinates if mdlr.tracker is None

    def init_mem(self, n=None):
        """

        Parameters
        ----------
        n : number of frames (mostly for testing purposes to avoid large memory allocation)

        Returns
        -------

        """
        # global apData

        n = n or len(self.data)
        nstars = len(self.tracker.use_labels)
        naps = np.size(self.maker.r)
        #nfit = len(self.modeller.use_labels)

        # reference star coordinates
        self.coords = SyncedArray(shape=(n, 2))

        # NOTE: You should check how efficient these memory structures are.
        # We might be spending a lot of our time synching access??

        # HACK: Initialize shared memory with nans...
        SyncedArray.__new__.__defaults__ = (None, None, np.nan, ctypes.c_double)  # lazy HACK

        apData = self.apData = AttrDict()
        apData.bg = SyncedArray(shape=(n, nstars))
        apData.flux = SyncedArray(shape=(n, nstars, naps))

        apData.sigma_xy = SyncedArray(shape=(n, 2))  # TODO: for nstars (optionally) ???
        apData.rsky = SyncedArray(shape=(n, 2))
        apData.theta = SyncedArray(shape=(n,))
        # cog_data = np.empty((n, nstars, 2, window*window))

        self.modeller.init_mem(n)

    def model_selection(self, i, results):
        """
        Do model selection (per star) based on goodness of fit metric(s)
        """

        pars, paru, gof = results
        best_models, params, pstd = [], [], []
        # loop over stars
        for j, g in enumerate(gof.swapaxes(0, 1)):  # zip(pars, paru, gof)
            ix, mdl, msg = self.modeller.model_selection(g)
            if msg:
                self.logger.warning('%s (Frame %i, Star %i)', (msg, i, j))

            if ix is not None:
                self.logger.info('Best model: %s (Frame %i, Star %i)' % (mdl, i, j))

            # TODO: if best_model is self.db.bg:
            #     "logging.warning('Best model is BG')"
            #     "flux is upper limit?"

            # yield mdl, p
            best_models.append(mdl)
            params.append(pars[ix][j])
            pstd.append(paru[ix][j])
        return best_models, params, pstd

    def aperture_photometry(self, data, aps, skyaps):

        method = 'exact'

        # a quantity is needed for photutils
        udata = u.Quantity(data, copy=False)

        m3d = self.tracker.segm.to_boolean_3d()
        masks = m3d.any(0, keepdims=True) & ~m3d
        masks |= self.bad_pixel_mask

        Flux = np.empty(np.shape(aps))
        if Flux.ndim == 1:
            Flux = Flux[:, None]
        FluxBG = np.empty(np.shape(skyaps))
        for j, (ap, ann) in enumerate(zip(aps, skyaps)):
            mask = masks[j]

            # sky
            flxBG, flxBGu = ann.do_photometry(udata,
                                              # error,
                                              mask=mask,
                                              # effective_gain,#  must have same shape as data
                                              # TODO: ERROR ESTIMATE
                                              method=method)  # method='subpixel', subpixel=5)
            m = ap.to_mask(method)[0]
            area = (m.data * m.cutout(~mask)).sum()
            fluxBGpp = flxBG / area  # Background Flux per pixel
            flxBGppu = flxBGu / area
            FluxBG[j] = fluxBGpp

            # multi apertures ??
            for k, app in enumerate(np.atleast_1d(ap)):
                flux, flux_err = app.do_photometry(udata,
                                                   mask=mask,
                                                   # error, #TODO: ERROR ESTIMATE
                                                   # effective_gain,#  must have same shape as data
                                                   method=method)
                # get the area of the aperture excluding masked pixels
                m = ap.to_mask(method)[0]
                area = (m.data * m.cutout(~mask)).sum()

                Flux[j, k] = flux - (fluxBGpp * area)

        return Flux, FluxBG

    def save_params(self, i, coo):
        if self.tracker is not None:
            self.coords[i] = coo
            # self.sigma[i] =

    def estimate_max_shift(self, nframes, snr=5, npixels=7):
        """Estimate the maximal positional shift for stars"""
        step = len(self) // nframes  # take `nframes` frames evenly spaced across data set
        maxImage = self[::step].max(0)  #

        threshold = detect_threshold(maxImage, snr)  # detection at snr of 5
        segImage = detect_sources(maxImage, threshold, npixels)
        mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                          for (xs, ys) in segImage.slices], 0)

        # TODO: check for cosmic rays inside sky apertures!

        return mxshift, maxImage, segImage
