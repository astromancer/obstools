import ctypes
# from collections import Callable

import numpy as np
import astropy.units as u

from recipes.parallel.synched import SyncedArray
from recipes.dict import AttrDict
from recipes.logging import LoggingMixin


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
        residu, results = mdlr.background_subtract(data, imbg.mask)

        # track stars
        com = track(residu)
        # save coordinates in shared data array.
        self.coords[i] = com[track.ir]

        # PSF photometry
        # Calculate the standard deviation of the data distribution of each pixel
        data_std = np.ones_like(data)  # FIXME:
        # fit models
        p, pu, gof = mdlr.fit(residu, data_std, self.bad_pixel_mask, )
        # labels=track.bright)
        best_models, params, pstd = self.model_selection(i, p, pu, gof)

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
        # global modlr, apData

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

    def model_selection(self, i, pars, paru, gof):
        """
        Do model selection (per star) based on goodness of fit metric(s)
        """
        # loop over stars
        best_models, params, pstd = [], [], []
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

        m3d = self.tracker.segm.masks3D()
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


from grafico.imagine import FitsCubeDisplay
from matplotlib.widgets import CheckButtons

from .find.trackers import StarTracker


class GraphicalFrameProcessor(FitsCubeDisplay, LoggingMixin):
    # TODO: Inherit from mpl Animation??

    trackerClass = StarTracker
    marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)

    def __init__(self, filename, **kws):
        FitsCubeDisplay.__init__(self, filename, ap_prop_dict={}, ap_updater=None, **kws)

        self.outlines = None

        # toDO
        # self.playing = False

        # bbox = self.ax.get_position()
        # rect = bbox.x0, bbox.y1, 0.2, 0.2
        rect = 0.05, 0.825, 0.2, 0.2
        axb = self.figure.add_axes(rect, xticks=[], yticks=[])
        self.image_buttons = CheckButtons(axb, ('Tracking Regions', 'Data'), (False, True))
        self.image_buttons.on_clicked(self.image_button_action)

        # window slices
        # rect = 0.25, 0.825, 0.2, 0.2
        # axb = self.figure.add_axes(rect, xticks=[], yticks=[])
        # self.window_buttons = CheckButtons(axb, ('Windows', ), (False, ))
        # self.window_buttons.on_clicked(self.toggle_windows)

        self.tracker = None
        self._xmarks = None
        # self._windows = None

    # def toggle_windows(self, label):


    def image_button_action(self, label):
        image = self.get_image_data(self.frame)
        self.imagePlot.set_data(image)
        self.figure.canvas.draw()

    def init_tracker(self, first, **findkws):
        if isinstance(first, int):
            first = slice(first)
        img = np.median(self.data[first], 0)
        self.tracker = self.trackerClass.from_image(img, **findkws)

        return self.tracker, img

    @property
    def xmarks(self):
        # dynamically create track marks when first getting this property
        if self._xmarks is None:
            self._xmarks, = self.ax.plot(*self.tracker.rcoo[:, ::-1].T,
                                         **self.marker_properties)
        return self._xmarks

    def get_image_data(self, i):
        tr, d = self.image_buttons.get_status()
        image = self.data[i]
        trk = self.tracker
        if (tr and d):
            mask = trk.segm.to_mask(trk.use_labels)
            data = np.ma.masked_array(image, mask=mask)
            return data
        elif tr:  # and not d
            return trk.segm.data
        else:
            return image

    def set_frame(self, i, draw=True):
        self.logger.debug('set_frame: %s', i)

        i = FitsCubeDisplay.set_frame(self, i, False)
        # needs_drawing = self._needs_drawing()

        # data = self.get_image_data(i)


        centroids = self.tracker(self.data[i])  # unmask and track
        self.xmarks.set_data(centroids[:, ::-1].T)

        if self.outlines is not None:
            segments = []
            off = self.tracker.offset[::-1]
            for seg in self.outlineData:
                segments.append(seg + off)
            self.outlines.set_segments(segments)

            # if self.use_blit:
            #     self.draw_blit(needs_drawing)

    def show_outlines(self, **kws):
        from matplotlib.collections import LineCollection
        from matplotlib._contour import QuadContourGenerator

        segm = self.tracker.segm
        data = segm.data
        outlines = []
        for s in segm.slices:
            sy, sx = s
            e = np.array([[sx.start - 1, sx.stop + 1],
                          [sy.start - 1, sy.stop + 1]])
            im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]
            f = lambda x, y: im[int(y), int(x)]
            g = np.vectorize(f)

            yd, xd = im.shape
            x = np.linspace(0, xd, xd * 25)
            y = np.linspace(0, yd, yd * 25)
            X, Y = np.meshgrid(x[:-1], y[:-1])
            Z = g(X[:-1], Y[:-1])

            gen = QuadContourGenerator(X[:-1], Y[:-1], Z, None, False, 0)
            c, = gen.create_contour(0)
            outlines.append(c + e[:, 0] - 0.5)

        col = LineCollection(outlines, **kws)
        self.ax.add_collection(col)

        self.outlineData = outlines
        self.outlines = col
        return col

    def play(self):
        if self.playing:
            return

    def pause(self):
        'todo'
