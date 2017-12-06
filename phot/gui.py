import ctypes
import logging
from collections import namedtuple #Callable

import numpy as np
import astropy.units as u

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Ellipse as _Ellipse, Annulus
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.widgets import RadioButtons, CheckButtons
from matplotlib._contour import QuadContourGenerator

from obstools.aps import ApertureCollection, SkyApertures

from recipes.parallel.synced import SyncedArray
from recipes.dict import AttrDict
from recipes.logging import LoggingMixin
from recipes.list import flatten



def lm_extract_values_stderr(pars):
    return np.transpose([(p.value, p.stderr) for p in pars.values()])


def rotation_matrix_2D(theta):
    """Rotation matrix"""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin],
                     [sin, cos]])


def rotate_2D(xy, theta):
    return rotation_matrix_2D(theta) @ xy


def edge_proximity(self, xy):

    xyprox = np.subtract(xy, self.center)
    x, y = xyprox.T
    theta = np.arctan2(y,x) # angle between position and ellipse centre
    alpha = np.radians(self.angle)
    phi = theta - alpha
    a, b = self.a, self.b
    # radial equation for ellipse
    r = (a * b) / np.sqrt(np.square((b * np.cos(phi),
                                     a * np.sin(phi))).sum())
    # radial distance of (x, y)
    rp = np.sqrt(np.square(xyprox).sum())
    return rp - r


def ellipse_picker(self, event):
    mouse_position = (event.x, event.y)
    if None in mouse_position:
        return False, {}

    hit = np.abs(edge_proximity(self, mouse_position)) < 3
    return hit, {}


def annulus_picker(self, event):

    mouse_position = (event.x, event.y)

    if None in mouse_position:
        return False, {}

    dr = edge_proximity(self, mouse_position)
    tol = 3
    hit = (tol > dr > 0) or (0 > dr > tol - self.width)
    return hit, {}


def linecol_picker(self, event):
    # print('pickr ' * 10)
    # print(event)
    xy = event.xdata, event.ydata

    segs = self.get_segments()
    ind = []
    for i, s in enumerate(segs):
        r = np.sqrt(np.square(s - xy).sum(1))
        if np.any(r < 1):
            ind.append(i)

    return len(ind) > 0, dict(ind=ind)


def binary_contours(b):
    """Image contour around mask pixels"""

    f = lambda x, y: b[int(y), int(x)]
    g = np.vectorize(f)

    yd, xd = b.shape
    x = np.linspace(0, xd, xd * 25)
    y = np.linspace(0, yd, yd * 25)
    X, Y = np.meshgrid(x[:-1], y[:-1])
    Z = g(X[:-1], Y[:-1])

    gen = QuadContourGenerator(X[:-1], Y[:-1], Z, None, False, 0)
    c = gen.create_contour(0)
    return c



class Ellipse(_Ellipse):
    @property
    def a(self):
        """semi-major axis"""
        return self.width / 2

    @property
    def b(self):
        """semi-minor axis"""
        return self.height / 2


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


class FrameProcessorGUI(FitsCubeDisplay, LoggingMixin):
    # TODO: Inherit from mpl Animation??

    _xpcom = dict(marker='x', ms=6, ls='none', picker=5)
    xcomProp = dict(mec='r', **_xpcom)
    xpeakProp = dict(mec='k', **_xpcom)
    xfitProp = dict(mec='g', **_xpcom)

    apStarProp = dict(ec='c', ls='-', fc='none', lw=1)
    apSkyProp = dict(ec='b', fc='none', lw=1)
    # winProp = dict(ec='g', fc='none', ls=':', lw=1.5, picker=5)
    # TODO: mpl collections - make the shortcuts work!
    winProp = dict(edgecolor='g', facecolor='none',
                   linestyle=':', linewidth=1.5, picker=5)

    def __init__(self, proc, **kws):

        filename = proc.data.filename
        FitsCubeDisplay.__init__(self, filename, connect=False, **kws)

        self.proc = proc

        # create position markers (initially empty)
        marks = []
        for prp in (self.xcomProp, self.xpeakProp, self.xfitProp):
            mrk, = self.ax.plot([], **prp)
            marks.append(mrk)
        Markers = namedtuple('Markers', ('com', 'peak', 'mdl'))
        self.markers = Markers(*marks)

        # create apertures
        Aps = namedtuple('Apertures', ('stars', 'sky'))
        self.aps = Aps(ApertureCollection(**self.apStarProp),
                       SkyApertures(**self.apSkyProp))

        for aps in self.aps:
            aps.add_to_axes(self.ax)
        # self.aps.stars.add_to_axes(self.ax)

        #
        self.windows = self.show_windows()

        # self.outlines = None
        self.outlines = self.show_outlines(color='r')
        self.outlineData = self.outlines.get_segments()

        # add the centroid markers to the slider update so they are drawn when clim changed
        self.sliders.lower.on_changed.add(self._slider_move)
        self.sliders.upper.on_changed.add(self._slider_move)
        # TODO: generic on_changed method for the sliders that adds to both

        # add gui buttons
        art = self.add_buttons()

        # update everything to display the first frame
        self.update(0)

        # bg grid #HACK
        self._bggrid = proc.modeller.bg.grid_from_data(self.data[0])

        # toDO
        # self.playing = False

        self.connect()

    def _slider_move(self, x, y):
        draw_list = []
        # art = self.markers + self.aps + (self.windows, )
        for mrk in flatten((self.markers, self.aps, self.windows)):
            if mrk.get_visible():
                draw_list.append(mrk)

        return draw_list

    # def toggle_windows(self, label):

    def init_figure(self, **kws):

        import matplotlib.pylab as plt
        from matplotlib.gridspec import GridSpec
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ax = kws.pop('ax', None)
        title = kws.pop('title', None)
        autoscale_figure = kws.pop('autoscale_figure', True)
        # sidebar = kws.pop('sidebar', True)

        # create axes if required
        if ax is None:
            if autoscale_figure:
                # automatically determine the figure size based on the data
                figsize = self.guess_figsize(self.data)
            else:
                figsize = None

            fig = plt.figure(figsize=figsize)

            self._gs = gs = GridSpec(2, 5,
                              left=0.05, right=0.95,
                              top=0.98, bottom=0.05,
                              hspace=0, wspace=0,
                              height_ratios=(2, 1))

            ax = fig.add_subplot(gs[0, :])

            # axes = self.init_axes(fig)
        # else:
            # axes = namedtuple('AxesContainer', ('image',))(ax)

        self.divider = make_axes_locatable(ax)
        # ax = axes.image
        # set the axes title if given
        if not title is None:
            ax.set_title(title)

        # setup coordinate display
        ax.format_coord = self.cooDisplayFormatter
        return ax.figure, ax

    def guess_figsize(self, data):
        size = super().guess_figsize(data)
        # create a bit more space below the figure for the frame nr indicator
        size[1] += 1.5
        # logging.debug('CubeDisplayBase: Guessed figure size: (%.1f, %.1f)', *size)
        return size

    def add_buttons(self):
        """"""
        ttlprp = dict(fontweight='bold', size='large')
        self.lax = lax = self.figure.add_subplot(self._gs[1, :2], frameon=False,
                                      xticks=[], yticks=[])
        lax.set_title('Markers', **ttlprp)
        proxies = self.add_legend()
        art = self.aps + (self.windows, ) + self.markers + (self.outlines,)
        visible = (True, True, True, True, True, True, False)
        self.lgb = LegendGuiBase(self.figure, art, proxies, visible)

        rax1 = self.figure.add_subplot(self._gs[1, 2], aspect='equal', frameon=False)
        self.image_selector = RadioButtons(
            rax1, ('Raw', 'Calibrated', 'BG subtracted'))
        rax1.set_title('Image Data', **ttlprp)
        self.image_selector.on_clicked(self.image_button_action)

        rax2 = self.figure.add_subplot(self._gs[1, 3], aspect='equal', frameon=False)
        self.mask_selector = CheckButtons(
            rax2, ('Bad pixels', 'Stars'), (False, False))
            # todo: right stars / faint stars
        rax2.set_title('Masks', **ttlprp)
        self.mask_selector.on_clicked(self.image_button_action)

        # fig.subplots_adjust(top=0.83)
        fig = self.figure
        figsize = fig.get_size_inches() + [0, 1]
        fig.set_size_inches(figsize)

    def image_button_action(self, label):
        print('button i')
        image = self.get_image_data(self.frame)
        self.imagePlot.set_data(image) # FIXME: update histogram!
        self.figure.canvas.draw()

    # def mask_button_action(self, label):
    #     print('button m')
    #     image = self.get_image_data(self.frame)
    #     self.imagePlot.set_data(image)
    #     self.figure.canvas.draw()

    def get_image_data(self, i):

        s = self.image_selector.value_selected
        # get image data to display
        image = self.data[i]
        if s == 'Calibrated':
            logging.warning('need calibration image')

        if s == 'BG subtracted':
            mdlr = self.proc.modeller
            mdl = mdlr.bg
            p = mdlr.data[mdl].params[i, 0]
            image = mdl.residuals(p, image, self._bggrid)

        mask = self.get_image_mask()

        return np.ma.masked_array(image, mask=mask)

    def get_image_mask(self):
        # apply masks
        trk = self.proc.tracker
        stat = self.mask_selector.get_status()
        mask = False
        if stat[0]:
            mask |= self.proc.bad_pixel_mask
        if stat[1]:
            mask |= trk.segm.to_mask(trk.use_labels)
        return mask

    def update(self, i, draw=True):
        self.logger.debug('set_frame: %s', i)

        # get image
        i, image = FitsCubeDisplay.update(self, i)

        #
        proc = self.proc
        trk = proc.tracker

        coords = proc.coords[i]
        centroids = (coords + trk.rvec)[:, ::-1]

        # update markers
        mprx = self.proxies.markers
        if mprx.com.get_visible():
            self.markers.com.set_data(centroids.T)

        if mprx.peak.get_visible():
            peaks = trk.segm.imax(image, trk.use_labels)
            self.markers.peak.set_data(peaks[:, ::-1].T)

        if mprx.mdl.get_visible():
            mdlr = self.proc.modeller
            mdl = mdlr.models[0]
            cxf = mdlr.data[mdl].params[i, :, :2].T
            self.markers.mdl.set_data(cxf)

        # update apertures
        self.aps.stars.coords = centroids

        sxy = proc.apData.sigma_xy[i]
        angle = proc.apData.theta[i]
        self.aps.stars.a, self.aps.stars.b = sxy * proc.maker.r
        self.aps.stars.angles = angle

        # inner and outer semi-major and -minor for sky
        a, b = proc.maker.rsky * sxy[None].T
        self.aps.sky.coords = centroids
        self.aps.sky.angles = angle
        self.aps.sky.a = a
        self.aps.sky.b = b
        # self.aps.sky.outer.a, self.aps.sky.outer.b = skyout

        # contour tracking regions
        if self.outlines is not None:
            segments = []
            off = trk.offset[::-1]
            for seg in self.outlineData:
                segments.append(seg + off)
            self.outlines.set_segments(segments)

            # if self.use_blit:
            #     self.draw_blit(needs_drawing)

    # def update_plots(self, i):
    #
    #     proc = self.proc

    def add_legend(self):

        #TODO: better to have titles inside axes to they don't begin overlapping
        #  on figure resize

        tr = None # NOTE: having the artist transform with the axes would
        # probably be better, but hard to get ellipse to draw correctly then
        common = dict(transform=tr)

        # Apertures from model fits
        # epos = np.array((0.1, 0.8))
        lax = self.lax
        epos = lax.transAxes.transform((0.075, 0.85))
        # sxy = (4, 5)

        # r = np.c_[rs].T * sxy
        r = (16, 20)
        a = 45

        apStar = Ellipse(epos, *r, a, **self.apStarProp, **common,
                         picker=ellipse_picker)  # aperture
        #apSkyin = Ellipse(epos, *r[1], a, **self.apSkyProp, **common, picker=ellipse_picker)
        #apSkyout = Ellipse(epos, *r[2], a, **self.apSkyProp, **common, picker=ellipse_picker)

        w = 7
        rxy = (21, 25)
        skyprx = Annulus(epos, rxy, w, a, **self.apSkyProp, **common,
                         picker=annulus_picker)

        # from obstools.aps import ApertureCollection
        #
        # qpos = np.zeros((2,2))#lax.transAxes.transform([(0.075, 0.85)] * 2)
        # skyprx = ApertureCollection(*r[:2].T / 100, a, coords=qpos, **self.apSkyProp)
        # # skyprx.add_to_axes(lax)
        # skyprx._transOffset = lax.transAxes
        # lax.add_collection(skyprx)

        rw = rh = 25
        rpos = lax.transAxes.transform((0.075, 0.5)) - rw / 2
        rect = Rectangle(rpos, rw, rh, **self.winProp, **common)  # windows
        xfit = Line2D(*epos[None].T, **self.xfitProp, **common)  # fit position

        # skyprx = SkyAps(apSkyin, apSkyout)
        # TODO: mpl.Container?
        aprx = type(self.aps)(apStar, skyprx)

        # Centre markers
        xpos = lax.transAxes.transform([0.6, 0.85])
        xcom = Line2D(*xpos[None].T, **self.xcomProp, **common)  # CoM markers
        xpkpos = lax.transAxes.transform([0.6, 0.5])
        xpk = Line2D(*xpkpos[None].T, **self.xpeakProp, **common)  # CoM markers
        mprx = type(self.markers)(xcom, xpk, xfit)

        # tracking region contours proxy
        r = 10
        b = np.square(np.mgrid[:r, :r] - r / 2).sum(0) < (r / 3) ** 2
        c = binary_contours(b)
        # c = np.subtract(c, r/2 + 0.5) + xpos
        #xpos =
        # c = np.subtract(c, r / 2 + 0.5) * 4 + np.squeeze(xcom.get_data())# / 2
        # c = np.add(c, xpos)
        c = np.subtract(c, r / 2 + 0.5) * 4.5 + (373.065, 133.237)

        cntr = Line2D(*c[0].T, color='r', transform=None, picker=5)

        # cntr = LineCollection(c, color='r',
        #                       picker=linecol_picker, transform=None)

        # collect proxy art
        Prox = namedtuple('Proxies', ('markers', 'aps', 'win', 'contours'))
        self.proxies = Prox(mprx, aprx, rect, cntr)

        # artists = [apStar, apSkyin, apSkyout, rect, xfit, xcom, xpk]
        artists = aprx + (rect,) + mprx + (cntr,)
        for art in artists:
            lax.add_artist(art)

        txtoff = (40, 0)
        txtprp = dict(fontdict=dict(weight='bold', size='medium'),
                      transform=None, va='center', clip_on=True)

        rs = 4, 7, 10
        txt = 'Auto aps.\n' + r'$(r_{\bigstar}=%g\sigma$,' \
                              r' $r_{sky}=(%g\sigma, %g\sigma))$' % rs
        text = lax.text(*(epos + txtoff), txt, **txtprp)

        pos = rpos + (rect.get_width() / 2, rect.get_height() / 2)
        lax.text(*(pos + txtoff), 'Model windows', **txtprp)

        #
        lax.text(*(xpos + txtoff), 'Centroids', **txtprp)
        lax.text(*(xpkpos + txtoff), 'Peaks', **txtprp)

        return artists

    def show_windows(self):
        """
        Create PatchCollection for Rectangular windows

        Returns
        -------

        """
        # model windows
        trk = self.proc.tracker
        segm = trk.segm
        ix = segm.indices(trk.use_labels)  # TODO: method of StarTracker class
        slices = np.take(segm.slices, ix, 0)
        rect = []
        for (y, x) in slices:
            xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
            w = x.stop - x.start
            h = y.stop - y.start
            r = Rectangle(xy, w, h)
            rect.append(r)

        windows = PatchCollection(rect, **self.winProp)
        self.ax.add_collection(windows)
        return windows

    def show_outlines(self, **kws):
        """
        Create LineColection that delineates the tracking regions

        Parameters
        ----------
        kws

        Returns
        -------

        """
        # TODO: method of TrackerGui class

        trk = self.proc.tracker
        segm = self.proc.tracker.segm
        data = segm.data
        outlines = []

        # trk.segm.thumbnails(labels=trk.use_labels)
        ix = trk.segm.indices(trk.use_labels)
        slices = np.take(segm.slices, ix, 0)

        for s in slices:
            try:
                sy, sx = s
                e = np.array([[sx.start - 1, sx.stop + 1],
                              [sy.start - 1, sy.stop + 1]])
                e = np.clip(e, 0, np.inf).astype(int)
                # print(e)
                im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]

                contours = binary_contours(im)
                for c in contours:
                    outlines.append(c + e[:, 0] - 0.5)
            except Exception as err:
                from IPython import embed
                embed()

        col = LineCollection(outlines, **kws)
        self.ax.add_collection(col)

        return col

    def connect(self):

        FitsCubeDisplay.connect(self)
        self.lgb.connect()

    # def add_legend(self):
    #     from matplotlib.lines import Line2D
    #     from matplotlib.patches import Rectangle, Ellipse, Circle
    #     from matplotlib.legend_handler import HandlerPatch, HandlerLine2D
    #
    #     def handleSquare(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    #         xy = (xdescent, ydescent - width / 3)
    #         return Rectangle(xy, width, width)
    #
    #     def handleEllipse(w, h, angle):
    #         def _handler(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    #             xy = (xdescent + width / 2, ydescent + height / 2)
    #             return Ellipse(xy, w, h, angle=angle, lw=1)
    #         return _handler
    #
    #     # def handleCircle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    #     #     w = width / 2
    #     #     return Circle((xdescent + width / 2, ydescent + width / 8), width / 3.5, lw=1)
    #
    #     xcomProp = dict(mec='r', marker='x', ms=6, ls='none')
    #     xfitProp = dict(mec='g', mfc='g', marker='o', ms=3, ls='none')
    #     xpeakProp = dict(mec='k', marker='x', ms=6, ls='none')
    #
    #     apStarProp = dict(ec='c', ls='-', fc='none', lw=1)
    #     apSkyProp = dict(ec='b', fc='none', lw=1)
    #     winProp = dict(ec='g', fc='none', ls=':', lw=1.5)
    #
    #     # Apertures from model fits
    #     apStar = Ellipse((0, 0), 1, 1, **apStarProp)  # aperture
    #     apSkyin = Ellipse((0, 0), 1, 1, **apSkyProp)
    #     apSkyout = Ellipse((0, 0), 1, 1, **apSkyProp)
    #     rect = Rectangle((0, 0), 1.4, 1.4, **winProp)  # windows
    #     xfit = Line2D([0], [0], **xfitProp)  # fit position
    #
    #     # Circular aps
    #     xcom = Line2D([0], [0], **xcomProp)  # CoM markers
    #
    #     sx, sy = 3, 3.5 # TODO: from model?
    #     rs = r0, r1, r2 = 4, 7, 10
    #     a = 45
    #     proxies = (((apStar, apSkyin, apSkyout, xfit),
    #                 r'Auto aps. ($r_{\bigstar}=%g\sigma$, $r_{sky}=(%g\sigma, %g\sigma))$' % rs),
    #                (rect, 'Model windows'),
    #                (xcom, 'Centroid (CoM)'))
    #
    #     handler_map = {  # Line2D : HandlerDelegateLine2D(),
    #         rect: HandlerPatch(handleSquare),
    #         apStar: HandlerPatch(handleEllipse(sx * r0, sy * r0, a)),
    #         apSkyin: HandlerPatch(handleEllipse(sx * r1, sy * r1, a)),
    #         apSkyout: HandlerPatch(handleEllipse(sx * r2, sy * r2, a)),
    #         # apCir: HandlerPatch(handleCircle)
    #     }
    #
    #     leg1 = self.ax.legend(*zip(*proxies),  # proxies, labels,
    #                           #title='Markers',
    #                           loc=3, ncol=2,
    #                           labelspacing=2,
    #                           handletextpad=1,
    #                           # borderaxespad=-5,
    #                           borderpad=1,
    #                           framealpha=0.5,
    #                           prop=dict(weight='bold', size='medium'),
    #                           handler_map=handler_map,
    #                           bbox_to_anchor=(0, -1.35),
    #                           # bbox_transform=fig.transFigure,
    #                           )
    #     leg1.set_title('Markers', dict(weight='bold', size='large'))
    #     leg1.draggable()
    #
    #     fig = self.figure
    #     # fig.subplots_adjust(top=0.83)
    #     figsize = fig.get_size_inches() + [0, 1]
    #     fig.set_size_inches(figsize)
    #
    #     return leg1

    # def play(self):
    #     if self.playing:
    #         return
    #
    # def pause(self):
        'todo'

    # def





from grafico.interactive import ConnectionMixin, mpl_connect


class LegendGuiBase(ConnectionMixin):  # TODO: move to separate script....
    """
    Enables toggling marker / bar / cap visibility by selecting on the legend.
    """
    def __init__(self, figure, art, proxies, states=None, use_blit=False):
        """enable legend picking"""

        assert len(art) == len(proxies), 'Unequal number of artists and proxies'
        self.artists = art
        self.proxies = proxies
        if states is None:
            states = np.ones(len(art), bool)
        assert len(art) == len(states), 'Unequal number of artists and states'

        # initialize auto-connect
        ConnectionMixin.__init__(self, figure.canvas)

        self.use_blit = use_blit
        if use_blit and self.canvas.supports_blit:
            self.use_blit = True

        # create mapping between the legend artists (markers), and the
        # original (axes) artists

        self.to_orig = {}
        for handel, origart, stat in zip(proxies, self.artists, states):
            # if handel.get_picker() is None:
            #     logging.warning('Setting pick radius=10 for %s' % handel)
            self.to_orig[handel] = origart
            if not stat:
                self.toggle_vis(origart, handel)

    @mpl_connect('pick_event')
    def on_pick(self, event):
        """Pick event handler."""
        print('pick')
        prx = event.artist
        if prx in self.to_orig:
            art = self.to_orig[event.artist]
            self.toggle_vis(art, prx)
            print('toggled', art)

    def toggle_vis(self, art, proxy):
        """
        on the pick event, find the orig line corresponding to the
        legend proxy line, and toggle the visibility.
        """
        fig = self.canvas.figure
        if self.use_blit:
            art.set_animated = True
            proxy.set_animated = True
            background = fig.canvas.copy_from_bbox(fig.bbox)

        # Toggle vis of axes artists
        vis = not art.get_visible()
        art.set_visible(vis)

        # set alpha of legend artist
        proxy.set_alpha(1.0 if vis else 0.2)

        if self.use_blit:
            fig.canvas.restore_region(background)
            art.draw(fig.canvas.renderer)
            proxy.draw(fig.canvas.renderer)
            fig.canvas.blit(fig.bbox)
        else:
            fig.canvas.draw()