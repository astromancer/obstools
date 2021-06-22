

# std libs
import logging
from collections import namedtuple, defaultdict

# third-party libs
import numpy as np
from matplotlib.lines import Line2D
from matplotlib._contour import QuadContourGenerator
from matplotlib.widgets import RadioButtons, CheckButtons
from matplotlib.patches import Rectangle, Ellipse as _Ellipse
from matplotlib.collections import LineCollection, PatchCollection

# local libs
from recipes.lists import flatten
from recipes.dicts import AttrReadItem
from recipes.logging import LoggingMixin
from obstools.aps import ApertureCollection, SkyApertures
from scrawl.imagine import VideoDisplay, VideoDisplayA, VideoDisplayX






# TODO: better spacing between legend entries
# TODO: legend art being clipped


# def lm_extract_values_stderr(pars):
#     return np.transpose([(p.value, p.stderr) for p in pars.values()])


def edge_proximity(self, xy):
    xyprox = np.subtract(xy, self.center)
    x, y = xyprox.T
    theta = np.arctan2(y, x)  # angle between position and ellipse centre
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


class TrackerGui(VideoDisplayX):

    def __init__(self, data, tracker, **kws):
        VideoDisplayX.__init__(self, data, **kws)
        self.tracker = tracker

    def get_coords(self, i):
        return (self.tracker.shifts[i] + self.tracker.rcoo).T[::-1]


ApertureContainer = namedtuple('ApertureContainer', ('stars', 'sky'))


def update_aps(aps, coords, abc):
    aps.coords = coords
    aps.a, aps.b, aps.angles = abc

update_aps_stars = update_aps

def update_aps_sky(aps, coords, abc):
    aps.coords = coords
    a_sky_in, a_sky_out, b_sky_out = abc
    b_sky_in = a_sky_in * (b_sky_out / a_sky_out)
    aps.a = a_sky_in, a_sky_out
    aps.b = b_sky_in, b_sky_out


class ApertureVizGui0(VideoDisplayA):
    apPropsCommon = dict(fc='none', lw=1, picker=False)
    apPropsSky = dict(ec='c')
    # apCMap = cm.get_cmap('Reds)

    def __init__(self, data, coords, appars, groups_sizes, **kws):
        self.n_groups = appars.stars.shape[1]
        super().__init__(data, coords, **kws)
        self.appars = appars
        self._ix = np.split(np.arange(coords.shape[1]), np.cumsum(groups_sizes))

    def create_apertures(self, **props):
        apd = defaultdict(list)
        propList = self.apProps, self.apPropsSky
        kls = ApertureCollection, SkyApertures
        for i, w in enumerate(('stars', 'sky')):
            props = propList[i]
            props.setdefault('animated', self.use_blit)

            for j in range(self.n_groups):
                # color = next(self.)
                # props.update(ec=color)
                aps = kls[i](**props)
                apd[w].append(aps)

                # add apertures to axes.  will not display if coords not given
                aps.add_to_axes(self.ax)

        return AttrReadItem(apd)

    # def get_coords(self, i):
    #     return tracker.get_coords(i).T[::-1]

    def update_apertures(self, i, *args, **kws):
        coords = args[0]
        for i, w in enumerate(('stars', 'sky')):
            aps = self.aps[w]
            pars = self.appars[w][i]
            updater = eval(f'update_aps_{w}')
            for j in range(self.n_groups):
                updater(aps[j], coords[self._ix[j]], pars[j])

        return self.aps


class ApertureVizBase(VideoDisplayA):
    apPropsSky = dict(ec='b', fc='none', lw=1, picker=False)

    def __init__(self, data, coords, appars, skypars=None, **kws):
        VideoDisplayA.__init__(self, data, coords, **kws)

        self.appars = appars
        self.skypars = skypars

        self.aps = ApertureContainer(
                self.aps,
                SkyApertures(**self.apPropsSky)
        )

        self.aps.sky.add_to_axes(self.ax)

    def update_apertures(self, i, *args, **kws):
        coords, appars, skypars = args
        for aps, pars in zip(self.aps, (appars, skypars)):
            aps.coords = coords
            aps.a, aps.b, aps.angles = pars.T

        return self.aps

    def update(self, i, draw=True):
        # get all the artists that changed by calling parent update
        draw_list = VideoDisplay.update(self, i, False)
        #
        coo = self.get_coords(i)
        if coo is not None:
            self.marks.set_data(coo)
            draw_list.append(self.marks)

        appars = self.appars[i]
        skypars = None
        if self.skypars is not None:
            skypars = self.skypars[i]

        art = self.update_apertures(i, coo.T, appars, skypars)
        draw_list.append(art)

        return draw_list


class ApertureVizGui(VideoDisplayA):
    apPropsSky = dict(ec='b', fc='none', lw=1, picker=False)

    def __init__(self, data, tracker, appars, skypars=None, **kws):
        VideoDisplayA.__init__(self, data, None, **kws)

        self.tracker = tracker
        self.appars = appars
        self.skypars = skypars

        # create apertures
        r0 = np.zeros(tracker.nsegs)
        c0 = tracker.rcoo_xy
        θ = np.zeros(tracker.nsegs)
        #
        a = np.zeros(tracker.nsegs * 2)
        # b = np.zeros(tracker.nsegs * 2)
        θsky = np.zeros(tracker.nsegs * 2)
        self.aps = ApertureContainer(
                ApertureCollection(coords=c0, r=r0, angles=θ, **self.apProps),
                SkyApertures(coords=c0, r=a, angles=θsky, **self.apPropsSky)
        )

        # add ApertureCollection to axes
        for aps in self.aps:
            aps.add_to_axes(self.ax)

    def get_coords(self, i):
        return (self.tracker.shifts[i] + self.tracker.rcoo).T[::-1]

    def update_apertures(self, i, *args, **kws):

        coords, appars, skypars = args

        # update apertures
        self.aps.stars.coords = coords
        self.aps.sky.coords = coords  # duplicates coordinates automatically
        for g, lbls in enumerate(self.tracker.groups.values()):
            # NOTE: better to have one ApertureCollection per group to get
            #  drawing optimizations
            ix = lbls - 1
            a, b, theta = appars[g]
            self.aps.stars.a[ix] = a
            self.aps.stars.b[ix] = b
            self.aps.stars.angles[ix] = theta

            a_sky_in, a_sky_out, b_sky_out = skypars[g]
            b_sky_in = a_sky_in * (b_sky_out / a_sky_out)
            # print('a_sky_in, a_sky_out', a_sky_in, a_sky_out)
            # print('b_sky_in, b_sky_out', b_sky_in, b_sky_out)
            self.aps.sky.a[2 * ix] = a_sky_in
            self.aps.sky.a[2 * ix + 1] = a_sky_out
            self.aps.sky.b[2 * ix] = b_sky_in
            self.aps.sky.b[2 * ix + 1] = b_sky_out
            # self.aps.sky.a[ix] = a_sky_in#, a_sky_out
            # self.aps.sky.b[ix] = b_sky_in#, b_sky_out
            self.aps.sky.angles[ix] = theta

        return self.aps

    def update(self, i, draw=True):
        # get all the artists that changed by calling parent update
        i = int(i)
        draw_list = VideoDisplay.update(self, i, False)
        #
        coo = self.get_coords(i)
        if coo is not None:
            self.marks.set_data(coo)
            draw_list.append(self.marks)

        appars = self.appars[i]
        skypars = None
        if self.skypars is not None:
            skypars = self.skypars[i]

        art = self.update_apertures(i, coo.T, appars, skypars)
        draw_list.append(art)

        return draw_list


class FrameProcessorGUI(VideoDisplay, LoggingMixin):
    # TODO split off basic gui without legend
    # TODO: Inherit from mpl Animation??
    # TODO: grey out buttons (calibrated / background) if option not avail

    _xpcom = dict(marker='x', ms=6, ls='none', picker=5)
    xcomProp = dict(mec='r', **_xpcom)
    xpeakProp = dict(mec='k', **_xpcom)
    xfitProp = dict(mec='g', **_xpcom)

    apStarProp = dict(ec='g', ls='-', fc='none', lw=1)
    apSkyProp = dict(ec='c', fc='none', lw=1)
    # winProp = dict(ec='g', fc='none', ls=':', lw=1.5, picker=5)

    # TODO: mpl collections - make the shortcuts work!
    winProp = dict(edgecolor='g', facecolor='none',
                   linestyle=':', linewidth=1.5, picker=5)

    def __init__(self, data, coords, tracker, mdlr, apdata, residu=None, **kws):

        # filename = cube.filename
        VideoDisplay.__init__(self, data, connect=False, **kws)

        # self.proc = proc
        self.coords = coords
        self.tracker = tracker
        self.modeller = mdlr
        self.apData = apdata
        self.residu = residu

        # create position markers (initially empty)
        marks = []
        for prp in (self.xcomProp, self.xpeakProp, self.xfitProp):
            mrk, = self.ax.plot([], **prp)
            marks.append(mrk)
        Markers = namedtuple('Markers', ('com', 'peak', 'mdl'))
        self.markers = Markers(*marks)

        # create apertures
        Aps = namedtuple('Apertures', ('stars', 'sky'))
        r0 = np.zeros(tracker.nsegs)
        c0 = tracker.rcoo_xy
        θ = np.zeros(tracker.nsegs)
        self.aps = Aps(
                ApertureCollection(r=r0, coords=c0, angles=θ,
                                   **self.apStarProp),
                SkyApertures(r=r0, coords=c0, angles=θ, **self.apSkyProp))

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
        # TODO: generic on_move method for the sliders that adds to both

        # add gui buttons
        art = self.add_buttons()

        # update everything to display the first frame
        self.update(0)

        # bg grid #HACK
        # self._bggrid = self.modeller.bg.grid_from_data(self.data[0])

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
        # logging.debug('VideoDisplayBase: Guessed figure size: (%.1f, %.1f)', *size)
        return size

    def add_buttons(self):
        """"""
        ttlprp = dict(fontweight='bold', size='large')
        self.lax = lax = self.figure.add_subplot(self._gs[1, :2], frameon=False,
                                                 xticks=[], yticks=[])
        lax.set_title('Markers', **ttlprp)
        proxies = self.add_legend()
        art = self.aps + (self.windows,) + self.markers + (self.outlines,)
        fitvis = bool(self.modeller.nmodels)
        visible = (True, True, fitvis, True, False, fitvis, False)
        self.lgb = LegendGuiBase(self.figure, art, proxies, visible)

        rax1 = self.figure.add_subplot(self._gs[1, 2], aspect='equal',
                                       frameon=False)
        self.image_selector = RadioButtons(  # FIXME: should be check buttons
                rax1, ('Raw', 'Calibrated', 'BG subtracted'))
        rax1.set_title('Image Data', **ttlprp)
        self.image_selector.on_clicked(self.image_button_action)

        rax2 = self.figure.add_subplot(self._gs[1, 3], aspect='equal',
                                       frameon=False)
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
        self.imagePlot.set_data(image)  # FIXME: update histogram!
        self.figure.canvas.draw()

    # def mask_button_action(self, label):
    #     print('button m')
    #     image = self.get_image_data(self.frame)
    #     self.imagePlot.set_data(image)
    #     self.figure.canvas.draw()

    def get_image_data(self, i):

        s = self.image_selector.value_selected
        # get image data to display

        if s == 'Raw':
            image = self.data[i]
        elif s == 'BG subtracted':
            image = self.residu[i]
        elif s == 'Calibrated':
            logging.warning('need calibration image')
            image = self.data[i]

        mask = self.get_image_mask()

        return np.ma.masked_array(image, mask=mask)

    def get_image_mask(self):
        # apply masks
        trk = self.tracker
        stat = self.mask_selector.get_status()
        mask = False
        if stat[0]:
            mask |= self.tracker.bad_pixel_mask
        if stat[1]:
            mask |= trk.segm.to_boolean(trk.use_labels)
        return mask

    def update(self, i, draw=True):
        self.logger.debug('set_frame: %s', i)

        # get image
        i, image = VideoDisplay.update(self, i)

        # proc = self.proc
        trk = self.tracker

        coords = self.coords[i]
        centroids = (coords + trk.rvec)[:, ::-1]

        # update markers
        mprx = self.proxies.markers
        if mprx.com.get_visible():
            self.markers.com.set_data(centroids.T)

        if mprx.peak.get_visible():
            peaks = trk.segm.imax(image, trk.use_labels)
            self.markers.peak.set_data(peaks[:, ::-1].T)

        if self.modeller.nmodels and mprx.mdl.get_visible():
            # mdl = mdlr.models[0]
            cxf = self.modeller.data[0].params[i, :, :2].T
            self.markers.mdl.set_data(cxf)

        # update apertures
        self.aps.stars.coords = centroids
        self.aps.sky.coords = centroids  # duplicates coordinates automatically
        for g, lbls in enumerate(self.tracker.groups):
            # NOTE: better to have one ApertureCollection per group to get
            # drawing optimiztions
            ix = lbls - 1
            a, b, theta = self.apData.stars[i, g]
            self.aps.stars.a[ix] = a
            self.aps.stars.b[ix] = b
            self.aps.stars.angles[ix] = theta

        a_sky_in, a_sky_out, b_sky_out = self.apData.sky[i, g]
        b_sky_in = a_sky_in * (b_sky_out / a_sky_out)
        self.aps.sky.a = a_sky_in, a_sky_out
        self.aps.sky.b = b_sky_in, b_sky_out
        self.aps.sky.angles = theta

        # sxy = self.apData.sigmaXY[i]
        # angle = self.apData.theta[i]
        # self.aps.stars.a, self.aps.stars.b = sxy * self.apData.r
        # self.aps.stars.angles = angle

        # inner and outer semi-major and -minor for sky
        # a, b = self.apData.rsky * sxy[None].T
        # self.aps.sky.coords = centroids
        # self.aps.sky.angles = angle
        # self.aps.sky.a = a
        # self.aps.sky.b = b
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

        # TODO: better to have titles inside axes so they don't begin
        # overlapping on figure resize
        # Toggle label visibility with art

        tr = None  # NOTE: having the artist transform with the axes would
        # probably be better, but hard to get ellipse to draw correctly then
        common = dict(transform=tr)

        # "legend" texts properties
        txtoff = (40, 0)
        txtprp = dict(fontdict=dict(weight='bold', size='medium'),
                      transform=None, va='center', clip_on=True)

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
        # apSkyin = Ellipse(epos, *r[1], a, **self.apSkyProp, **common, picker=ellipse_picker)
        # apSkyout = Ellipse(epos, *r[2], a, **self.apSkyProp, **common, picker=ellipse_picker)

        w = 7
        rxy = (21, 25)
        skyprx = Annulus(epos, rxy, w, a, **self.apSkyProp, **common,
                         picker=annulus_picker)

        # rs = 4, 7, 10
        # txt = 'Auto aps.' \n' + r'$(r_{\bigstar}=%g\sigma$,' \
        # r' $r_{sky}=(%g\sigma, %g\sigma))$' % rs
        txt = 'Auto aps.'
        text = lax.text(*(epos + txtoff), txt, **txtprp)

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

        pos = rpos + (rect.get_width() / 2, rect.get_height() / 2)
        lax.text(*(pos + txtoff), 'Model windows', **txtprp)

        # skyprx = SkyAps(apSkyin, apSkyout)
        # TODO: mpl.Container?
        aprx = type(self.aps)(apStar, skyprx)

        # CoM markers
        xpos = lax.transAxes.transform([0.6, 0.85])
        xcom = Line2D(*xpos[None].T, **self.xcomProp, **common)
        lax.text(*(xpos + txtoff), 'Centroids', **txtprp)
        # peak position markers
        xpkpos = lax.transAxes.transform([0.6, 0.5])
        xpk = Line2D(*xpkpos[None].T, **self.xpeakProp, **common)
        lax.text(*(xpkpos + txtoff), 'Peaks', **txtprp)
        # fit position
        xfitpos = lax.transAxes.transform([0.6, 0.15])
        xfit = Line2D(*xfitpos[None].T, **self.xfitProp, **common)
        lax.text(*(xfitpos + txtoff), 'Fit pos.', **txtprp)
        # collect in namedtuple
        mprx = type(self.markers)(xcom, xpk, xfit)

        # tracking region contours proxy
        r = 10
        b = np.square(np.mgrid[:r, :r] - r / 2).sum(0) < (r / 3) ** 2
        c = binary_contours(b)
        # c = np.subtract(c, r/2 + 0.5) + xpos
        # xpos =
        # c = np.subtract(c, r / 2 + 0.5) * 4 + np.squeeze(xcom.get_data())# / 2
        # c = np.add(c, xpos)
        c = np.subtract(c, r / 2 + 0.5) * 4.5 + (269, 96)

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

        return artists

    def show_windows(self):
        """
        Create PatchCollection for Rectangular windows

        Returns
        -------

        """
        # model windows
        trk = self.tracker
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
        Create LineCollection that delineates the tracking regions

        Parameters
        ----------
        kws

        Returns
        -------

        """
        # TODO: method of TrackerGui class

        trk = self.tracker
        segm = self.tracker.segm
        data = segm.data
        outlines = []

        # trk.seg.thumbnails(labels=trk.use_labels)
        ix = trk.segm.indices(trk.use_labels)
        slices = np.take(segm.slices, ix, 0)

        for s in slices:
            sy, sx = s
            e = np.array([[sx.start - 1, sx.stop + 1],
                            [sy.start - 1, sy.stop + 1]])
            e = np.clip(e, 0, np.inf).astype(int)
            # print(e)
            im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]

            contours = binary_contours(im)
            for c in contours:
                outlines.append(c + e[:, 0] - 0.5)


        col = LineCollection(outlines, **kws)
        self.ax.add_collection(col)

        return col

    def connect(self):

        VideoDisplay.connect(self)
        self.lgb.connect()

    def add_legend(self):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle, Ellipse, Circle
        from matplotlib.legend_handler import HandlerPatch, HandlerLine2D

        def handleSquare(legend, orig_handle, xdescent, ydescent, width, height,
                         fontsize):
            xy = (xdescent, ydescent - width / 3)
            return Rectangle(xy, width, width)

        def handleEllipse(w, h, angle):
            def _handler(legend, orig_handle, xdescent, ydescent, width, height,
                         fontsize):
                xy = (xdescent + width / 2, ydescent + height / 2)
                return Ellipse(xy, w, h, angle=angle, lw=1)

            return _handler

        # def handleCircle(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        #     w = width / 2
        #     return Circle((xdescent + width / 2, ydescent + width / 8), width / 3.5, lw=1)

        # Apertures from model fits
        apStar = Ellipse((0, 0), 1, 1, **apStarProp)  # aperture
        apSkyin = Ellipse((0, 0), 1, 1, **apSkyProp)
        apSkyout = Ellipse((0, 0), 1, 1, **apSkyProp)
        rect = Rectangle((0, 0), 1.4, 1.4, **winProp)  # windows
        xfit = Line2D([0], [0], **xfitProp)  # fit position

        # Circular aps
        xcom = Line2D([0], [0], **xcomProp)  # CoM markers

        sx, sy = 3, 3.5  # TODO: from model?
        rs = r0, r1, r2 = 4, 7, 10
        a = 45
        proxies = (((apStar, apSkyin, apSkyout, xfit),
                    r'Auto aps. ($r_{\bigstar}=%g\sigma$, $r_{sky}=(%g\sigma, %g\sigma))$' % rs),
                   (rect, 'Model windows'),
                   (xcom, 'Centroid (CoM)'))

        handler_map = {  # Line2D : HandlerDelegateLine2D(),
            rect: HandlerPatch(handleSquare),
            apStar: HandlerPatch(handleEllipse(sx * r0, sy * r0, a)),
            apSkyin: HandlerPatch(handleEllipse(sx * r1, sy * r1, a)),
            apSkyout: HandlerPatch(handleEllipse(sx * r2, sy * r2, a)),
            # apCir: HandlerPatch(handleCircle)
        }

        leg1 = self.ax.legend(*zip(*proxies),  # proxies, labels,
                              # title='Markers',
                              loc=3, ncol=2,
                              labelspacing=2,
                              handletextpad=1,
                              # borderaxespad=-5,
                              borderpad=1,
                              framealpha=0.5,
                              prop=dict(weight='bold', size='medium'),
                              handler_map=handler_map,
                              bbox_to_anchor=(0, -1.35),
                              # bbox_transform=fig.transFigure,
                              )
        leg1.set_title('Markers', dict(weight='bold', size='large'))
        leg1.draggable()

        fig = self.figure
        # fig.subplots_adjust(top=0.83)
        figsize = fig.get_size_inches() + [0, 1]
        fig.set_size_inches(figsize)

        return leg1

    # def play(self):
    #     if self.playing:
    #         return
    #
    # def pause(self):
    # 'todo'

    # def




class LegendGuiBase(ConnectionMixin):
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

# class FrameProcessor(LoggingMixin):
#     # @classmethod
#     # def from_fits(self, filename, **options):
#     #     ''
#
#     def __init__(self, datacube, tracker=None, modeller=None, apmaker=None,
#                  bad_pixel_mask=None):
#
#         self.data = datacube
#         self.tracker = tracker
#         self.modeller = modeller
#         self.maker = apmaker
#         self.bad_pixel_mask = bad_pixel_mask
#
#     def __call__(self, i):
#
#         data = self.data[i]
#         track = self.tracker
#         mdlr = self.modeller
#         mkr = self.maker
#         apD = self.apData
#
#         # prep background image
#         imbg = track.background(data)
#
#         # fit and subtract background
#         residu, p_bg = mdlr.background_subtract(data, imbg.mask)
#         dat = mdlr.data[mdlr.bg]
#         p, pstd = lm_extract_values_stderr(p_bg)
#         # try:
#         dat.params[i] = p
#         dat.params_std[i] = pstd
#         # except Exception as err:
#         #     print(p, pstd)
#         #     print(dat.params[i]._shared)
#         #     print(dat.params_std[i]._shared)
#
#         # track stars
#         com = track(residu)
#         # save coordinates in shared data array.
#         self.coords[i] = com[track.ir]
#
#         # PSF photometry
#         # Calculate the standard deviation of the data distribution of each pixel
#         data_std = np.ones_like(data)  # FIXME:
#         # fit models
#         results = mdlr.fit(residu, data_std, self.bad_pixel_mask, )
#         # save params
#         mdlr.save_params(i, results)
#         # model selection for each star
#         best_models, params, pstd = self.model_selection(i, results)
#
#         # PSF-guided aperture photometry
#         # create scaled apertures from models
#         appars = mkr.combine_results(best_models, params, axis=0)  # coo_fit, sigma_xy, theta
#         aps = mkr.create_apertures(com, appars)
#         apsky = mkr.create_apertures(com, appars, sky=True)
#
#         # save appars
#         apD.sigma_xy[i], apD.theta[i] = appars[1:]
#
#         # do background subtracted aperture photometry
#         flx, flxBG = self.aperture_photometry(residu, aps, apsky)
#         apD.flux[i], apD.bg[i] = flx, flxBG
#
#         # save coordinates in shared data array.
#         # if
#         # self.coords[i] = coo_fit
#         # only overwrites coordinates if mdlr.tracker is None
#
#     def init_mem(self, n=None):
#         """
#
#         Parameters
#         ----------
#         n : number of frames (mostly for testing purposes to avoid large memory allocation)
#
#         Returns
#         -------
#
#         """
#         # global apData
#
#         n = n or len(self.data)
#         nstars = len(self.tracker.use_labels)
#         naps = np.size(self.maker.r)
#         # nfit = len(self.modeller.use_labels)
#
#         # reference star coordinates
#         self.coords = SyncedArray(shape=(n, 2))
#
#         # NOTE: You should check how efficient these memory structures are.
#         # We might be spending a lot of our time synching access??
#
#         # HACK: Initialize shared memory with nans...
#         SyncedArray.__new__.__defaults__ = (None, None, np.nan, ctypes.c_double)  # lazy HACK
#
#         apData = self.apData = AttrDict()
#         apData.bg = SyncedArray(shape=(n, nstars))
#         apData.flux = SyncedArray(shape=(n, nstars, naps))
#
#         apData.sigma_xy = SyncedArray(shape=(n, 2))  # TODO: for nstars (optionally) ???
#         apData.rsky = SyncedArray(shape=(n, 2))
#         apData.theta = SyncedArray(shape=(n,))
#         # cog_data = np.empty((n, nstars, 2, window*window))
#
#         self.modeller.init_mem(n)
#
#     def model_selection(self, i, results):
#         """
#         Do model selection (per star) based on goodness of fit metric(s)
#         """
#
#         pars, paru, gof = results
#         best_models, params, pstd = [], [], []
#         # loop over stars
#         for j, g in enumerate(gof.swapaxes(0, 1)):  # zip(pars, paru, gof)
#             ix, mdl, msg = self.modeller.model_selection(g)
#             if msg:
#                 self.logger.warning('%s (Frame %i, Star %i)', (msg, i, j))
#
#             if ix is not None:
#                 self.logger.info('Best model: %s (Frame %i, Star %i)' % (mdl, i, j))
#
#             # TODO: if best_model is self.db.bg:
#             #     "logging.warning('Best model is BG')"
#             #     "flux is upper limit?"
#
#             # yield mdl, p
#             best_models.append(mdl)
#             params.append(pars[ix][j])
#             pstd.append(paru[ix][j])
#         return best_models, params, pstd
#
#     def aperture_photometry(self, data, aps, skyaps):
#
#         method = 'exact'
#
#         # a quantity is needed for photutils
#         udata = u.Quantity(data, copy=False)
#
#         m3d = self.tracker.seg.to_boolean_3d()
#         masks = m3d.any(0, keepdims=True) & ~m3d
#         masks |= self.bad_pixel_mask
#
#         Flux = np.empty(np.shape(aps))
#         if Flux.ndim == 1:
#             Flux = Flux[:, None]
#         FluxBG = np.empty(np.shape(skyaps))
#         for j, (ap, ann) in enumerate(zip(aps, skyaps)):
#             mask = masks[j]
#
#             # sky
#             flxBG, flxBGu = ann.do_photometry(udata,
#                                               # error,
#                                               mask=mask,
#                                               # effective_gain,#  must have same shape as data
#                                               # TODO: ERROR ESTIMATE
#                                               method=method)  # method='subpixel', subpixel=5)
#             m = ap.to_boolean(method)[0]
#             area = (m.data * m.cutout(~mask)).sum()
#             fluxBGpp = flxBG / area  # Background Flux per pixel
#             flxBGppu = flxBGu / area
#             FluxBG[j] = fluxBGpp
#
#             # multi apertures ??
#             for k, app in enumerate(np.atleast_1d(ap)):
#                 flux, flux_err = app.do_photometry(udata,
#                                                    mask=mask,
#                                                    # error, #TODO: ERROR ESTIMATE
#                                                    # effective_gain,#  must have same shape as data
#                                                    method=method)
#                 # get the area of the aperture excluding masked pixels
#                 m = ap.to_boolean(method)[0]
#                 area = (m.data * m.cutout(~mask)).sum()
#
#                 Flux[j, k] = flux - (fluxBGpp * area)
#
#         return Flux, FluxBG
#
#     def save_params(self, i, coo):
#         if self.tracker is not None:
#             self.coords[i] = coo
#             # self.sigma[i] =
#
#     def check_image_drift(self, nframes, snr=5, npixels=7):
#         """Estimate the maximal positional shift for stars"""
#         step = len(self) // nframes  # take `nframes` frames evenly spaced across data set
#         maxImage = self[::step].max(0)  #
#
#         threshold = detect_threshold(maxImage, snr)  # detection at snr of 5
#         segImage = detect_sources(maxImage, threshold, npixels)
#         mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
#                           for (xs, ys) in segImage.slices], 0)
#
#         # TODO: check for cosmic rays inside sky apertures!
#
#         return mxshift, maxImage, segImage


# from scrawl.imagine import FitsCubeDisplay


# class FrameDisplay(FitsCubeDisplay):
#     # TODO: blit
#     # TODO: let the home button restore the original config
#
#     # TODO: enable scroll through - ie inherit from VideoDisplayA
#     #     - middle mouse to switch between prePlot and current frame
#     #     - toggle legend elements
#
#     # TODO: make sure annotation appear on image area or on top of other stars
#
#     def __init__(self, filename, *args, **kwargs):
#         # self.foundCoords = found # TODO: let model carry centroid coords?
#         FitsCubeDisplay.__init__(self, filename, *args, **kwargs)
#
#         # FIXME:  this is not always appropriate NOTE: won't have to do this if you use wcs
#         # self.ax.invert_xaxis()    # so that it matches sky orientation
#
#     def add_aperture_from_model(self, model, params, r_scale_sigma, rsky_sigma,
#                                 **kws):
#         # apertures from elliptical fit
#
#         apColour = 'g'
#         aps, ap_data = from_params(model, params, r_scale_sigma,
#                                    ec=apColour, lw=1, ls='--', **kws)
#         aps.axadd(self.ax)
#         # aps.annotate(color=apColour, size='small')
#
#         # apertures based on finder + scaled by fit
#         # from obstools.aps import ApertureCollection
#         apColMdl = 'c'
#         sigma_xy = ap_data[:, 2:4]
#         r = np.nanmean(sigma_xy) * r_scale_sigma * np.ones(len(foundCoords))
#         aps2 = ApertureCollection(coords=foundCoords[:, ::-1], radii=r,
#                                   ec=apColMdl, ls='--', lw=1)
#         aps2.axadd(self.ax)
#         # aps2.annotate(color=apColMdl, size='small')
#
#         # skyaps
#         rsky = np.multiply(np.nanmean(sigma_xy), rsky_sigma)
#         # rsky = np.nanmean(sigma_xy) * rsky_sigma * np.ones_like(foundCoords)
#         coosky = [foundCoords[:, ::-1]] * 2
#         coosky = np.vstack(list(zip(*coosky)))
#         apssky = ApertureCollection(coords=coosky, radii=rsky,
#                                     ec='b', ls='-', lw=1)
#         apssky.axadd(self.ax)
#
#         # Mark aperture centers
#         self.ax.plot(*params[:, 1::-1].T, 'x', color=apColour)
#
#     def mark_found(self, xy, style='rx'):
#         # Mark coordinates from finder algorithm
#         return self.ax.plot(*xy, style)
#
#     def add_windows(self, xy, window, sdist=None, enumerate=True):
#         from matplotlib.patches import Rectangle
#         from matplotlib.collections import PatchCollection
#
#         n = len(xy)
#         if sdist is None:
#             from scipy.spatial.distance import cdist
#             sdist = cdist(xy, xy)
#
#         # since the distance matrix is symmetric, ignore lower half
#         try:
#             sdist[np.tril_indices(n)] = np.inf
#         except:
#             print('BROKEN with add_windows ' * 100)
#             embed()
#             raise
#         ix = np.where(sdist < window / 2)
#         overlapped = np.unique(ix)
#
#         # corners
#         llc = xy - window / 2
#         urc = llc + window
#         # patches
#         patches = [Rectangle(coo, window, window) for coo in llc]
#         # colours
#         c = np.array(['g'] * n)
#         c[overlapped] = 'r'
#         rectangles = PatchCollection(patches,
#                                      edgecolor=c, facecolor='none',
#                                      lw=1, linestyle=':')
#         self.ax.add_collection(rectangles)
#
#         # annotation
#         text = []
#         if enumerate:
#             # names = np.arange(n).astype(str)
#             for i in range(n):
#                 txt = self.ax.text(*urc[i], str(i), color=c[i])
#                 text.append(txt)
#                 # ax.annotate(str(i), urc[i], (0,0), color=c[i],
#                 #                  transform=ax.transData)
#
#             # print('enum_'*100)
#             # embed()
#         return rectangles, text
#
#     def add_detection_outlines(self, outlines):
#         from matplotlib.colors import to_rgba
#         overlay = np.empty(self.data.shape + (4,))
#         overlay[...] = to_rgba('0.8', 0)
#         overlay[..., -1][~outlines.mask] = 1
#         # ax.hold(True) # triggers MatplotlibDeprecationWarning
#         self.ax.imshow(overlay)
#
#     def add_vectors(self, vectors, ref=None):
#         if ref is None:
#             'cannot plot vectors without reference star'
#         Y, X = self.foundCoords[ref]
#         V, U = vectors.T
#         self.ax.quiver(X, Y, U, V, color='r', scale_units='xy', scale=1,
#                        alpha=0.6)
#
#     def add_legend(self):
#         from matplotlib.lines import Line2D
#         from matplotlib.patches import Rectangle, Ellipse, Circle
#         from matplotlib.legend_handler import HandlerPatch, HandlerLine2D
#
#         def handleSquare(legend, orig_handle, xdescent, ydescent, width, height,
#                          fontsize):
#             w = width  # / 1.75
#             xy = (xdescent, ydescent - width / 3)
#             return Rectangle(xy, w, w)
#
#         def handleEllipse(legend, orig_handle, xdescent, ydescent, width,
#                           height, fontsize):
#             return Ellipse((xdescent + width / 2, ydescent + height / 2),
#                            width / 1.25, height * 1.25, angle=45, lw=1)
#
#         def handleCircle(legend, orig_handle, xdescent, ydescent, width, height,
#                          fontsize):
#             w = width / 2
#             return Circle((xdescent + width / 2, ydescent + width / 8),
#                           width / 3.5, lw=1)
#
#         apColCir = 'c'
#         apColEll = 'g'
#         apColSky = 'b'
#
#         # Sky annulus
#         pnts = [0], [0]
#         kws = dict(c=apColSky, marker='o', ls='None', mfc='None')
#         annulus = (Line2D(*pnts, ms=6, **kws),
#                    Line2D(*pnts, ms=14, **kws))
#
#         # Fitting
#         apEll = Ellipse((0, 0), 0.15, 0.15, ls='--', ec=apColEll, fc='none',
#                         lw=1)  # aperture
#         rect = Rectangle((0, 0), 1, 1, ec=apColEll, fc='none', ls=':')  # window
#         xfit = Line2D(*pnts, mec=apColEll, marker='x', ms=3,
#                       ls='none')  # fit position
#
#         # Circular aps
#         apCir = Circle((0, 0), 1, ls='--', ec=apColCir, fc='none', lw=1)
#         xcom = Line2D(*pnts, mec=apColCir, marker='x', ms=6,
#                       ls='none')  # CoM markers
#
#         proxies = (((apEll, xfit, rect), 'Elliptical aps. ($%g\sigma$)' % 3),
#                    (apCir, 'Circular aps. ($%g\sigma$)' % 3),
#                    (annulus, 'Sky annulus'),
#                    # (rect, 'Fitting window'),
#                    # ( xfit, 'Fit position'),
#                    (xcom, 'Centre of Mass'))
#
#         # proxies = [apfit, annulus, rect, xfit, xcom] # markers in nested tuples are plotted over each other in the legend YAY!
#         # labels = [, 'Sky annulus', 'Fitting window', 'Fit position', 'Centre of Mass']
#
#         handler_map = {  # Line2D : HandlerDelegateLine2D(),
#             rect: HandlerPatch(handleSquare),
#             apEll: HandlerPatch(handleEllipse),
#             apCir: HandlerPatch(handleCircle)
#         }
#
#         leg1 = self.ax.legend(*zip(*proxies),  # proxies, labels,
#                               framealpha=0.5, ncol=2, loc=3,
#                               handler_map=handler_map,
#                               bbox_to_anchor=(0., 1.02, 1., .102),
#                               # bbox_to_anchor=(0, 0.5),
#                               # bbox_transform=fig.transFigure,
#                               )
#         leg1.draggable()
#
#         fig = self.figure
#         fig.subplots_adjust(top=0.83)
#         figsize = fig.get_size_inches() + [0, 1]
#         fig.set_size_inches(figsize)
#
#
# # from scrawl.imagine import FitsCubeDisplay
#
#
# def displayCube(fitsfile, coords, rvec=None):
#     # TODO: colour the specific stars used to track differently
#
#     def updater(aps, i):
#         cxx = coords[i, None]
#         if rvec is not None:
#             cxx = cxx + rvec  # relative positions of all other stars
#         else:
#             cxx = cxx[None]  # make 2d else aps doesn't work
#         # print('setting:', aps, i, cxx)
#         aps.coords = cxx[:, ::-1]
#
#     im = FitsCubeDisplay(fitsfile, {}, updater,
#                          autoscale_figure=False, sidebar=False)
#     im.ax.invert_xaxis()  # so that it matches sky orientation
#     return im
