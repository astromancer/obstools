"""
Tools for visualising object tracks across the night sky
"""


# std libs
from recipes.logging import LoggingMixin
import time
import inspect
import logging
import threading
import itertools as itt
from pathlib import Path
from functools import partial
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict

# third-party libs
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates.name_resolve import NameResolveError
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, get_sun,
                                 get_moon, jparser)
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date
from matplotlib.transforms import (Transform, IdentityTransform, Affine2D,
                                   blended_transform_factory as btf)
from addict.addict import Dict
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from recipes.decor import memoize
from graphing.ticks import DegreeFormatter, TransFormatter
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local libs
from motley import profiling
from recipes.containers.lists import sorter
from ..utils import get_coordinates, get_site
import more_itertools as mit


# TODO: enable different projections, like Mercator etc...
# FIXME: crashes randomly due to threading problems:
#   QWidget::repaint: Recursive repaint detected
#   QBackingStore::endPaint() called with active painter on backingstore paint device


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# def OOOOh(t):
#   """hours since midnight"""
#   return (t - midnight).sec / 3600


def local_time_str(t, tz=2 * u.hour, precision='m'):
    scales = (24, 60, 60)
    ix = 'hms'.index(precision)
    t = (t + tz).to_datetime().timetuple()
    tt = np.round(t[3:4 + ix])
    inc = tt // scales[:ix + 1]
    tt = (tt + inc) % scales[:ix + 1]
    fmt = ':'.join(('{:02,d}',) * (ix + 1))
    return fmt.format(*tt)

    # timeTxt = (t+tz).iso.split()[1].split('.')[0]   # up to seconds
    # timeTxt = ':'.join(timeTxt.split(':')[:ix+1])   # up to requested precision
    # return (t+tz).iso.split()[1].split('.')[0]


def vertical_txt(ax, s, t, y=1, precision='m', **kw):
    va = 'top'
    if y == 'top':
        y, va = 1, y
    if y == 'bottom':
        y, va = 0.01, y

    s = '%s %s SAST' % (s, local_time_str(t, precision=precision))
    txt = ax.text(t.plot_date, y, s,
                  rotation=90, ha='right', va=va,
                  transform=btf(ax.transData, ax.transAxes),
                  clip_on=True,
                  **kw)
    return txt


def nearest_midnight_date(switchHour=9):
    """
    default behaviour of this function changes depending on the time of day
    when called:
    if calling during early morning hours (presumably at telescope):
        time returned is current date 00:00:00
    if calling during afternoon hours:
        time returned is midnight of the next day
    """
    now = datetime.now()  # current local time
    day_inc = int(now.hour > switchHour)  # int((now.hour - 12) > 12)
    midnight = datetime(now.year, now.month, now.day, 0, 0, 0)
    return midnight + timedelta(day_inc)


def nearest_midnight_time():
    """Return time of nearest midnight utc"""
    return Time(nearest_midnight_date())


def sidereal_transform(date, longitude):
    """
    Initialize matplotlib transform for local time - sidereal time conversion
    """

    midnight = Time(date)  # midnight UTC

    midSid = midnight.sidereal_time('mean', longitude)
    # offset from local time
    offset = midSid.hour / 24
    # used to convert to origin of plot_date coordinates
    p0 = midnight.plot_date
    xy0 = (-p0, 0)
    xy1 = (p0 + offset, 0)
    # A mean sidereal day is 23 hours, 56 minutes, 4.0916 seconds
    # (23.9344699 hours or 0.99726958 mean solar days)
    scale = 366.25 / 365.25
    return Affine2D().translate(*xy0).scale(scale).translate(*xy1).inverted()


def short_name(name):
    if jparser.search(name):
        return jparser.shorten(name)
    return name


def set_visible(artists, state=True):
    for art in mit.collapse(artists):
        art.set_visible(state)

# ******************************************************************************


class Sun(object):
    """
    Object that encapsulates the visibility of the sun for the given frames
    """

    def __init__(self, frames, midnight):

        t = frames.obstime
        self.coords = get_sun(t).transform_to(frames)  # WARNING: slow!!!!
        # get dawn / dusk times
        self.dusk, self.dawn = self.get_rise_set(t, midnight)
        self.set, self.rise = self.dusk['sunset'], self.dawn['sunrise']

    def get_rise_set(self, t, midnight):
        """calculate dawn / dusk / twilight times"""

        # We interpolate the calculated sun positions to get dusk/dawn times.  Should still be accurate to ~1s
        h = (t - midnight).to('h').value  # hours since midnight ut
        ip = interp1d(h, self.coords.alt.degree)

        def twilight(h, angle):
            """civil / nautical / astronomical twilight solver"""
            return ip(h) + angle

        angles = np.arange(0, 4) * 6.
        solver = partial(brentq, twilight)
        hdusk = np.vectorize(solver)(-11.99, 0, angles)
        hdawn = np.vectorize(solver)(0, 11.99, angles)

        dusk = midnight + hdusk * u.h
        dawn = midnight + hdawn * u.h

        up = OrderedDict(), OrderedDict()
        which = ['sun', 'civil', 'nautical', 'astronomical']
        when = [['set', 'rise'], ['dusk', 'dawn']]
        for i, times in enumerate(zip(dusk, dawn)):
            j = bool(i)
            for k, t in enumerate(times):
                words = (' ' * j).join((which[i], when[j][k]))
                up[k][words] = t

        return up


class Moon(object):
    """
    Object that encapsulates the visibility of the moon for the given frames
    """

    def __init__(self, frames, midnight):
        # get moon rise/set times, phase, illumination etc...
        t = frames.obstime
        self.coords = get_moon(t).transform_to(frames)  # WARNING: slow!!!!
        self.up = self.get_rise_set(t, midnight)
        self.phase, self.illumination = self.get_phase(
            midnight, frames.location)

    def get_rise_set(self, t, midnight):
        """get moon rising and setting times"""

        h = (t - midnight).to('h').value  # hours since midnight ut
        malt = self.coords.alt.degree
        # interpolator
        ip = interp1d(h, malt)

        # find horizon crossing interval (# index of sign changes)
        smalt = np.sign(malt)
        wsc = np.where(abs(smalt - np.roll(smalt, -1))[:-1] == 2)[0]
        ix_ = np.c_[wsc - 1, wsc + 1]  # check for sign changes: malt[ix_]
        up = {}

        for ix in ix_:
            crossint = h[ix]  # hour interval when moon crosses horizon
            rise_or_set = np.subtract(*malt[ix]) > 1
            s = 'moon{}'.format(['rise', 'set'][rise_or_set])
            hcross = brentq(ip, *crossint)
            up[s] = midnight + hcross * u.h

        return up

    def get_phase(self, t, location):
        """calculate moon phase and illumination at local midnight"""

        altaz = AltAz(obstime=t, location=location)
        moon = get_moon(t).transform_to(altaz)
        sun = get_sun(t).transform_to(altaz)

        #
        # elongation = sun.separation(moon)  # BORKS!!!
        elongation = sun.frame.separation(moon.frame)

        # phase angle at obstime
        phase = np.arctan2(sun.distance * np.sin(elongation),
                           moon.distance - sun.distance * np.cos(elongation))
        illumination = (1 + np.cos(phase)) / 2.0

        return phase.value, illumination.value

    def get_marker(self):  # NOTE: could be a function
        """
        Create a marker for the moon that recreates it's current shape in the
        sky based on the phase

        Returns
        -------
        `matplotlib.path.Path` instance
        """
        # if close to full, just return the standard filled circle
        if np.abs(self.phase - np.pi) % np.pi < 0.1:
            return 'o'

        theta = np.linspace(np.pi / 2, 3 * np.pi / 2)
        xv, yv = np.cos(theta), np.sin(theta)
        # xv, yv = circle.vertices.T
        # lx = xv < 0
        angle = np.pi - self.phase
        xp = xv * np.cos(angle)

        x = np.r_[xv, xp]
        y = np.r_[yv, yv[::-1]]
        verts = np.c_[x, y]
        codes = np.ones(len(x)) * 4
        codes[0] = 1
        return mplPath(verts, codes)


class SeczTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = False
    has_inverse = False

    def transform_non_affine(self, alt):
        return 1. / np.cos(np.radians(90. - alt))


class SeczFormatter(TransFormatter):
    _transform = SeczTransform()

    def __call__(self, x, pos=None):
        # ignore negative numbers (below horizon)
        if (x < 0):
            return ''

        return TransFormatter.__call__(self, x, pos)


class VizAxes(SubplotHost):
    """The standard axes class for visibility tracks"""

    # def __init__(self, *args, **kw):

    ##self.ytrans = SeczTransform()
    ##self._aux_trans = btf(ReciprocalTransform(), IdentityTransform())

    # kws.pop('site')

    # date = '2016-07-08'
    # lon = viz.siteLoc.longitude
    # sid_trans = sidereal_transform(date, lon, 2)
    # aux_trans = btf(sid_trans, IdentityTransform())

    # SubplotHost.__init__(self, *args, **kw)
    # self.parasite = self.twin(aux_trans)

    def setup_ticks(self):
        # Tick setup for both axes
        minorTickSize = 8
        for axis in (self.yaxis, self.parasite.yaxis):
            axis.set_tick_params('both', tickdir='out')
            axis.set_tick_params('minor', labelsize=minorTickSize, pad=5)

        # TODO:  colors='' #For sidereal time axis
        self.xaxis.set_tick_params('major', pad=15)
        # self.yaxis.set_tick_params('minor', labelsize=minorTickSize, pad=5)

        # Tick setup for main axes
        #         self.xaxis.set_tick_params('major', pad=10)
        #         self.yaxis.set_tick_params('minor', labelsize=6, pad=5)
        dloc = AutoDateLocator()
        # self.xaxis.tick_bottom()
        self.xaxis.set_major_locator(dloc)
        self.xaxis.set_minor_locator(AutoMinorLocator())
        fmt = AutoDateFormatter(dloc)
        fmt.scaled[1 / 24] = '%H:%M'
        self.xaxis.set_major_formatter(fmt)

        self.yaxis.set_minor_locator(AutoMinorLocator())
        self.yaxis.set_major_formatter(DegreeFormatter())
        self.yaxis.set_minor_formatter(DegreeFormatter())

        # Tick setup for main axes

        # self.parasite.axis['right'].major_ticklabels.set_visible(False)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def set_formatters(self):

        # self.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        #  self.parasite.xaxis.set_minor_locator(AutoMinorLocator())
        self.parasite.xaxis.tick_top()
        self.parasite.xaxis.offsetText.set_visible(False)

        #
        dloc = AutoDateLocator()
        self.parasite.xaxis.set_major_locator(dloc)
        fmt = AutoDateFormatter(dloc)
        fmt.scaled[1 / 24] = '%H:%M'
        self.parasite.xaxis.set_major_formatter(fmt)
        # self.parasite.xaxis.set_minor_locator(AutoMinorLocator())

        # fine grained formatting for coord display subtext
        fineGrainDateFmt = AutoDateFormatter(dloc)
        fineGrainDateFmt.scaled[1 / 24] = '%H:%M:%S'
        self._xcoord_formatter = fineGrainDateFmt
        self._ycoord_formatter = DegreeFormatter(precision=2)

        # ticks for airmass axis
        self.parasite.yaxis.tick_right()
        airmassFmt = SeczFormatter(precision=2)
        self.parasite.yaxis.set_major_formatter(airmassFmt)

        self.parasite.yaxis.set_minor_locator(AutoMinorLocator())
        self.parasite.yaxis.set_minor_formatter(airmassFmt)

        # def set_locators(self):
        # formatter_factory(AutoMinorLocator(n=5))
        # self.xaxis.set_minor_locator(AutoMinorLocator(n=5))

        # self.parasite.xaxis.set_minor_locator(AutoMinorLocator(n=5))

    def format_coord(self, x, y):
        # print('fmt')
        # s = super().format_coord(x, y)
        xs = self._xcoord_formatter(x)
        ys = self._ycoord_formatter(y)

        xt, _ = self.parasite.transAux.inverted().transform([x, y])
        xts = self._xcoord_formatter(xt)

        fmt = self.parasite.yaxis.get_major_formatter()
        _, yt = fmt._transform.transform([x, y])  # HACK!!

        return f'UTC={xs}\talt={ys}\tsid.T={xts}\tairmass={yt:.3f}'





class VizPlot(LoggingMixin):
    """
    A tool for plotting visibility tracks of astronomical objects to aid
    observational planning and scheduling.

    Initializing the class without arguments will set up the plot for the
    current date and default location (sutherland). Date, location and targets
    can optionally be passed upon initialization.

    Objects tracks can be added to the plot by passing an object name to the
    `add_target` method. In the case of objects in the SIMBAD database, or
    objects which have coordinates in their names (eg. SDSS J015543.40+002807.2)
    the coordinates will be resolved automatically. Objects for which the name
    cannot be resolved into coordinates (eg. recent uncatalogued transient
    sources, new exoplanet candidates or whatever) can be added by explicitly
    passing coordinates.

    To aid visualizing many objects simultaneously, hovering over entries in the
    legend will highlight the object track in the main plot. Tracks can be
    hidden by clicking on the corresponing legend entry Current time is
    indicated by a vertical line (dashed green) that updates at a specified
    interval. To activate the dynamic features (highlight, current time
    indicator line), use the `connect` method.

    Examples
    --------
    >>> vis = VizPlot()
    >>> vis.add_target('FO Aqr')
    >>> vis.add_target('NOI-105276', '20:31:25.8 -19:08:35.0')
    >>> vis.add_targets('SN 1987A', 'M31')
    >>> vis.connect()
    """

    # cmap that will be used for track colours
    default_cmap = 'jet'
    # whether to abbreviate target names containing J coordinates
    shortenJnames = True
    n_points_track = 250

    # TODO: figure cross-hairs indicating altitude etc of objects on hover.
    # TODO: indicate 'current' distance from moon when hovering
    # TODO: let text labels for tracks curve along the track - long labels
    # TODO: Moon / sun pickable
    # FIXME: legend being cut off

    # @profiling.histogram()
    def __init__(self, targets=None, date=None, site='sutherland', tz=2 * u.h,
                 **options):  # res
        """

        Parameters
        ----------
        date
        site
        targets
        tz
        options
        """

        self.siteName = site.title()
        self.siteLoc = get_site(self.siteName)
        # TODO from time import timezone
        self.tz = tz
        # TODO: can you get this from the site location??
        # http://stackoverflow.com/questions/16086962/how-to-get-a-time-zone-from-a-location-using-latitude-and-longitude-coordinates
        # https://developers.google.com/maps/documentation/timezone/start

        self.targetCoords = {}
        self.tracks = {}
        self.plots = OrderedDict()
        self.labels = defaultdict(list)
        self.texts = Dict()
        self.vlines = Dict()
        self.shortNames = {}
        self.legLineMap = {}

        # blitting setup
        self.saving = False  # save background on draw
        self.use_blit = options.get('use_blit', True)
        self.count = 0

        if date:
            self.date = Time(date).to_datetime()
        else:
            self.date = nearest_midnight_date()

        # local midnight in UTC
        self.midnight = midnight = Time(self.date) - tz

        # time variable
        self.hours = h = np.linspace(-12, 12, self.n_points_track) * u.h
        self.t = t = midnight + h
        self.tp = t.plot_date
        self.frames = frames = AltAz(obstime=t, location=self.siteLoc)

        self._legend_fix_req = False

        # collect name, coordinates in dict
        if targets is not None:
            self.add_targets(targets)

        # TODO: the next two lines are slow!
        # tODO: do in thread / or interpolate position / memoize
        # Get sun, moon coordinates
        # frames = self.frames[:]
        self.sun = Sun(frames, midnight)
        self.moon = Moon(frames, midnight)

        # self._drawn = False
        self.figure, self.ax = self.setup_figure()
        self.canvas = self.figure.canvas
        ax = self.ax

        self.highlighted = None

        # current time indicator
        # animated=True to prevent redrawing the canvas
        self.vlines.now, = ax.plot([self.midnight.plot_date] * 2, [0, 1],
                                   ls=':', c='g',
                                   transform=btf(ax.transData, ax.transAxes),
                                   animated=True)
        self.texts.now.sast = vertical_txt(ax, '', t[0], y='bottom', color='g',
                                           animated=True)
        self.texts.now.sidT = vertical_txt(ax, '', t[0], y='top', color='c',
                                           animated=True)

        # set all tracks and their labels as well as current time text invisible
        # before drawing, so we can blit
        self.currentTimeArt = (self.vlines.now, self.texts.now.sast,
                               self.texts.now.sidT)
        self._vart = (self.currentTimeArt,
                      self.plots.values(), self.labels.values(),
                      self.legLineMap)
        set_visible(self._vart, False)

        self.currentTimeThreadAlive = threading.Event()
        self.currentTimeThread = threading.Thread(
            target=self.show_current_time,
            args=(self.currentTimeThreadAlive,))

        # HACK
        self.cid = self.canvas.mpl_connect('draw_event', self._on_first_draw)
        self.canvas.mpl_connect('draw_event', self._on_draw)
        # At this point the background with everything except the tracks is saved

    def show_current_time(self, alive, interval=3):
        """thread to update line indicating current time"""

        while not alive.isSet():
            # print('updating current time')
            self.update_current_time()
            time.sleep(interval)

    def update_current_time(self, draw=True):
        now = Time.now()
        t = now.plot_date
        # update line position
        self.vlines.now.set_xdata([t, t])

        # update SAST text
        sastTxt = f"{local_time_str(now, precision='m')} SAST"
        self.texts.now.sast.set_text(sastTxt)
        self.texts.now.sast.set_position((t, 0.01))
        # update Sid.T text
        sidT = now.sidereal_time('mean', self.siteLoc.lon)
        sidTTxt = f"{sidT.to_string(sep=':')[:5]} Sid.T"
        self.texts.now.sidT.set_text(sidTTxt)
        self.texts.now.sidT.set_position((t, 1))

        # blit the figure if the current time is within range
        tmin, tmax = self.ax.get_xlim()
        if ((tmin < t) & (t < tmax)) and draw:
            # blit figure
            set_visible(self.currentTimeArt)
            self.draw_blit(self.currentTimeArt, bg=self.background2)

    def add_coordinate(self, name, coo=None):
        """
        Resolve coordinates from object name and add to cache. If coordinates 
        provided, simply add to list with associated name.

        Parameters
        ----------
        name : str
            The object name
        coo : str or SkyCoord, optional
            The object coordinates (right ascention, declination) as a str that
            be resolved by SkyCoord, or a SkyCoord object.

        Returns
        -------
        SkyCoord
            The object coordinates
        """

        if name in self.targetCoords:
            self.logger.info('%s already in target list' % name)
            return False

        if coo is None:
            coo = get_coordinates(name)
            if self.shortenJnames:
                self.shortNames[name] = short_name(name)
        else:
            coo = get_coordinates(coo)

        # add to list
        if coo:
            self.targetCoords[name] = coo

        return coo

    def add_coordinates(self, names):
        for name in names:
            self.add_coordinate(name)

    def add_target(self, name, coo=None, update=True):
        success = self.add_coordinate(name, coo)
        if success:
            self.plot_track(name)
            if update:
                self.do_legend()
                self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

    def remove_target(self, name):
        """
        Remove the visibility track of target with name `name` from the plot

        Parameters
        ----------
        name : str
            The target name. Will result in an error if the target name is not
            included in the plot
        """
        if name in self.targetCoords:
            for dic in (self.targetCoords, self.tracks, self.plots,
                        self.labels, self.shortNames):
                dic.pop(name)

    def add_targets(self, names, coords=()):
        """
        Add visibility tracks of targets to the plot

        Parameters
        ----------
        names : dict or list
            The names of the objects. Will attempt to resolve the name via a
            sesame query if coordinates are not provided
        coords : list, optional
            Sequence of coordinates (str or SkyCoord), by default ().  If
            provided, these coordinates will be used and no lookup for the
            name will occur.
        """
        if isinstance(names, dict):
            itr = names.items()
        else:
            itr = itt.zip_longest(names, coords)

        for name, coo in itr:
            self.add_target(name, coo, update=False)

        self.do_legend()
        self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

        # self.cid = self.canvas.mpl_connect('draw_event', self.fix_legend())

    def plot_track(self, name):
        """
        Calculate and plot the visibility track for an object whose coordinates
        has already been resolved and cached.
        """
        
        coords = self.targetCoords[name]

        if name not in self.tracks:
            # compute trajectory
            altaz = coords.transform_to(self.frames)
            alt = self.tracks[name] = altaz.alt.degree
        else:
            # recall trajectory
            alt = self.tracks[name]

        trg_pl, = self.ax.plot(self.tp, alt, label=name)  # , animated=True)
        self.plots[name] = trg_pl

        # add name text at axes edges
        texts = self.add_annotation(name)
        return trg_pl

    def add_annotation(self, name):
        """
        Add target name to visibility curve - helps more easily distinguish
        targets when plotting numerous curves
        """

        ax = self.ax
        alt = self.tracks[name]
        # find where the curve intersects the edge of the axes
        y0, y1 = ax.get_ylim()
        ly = (y0 < alt) & (alt < y1)
        # region within axes (possibly manyfold for tracks that pass out, and then in again)
        # gives edge intersecting points in boolean array
        li = (ly & self._lt).astype(int)
        # first points inside axes (might be more than one per curve)
        l0 = (li - np.roll(li, 1)) == 1
        l1 = (li - np.roll(li, -1)) == 1  # last points inside axes (")
        first, = np.where(l0)
        last, = np.where(l1)
        ixSeg = np.c_[first, last]

        # same colour for line and text
        colour = self.plots[name].get_color()
        kws = dict(color=colour, size='small', fontweight='bold',
                   rotation_mode='anchor', clip_on=True)  # , animated=True)

        # remove labels if exists
        for text in self.labels.pop(name, ()):
            text.remove()

        # decide whether to add one label or two per segment
        for i, (il, iu) in enumerate(ixSeg):
            # determine the slope and length of curve segments at all points within axes
            x, y = self.tp[il:iu + 1], alt[il:iu + 1]
            dx, dy = np.diff((x, y))
            segL = np.sqrt(np.square(np.diff([x, y])).sum(0))
            angles = np.degrees(np.arctan2(dy, dx))
            angles = ax.transData.transform_angles(angles, np.c_[x, y][:-1])
            # TODO: BUGREPORT transform_angles.
            # docstring says: 'The *angles* must be a column vector (i.e., numpy array).'
            # this is wrong since the function *DOES NOT WORK* when angles is column vector!!

            # create the text
            shortName = self.shortNames.get(name, name)
            text = ax.text(x[0], y[0] + 0.5, shortName,
                           ha='left', rotation=angles[0], **kws)
            self.labels[name].append(text)

            bb = text.get_window_extent(ax.figure.canvas.get_renderer())
            c = bb.corners()[[0, -1]]  # lower left, top right
            xyD = ax.transData.inverted().transform(c)
            # length of rendered text in data space
            txtL = np.sqrt(np.square(np.diff(xyD.T)).sum())
            # space = segL.sum()  # curve length of track segment
            # if entry & exit point are close to oe another, only make 1 label
            if txtL * 2 < segL.sum():
                 # second label for segment
                text = ax.text(x[-1], y[-1] + 0.5, name,
                               ha='right', rotation=angles[-1], **kws)
                self.labels[name].append(text)

        # TODO: Curved text
        # - interpolators for each track
        # split text at n positions, where n is decided based on str length
        # place each text segment at position along line with angle of slope at
        #  that position
        return self.labels[name]

    def _on_resize(self, event):

        # this will happen before draw
        self.save_background()

        # redo the track labels (angles will have changed)
        for name in self.tracks.keys():
            self.add_annotation(name)

        # now the canvas will redraw automatically
        set_visible(self._vart)
        self.saving = True

    def setup_figure(self, figsize=(15, 7.5)):

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.94,
                            left=0.05,
                            right=0.8,
                            bottom=0.075)
        # setup axes with
        ax = VizAxes(fig, 111)
        sid_trans = sidereal_transform(self.date, self.siteLoc.lon)  # gitude)
        aux_trans = btf(sid_trans, IdentityTransform())
        ax.parasite = ax.twin(aux_trans)

        # ax.autoscale(False)
        # ax.parasite.autoscale(False)

        ax.setup_ticks()
        fig.add_subplot(ax)

        # horizon line
        horizon = ax.axhline(0, 0, 1, color='0.85')

        # Shade twilight / night
        sun, moon = self.sun, self.moon
        for i, twilight in enumerate(zip(sun.dusk.items(), sun.dawn.items())):
            desc, times = zip(*twilight)

            ax.axvspan(*Time(times).plot_date, color=str(0.25 * (3 - i)))

            for words, t in zip(desc, times):
                vertical_txt(ax, words, t, color=str(0.33 * i))

        # Indicate moonrise / set
        intervals = list(zip(sun.dusk.values(), sun.dawn.values()))[::-1]
        cols = ['y', 'y', 'orange', 'orange']
        c = 'y'  # in case the `moon.up` dict is empty
        for rise_set, time in moon.up.items():
            # pick colours that are readable against grey background
            for i, (r, s) in enumerate(intervals):
                if r < time < s:
                    c = cols[i]

            self.vlines[rise_set] = ax.axvline(time.plot_date, c=c, ls='--')
            self.texts[rise_set] = vertical_txt(
                ax, rise_set, time, y='bottom', color=c)

        # TODO: enable picking for sun / moon
        sun_pl, = ax.plot(self.tp, sun.coords.alt,
                          'orangered', ls='none', markevery=2,
                          marker='o', ms=10,
                          label='sun')
        moon_pl, = ax.plot(self.tp, moon.coords.alt,
                           'yellow', ls='none', markevery=2,
                           marker=moon.get_marker(), ms=10,
                           label='moon ({:.0%})'.format(moon.illumination))

        # site / date info text
        ax.text(0, 1.04, self.date_info_txt(self.midnight),
                fontweight='bold', ha='left', transform=ax.transAxes)
        ax.text(1, 1.04, self.site_info_txt(), fontweight='bold',
                ha='right', transform=ax.transAxes)

        # setup axes
        # dloc = AutoDateLocator()
        # ax.xaxis.set_major_locator(dloc)
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_major_formatter(DegreeFormatter())
        # ax.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        # ax.yaxis.set_minor_formatter(DegreeFormatter())

        just_before_sunset = (sun.set - 0.25 * u.h).plot_date
        just_after_sunrise = (sun.rise + 0.25 * u.h).plot_date
        ax.set_xlim(just_before_sunset, just_after_sunrise)
        ax.set_ylim(-10, 90)

        # HACK
        # qq = aux_trans.transform(np.c_[ax.get_xlim(), ax.get_ylim()])
        # ax.parasite.viewLim.intervalx = qq[:,0]

        # which part of the visibility curves are visible within axes
        self._lt = (just_before_sunset < self.tp) & (
            self.tp < just_after_sunrise)

        # labels for axes
        ax.set_ylabel('Altitude', fontweight='bold')
        ax.parasite.set_ylabel('Airmass', fontweight='bold')

        # sun / moon legend
        leg = ax.legend(bbox_to_anchor=(1.05, 0), loc=3,
                        borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')
        ax.add_artist(leg)

        ax.grid()

        return fig, ax

    def site_info_txt(self):
        # eg:
        lat = self.siteLoc.lat  # itude
        lon = self.siteLoc.lon  # gitude
        ns = 'NS'[bool(lat > 0)]
        ew = 'WE'[bool(lon > 0)]

        lat = abs(lat).to_string(precision=0, format='latex')
        lon = abs(lon).to_string(precision=0, format='latex')
        h = '{0.value:.0f} {0.unit:s}'.format(self.siteLoc.height)
        return f'{self.siteName} @ {lat} {ns}; {lon} {ew}; {h}'
        # return '{} @ {} {}; {} {}; {}'.format(self.siteName, lat, ns, lon, ew, h)

    # ==============================================================================
    @staticmethod
    def date_info_txt(t):
        dayname = WEEKDAYS[t.datetime.weekday()]
        datestr = t.iso.split()[0]
        return ', '.join((dayname, datestr))

    def make_time_ticks(self):
        # Create ticklabels for SAST
        ax = self.ax
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklocs = ax.xaxis.get_majorticklocs()
        fmt = ax.xaxis.get_major_formatter()
        sast_lbls = map(fmt, (Time(num2date(xticklocs)) + self.tz).plot_date)

        for sast, tck in zip(sast_lbls, xticklabels):
            # FIXME: does not update when zooming / panning!  better to have another parasite?
            xy = x, y = tck.get_position()
            trans = tck.get_transform()
            self.ax.text(*xy, sast,
                         color='g',
                         transform=trans,
                         va='bottom', ha=tck.get_ha())

        # UTC / SAST  axes labels
        btrans = btf(ax.transAxes, trans)
        x = self._get_tick_x(tck)
        self.ax.text(x, y, 'SAST',
                     color='g', fontweight='bold',
                     va='bottom', ha='left',
                     transform=btrans)
        self.ax.text(x, y, 'UTC',
                     color='k', fontweight='bold',
                     va='top', ha='left',
                     transform=btrans)

        # change color of sidereal time labels
        for tck in self.ax.parasite.get_xmajorticklabels():
            tck.set_color('c')

        # sidereal time label
        btrans = btf(ax.transAxes, tck.get_transform())
        x, y = tck.get_position()
        x = self._get_tick_x(tck)
        self.ax.text(x, y, 'Sid.T.',
                     color='c', fontweight='bold',
                     va='bottom', ha='left',
                     transform=btrans)

    def _get_tick_x(self, tck):
        bb = tck.get_window_extent()
        c = bb.corners()[[0, 2]]  # lower left, lower right
        xyAx = self.ax.transAxes.inverted().transform(c)  # text box in axes coordinates
        w = np.diff(xyAx[:, 0])  # textbox width in axes coordinates
        return max(xyAx[1, 0] + w * 0.1, 1)

    def plot_viz(self, **kws):
        # TODO: MAYBE option for colour coding azimuth
        ax = self.ax

        cmap = kws.get('cmap')
        colours = kws.get('colours', [])
        sort = kws.get('sort', True)

        self.set_colour_cycle(colours, cmap)
        names, coords = zip(*self.targetCoords.items())
        if sort:
            coords, names = sorter(coords, names, key=lambda coo: coo.ra)

        for name in names:
            self.plot_track(name)

        # self.plots += [sun_pl, moon_pl]

        self.do_legend()

    # alias
    plot_vis = plot_viz

    def set_colour_cycle(self, colours=[], cmap=None):
        # Ensure we plot with unique colours
        N = len(self.targetCoords)

        # default behaviour - use colourmap
        if (not len(colours)) and (cmap is None):
            cmap = self.default_cmap

        # if colour map given or colours not given and default colour sequence insufficien
        if ((cmap is not None)  # colour map given - superceeds colours arg
                or ((not len(colours))  # colours not given
                    and len(rcParams['axes.prop_cycle']) < N)):  # default colour sequence insufficient for uniqueness
            cm = plt.get_cmap(cmap)
            colours = cm(np.linspace(0, 1, N))

        # Colours provided explicitly and no colourmap given
        elif len(colours) < N:
            'Given colour sequence less than number of time series. Colours will repeat'

        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            self.ax.set_prop_cycle(ccyc)

    def get_legend_labels(self, show_coords=True):
        # if show_coords:
        # add amsmath so we can make short minus in math mode
        # from matplotlib import rcParams
        # rcParams["text.latex.preamble"] = [r'\usepackage{amsmath}']#,
        # r'\mathchardef\mhyphen="2D']

        labels = []
        for name, coo in self.targetCoords.items():
            name = self.shortNames.get(name, name)
            if show_coords:
                #  math mode for boldness, thus need special space
                name = name.replace(' ', '\:')  # .replace('-', r'\mbox{-}')
                name = '$\mathbf{%s}$' % name
                cs = coo.to_string('hmsdms', precision=0, format='latex')
                lbl = '\n'.join((name, cs.replace(' ', '; ')))
            else:
                lbl = name
            labels.append(lbl)
        return labels

    def do_legend(self, show_coords=True):
        # TODO: maybe use your DynamicLegend class ??
        """
        Create a legend with pickable elements to toggle visibility of tracks
        """
        labels = self.get_legend_labels(show_coords)
        leg = self.ax.legend(list(self.plots.values()), labels,
                             bbox_to_anchor=(1.05, 1), loc=2,
                             borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')

        # if show_coords:
        #     for txt in leg.texts:
        #         txt.set_usetex(True)
        #  TODO: mpl feature request to usetex in legend

        # self.legLineMap = {}
        for legline, origline in zip(leg.get_lines(), self.plots.values()):
            legline.set_pickradius(10)
            legline.set_picker(True)
            self.legLineMap[legline] = origline

        self.legLineMapInv = dict(
            zip(self.legLineMap.values(), self.legLineMap.keys()))

        # re-space legend on next draw
        # connect = self.canvas.mpl_connect
        # self._lfcid = connect('draw_event', self.fix_legend)

        self._legend_fix_req = True

    def fix_legend(self):
        # Shift axes position to fit the legend nicely (long names get clipped)

        self.logger.debug('Fixing Legend')

        leg = self.ax.get_legend()
        bb = leg.get_window_extent(self.figure._cachedRenderer)
        # lower left, lower right of legend in display coordinates
        c = bb.corners()[[0, 2]]
        xyAx = self.ax.transAxes.inverted().transform(c)
        wAx = np.diff(xyAx[:, 0])  # textbox width in axes coordinates
        right = 1 - wAx * 1.05
        self.figure.subplots_adjust(right=right)

        # write name width and axes right position to file - eventually we can use this to guess right and avoid this function
        # with (moduleDir / '.legend.fix').open('a') as fp:
        #     longest_name = max(map(len, self.targetCoords.keys()))
        #     fp.write(str((longest_name, right)))
        #     fp.write('\n')

        # self.canvas.mpl_disconnect(self._lfcid)

    def _on_pick(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility

        # first remove any highlighting due to hover
        legline = event.artist
        track = self.legLineMap[event.artist]
        name = track.get_label()

        if self.highlighted:
            self.highlight(name, 0.5, False)  # restore
            self.highlighted = None

        vis = not track.get_visible()
        set_visible((track, self.labels[name]), vis)

        # Change the alpha on the line in the legend so we can see what lines
        # have been toggled
        legline.set_alpha((0.2, 1.0)[vis])
        self.draw_blit(self._vart)

    def _on_motion(self, event):
        # TODO: move this to the DynamicLegend class!!!!!!

        leg = self.ax.get_legend()
        in_legend, props = leg.contains(event)
        hit = False
        if in_legend:
            for legLine in leg.get_lines():
                name = legLine.get_label()
                hit, props = legLine.contains(event)
                if hit:
                    break

            if hit:
                if self.highlighted:
                    # another track still highlighted - restore
                    self.highlight(self.highlighted, 0.5, False)

                # new track to highlight
                self.highlighted = name
                self.highlight(name)

        # not hovering over anything, but something remains highlighted
        if self.highlighted and not hit:
            # restore
            self.highlight(self.highlighted, 0.5)
            self.highlighted = None

    def highlight(self, name, factor=2, draw=True):
        """ """

        plotLine = self.plots[name]
        # if line has been de-selected, no need to hightlight
        if not plotLine.get_visible():
            return

        legLine = self.legLineMapInv[plotLine]
        legLine.set_lw(legLine.get_lw() * factor)  # thicken legend line
        plotLine.set_lw(plotLine.get_lw() * factor)  # thicken track line
        # plot over everything else
        plotLine.set_zorder(plotLine.get_zorder() * factor)

        # update current time else it will disappear
        self.currentTimeThreadAlive.clear()  # suspend the current time thread
        self.update_current_time(draw=False)  # will draw below

        if draw:
            self.draw_blit(self._vart)

        # re-activate the current time thread
        self.currentTimeThreadAlive.set()

    def _on_draw(self, event):
        self.count += 1
        if self.saving:
            self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

        # # fix the legend spacing if necessary
        # if self._legend_fix_req:
        #     self.fix_legend()
        #     self._legend_fix_req = False

    def _on_first_draw(self, event):
        # HACK! this method creates the SAST labels upon the first call to draw
        # as well as starting the currentTimeThread
        self.logger.debug('FIRST DRAW')

        fig = self.figure
        canvas = fig.canvas

        # disconnect callback to this function
        canvas.mpl_disconnect(self.cid)

        # ticks
        self.make_time_ticks()

        # save background without tracks
        self.save_background()
        set_visible(self._vart, True)
        self.draw_blit(*self._vart)
        # save background with tracks
        self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

        # canvas.draw()
        self.currentTimeThread.start()

    def save_background(self, event=None):
        # save the background for blitting
        self.logger.debug('save_background')
        # make tracks invisible
        set_visible(self._vart, False)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

    def draw_blit(self, *artists, bg=None):
        self.canvas.restore_region(bg or self.background)
        renderer = self.canvas.get_renderer()
        for art in mit.collapse(artists):
            art.draw(renderer)
        self.canvas.blit(self.figure.bbox)

    def closing(self, event):
        # stop the currentTimeThread
        self.currentTimeThreadAlive.set()
        # self.currentTimeThread.join()  # NOTE: will potentially wait for 30s

    def connect(self):
        # Setup legend picking
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('close_event', self.closing)
        self.canvas.mpl_connect('resize_event', self._on_resize)


# alias
VisPlot = VizPlot
