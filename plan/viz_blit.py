"""
Tools for visualising object tracks across the night sky
"""

import time
import inspect
import logging
import threading
from functools import partial
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# Import the packages necessary for finding coordinates and making coordinate transformations
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz,
                                 get_sun, get_moon)
from astropy.coordinates.name_resolve import NameResolveError

from obstools.jparser import Jparser

import matplotlib.pyplot as plt
from matplotlib import rcParams
# from matplotlib.lines import Line2D
from matplotlib.path import Path as mplPath
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date
from matplotlib.transforms import (Transform, IdentityTransform, Affine2D,
                                   blended_transform_factory as btf)
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local libs
from grafico.ticks import DegreeFormatter, TransFormatter
from recipes.list import sorter

from decor.misc import persistant_memoizer
from decor import profiler
# profiler = profile()        #truncate_lines=50

# ====================================================================================================
# setup persistant coordinate cache - faster object coordinate retrieval via sesame query
here = inspect.getfile(inspect.currentframe())
moduleDir = Path(here).parent
cooCacheName = '.coordcache'
siteCacheName = '.sitecache'
cooCachePath = moduleDir / cooCacheName
siteCachePath = moduleDir / siteCacheName


@persistant_memoizer(cooCachePath)
def resolver(name):
    """Get the target coordinates from object name if known"""
    # try extract J coordinates from name.  We do this first, since it is faster than a sesame query
    try:
        return Jparser(name).skycoord()
    except ValueError:
        pass

    # try use Simbad to resolve object names and retrieve coordinates.
    return SkyCoord.from_name(name)


@persistant_memoizer(siteCachePath)
def get_site(name):
    return EarthLocation.of_site(name)


# ====================================================================================================
# def OOOOh(t):
# """hours since midnight"""
# return (t - midnight).sec / 3600

# ====================================================================================================
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


# ====================================================================================================
def get_sid_trans(date, longitude):
    """Initialize matplotlib transform for local time - sidereal time conversion"""
    midnight = Time(date)  # midnight UTC
    midSid = midnight.sidereal_time('mean', longitude)
    # offset from local time
    offset = midSid.hour / 24
    # used to convert to origin of plot_date coordinates
    p0 = midnight.plot_date
    # A mean sidereal day is 23 hours, 56 minutes, 4.0916 seconds (23.9344699 hours or 0.99726958 mean solar days)
    scale = 366.25 / 365.25

    return Affine2D().translate(-p0, 0).scale(scale).translate(p0 + offset, 0).inverted()


# ****************************************************************************************************
class SeczTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = False
    has_inverse = False

    def transform_non_affine(self, alt):
        return 1. / np.cos(np.radians(90. - alt))


# ****************************************************************************************************
class SeczFormatter(TransFormatter):
    _transform = SeczTransform()


# ****************************************************************************************************
class VizAxes(SubplotHost):
    """The standard axes class for visibility tracks"""


    # def __init__(self, *args, **kw):

    ##self.ytrans = SeczTransform()
    ##self._aux_trans = btf(ReciprocalTransform(), IdentityTransform())

    # kws.pop('site')

    # date = '2016-07-08'
    # lon = viz.siteLoc.longitude
    # sid_trans = get_sid_trans(date, lon, 2)
    # aux_trans = btf(sid_trans, IdentityTransform())

    # SubplotHost.__init__(self, *args, **kw)
    # self.parasite = self.twin(aux_trans)


    def setup_ticks(self):
        # Tick setup for both axes
        minorTickSize = 8
        for axis in (self.yaxis, self.parasite.yaxis):
            axis.set_tick_params('both', tickdir='out')
            # axis.set_tick_params('minor', labelsize=minorTickSize, pad=0)

        # TODO:  colors='' #For sidereal time axis
        self.xaxis.set_tick_params('major', pad=10)
        self.yaxis.set_tick_params('minor', labelsize=6, pad=5)

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


        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # def set_formatters(self):

        # self.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        self.parasite.xaxis.tick_top()
        self.parasite.xaxis.offsetText.set_visible(False)
        #
        # class FuckYou(AutoDateLocator):
        #     def autoscale(self):
        #         print('fuck you')


        # dloc = FuckYou()
        dloc = AutoDateLocator()
        self.parasite.xaxis.set_major_locator(dloc)
        fmt = AutoDateFormatter(dloc)
        fmt.scaled[1 / 24] = '%H:%M'
        self.parasite.xaxis.set_major_formatter(fmt)
        # self.parasite.xaxis.set_minor_locator(AutoMinorLocator())

        # fine grained formatting for coord display subtext
        fgfmt = AutoDateFormatter(dloc)
        fgfmt.scaled[1 / 24] = '%H:%M:%S'
        self._xcoord_formatter = fgfmt
        self._ycoord_formatter = DegreeFormatter(precision=2)

        # Consider ticking independently?
        self.parasite.yaxis.tick_right()
        self.parasite.yaxis.set_major_formatter(SeczFormatter())

        # self.parasite.yaxis.set_minor_locator(AutoMinorLocator())
        self.parasite.yaxis.set_minor_formatter(SeczFormatter())



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

        _, yt = self.parasite.yaxis.get_major_formatter()._transform.transform([x, y])  # HACK!!
        yts = '%.3f' % yt

        return 'UTC=%s\talt=%s\tsid.T=%s\tairmass=%s' % (xs, ys, xts, yts)


class Sun():

    def __init__(self, frames, midnight):

        t = frames.obstime
        self.coords = get_sun(t).transform_to(frames)  # WARNING: slow!!!!
        # get dawn / dusk times
        self.dusk, self.dawn = self.get_rise_set(t, midnight)
        self.set, self.rise = self.dusk['sunset'], self.dawn['sunrise']


    def get_rise_set(self, t, midnight):
        """calculate dawn / dusk / twighlight times"""

        # We interpolate the calculated sun positions to get dusk/dawn times.  Should still be accurate to ~1s
        h = (t - midnight).to('h').value  # hours since midnight ut
        ip = interp1d(h, self.coords.alt.degree)

        def twilight(h, angle):
            """civil / nautical / astronomical twighlight solver"""
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


class Moon():
    def __init__(self, frames, midnight):
        # get moon rise/set times, phase, illumination etc...
        t = frames.obstime
        self.coords = get_moon(t).transform_to(frames)  # WARNING: slow!!!!
        self.up = self.get_rise_set(t, midnight)
        self.phase, self.illumination = self.get_phase(midnight, frames.location)


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
        elongation = sun.separation(moon)
        # phase angle at obstime
        phase = np.arctan2(sun.distance * np.sin(elongation),
                           moon.distance - sun.distance * np.cos(elongation))
        illumination = (1 + np.cos(phase)) / 2.0

        return phase.value, illumination.value


    def get_marker(self):  # NOTE: could be a fuction

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



# ****************************************************************************************************
class VisPlot():
    """
    A tool for plotting visibility tracks of astronomical objects to aid
    observational planning and scheduling.

    Initializing the class without arguments will set up the plot for the
    current date and default location (sutherland). Date, location and targets
    can optionally be passed upon initialization.

    Objects tracks can be added to the plot by passing an object name to the
    `add_target` method. In the case of objects in the SIMBAD database,
    or objects which have coordinates in their names (eg. SDSS J015543.40+002807.2)
    the coordinates will be resolved automatically. Objects for which the name
    cannot be resolved into coordinates (eg. recent uncatalogued transient sources)
    can be added by explicitly passing coordinates.

    To aid visualizing many objects simultaneously, hovering over entries in
    the legend will highlight the object track in the main plot.
    Tracks can be hidden by clicking on the corresponing legend entry
    Current time is indicated by a vertical line (dashed green) that updates
    at a specified interval.
    To activate the dynamic features (highlight, current time indicator line),
    use the `connect` method.

    Examples
    --------
    >>> vis = VisPlot()
    >>> vis.add_target('FO Aqr')
    >>> vis.add_targets('SN 1987A', 'M31')
    >>> vis.connect()
    """

    default_cmap = 'jet'

    # TODO: figure cross-hairs indicating altitude etc of objects on hover.
    # TODO: indicate 'current' distance from moon when hovering
    # TODO: Option for displaying coordinates in legend
    # TODO: text for current time
    # TODO: let text labels for tracks curve along the track - long labels
    # TODO: Moon / sun pickable
    # FIXME: legend being cut off
    #

    # FIXME: hover highlight not playing well with current time indicator
    # FIXME: Moonrise text at bottom


    # @profiler.histogram
    def __init__(self, date=None, site='sutherland', targets=None, tz=2 * u.h, **options):  # res
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

        #
        self.use_blit = options.get('use_blit', True)

        if date:
            self.date = Time(date).to_datetime()
        else:
            self.date = nearest_midnight_date()

        self.midnight = midnight = Time(self.date) - tz  # local midnight in UTC
        # TODO: efficiency here.  Don't need to plot everything over such a large range
        self.hours = h = np.linspace(-12, 12, 150) * u.h  # time variable
        self.t = t = midnight + h
        self.tp = t.plot_date
        self.frames = frames = AltAz(obstime=t, location=self.siteLoc)

        # collect name, coordinates in dict
        if not targets is None:
            # FIXME: this will only add the coordinates, and not plot them (which is what we want for blitting
            # FIXME: however, we need to flag here that plotting has not been done yet
            self.add_coordinates(targets)

        # TODO: the next two lines are slow!
        # tODO: do in thread / or interpolate position / memoize
        # Get sun, moon coordinates  #TODO: other bright stars / planets
        self.sun = Sun(frames, midnight)
        self.moon = Moon(frames, midnight)

        # self._drawn = False
        self.setup_figure()



        self.highlighted = None

        # current time indicator
        self.nowLine, = self.ax.plot([self.midnight.plot_date] * 2, [0, 1],
                                     ls=':', c='g',
                                     transform=btf(self.ax.transData, self.ax.transAxes),
                                     animated=True)  # prevent redrawing the canvas
        self.nowText = self.vertical_txt('', t[0], y='bottom', color='g', animated=True)
        self.nowLineThreadAlive = threading.Event()
        self.nowLineThread = threading.Thread(target=self.show_current_time,
                                              args=(self.nowLineThreadAlive,))
        # TODO: text to indicate current time UTC / sidereal

        # HACK
        self.cid = self.figure.canvas.mpl_connect('draw_event', self._on_first_draw)
        # At this point the background with everything except the tracks is saved


    def show_current_time(self, alive, interval=30):
        """thread to update line indicating current time"""
        while not alive.isSet():
            # print('updating current time')
            time.sleep(interval)
            now = Time.now()
            s = local_time_str(now)
            t = now.plot_date
            self.nowLine.set_xdata([t, t])
            self.nowText.set_text(s)
            tmin, tmax = self.ax.get_xlim()
            if (tmin < t) & (t < tmax):
                # blit figure
                self.figure.canvas.restore_region(self.background)
                renderer = self.figure.canvas.get_renderer()
                self.nowLine.draw(renderer)
                self.nowText.draw(renderer)
                self.figure.canvas.blit(self.figure.bbox)


    def add_coordinate(self, name, coo=None):
        """
        Resolve coordinates from object name and add to cache. Issue a warning when name cannot be resolved.
        If coordinates provided, simply add to list with associated name
        :returns True if successfully resolved name
        """

        if name in self.targetCoords:
            logging.info('%s already in target list' % name)
            return False

        success = True
        if coo is not None:
            self.targetCoords[name] = SkyCoord(*coo, unit=('h', 'deg'))
        else:
            try:
                self.targetCoords[name] = resolver(name)
            except NameResolveError as err:
                # name not in SIMBAD database
                success = False
                logging.warning(str(err))

        return success


    def add_target(self, name, coo=None, do_legend=True):
        success = self.add_coordinate(name, coo)
        if success:
            self.add_curve(name)
            if do_legend:
                self.do_legend()


    def remove_target(self, name):
        if name in self.targetCoords:
            self.targetCoords.pop(name)
            self.tracks.pop(name)
            self.plots.pop(name)
            self.labels.pop(name)


    def add_targets(self, names):
        for name in names:
            self.add_target(name, do_legend=False)

        self.do_legend()

        # self.cid = self.figure.canvas.mpl_connect('draw_event', self.fix_legend())


    def add_coordinates(self, names):
        for name in names:
            self.add_coordinate(name)


    def add_curve(self, name):
        """Calculate track and plot"""
        # NOTE there is a difference between adding and plotting. #TODO: choose better name. refactor
        coords = self.targetCoords[name]

        if not name in self.tracks:
            # compute trajectory
            altaz = coords.transform_to(self.frames)
            alt = self.tracks[name] = altaz.alt.degree
        else:
            # recall trajectory
            alt = self.tracks[name]

        trg_pl, = self.ax.plot(self.tp, alt, label=name)
        self.plots[name] = trg_pl

        # add name text at axes edges
        self.add_annotation(name)


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
        li = (ly & self._lt).astype(int)    # gives edge intersecting points in boolean array
        l0 = (li - np.roll(li, 1)) == 1     # first points inside axes (might be more than one per curve)
        l1 = (li - np.roll(li, -1)) == 1    # last points inside axes (")
        first, = np.where(l0)
        last, = np.where(l1)
        ixSeg = np.c_[first, last]

        # same colour for line and text
        colour = self.plots[name].get_color()

        # decide whether to add one label or two per segment
        for i, (il, iu) in enumerate(ixSeg):
            # determine the slope and length of curve segments at all points within axes
            x, y = self.tp[il:iu + 1], alt[il:iu + 1]
            dx, dy = np.diff((x, y))
            segL = np.sqrt(np.square(np.diff([x, y])).sum(0))
            angles = np.degrees(np.arctan2(dy, dx))
            angles = ax.transData.transform_angles(angles, np.c_[x, y][:-1])
            # NOTE: BUGREPORT transform_angles.
            # NOTE docstring says: 'The *angles* must be a column vector (i.e., numpy array).'
            # NOTE: this is wrong since the function *DOES NOT WORK* when angles is column vector!!

            # create the text
            text = ax.text(x[0], y[0] + 0.5, name, color=colour,
                           size='small', fontweight='bold', ha='left',
                           rotation=angles[0], rotation_mode='anchor',
                           clip_on=True)
            self.labels[name].append(text)
            bb = text.get_window_extent(ax.figure.canvas.get_renderer())
            c = bb.corners()[[0, -1]]  # lower left, top right
            xyD = ax.transData.inverted().transform(c)
            txtL = np.sqrt(np.square(np.diff(xyD.T)).sum())  # length of rendered text in data space
            space = segL.sum()  # curve length of track segment
            close = txtL * 2 > space  # if entry & exit point are close, only make 1 label

            # second label for segment
            if not close:
                text = ax.text(x[-1], y[-1] + 0.5, name, color=colour,
                               size='small', fontweight='bold', ha='right',
                               rotation=angles[-1], rotation_mode='anchor',
                               clip_on=True)
                self.labels[name].append(text)

        # TODO: Curved text
        # - interpolators for each track
        # split text at n positions, where n is decided based on str length
        # place each text segment at position along line with angle of slope at that position

        # FIXME: labels don't stay in right position when zooming / resizing.  will need custom callback to fix this...


    def setup_figure(self):

        self.figure = fig = plt.figure(figsize=(18, 10))
        fig.subplots_adjust(top=0.94,
                            left=0.05,
                            right=0.85,
                            bottom=0.05)
        # setup axes with
        self.ax = ax = VizAxes(fig, 111)
        sid_trans = get_sid_trans(self.date, self.siteLoc.lon)  # gitude)
        aux_trans = btf(sid_trans, IdentityTransform())
        ax.parasite = ax.twin(aux_trans)

        # ax.autoscale(False)
        # ax.parasite.autoscale(False)


        ax.setup_ticks()
        fig.add_subplot(ax)

        # horizon line
        horizon = ax.axhline(0, 0, 1, color='0.85')

        # Shade twilight / night
        for i, twilight in enumerate(zip(self.sun.dusk.items(), self.sun.dawn.items())):
            desc, times = zip(*twilight)
            ax.axvspan(*Time(times).plot_date, color=str(0.25 * (3 - i)))
            for words, t in zip(desc, times):
                self.vertical_txt(words, t, color=str(0.33 * i))

        # Indicate moonrise / set
        for rise_set, time in self.moon.up.items():
            ax.axvline(time.plot_date, c='y', ls='--')
            self.vertical_txt(rise_set, time, y='bottom', color='y')

        # TODO: enable picking for sun / moon
        sun_pl, = ax.plot(self.tp, self.sun.coords.alt,
                          'orangered', ls='none', markevery=2,
                          marker='o', ms=10,
                          label='sun')
        moon_pl, = ax.plot(self.tp, self.moon.coords.alt,
                           'yellow', ls='none', markevery=2,
                           marker=self.moon.get_marker(), ms=10,
                           label='moon ({:.0%})'.format(self.moon.illumination))

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

        just_before_sunset = (self.sun.set - 0.25 * u.h).plot_date
        just_after_sunrise = (self.sun.rise + 0.25 * u.h).plot_date
        ax.set_xlim(just_before_sunset, just_after_sunrise)
        ax.set_ylim(-10, 90)

        # HACK
        # qq = aux_trans.transform(np.c_[ax.get_xlim(), ax.get_ylim()])
        # ax.parasite.viewLim.intervalx = qq[:,0]

        # which part of the visibility curves are visible within axes
        self._lt = (just_before_sunset < self.tp) & (self.tp < just_after_sunrise)

        # labels for axes
        ax.set_ylabel('Altitude', fontweight='bold')
        ax.parasite.set_ylabel('Airmass', fontweight='bold')

        # sun / moon legend
        leg = self.ax.legend(bbox_to_anchor=(1.05, 0), loc=3,
                             borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')
        self.ax.add_artist(leg)

        ax.grid()

    # ====================================================================================================
    # @staticmethod
    def vertical_txt(self, s, t, y=1, **kw):
        va = 'top'
        if y == 'top':
            y, va = 1, y
        if y == 'bottom':
            y, va = 0, y

        ax = self.ax
        txt = ax.text(t.plot_date, y, '{} {} SAST'.format(s, local_time_str(t)),
                      rotation=90, ha='right', va=va,
                      transform=btf(ax.transData, ax.transAxes),
                      clip_on=True,
                      **kw)
        return txt


    def site_info_txt(self):
        # eg:
        lat = self.siteLoc.lat  # itude
        lon = self.siteLoc.lon  # gitude
        ns = 'NS'[bool(lat > 0)]
        ew = 'WE'[bool(lon > 0)]

        lat = abs(lat).to_string(precision=0, format='latex')
        lon = abs(lon).to_string(precision=0, format='latex')
        h = '{0.value:.0f} {0.unit:s}'.format(self.siteLoc.height)
        return '{} @ {} {}; {} {}; {}'.format(self.siteName, lat, ns, lon, ew, h)

    # ====================================================================================================
    @staticmethod
    def date_info_txt(t):
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dayname = weekdays[t.datetime.weekday()]
        datestr = t.iso.split()[0]
        return ', '.join((dayname, datestr))


    def make_time_ticks(self):
        # Create ticklabels for SAST
        ax = self.ax
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklocs = ax.xaxis.get_majorticklocs()
        fmt = ax.xaxis.get_major_formatter()
        sast_lbls = list(map(fmt, (Time(num2date(xticklocs)) + self.tz).plot_date))

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


    def plot_vis(self, **kws):  # MAYBE option for colour coding azimuth
        ax = self.ax

        cmap = kws.get('cmap')
        colours = kws.get('colours', [])
        sort = kws.get('sort', True)

        self.set_colour_cycle(colours, cmap)
        names, coords = zip(*self.targetCoords.items())
        if sort:
            coords, names = sorter(coords, names, key=lambda coo: coo.ra)

        for name in names:
            self.add_curve(name)

        # self.plots += [sun_pl, moon_pl]

        self.do_legend()


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
        # TODO: maybe have option for shortening the J---- names
        # if show_coords:
            # add amsmath so we can make short minus in math mode
            # from matplotlib import rcParams
            # rcParams["text.latex.preamble"] = [r'\usepackage{amsmath}']#,
                                               #r'\mathchardef\mhyphen="2D']

        labels = []
        for name, coo in self.targetCoords.items():
            if show_coords:
                name = name.replace(' ', '\:')#.replace('-', r'\mbox{-}')  # math mode for boldness, thus need special space
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
        leg = self.ax.legend(self.plots.values(), labels,
                             bbox_to_anchor=(1.05, 1), loc=2,
                             borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')

        # if show_coords:
        #     for txt in leg.texts:
        #         txt.set_usetex(True)  # TODO: mpl feature request to usetex in legend

        self.legLineMap = {}
        for legline, origline in zip(leg.get_lines(), self.plots.values()):
            legline.set_picker(5)  # 5 pts tolerance
            self.legLineMap[legline] = origline

        self.legLineMapInv = dict(zip(self.legLineMap.values(), self.legLineMap.keys()))

        # re-space legend on next draw
        # connect = self.figure.canvas.mpl_connect
        # self._lfcid = connect('draw_event', self.fix_legend)


    def fix_legend(self, event=None):
        # Shift axes position to fit the legend nicely (long names get clipped)

        logging.info('Fixing Legend')

        leg = self.ax.get_legend()
        bb = leg.get_window_extent(self.figure._cachedRenderer)
        c = bb.corners()[[0, 2]]  # lower left, lower right of legend in display coordinates
        xyAx = self.ax.transAxes.inverted().transform(c)
        wAx = np.diff(xyAx[:, 0])  # textbox width in axes coordinates
        right = 1 - wAx * 1.05
        self.figure.subplots_adjust(right=right)

        # write name width and axes right position to file - eventually we can use this to guess right and avoid this function
        with (moduleDir / '.legend.fix').open('a') as fp:
            longest_name = max(map(len, self.targetCoords.keys()))
            fp.write(str((longest_name, right)))
            fp.write('\n')

        # self.figure.canvas.mpl_disconnect(self._lfcid)

    def _on_pick(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = self.legLineMap[legline]
        texts = self.labels[origline.get_label()]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        for txt in texts:
            txt.set_visible(vis)

        # Change the alpha on the line in the legend so we can see what lines have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)

        self.figure.canvas.draw()  # TODO: blit


    def highlight(self, legLine, plotLine, factor=2):
        """ """
        # FIXME: stays permenantly highlighted when clicking to remove

        legLine.set_lw(legLine.get_lw() * factor)  # thicken legend line
        plotLine.set_lw(plotLine.get_lw() * factor)  # thicken track line
        plotLine.set_zorder(plotLine.get_zorder() * factor)  # plot over everything else

        self.figure.canvas.restore_region(self.background)
        renderer = self.figure.canvas.get_renderer()
        plotLine.draw(renderer)
        legLine.draw(renderer)
        self.figure.canvas.blit(self.figure.bbox)


    def _on_motion(self, event):  # TODO: move this to the DynamicLegend class!!!!!!

        leg = self.ax.get_legend()
        in_legend, props = leg.contains(event)
        hit = False
        if in_legend:
            # TODO: save background here.. including nowLine so that it remains when we blit

            # t = time.time()
            # print('IN LEGEND', t)
            for legLine in leg.get_lines():
                name = legLine.get_label()
                hit, props = legLine.contains(event)
                if hit:
                    # print('HIT', name)
                    break

            if hit and self.highlighted != name:  # new track to highlight
                # print('NEW HIGHLIGHT')
                self.highlighted = name
                track = self.plots[name]
                self.highlight(legLine, track)

        # not hovering over anything, but something remains highlighted
        if self.highlighted and not hit:
            # print('RESTORE', self.highlighted)
            track = self.plots[self.highlighted]
            legLine = self.legLineMapInv[track]
            self.highlight(legLine, track, 0.5)  # restore
            self.highlighted = None


    def _on_first_draw(self, event):
        # HACK! this method creates the SAST labels upon the first call to draw
        logging.debug('FIRST DRAW')
        fig = self.figure
        canvas = fig.canvas

        self.make_time_ticks()
        self.nowLineThread.start()

        # fix the legend if necessary
        if self.tracks:  # some tracks have been added ==> we have a legend
            self.fix_legend()

        # disconnect callback to this function
        canvas.mpl_disconnect(self.cid)

        # connect callback for saving background
        self.cid = self.figure.canvas.mpl_connect('draw_event', self.save_background)
        # canvas.draw()


    def save_background(self, event):

        logging.debug('save_background')
        # print(vars(event))

        # if self.use_blit:
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)  # save the background for blitting


    def closing(self, event):
        self.nowLineThreadAlive.set()


    def connect(self):
        # Setup legend picking
        self.figure.canvas.mpl_connect('pick_event', self._on_pick)
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.figure.canvas.mpl_connect('close_event', self.closing)
