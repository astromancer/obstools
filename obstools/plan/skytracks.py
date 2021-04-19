"""
Tools for visualising object tracks across the night sky
"""


# std libs
from astropy.utils import lazyproperty
import functools as ftl
from PyQt5 import QtCore
from recipes.logging import LoggingMixin
import time
import inspect
import logging
import threading
import itertools as itt
from pathlib import Path
from functools import partial
from datetime import datetime
from collections import OrderedDict, defaultdict

# third-party libs
import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates.name_resolve import NameResolveError
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz, get_sun,
                                 get_moon, jparser)
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.path import Path as mplPath
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import (AutoDateFormatter, AutoDateLocator, num2date,
                              get_epoch)
from matplotlib.transforms import (Transform, IdentityTransform, Affine2D,
                                   blended_transform_factory as btf)
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from addict.addict import Dict
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from recipes import memoize
from recipes.containers.dicts import DefaultOrderedDict
from scrawl.ticks import DegreeFormatter, TransFormatter
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

# local libs
# from motley import profiling
from recipes.containers.lists import sortmore
import recipes.pprint as ppr
from recipes.string import rreplace
from ..utils import get_coordinates, get_site
from .limits import TelescopeLimits
import more_itertools as mit
from .utils import nearest_midnight_date, get_midnight


# TODO: enable different projections, like Mercator etc...
# FIXME: ghost lines in the legend after remove_target

# cache
cachePath = Path.home() / '.cache/obstools'  # NOTE only for linux!
celestialCache = cachePath / 'celestials.pkl'
frameCache = cachePath / 'frames.pkl'

#
TODAY = nearest_midnight_date()
HOME_SITE = 'SAAO'
TIMEZONE = +2 * u.hour  # SAST is UTC + 2
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
SECONDS_PER_DAY = 86400
SIDEREAL_RATE = (366.24 / 365.24)
# A mean sidereal day is 23 hours, 56 minutes, 4.0916 seconds
# (23.9344699 hours or 0.99726958 mean solar days)
HOUR_RANGE = (-12, 12)  # default range for plotting

# '\degree' symbol for latex
rcParams['text.latex.preamble'] = r'\usepackage{gensymb}'


def site_info_txt(site, tel):
    # eg:
    tel_name = f'{tel:g}m' if tel else ''
    return (f'{site.name} {tel_name} @ '
            f'{dms(site.lat, "NS")}; {dms(site.lon, "WE")}; '
            f'{site.height.value:.0f} {site.height.unit}')


def date_info_txt(t):
    # FIXME: use t.strftime
    return ', '.join((WEEKDAYS[t.datetime.weekday()],
                      t.iso.split()[0]))


def hms(angle, **kws):
    return angle.to_string(u.hourangle,
                           **{**dict(precision=0,
                                     format='latex'),
                              **kws})


def dms(angle, cardinal='', **kws):
    # better sexagesimal formatting for angles
    # * put minus outside of math mode so it is rendered smaller (aesthetic)
    # * use \degree from gensymb package (spacing better!)
    # * add space after arcminute symbol (aesthetic)
    sep = ''
    if cardinal:
        cardinal = cardinal[int(angle > 0)]
        angle = abs(angle)
        sep = ' '
    return sep.join((rreplace(angle.to_string(u.deg,
                                              **{**dict(precision=0,
                                                        format='latex'),
                                                 **kws}),
                              {'$-': '-$',
                               r'{}^\prime': r'{{}^\prime\,}',
                               r'^\circ': r'\degree'}),
                     cardinal))


def hmsdms(coords, **kws):
    return f'{hms(coords.ra)}; {dms(coords.dec)}'


def mathbold(string):
    return f'$\\mathbf{{{string}}}$'


def local_time_str(t, precision='m0', tz=TIMEZONE):
    """
    Convert Time `t` to a sexagesimal string representing the number of hours
    since local midnight, eg: "03:14:59"

    Parameters
    ----------
    t : Time
        UTC time
    precision : str, optional
        [description], by default 'm0'
    tz : [type], optional
        [description], by default TIMEZONE

    Returns
    -------
    [type]
        [description]
    """
    t = t + tz - Time(t.isot.split('T')[0])
    t = t.to('s').value % SECONDS_PER_DAY  # prevent times > 24h
    tz_name = 'SAST'
    # from dateutil.tz import tzlocal
    # datetime.now(tzlocal()).tzname()  #  'SAST'
    return f'{ppr.hms(t, precision, sep=":")} {tz_name}'


def vertical_txt(ax, s, t, y=1, precision='m0', **kws):
    """
    Show text `s` in vertical orientation on axes `ax` at time position `t`
    """
    va = 'top'
    if y == 'top':
        y, va = 1, y

    if y == 'bottom':
        y, va = 0.01, y

    s = ' '.join(filter(None, (s, local_time_str(t, precision))))
    return ax.text(t.plot_date, y, s,
                   rotation=90, ha='right', va=va,
                   transform=btf(ax.transData, ax.transAxes),
                   clip_on=True, **kws)
    # TODO: add emphasis if text in twilight region
    # import matplotlib.patheffects as path_effects
    # txt.set_path_effects(
    #         [path_effects.Stroke(linewidth=2, foreground='black'),
    #          path_effects.Normal()])


def sidereal_transform(t, longitude):
    """
    Initialize matplotlib transform for local time -> sidereal time conversion

    Parameters
    ----------
    t : Time
        Reference time
    longitude : float
        Longitude of the observer

    Returns
    -------
    matplotlib.transforms.Affine2D
        Transformation from local time to sidereal time
    """

    # midnight = Time(date)  # midnight UTC
    mid_sid = t.sidereal_time('mean', longitude)
    # offset from local time
    # used to convert to origin of plot_date coordinates
    p0 = t.plot_date
    # xy0 = (-p0, 0)
    # xy1 = (p0 + mid_sid.hour / 24, 0)

    # scale = 366.24 / 365.24
    return Affine2D().translate(-p0, 0).scale(SIDEREAL_RATE).translate(
        p0 + mid_sid.hour / 24, 0).inverted()


def short_name(name):
    if jparser.search(name):
        return jparser.shorten(name)
    return name


def set_visible(artists, state=True):
    for art in mit.collapse(artists):
        art.set_visible(state)


# ******************************************************************************
@memoize.to_file(celestialCache)
class CelestialTrack:
    """
    Class representing the visibility of a celestial body on a given date at a
    specific site
    """

    # whether to abbreviate target names containing J coordinates
    shorten_Jnames = True
    n_points = 100
    name = short_name = None    # place holders

    def __init__(self, site=HOME_SITE, date=TODAY, hours=HOUR_RANGE,
                 n_points=n_points, limits=None):
        """
        Create the track for this celestial body on the `date` at `site`.

        This constructor uses a persistant memoization cache to optimize
        computation of on-sky positions on a certain date with respect to a
        given site.

        Parameters
        ----------
        site : str or EarthLocation
            The site of the observer
        date : str
            The date on which to compute the position for the object

        """

        self.site = get_site(site)
        self.range = hours  # np.add(hours, -TIMEZONE.value)
        self.n_points = int(n_points)
        # self.coords = self.get_coords()
        self.set_date(date)
        self.limits = limits

        # art
        self.curves = []
        self.labels = []

    def __iter__(self):
        """Yields all the matplotlib arists associated with this track"""
        yield from (self.curves, self.labels)

    @lazyproperty
    def coords(self):
        raise NotImplementedError  # subclass should overwrite

    @property
    def art(self):
        return self.curves, self.labels

    def set_date(self, date):
        self.date, self.midnight, self.mid_sid = get_midnight(
            date, self.site.lon)

        # Compute the trajectory
        # WARNING: next line slow!!!!
        self.track = self.get_track(self.range, self.n_points)
        self.interpolator = interp1d(self.hours, self.track.alt.degree)

    def get_track(self, hours=HOUR_RANGE, n_points=n_points):
        # TODO: multiprocess for performance ?
        self.hours = np.linspace(*hours, n_points)
        t = self.midnight + self.hours * u.hour
        frames = AltAz(obstime=t, location=self.site)
        return self.coords.transform_to(frames)

    def rises(self):
        return self.get_rise_set()['rise'][0]

    def sets(self):
        return self.get_rise_set()['set'][0]

    # @memoize.to_file()
    def get_rise_set(self, altitude=(0, )):
        """get rising and setting times"""

        # Function to help calculate rise set times as well as times for
        # civil / nautical / astronomical twilight

        # site=None, date=TODAY,
        # site = site or self.site
        # date = date or self.date

        # We interpolate the calculated positions to get rise/set times.
        # Should still be accurate to ~1s and is fast (x10 times faster than
        # astroplan.Observer.get_sun_rise for roughly the same accuracy)
        # def crossing(hour, altitude=0):
        #     return interpolator(hour) + altitude

        # _, midnight, _ = get_midnight(date, get_site(site).lon)
        # frames = get_frames(date, site, hours, n_points)
        # alt = self.coords.transform_to(frames).alt.degree  # SLOW
        # track = self.get_track(hours, n_points)
        alt = self.track.alt.degree
        # hours = (track.obstime - midnight).to('h').value
        # interpolator = interp1d(hours, alt)
        # solver = partial(brentq, crossing)

        # find horizon crossing interval (index of sign changes)
        up = {}
        for i, word in enumerate(['rise', 'set']):
            times = []
            for x in np.atleast_1d(altitude):
                altx = alt - x
                # check for sign changes:
                wsc, = np.where(np.diff(np.sign(altx)) == [2, -2][i])
                if not wsc.any():
                    # never rises or never sets in interval
                    times.append(None)
                    continue

                # hour interval when object crosses horizon
                interpolator = interp1d(self.hours, altx)
                hcross = brentq(interpolator, *self.hours[wsc + [-1, 1]])
                times.append(self.midnight + hcross * u.hour)

            if len(times) == 1:
                times = times.pop()
            up[word] = times

        return up

    def get_visible_ha(self, where='both', which='soft'):
        return self.limits.get_visible_ha(
            self.coords.dec.deg, 'both', which)

    def ha_to_ut(self, ha):
        """Time UTC when this source is at hour angle `ha`"""
        return self.midnight + TimeDelta(self.ha_to_hour(ha) / 24)

    def ha_to_hour(self, ha):
        """
        Local time in hours from midnight when this source is at hour angle `ha`
        """
        sidt = self.coords.ra.hour + ha
        h = (sidt - self.mid_sid.value) / SIDEREAL_RATE
        # return (h + 24) % 24
        # if np.any(h > 12):
        #     return h - 24
        if np.any(h < -12):
            h += 24
        if np.any(h > 12):
            h -= 24
        # h[h < -12] += 24
        return h

    def plot(self, ax, annotate=True, **kws):
        # site=HOME_SITE, date=TODAY, limits=None,
        # track = self.get_track(hours, n_points)

        # site = get_site(site)
        # _, midnight, mid_sid = get_midnight(date, site.lon)
        # frames = get_frames(date, site, hours, n)
        # altaz = self.coords.transform_to(frames)
        # y = altaz.alt
        # t = frames.obstime.plot_date

        t = self.track.obstime.plot_date
        y = self.track.alt.degree
        i00, i11 = 0, -1
        colour = None

        art = self.curves = []
        kws = {**dict(color=None), **kws}
        if self.limits:
            # TODO: this part could be in ObjTrack.
            # get critical hour angles
            ha = np.ravel([self.get_visible_ha('both', which)
                           for which in ('hard', 'soft')])
            ha.sort()

            # get local hour of critical time
            h = np.clip(self.ha_to_hour(ha), *self.range)
            t_crit = self.ha_to_ut(ha).plot_date
            y_crit = self.interpolator(h)
            idx = np.hstack([np.digitize(t_crit, t), len(t)])

            i0, t0, y0 = 0, None, None
            yy, tt = defaultdict(list), defaultdict(list)
            for i, (i1, t1, y1) in enumerate(
                    itt.zip_longest(idx, t_crit, y_crit), -2):
                # get line segments
                i = abs(i)
                tt[i].extend(mit.collapse([t0, t[i0:i1], t1, None]))
                yy[i].extend(mit.collapse([y0, y[i0:i1], y1, None]))
                i0, t0, y0 = i1, t1, y1

            # main / soft / hard
            label = self.name
            for i, dashes in enumerate(((), (4, 2), (2, 7))):
                # plot
                t, y = tt[i],  yy[i]
                line, = ax.plot(t, y,
                                **{**kws,
                                   # mark critical points
                                   **dict(label=label,
                                          dashes=dashes,
                                          marker='o',
                                          markevery=len(t) - 2,
                                          mfc=['none', colour][bool(i - 1)],
                                          mew=1.5,
                                          ms=5.5)})
                colour = kws['color'] = line.get_color()
                art.append(line)
                label = None

        else:
            # main track
            kws.setdefault('ls', '-')
            art.extend(ax.plot(t, y, **kws))
            colour = art[-1].get_color()

        if annotate:
            self.annotate(ax, color=colour)
        return art[::-1]  # reorder so main track is first (for legend)

    def annotate(self, ax, **kws):
        """
        Add target name to visibility curve - helps more easily distinguish
        targets when plotting numerous curves
        """

        # remove labels if pre-existing
        for text in self.labels:
            text.remove()

        t = self.track.obstime.plot_date
        y = self.track.alt.degree

        # find where the curve intersects the edge of the axes
        y0, y1 = ax.get_ylim()
        x0, x1 = ax.get_xlim()
        # region within axes (possibly multiple sections for tracks that pass
        # out, and then in again)
        l = (((y0 < y) & (y < y1)) &
             ((x0 < t) & (t < x1))).astype(int)
        # first points inside axes
        # last points inside axes (")
        (first,), (last, ) = (np.where((l - np.roll(l, i)) == 1)
                              for i in (1, -1))

        # same colour for line and text
        # colour = self.plots[name].get_color()
        kws = {**dict(size=10,
                      fontweight='black',
                      rotation_mode='anchor',
                      clip_on=True,
                      # color=colour,
                      # animated=True
                      ), **kws}

        # decide whether to add one label or two per segment
        labels = []
        for i, (i0, i1) in enumerate(zip(first, last)):
            # determine the slope and length of curve segments at all points
            #  within axes
            x, yy = t[i0:i1 + 1], y[i0:i1 + 1]
            dx, dy = np.diff((x, yy))
            length = np.sqrt(np.square(np.diff([x, yy])).sum(0))
            angles = np.degrees(np.arctan2(dy, dx))
            angles = ax.transData.transform_angles(
                angles, np.c_[x, yy][:-1])
            # TODO: BUGREPORT transform_angles.
            # docstring says: 'The *angles* must be a column vector (i.e., numpy
            # array).' this is wrong since the function *DOES NOT WORK* when
            # angles is column vector!!

            # create the text
            text = ax.text(x[0], yy[0] + 0.5, self.short_name,
                           ha='left', rotation=angles[0], **kws)
            labels.append(text)
            # get textbox lower left, upper right corners
            bb = text.get_window_extent(ax.figure.canvas.get_renderer())
            xyc = ax.transData.inverted().transform(bb.corners()[[0, -1]])
            # length of rendered text in data space
            text_length = np.sqrt(np.square(np.diff(xyc.T)).sum())
            # space = segL.sum()  # curve length of track segment
            # if entry & exit point are close to oe another, only make 1 label
            if text_length * 2 < length.sum():
                # second label for segment
                text = ax.text(x[-1], yy[-1] + 0.5, self.name,
                               ha='right', rotation=angles[-1], **kws)
                labels.append(text)

        # TODO: Curved text
        # - interpolators for each track
        # split text at n positions, where n is decided based on str length
        # place each text segment at position along line with angle of slope at
        #  that position
        self.labels = labels
        return labels


class ObjTrack(CelestialTrack):

    n_points = CelestialTrack.n_points

    def __init__(self, name, coords=None, site=HOME_SITE, date=TODAY,
                 hours=HOUR_RANGE, n_points=n_points, limits=None):  #
        self.name = name
        self.short_name = short_name(name)
        self._coords = coords

        #
        super().__init__(site, date, hours, n_points, limits)

    # def set_limits(self, tel):
    #     # set observing limits for site / telescope
    #     if tel:
    #         self.limits = TelescopeLimits(tel)

    @lazyproperty
    def coords(self):
        # resolve coordinates
        # TODO: build raises=True, so you can optionally pass
        coords = get_coordinates(self.name if self._coords is None
                                 else self._coords)
        if coords is None:
            raise ValueError(f'Could not resolve coordinates {coords!r}')

        return coords


class EclipticBody(CelestialTrack):

    plot_kws = dict(ls='none',
                    ms=10)

    @lazyproperty
    def coords(self):
        self.hours = np.linspace(*self.range, self.n_points)
        t = self.midnight + self.hours * u.hour
        return self.get_coords(t)

    def plot(self, ax, annotate=False, **kws):
        return super().plot(ax, annotate, **{**self.plot_kws, **kws})


class Sun(EclipticBody):
    """
    Object that encapsulates the visibility of the sun on a given date at a
    specific location
    """
    name = 'sun'
    n_points = 100
    get_coords = staticmethod(get_sun)
    plot_kws = {**EclipticBody.plot_kws,
                **dict(color='orangered',
                       marker='o')}

    def __init__(self, site=HOME_SITE, date=TODAY, hours=HOUR_RANGE,
                 n_points=n_points):
        #
        super().__init__(site, date, hours, n_points)

        # get dawn / dusk times
        self.dusk, self.dawn = self.get_rise_set()
        self.set, self.rise = self.dusk['sunset'], self.dawn['sunrise']

    def get_rise_set(self, altitude=(0, -6, -12, -18)):
        """calculate dawn / dusk / twilight times"""

        rise_set = super().get_rise_set(altitude)

        # angles = np.arange(0, 4) * 6.
        # solver = partial(brentq, self._crossing)
        # hdusk = np.vectorize(solver)(-11.99, 0, angles)
        # hdawn = np.vectorize(solver)(0, 11.99, angles)

        up = OrderedDict(), OrderedDict()
        when = [['set', 'dusk'], ['rise', 'dawn']]
        which = ['sun', 'civil', 'nautical', 'astronomical']
        for i, key in enumerate(sorted(rise_set.keys())[::-1]):
            for k, time in enumerate(rise_set[key]):
                j = bool(k)
                words = (' ' * j).join((which[k], when[i][j]))
                up[i][words] = time
                # set the which string as an attribute on the OrderedDict for
                # convenient access later on
                setattr(up[i], which[k], time)

        return up


class Moon(EclipticBody):
    """
    Object that encapsulates the visibility of the moon for the given frames
    """
    name = 'moon'
    get_coords = staticmethod(get_moon)
    n_points = 100

    def __init__(self, site=HOME_SITE, date=TODAY, hours=HOUR_RANGE,
                 n_points=n_points):
        # get moon rise/set times, phase, illumination etc...
        super().__init__(site, date, hours, n_points)

        # self.midnight_coords = get_moon(self.midnight).transform_to(
        #     AltAz(obstime=self.midnight, location=self.site))

        # compute mignight frame for average illumination
        self.up = self.get_rise_set()
        # moon phase and illumination at local midnight
        self.phase, self.illumination = self.get_phase()

    # @staticmethod
    def get_phase(self, t=None):
        """calculate moon phase and illumination for time `t` at `location`"""

        t = t or self.midnight

        frame = AltAz(obstime=t, location=self.site)
        moon = get_moon(t).transform_to(frame)
        sun = get_sun(t).transform_to(frame)

        # elongation = sun.separation(moon)  # BORKS!!!
        elongation = sun.frame.separation(moon.frame)

        # phase angle at obstime
        phase = np.arctan2(sun.distance * np.sin(elongation),
                           moon.distance - sun.distance * np.cos(elongation))
        illumination = (1 + np.cos(phase)) / 2.0
        return phase.value, illumination.value

    def get_distance(self, coords, t=None):
        t = t or self.midnight
        frame = AltAz(obstime=t, location=self.site)
        moon = get_moon(t).transform_to(frame)
        return moon.frame.separation(coords)

    def get_rise_set(self, altitude=(0, )):
        # make keys 'moonrise', 'moonset'
        return {f'moon{w}': t for w, t in
                super().get_rise_set(altitude).items()}

    def get_marker(self):
        """
        Create a marker for the moon that recreates it's current shape in the
        sky based on the phase

        Returns
        -------
        matplotlib.path.Path
        """
        # if close to full, just return the standard filled circle
        angle = np.pi - self.phase
        if np.abs(angle) % np.pi < 0.1:
            return 'o'

        theta = np.linspace(np.pi / 2, 3 * np.pi / 2)
        xv, yv = np.cos(theta), np.sin(theta)
        # xv, yv = circle.vertices.T
        # lx = xv < 0
        xp = xv * np.cos(angle)

        x = np.r_[xv, xp]
        y = np.r_[yv, yv[::-1]]
        verts = np.c_[x, y]
        codes = np.ones(len(x)) * 4
        codes[0] = 1
        return mplPath(verts, codes)

    def plot(self, ax, annotate=False, **kws):
        return super().plot(ax, annotate,
                            **{**dict(color='yellow',
                                      marker=self.get_marker(),
                                      label=f'moon ({self.illumination:.0%})'),
                               **kws})


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


class Clock(LoggingMixin):
    """
    Vertical line and clock on axes indicating current time
    """

    def __init__(self, ax, t0, lon, precision='s0'):
        #
        self.ax = ax
        self.lon = lon
        self.precision = precision

        # plot line
        # animated=True to prevent redrawing the canvas
        self.line, = ax.plot([t0.plot_date] * 2, [0, 1],
                             ls=':', c='g',
                             transform=btf(ax.transData, ax.transAxes),
                             animated=True)
        self.sast = vertical_txt(ax, '', t0, y='bottom', color='g',
                                 animated=True, fontweight='bold')
        self.sidt = vertical_txt(ax, '', t0, y='top', color='c',
                                 animated=True, fontweight='bold')
        # not visible initially
        # set_visible(self, False)

        # threading Event controls whether
        self.alive = threading.Event()

    def __iter__(self):
        """All artists associated with the clock"""
        yield from (self.line, self.sast, self.sidt)

    def update(self):
        """
        Update the current time line and texts
        """
        # print('updating')

        now = Time.now()
        t = now.plot_date
        # update line position
        self.line.set_xdata([t, t])

        # update SAST text
        sast = local_time_str(now, self.precision)
        self.sast.set_text(sast)
        self.sast.set_position((t, 0.01))

        # update Sid.T text
        sidt = now.sidereal_time('mean', self.lon).hour * 3600
        sidt = f"{ppr.hms(sidt, self.precision, ':')} Sid.T"
        self.sidt.set_text(sidt)
        self.sidt.set_position((t, 1))

        # blit the figure if the current time is within range
        tmin, tmax = self.ax.get_xlim()
        if (tmin < t) & (t < tmax):
            set_visible(self)

        # print('done update')


class ClockWork(QtCore.QObject):
    """Task that updates current timestamp"""

    signal = QtCore.pyqtSignal()

    def __init__(self, func, alive):
        super().__init__()
        # Ensure thread safety with pyqtSignal
        self.lock = threading.Lock()
        self.signal.connect(func)
        self.thread = threading.Thread(
            target=self.run, args=(alive,))

    def run(self, alive, interval=1):
        """thread to update line indicating current time"""
        while alive.is_set():
            self.signal.emit()
            time.sleep(interval)


class VizAxes(SubplotHost):
    """The standard axes class for visibility tracks"""

    # def __init__(self, *args, **kw):

    # self.ytrans = SeczTransform()
    # self._aux_trans = btf(ReciprocalTransform(), IdentityTransform())

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


class SkyTracks(LoggingMixin):
    """
    A tool for plotting visibility tracks of celestial objects to aid
    observational planning and scheduling.

    Initializing the class without arguments will set up the plot for the
    current date and default site (SAAO). Date, site and targets
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
    >>> viz = SkyTracks()
    >>> viz.add_target('FO Aqr')
    >>> viz.add_target('NOI-105276', '20:31:25.8 -19:08:35.0')
    >>> viz.add_targets('SN 1987A', 'M31')
    >>> viz.connect()
    """

    n_points_track = 200

    # FIXME: legend being cut off
    # FIXME Current time thread stops working after hover, and in general sporadically...
    # FIXME: from thread: SystemError: ../Objects/tupleobject.c:85: bad argument to internal function
    # TODO: ephemerides
    # TODO: non-overlapping labels
    # TODO: labels not legible if crossing twilight boundaries. emphasise
    # TODO: figure cross-hairs indicating altitude etc wrap of objects on hover
    # TODO: indicate 'current' distance from moon when hovering
    # TODO: let text labels for tracks curve along the track - long labels
    # TODO: Moon / sun pickable
    # TODO: hover bubble with name of track under mouse
    # TODO: watch_file
    # TODO: set_date !!
    # TODO: scroll through dates

    # @profiling.histogram()

    def __init__(self, targets=None, date=TODAY, site=HOME_SITE,
                 tel=None, tz=TIMEZONE, cmap='gist_ncar', colours=(),
                 use_blit=True):
        """
        Plot visibility tracks for a list of `targets` on a `date` at the an
        observing `site`, optionally including telescope `tel` specific limits.

        Parameters
        ----------
        targets : [type], optional
            [description], by default None
        date : [type], optional
            If given, midnight time of that date will be used. If not given,
            either the upcoming midnight or previous midnight will be used
            depending on the current (call) time. The hour (in local time) which
            the switch from past to future occurs, is by default 9am. This
            ensures that the tracks for the intended date are plotted when using
            this class during observations in the morning hours at the
            telescope. see: `utils.nearest_midnight_date`
        site : [type], optional
            [description], by default HOME_SITE
        tel : [type], optional
            [description], by default None
        tz : [type], optional
            [description], by default TIMEZONE
        cmap : str, optional
            cmap that will be used for track colours, by default 'jet'
        colours : tuple, optional
            [description], by default ()
        """

        # 'gist_ncar', 'rainbow', 'tab20', 'tab20b'
        self.site = get_site(site)
        self.site.name = site if site.isupper() else site.title()

        # from dateutil.tz import tzlocal
        # t = datetime.now(tzlocal())
        self.tz = tz
        # TODO: can you get this from the site location??
        # http://stackoverflow.com/questions/16086962/how-to-get-a-time-zone-from-a-location-using-latitude-and-longitude-coordinates
        # https://developers.google.com/maps/documentation/timezone/start

        self.targets = {}
        self.texts = Dict()
        self.vlines = Dict()
        self.legLineMap = {}

        # local midnight in UTC
        self.date, self.midnight, self.mid_sid = get_midnight(
            date, self.site.lon)

        # Compute track for sun and moon
        # The next two lines are potentially slow running once for every new
        # (date, site) combo
        # getting sun, moon coordinates needs to happen before setup_figure
        self.sun = sun = Sun(site, date)
        self.time_range = Time([sun.set, sun.rise]) + [-1/4, 1/4] * u.hour
        self.hour_range = tuple((self.time_range - sun.midnight).to('h').value)

        # frames = get_frames(date, site, interval, self.n_points_track)
        self.moon = Moon(site, date)

        # Visibility limits for telescope / site
        self.limits = self.tel = None
        if tel:
            self.limits = TelescopeLimits(tel)
            self.tel = self.limits.tel

        # setup figure
        self.one_line_legend = False  # bool(one_line_legend)
        self.figure, self.ax = self.setup_figure()
        self.canvas = self.figure.canvas
        ax = self.ax

        # blitting setup
        self.saving = False  # save background on draw
        self.use_blit = bool(use_blit)
        self.count = 0

        # Plotting done here! Also collect name, coordinates in dict
        self.colours = colours
        self.cmap = get_cmap(cmap)
        if targets is not None:
            self.add_targets(targets)

        #
        self.legend0 = ax.get_legend()
        self._legend_fix_req = False
        self.highlighted = None

        # current time indicator.
        self.clock = Clock(ax, Time.now(), self.site.lon)
        self.clockWork = ClockWork(self.update_clock, self.clock.alive)

        # set all tracks and their labels as well as current time text invisible
        # before drawing, so we can blit
        set_visible((self.art, self.clock), False)

        # HACK
        self.cid = self.canvas.mpl_connect('draw_event', self._on_first_draw)
        self.canvas.mpl_connect('draw_event', self._on_draw)
        # At this point the background with everything except the tracks is saved

    @property
    def plots(self):
        for art in self.targets.values():
            yield art.curves

    @property
    def labels(self):
        for art in self.targets.values():
            yield art.labels

    @property
    def art(self):
        """All artists that need to be redrawn upon interaction"""
        return (self.targets.values(), self.legLineMap)

    def moon_distances(self, t=None):
        """
        Get distane to moon for all active targets

        Parameters
        ----------
        t : Time, optional
            time at which to calculate distances, defaults to local midnight

        Returns
        -------
        dict
            Keyed on source names, values are separation in degrees
        """
        moon = get_moon(t or self.midnight, self.site)
        # moon = self.moon.coords[len(self.moon.coords) // 2]
        return {name: moon.frame.separation(body.coords).value
                for name, body in self.targets.items()}

    def update_clock(self):
        self.clock.update()
        # redraw the canvas
        # with self.clockWork.lock:  # deadlock??
        # blit figure
        self.draw_blit(self.clock, bg=self.background2)

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
        itr = (names.items() if isinstance(names, dict)
               else itt.zip_longest(names, coords))

        for name, coo in itr:
            self.add_target(name, coo, update=False)

        self.apply_colours()
        self.do_legend()
        self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

        # self.cid = self.canvas.mpl_connect('draw_event', self.fix_legend())

    def add_target(self, name, coords=None, update=True):
        if name in self.targets:
            self.logger.info(f'{name} already in target list')
            return

        #
        self.targets[name] = track = ObjTrack(name, coords,
                                              self.site.name,
                                              self.midnight,
                                              self.hour_range,  # HOUR_RANGE
                                              self.n_points_track,
                                              self.limits)
        track.plot(self.ax)

        if update:
            self.apply_colours()
            self.do_legend()
            self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

    def remove_target(self, name):
        """
        Remove the visibility track of target with name `name` from the plot

        Parameters
        ----------
        name : str
            The object name
        """
        obj = self.targets.pop(name, None)
        if obj is not None:
            for art in mit.collapse(obj.art):
                art.remove()

    def _on_resize(self, event):

        # this will happen before draw
        self.save_background()

        # redo the track labels (angles will have changed)
        for name in self.targets.keys():
            self.add_annotation(name)

        # now the canvas will redraw automatically
        set_visible((self.art, self.clock))
        self.saving = True

    def setup_figure(self, figsize=(15, 7.5)):

        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(right=0.8,  # 0.65 if self.one_line_legend else 0.8,
                            top=0.94,
                            left=0.05,
                            bottom=0.075)
        # setup axes with
        ax = VizAxes(fig, 111)
        aux_trans = btf(sidereal_transform(self.midnight, self.site.lon),
                        IdentityTransform())
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
        colours = c, *_ = ['y', 'y', 'orange', 'orange']
        # in case the `moon.up` dict is empty
        for rise_set, time in moon.up.items():
            if time is None:
                continue

            # pick colours that are readable against grey background
            for i, (r, s) in enumerate(intervals):
                if r < time < s:
                    c = colours[i]
                    break

            # vertival line for moonrise
            self.vlines[rise_set] = ax.axvline(time.plot_date, c=c, ls='--')
            self.texts[rise_set] = vertical_txt(
                ax, rise_set, time, y='bottom', color=c, fontweight='bold')

        # TODO: enable picking for sun / moon
        sun_pl = sun.plot(ax)  # markevery=2,
        moon_pl = moon.plot(ax)

        # site / date info text
        kws = dict(fontweight='bold', transform=ax.transAxes)
        ax.text(0, 1.04, date_info_txt(self.midnight),
                ha='left', **kws)
        ax.text(1, 1.04, site_info_txt(self.site, self.tel),
                ha='right', **kws)

        # setup axes
        # dloc = AutoDateLocator()
        # ax.xaxis.set_major_locator(dloc)
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
        # ax.yaxis.set_major_formatter(DegreeFormatter())
        # ax.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        # ax.yaxis.set_minor_formatter(DegreeFormatter())

        # qrth = 0.25 * u.hour
        # just_before_sunset = (sun.set - qrth).plot_date
        # just_after_sunrise = (sun.rise + qrth).plot_date
        ax.set_xlim(*self.time_range.plot_date)
        ax.set_ylim(-5, 90)

        # HACK
        # qq = aux_trans.transform(np.c_[ax.get_xlim(), ax.get_ylim()])
        # ax.parasite.viewLim.intervalx = qq[:,0]

        # which part of the visibility curves are visible within axes
        # self._lt = ((just_before_sunset < self.tp) &
        #             (self.tp < just_after_sunrise))

        # labels for axes
        ax.set_ylabel('Altitude', fontweight='bold')
        ax.parasite.set_ylabel('Airmass', fontweight='bold')

        # legend at top for moon
        mc, = moon.curves
        mc.set_mec('k') # Give moon legend marker a black edge for aesthetic
        leg = ax.legend(bbox_to_anchor=(1.05, 1.015), loc=3,
                        borderaxespad=0., frameon=True)
        mc.set_mec(mc.get_mfc())               
        leg.get_frame().set_edgecolor('k')
        ax.add_artist(leg)

        ax.grid(ls=':')
        return fig, ax

    def set_cmap(cmap):
        self.cmap = plt.get_cmap(cmap)
        self.set_colours()

    def get_colours(self):
        if len(self.colours):
            return self.colours

        cm = plt.get_cmap(self.cmap)
        return cm(np.linspace(0, 1, len(self.targets)))

    def set_colours(self, colours):
        # colour sequence insufficient for uniqueness
        nc, nt = len(colours), len(self.targets)
        if nc < nt:
            self.logger.info('Given colour sequence less than number of '
                             f'plots ({nc} < {nt}). Colours will repeat.')

        self.colours = colours
        self.apply_colours()

    def apply_colours(self):
        for c, target in zip(self.get_colours(), self.targets.values()):
            for artist in mit.collapse(target.art):
                artist.set_color(c)

        # self.ax.set_prop_cycle(color=)

    def make_time_ticks(self):
        # Create ticklabels for SAST
        ax = self.ax
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklocs = ax.xaxis.get_majorticklocs()
        fmt = ax.xaxis.get_major_formatter()
        sast_lbls = map(fmt, xticklocs + self.tz.value / 24.)
        for sast, tck in zip(sast_lbls, xticklabels):
            # FIXME: does not update when zooming / panning!  better to have
            # another parasite axes
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

    # def plot(self, **kws):
    #     # TODO: MAYBE option for colour coding azimuth
    #     ax = self.ax

    #     cmap = kws.get('cmap')
    #     colours = kws.get('colours', [])

    #     self.set_colour_cycle(colours, cmap)
    #     names, coords = zip(*self.targets.items())

    #     for name in names:
    #         self.plot_track(name)

    #     # self.plots += [sun_pl, moon_pl]

    #     self.do_legend()

    def do_legend(self, show_coords=True, **kws):
        """
        Create a legend with pickable elements to toggle visibility of tracks.
        Objects will be sorted by right ascension so that those that rise first
        will appear on top of the list.
        """
        # TODO: maybe use your DynamicLegend class ??
        labels = self.get_legend_labels(show_coords, self.one_line_legend)
        coords = self.targets.values()
        lines, *_ = zip(*self.plots)
        lkw = dict(marker='o' if self.limits else None, ls='-', ms=5.5)
        proxies = [Line2D([], [], color=line.get_color(), **lkw)
                   for line in lines]

        _, proxies, labels = sortmore(coords, proxies, labels,
                                      key=lambda cx: cx.coords.ra.deg)

        leg = self.ax.legend(proxies, labels,
                             **{**dict(loc=2,
                                       bbox_to_anchor=(1.05, 1),
                                       borderaxespad=0.,
                                       frameon=True,
                                       fontsize=10,
                                       numpoints=2),
                                **kws})
        leg.get_frame().set_edgecolor('k')

        #
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_pickradius(10)
            legline.set_picker(True)
            self.legLineMap[legline] = origline

        self.legLineMapInv = dict(
            zip(self.legLineMap.values(), self.legLineMap.keys()))

        # re-space legend on next draw
        # connect = self.canvas.mpl_connect
        # self._lfcid = connect('draw_event', self.fix_legend)

        self._legend_fix_req = True

    def get_legend_labels(self, show_coords=True, one_line=True):
        # using latex math mode for bold names:
        # * transform spaces to math mode
        # * escape underscores in names to prevent accidental subscripts
        # * put semi-colon between ra and dec

        name_fixes = {'_': '\\_',
                      ' ': r'\:'}

        labels = []
        sep = ': ' if one_line else '\n'
        for name, track in self.targets.items():
            if show_coords:
                lbl = sep.join(
                    (mathbold(rreplace(track.short_name, name_fixes)),
                     hmsdms(track.coords)))
            else:
                lbl = track.short_name
            labels.append(lbl)
        return labels

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
        #     longest_name = max(map(len, self.targets.keys()))
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
        self.draw_blit(self.art)

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
        self.clock.alive.clear()  # suspend the current time thread
        self.clock.update()
        if draw:
            self.draw_blit(self.art)

        # re-activate the current time thread
        self.clock.alive.set()

    def _on_draw(self, event):
        self.count += 1
        if self.saving:
            self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)

        # # fix the legend spacing if necessary
        # if self._legend_fix_req:
        #     self.fix_legend()
        #     self._legend_fix_req = False

    def _on_first_draw(self, event):
        # This method creates the SAST labels upon the first call to draw
        # as well as starting the currentTime.thread
        # print('FIRST DRAW')
        fig = self.figure
        canvas = fig.canvas

        # disconnect callback to this function
        canvas.mpl_disconnect(self.cid)

        # ticks
        self.make_time_ticks()

        # save background without tracks
        self.save_background()
        set_visible(self.art, True)
        self.draw_blit(*self.art)
        # clock intentionally not drawn here - will draw in thread
        # save background with tracks
        self.background2 = self.canvas.copy_from_bbox(self.figure.bbox)
        # canvas.draw()

    def save_background(self, event=None):
        # save the background for blitting
        self.logger.debug('save_background')
        # make tracks invisible
        set_visible((self.art, self.clock), False)
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.figure.bbox)

    def draw_blit(self, *artists, bg=None):
        # print('draw_blit')
        self.canvas.restore_region(bg or self.background)
        renderer = self.canvas.get_renderer()
        for art in mit.collapse(artists):
            art.draw(renderer)
        self.canvas.blit(self.figure.bbox)

    def close(self, event=None):
        # stop the currentTime.thread
        self.clock.alive.clear()
        self.clockWork.thread.join(3)
        # NOTE: will potentially wait for `interval` seconds before closing

    def connect(self):
        # Setup legend picking
        # self.canvas.mpl_connect('pick_event', self._on_pick)
        # self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('close_event', self.close)
        self.canvas.mpl_connect('resize_event', self._on_resize)

        # start clock
        self.clock.alive.set()
        self.clockWork.thread.start()
