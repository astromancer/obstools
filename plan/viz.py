# coding: utf-8
from collections import OrderedDict

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from functools import partial

from datetime import datetime

#Import the packages necessary for finding coordinates and making coordinate transformations
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (SkyCoord, EarthLocation, AltAz,
                                 get_sun, get_moon)
from astropy.coordinates.name_resolve import NameResolveError

from obstools.jparser import Jparser

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.path import Path
from matplotlib.ticker import Formatter, AutoMinorLocator
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, date2num, num2date 
from matplotlib.transforms import (Transform, IdentityTransform, Affine2D, 
                                   blended_transform_factory as btf)
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

from grafico.ticks import DegreeFormatter, TransFormatter
from recipes.list import sorter

#from decor.misc import persistant_memoizer
from decor.profile import profile
profiler = profile()        #truncate_lines=50

#import json

#====================================================================================================
#coordinate_cache = '/home/hannes/work/obstools/plan/.coordcache'
#@persistant_memoizer(coordinate_cache)
#FIXME:TypeError: <SkyCoord (ICRS): (ra, dec) in deg (223.421083, -55.36075)>
#is not JSON serializable


def resolver(name):
    '''Get the target coordinates if known'''
    #try extract J coordinates from name
    try:
        return Jparser(name).skycoord()
    except ValueError:
        pass
    
    try:
        #uses Simbad to resolve object names and retrieve coordinates.
        return SkyCoord.from_name(name) 
    except NameResolveError as err:
        #name not in SIMBAD database
        warnings.warn(str(err))
        
        #NOTE: DO NOT MEMOIZE!!!!!!!!!!!!



#====================================================================================================
#def OOOOh(t):
    #'''hours since midnight'''
    #return (t - midnight).sec / 3600

#====================================================================================================
def local_time_str(t, tz=2*u.hour):
    return (t+tz).iso.split()[1].split('.')[0]

#====================================================================================================
def get_sid_trans(date, longitude):
    '''Initialize matplotlib transform for local time - sidereal time conversion'''
    midnight = Time(date)   #midnight UTC
    sidmid = midnight.sidereal_time('mean', longitude)
    #offset from local time
    offset = sidmid.hour / 24 
    #used to convert to origin of plot_date coordinates
    p0 = midnight.plot_date
    #A mean sidereal day is 23 hours, 56 minutes, 4.0916 seconds (23.9344699 hours or 0.99726958 mean solar days)
    scale = 366.25 / 365.25
    
    return Affine2D().translate(-p0, 0).scale(scale).translate(p0 + offset, 0)
    
    
    
#****************************************************************************************************
class SeczTransform(Transform): 
    input_dims = 1 
    output_dims = 1 
    is_separable = False 
    has_inverse = False

    def transform_non_affine(self, alt): 
        return 1. / np.cos(np.radians(90. - alt))

    
#****************************************************************************************************
class SeczFormatter(TransFormatter):
    _transform = SeczTransform()
    
    

class VizAxes(SubplotHost):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #def __init__(self, *args, **kw):
        
        ##self.ytrans = SeczTransform()
        ##self._aux_trans = btf(ReciprocalTransform(), IdentityTransform())
        
        #kws.pop('site')
        
        #date = '2016-07-08'
        #lon = viz.siteloc.longitude
        #sid_trans = get_sid_trans(date, lon, 2)
        #aux_trans = btf(sid_trans, IdentityTransform())
        
        #SubplotHost.__init__(self, *args, **kw)
        #self.parasite = self.twin(aux_trans)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_ticks(self):
        
        #Tick setup for both axes
        minorTickSize = 8
        for axis in (self.yaxis, self.parasite.yaxis):
            axis.set_tick_params('both', tickdir='out')
            #axis.set_tick_params('minor', labelsize=minorTickSize, pad=0)
        
        #TODO:  colors='' #For sidereal time axis
        
        self.xaxis.set_tick_params('major', pad=10)
        self.yaxis.set_tick_params('minor', labelsize=6, pad=5)
        
        
        #Tick setup for main axes
#         self.xaxis.set_tick_params('major', pad=10)
#         self.yaxis.set_tick_params('minor', labelsize=6, pad=5)
        dloc = AutoDateLocator()
        #self.xaxis.tick_bottom()
        self.xaxis.set_major_locator(dloc)
        self.xaxis.set_minor_locator(AutoMinorLocator())
        self.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        
        self.yaxis.set_minor_locator(AutoMinorLocator())
        self.yaxis.set_major_formatter(DegreeFormatter())
        self.yaxis.set_minor_formatter(DegreeFormatter())
        
        #Tick setup for main axes
        
        
        #self.parasite.axis['right'].major_ticklabels.set_visible(False)
        
        
        ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #def set_formatters(self):
            
        #self.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        self.parasite.xaxis.tick_top()
        self.parasite.xaxis.offsetText.set_visible(False)
        
        dloc = AutoDateLocator()
        self.parasite.xaxis.set_major_locator(dloc)
        #self.parasite.xaxis.set_minor_locator(AutoMinorLocator())
        self.parasite.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        
        #fine grained formatting for coord display subtext
        fgfmt = AutoDateFormatter(dloc)
        fgfmt.scaled[1/24] = '%H:%M:%S'
        self._xcoord_formatter = fgfmt
        
        self._ycoord_formatter = DegreeFormatter(precision=2)
        
        self.parasite.yaxis.tick_right()
        self.parasite.yaxis.set_major_formatter(SeczFormatter())
        
        #self.parasite.yaxis.set_minor_locator(AutoMinorLocator())
        self.parasite.yaxis.set_minor_formatter(SeczFormatter())
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #def set_locators(self):
        #formatter_factory(AutoMinorLocator(n=5))
        #self.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        
        #self.parasite.xaxis.set_minor_locator(AutoMinorLocator(n=5))
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def format_coord(self, x, y):
        #print('fmt')
        #s = super().format_coord(x, y)
        xs = self._xcoord_formatter(x)
        ys = self._ycoord_formatter(y)
        
        xt, _ = self.parasite.transAux.transform([x,y])
        xts = self._xcoord_formatter(xt)
        
        _, yt = self.parasite.yaxis.get_major_formatter()._transform.transform([x,y]) #HACK!!
        yts = '%.3f' % yt
            
        return "UTC=%s alt=%s\tsid.T=%s airmass=%s" % (xs, ys, xts, yts)


#****************************************************************************************************
class VisPlot():
    default_cmap = 'jet'
    
    @profiler.histogram
    def __init__(self, date=None, site='sutherland', targets=None, tz=2*u.h, **options): #res
        
        self.sitename = site.title()
        self.siteloc = EarthLocation.of_site(self.sitename)
        #TODO from time import timezone
        self.tz = tz                         #can you get this from the site location??
        #obs = Observer(self.siteloc)
        self.targets = {}
        self.trajectories = {}
        self.plots = OrderedDict()
        
        if not date:
            now = datetime.now()        #current local time
            #default behaviour of this function changes depending on the time of day.
            #if calling during early morning hours (at telescope) - let midnight refer to previous midnight
            #if calling during afternoon hours - let midnight refer to coming midnight
            d = now.day + (now.hour > 7)
            date = datetime(now.year, now.month, d, 0, 0, 0)
        else:
            raise NotImplementedError
        self.date = date
        
        self.midnight = midnight = Time(date) - tz    #midnight UTC in local time
        #TODO: efficiency here.  Dont need to plot everything over 
        self.hours = h = np.linspace(-12, 12, 250) * u.h      #variable time 
        self.t = t = midnight + h
        self.tp = t.plot_date
        self.frames = AltAz(obstime=t, location=self.siteloc)
        #self.tmoon
        
        #collect name, coordinates in dict

        if not targets is None:
            self.add_coordinates(targets)
        
        #Get sun, moon coordinates
        sun = get_sun(t)
        self.sun = sun.transform_to(self.frames)    #WARNING: slow!!!!
        #TODO: other bright stars / planets
        
        #get dawn / dusk times
        self.dusk, self.dawn = self.get_daylight()
        self.sunset, self.sunrise = self.dusk['sunset'], self.dawn['sunrise']
        
        #get moon rise/set times, phase, illumination etc...
        self.moon = get_moon(t).transform_to(self.frames)
        self.mooning = self.get_moonlight()
        self.moon_phase, self.moon_ill  = self.get_moon_phase()
        
        self.setup_figure()
        #HACK
        self.cid = self.figure.canvas.mpl_connect('draw_event', self._on_first_draw)
        
     
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_daylight(self, accuracy=1):
        '''calculate dawn / dusk times'''
        midnight = self.midnight

        #We interpolate the calculated sun positions to get dusk/dawn times.  Should still be accurate to ~1s
        ip = interp1d(self.hours, self.sun.alt.degree)
        def twilight(h, angle):
            return ip(h) + angle

        angles = np.arange(0, 4) * 6.
        solver = partial(brentq, twilight)
        hdusk = np.vectorize(solver)(-12, 0, angles)
        hdawn = np.vectorize(solver)(0, 12, angles)
        dusk = midnight + hdusk * u.h
        dawn = midnight + hdawn * u.h
        
        sunning = OrderedDict(), OrderedDict()
        which = ['sun', 'civil', 'nautical', 'astronomical']
        when = [['set', 'rise'], ['dusk', 'dawn']]
        for i, times in enumerate(zip(dusk, dawn)):
            j = bool(i)
            for k, t in enumerate(times):
                words = (' '*j).join((which[i], when[j][k]))
                sunning[k][words] = t
        
        return sunning
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_moonlight(self):
        '''get moon rising and setting times'''
        h = self.hours.value
        malt = self.moon.alt.degree
        #interpolator
        ip = interp1d(h, malt)
        
        #find horizon crossing interval
        smalt = np.sign(malt)
        wsc = np.where(abs(smalt - np.roll(smalt, -1)) == 2)[0]  #index of sign changes
        ix_ = np.c_[wsc-1, wsc+1]    #check for sign changes: malt[ix_]
        mooning = {}
        for ix in ix_:
            crossint = h[ix]         #hour interval when moon crosses horizon
            rise_or_set = np.subtract(*malt[ix]) > 1
            s = 'moon{}'.format(['rise', 'set'][rise_or_set])
            hcross = brentq(ip, *crossint)
            mooning[s] = self.midnight + hcross * u.h
        
        return mooning
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_moon_phase(self):
        '''calculate moon phase and illumination at local midnight'''
        midnight = self.midnight
        altaz = AltAz(location=self.siteloc, obstime=midnight)

        moon = get_moon(midnight)
        moon = moon.transform_to(altaz)
        sun = get_sun(midnight)
        sun = sun.transform_to(altaz)

        #
        elongation = sun.separation(moon)
        #phase angle at midnight 
        phase = np.arctan2(sun.distance * np.sin(elongation),
                            moon.distance - sun.distance * np.cos(elongation))

        ill = (1 + np.cos(phase)) / 2.0
        return phase.value, ill.value
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_moon_marker(self):      #NOTE: could be a fuction
        
        if  np.pi - self.moon_phase < 0.1:
            return 'o'
        
        theta = np.linspace(np.pi/2, 3*np.pi/2)
        xv, yv = np.cos(theta), np.sin(theta)
        #xv, yv = circle.vertices.T
        # lx = xv < 0
        angle = np.pi - self.moon_phase
        xp = xv * np.cos(angle)

        x = np.r_[xv, xp]
        y = np.r_[yv, yv[::-1]]
        verts = np.c_[x, y]
        codes = np.ones(len(x)) * 4
        codes[0] = 1
        return Path(verts, codes)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_coordinate(self, name):
        self.targets[name] = resolver(name)
                
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_target(self, name, _do_leg=True):
        self.add_coordinate(name)
        self.add_curve(name)
        if _do_leg:
            self.do_legend()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_targets(self, names):
        for name in names:
            self.add_target(name, False)
        
        self.do_legend()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_coordinates(self, names):
        for name in names:
            self.add_coordinate(name)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_curve(self, name):  #NOTE there is a difference between adding and plotting. #TODO: choose better name. refactor
        coords = self.targets[name]
        
        if not name in self.trajectories:
            #compute trajectory
            altaz = coords.transform_to(self.frames)
            alt = self.trajectories[name] = altaz.alt.degree
        else:
            #recall trajectory
            alt = self.trajectories[name]
        
        trg_pl, = self.ax.plot(self.tp, alt, label=name)
        self.plots[name] = trg_pl
        
        #add name text at axes edges
        self.add_annotation(name)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def add_annotation(self, name):
        '''
        Add target name to visibility curve - helps more easily distinguish 
        targets when plotting numerous curves
        '''
        alt = self.trajectories[name]
        #find where the curve intersects the edge of the axes
        y0, y1 = self.ax.get_ylim()
        ly = (y0 < alt) & (alt < y1)
        #region within axes
        li = (ly & self._lt).astype(int)   #gives edge intersecting points in boolean array
        l0 = (li - np.roll(li, 1)) == 1    #first points inside axes (might be more than one per curve)
        l1 = (li - np.roll(li, -1)) == 1   #last points inside axes (")
        
        #decide whether to add one label or two per segment
        first, = np.where(l0)
        last, = np.where(l1)
        segments = np.c_[first, last]
        #if entry & exit point are close, only make 1 label
        thresh = 10
        close = np.squeeze(np.diff(segments) < thresh)
        do_label = np.c_[[True]*len(first),
                        ~close]
        
        #determine the slope of each curve at edges
        w = segments.ravel()
        ls = np.c_[w, w+1].ravel()
        xy = np.c_[self.tp[ls], alt[ls]]
        dx, dy = np.diff(xy.reshape(-1, 2, 2), axis=1).squeeze().T
        angles = np.degrees(np.arctan2(dy, dx))
        angles = self.ax.transData.transform_angles(angles, xy[::2])
        angles = angles.reshape(-1,2)
        
        #add the text labels
        for i, segs in enumerate(segments):
            colour = self.plots[name].get_color()
            for j, (x, y, ang, yn) in enumerate(zip(self.tp[segs], alt[segs],
                                                    angles[i], do_label[i])):
                if yn:
                    ha = ['right', 'left'][(j+1)%2]
                    off =  2, 2 #* np.sign(-th)
                    txt = self.ax.annotate(name, (x,y), off, color=colour,
                                            textcoords='offset points', ha=ha,
                                            rotation=ang, rotation_mode='anchor',
                                            size='x-small', fontweight='bold', )
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def setup_figure(self):
        
        self.figure = fig = plt.figure(figsize=(18,10))
        fig.subplots_adjust(top=0.94,
                            left=0.05,
                            right=0.85,
                            bottom=0.05)
        #setup axes with 
        self.ax = ax = VizAxes(fig, 111)
        lon = self.siteloc.longitude
        sid_trans = get_sid_trans(self.date, lon)
        aux_trans = btf(sid_trans, IdentityTransform())
        ax.parasite = ax.twin(aux_trans)
        
        
        ax.setup_ticks()
        fig.add_subplot(ax)
        
        #horizon line
        horizon = ax.axhline(0, 0, 1, color='0.85')
        
        #Shade twighlight / night
        for i, twilight in enumerate(zip(self.dusk.items(), self.dawn.items())):
            desc, times = zip(*twilight)
            ax.axvspan(*Time(times).plot_date, color=str(0.25*(3-i)))
            for words, t in zip(desc, times):
                self.twilight_txt(ax, words, t, color=str(0.33*i))

        #Indicate moonrise/set
        for rise_set, time in self.mooning.items():
            ax.axvline(time.plot_date, c='y', ls='--')
            self.twilight_txt(ax, rise_set, time, color='y')
        
        #TODO: enable picking for sun / moon
        sun_pl, = ax.plot(self.tp, self.sun.alt, 
                         'orangered', ls='none', markevery=2,
                         marker='o', ms=10,
                         label='sun')
        moon_pl, = ax.plot(self.tp, self.moon.alt,
                          'yellow',  ls='none',  markevery=2,
                          marker=self.get_moon_marker(), ms=10,
                          label='moon ({:.0%})'.format(self.moon_ill))
        
        
        #site / date info text
        ax.text(0, 1.04, self.date_info_txt(self.midnight),
                fontweight='bold', ha='left', transform=ax.transAxes)
        ax.text(1, 1.04, self.obs_info_txt(), fontweight='bold', 
                ha='right', transform=ax.transAxes)


        #setup axes
        #dloc = AutoDateLocator()
        #ax.xaxis.set_major_locator(dloc)
        #ax.xaxis.set_minor_locator(AutoMinorLocator())
        #ax.yaxis.set_minor_locator(AutoMinorLocator())
        #ax.yaxis.set_major_formatter(DegreeFormatter())
        #ax.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        #ax.yaxis.set_minor_formatter(DegreeFormatter())
        
        just_before_sunset = (self.sunset - 0.25*u.h).plot_date
        just_after_sunrise = (self.sunrise + 0.25*u.h).plot_date
        ax.set_xlim(just_before_sunset, just_after_sunrise)
        ax.set_ylim(-10, 90)
        
        #which part of the visibility curves are visible within axes
        self._lt = (just_before_sunset < self.tp) & (self.tp < just_after_sunrise)
        
        #labels for axes
        ax.set_ylabel('Altitude', fontweight='bold')
        ax.parasite.set_ylabel('Airmass', fontweight='bold')
        #UTC / SAST     #TODO: align with labels instead of guessing coordinates...
        self.ax.text(1, -0.005, 'SAST',
                    color='g', fontweight='bold', 
                    va='top', ha='right',
                    transform=self.ax.transAxes)
        self.ax.text(1,-0.02, 'UTC', 
                    color='k', fontweight='bold',
                    va='top', ha='right', 
                    transform=self.ax.transAxes)
        #sidereal time label
        txt = self.ax.text(1, 1.01, 'Sid.T.',
                    color='c', fontweight='bold', 
                    va = 'bottom', ha='right',
                    transform=self.ax.transAxes)
        
        #sun / moon legend
        leg = self.ax.legend(bbox_to_anchor=(1.05, 0), loc=3,
                             borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')
        self.ax.add_artist(leg)
    
        ax.grid()
        
    #====================================================================================================
    @staticmethod
    def twilight_txt(ax, s, t, **kw):
        ax.text(t.plot_date, 1, '{} {} SAST'.format(s, local_time_str(t)),
                rotation=90, ha='right', va='top',
                transform=btf(ax.transData, ax.transAxes),
                clip_on=True,
                **kw)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def obs_info_txt(self):
        #eg: 
        lat = self.siteloc.latitude
        lon = self.siteloc.longitude
        ns = 'NS'[bool(lat>0)]
        ew = 'WE'[bool(lon>0)]

        lat = abs(lat).to_string(precision=0, format='latex')
        lon = abs(lon).to_string(precision=0, format='latex')
        h = '{0.value:.0f} {0.unit:s}'.format(self.siteloc.height)
        return '{} @ {} {}; {} {}; {}'.format(self.sitename, lat, ns, lon, ew, h)
    
    #====================================================================================================
    @staticmethod
    def date_info_txt(t):
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dayname = weekdays[t.datetime.weekday()]
        datestr = t.iso.split()[0]
        return ', '.join((dayname, datestr))
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def make_time_ticks(self):
        #Create ticklabels for SAST
        ax = self.ax
        xticklabels = ax.xaxis.get_majorticklabels()
        xticklocs = ax.xaxis.get_majorticklocs()
        fmt = ax.xaxis.get_major_formatter()
        sast_lbls = list(map(fmt, (Time(num2date(xticklocs)) + self.tz).plot_date))

        for sast, tlbl in zip(sast_lbls, xticklabels):
            self.ax.text(*tlbl.get_position(), sast, 
                         color='g',
                         transform=tlbl.get_transform(),
                         va='bottom', ha=tlbl.get_ha())
        
        #change color of sidereal time labels
        #for tck in self.ax.parasite.get_xmajorticklabels():
            #tck.set_color('c')
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_vis(self, **kws): #MAYBE option for colour coding azimuth
        ax = self.ax
        
        cmap    = kws.get('cmap')
        colours = kws.get('colours', [])
        sort    = kws.get('sort', True)
        
        self.set_colour_cycle(colours, cmap)
        
        
        names, coords = zip(*self.targets.items())
        if sort:
            coords, names = sorter(coords, names, key=lambda coo: coo.ra)
            
        for name in names:
            self.add_curve(name)
        
        #self.plots += [sun_pl, moon_pl]
        
        self.do_legend()
        
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def set_colour_cycle(self, colours=[], cmap=None):
        #Ensure we plot with unique colours
        N = len(self.targets)
        
        #default behaviour - use colourmap
        if (not len(colours)) and (cmap is None):
            cmap = self.default_cmap
        
        #if colour map given or colours not given and default colour sequence insufficien
        if ((cmap is not None) #colour map given - superceeds colours arg
        or ((not len(colours)) #colours not given
            and len(rcParams['axes.prop_cycle']) < N)): #default colour sequence insufficient for uniqueness
            cm =  plt.get_cmap(cmap)
            colours =  cm(np.linspace(0, 1, N))
        
        #Colours provided explicitly and no colourmap given
        elif len(colours) < N:
            'Given colour sequence less than number of time series. Colours will repeat'
        
        if len(colours):
            from cycler import cycler
            ccyc = cycler('color', colours)
            self.ax.set_prop_cycle(ccyc)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def do_legend(self):
        
        leg = self.ax.legend(self.plots.values(), self.plots.keys(),            #*reversed(zip(*self.plots.items()))
                             bbox_to_anchor=(1.05, 1), loc=2,
                             borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')  
        
        
        self.pickable = {}
        for legline, origline in zip(leg.get_lines(), self.plots.values()):
            legline.set_picker(5)  # 5 pts tolerance
            self.pickable[legline] = origline
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_pick(self, event):
        # on the pick event, find the orig line corresponding to the
        # legend proxy line, and toggle the visibility
        legline = event.artist
        origline = self.pickable[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        # Change the alpha on the line in the legend so we can see what lines have been toggled
        if vis:
            legline.set_alpha(1.0)
        else:
            legline.set_alpha(0.2)
        self.figure.canvas.draw()        #TODO: blit
    
     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _on_first_draw(self, event):
        #HACK! this method creates the SAST labels upon the first cal to draw
        #print('FIRST DRAW')
        fig = self.figure
        canvas = fig.canvas
        
        self.make_time_ticks()
        canvas.mpl_disconnect(self.cid)         #disconnect callback.
        
        canvas.draw()        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def connect(self):
        #Setup legend picking
        self.figure.canvas.mpl_connect('pick_event', self._on_pick)

