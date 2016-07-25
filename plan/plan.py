# coding: utf-8
import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter, AutoMinorLocator
from matplotlib.dates import date2num, num2date
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
from matplotlib.transforms import blended_transform_factory as btf

from datetime import datetime

#Import the packages necessary for finding coordinates and making coordinate transformations
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun

from astroplan import Observer

from grafico.rainbow import rainbow_text
#from decor.profile import profile
#profiler = profile()




class DegreeFormatter(Formatter):
    def __call__(self, x, pos=None):
        # \u00b0 : degree symbol
        return "%d\u00b0" % (x)


# In[6]:

def OOOOh(t):
    '''hours since midnight'''
    return (t - midnight).sec / 3600


def local_time_str(t, tz=2*u.hour):
    return (t+tz).iso.split()[1].split('.')[0]



#TODO: Airmass
#TODO: sidereal time
#TODO: moonrise, moonset
#TODO: 1.9m limits

class VisPlot():
    def __init__(self, date=None, site='sutherland', targets=None, tz=2*u.h, **options): #res
        
        self.sitename = site.title()
        self.siteloc = EarthLocation.of_site(self.sitename)
        self.tz = tz                         #can you get this from the site location??
        obs = Observer(self.siteloc)

        if not date:
            now = datetime.now()
            date = datetime(now.year, now.month, now.day, 0, 0, 0)
        
        self.midnight = midnight = Time(date) - tz    #midnight UTC in local time
        self.hours = h = np.linspace(-12, 12, 100) * u.h      #variable time 
        self.t = t = midnight + h
        self.tp = t.plot_date
        self.frames = AltAz(obstime=t, location=self.siteloc)
        
        #collect name, coordinates in dict
        self.targets = {}
        if not targets is None:
            self.add_coordinates(targets)
        
        #Get sun, moon coordinates
        self.sun = get_sun(t).transform_to(self.frames)
        self.moon = obs.moon_altaz(t)
        #TODO: other bright stars / planets
        
        #get dawn / dusk times
        self.dusk, self.dawn = self.get_daylight()
        self.sunset, self.sunrise = self.dusk[0], self.dawn[0]
        
        self.moonrise, self.moonset = self.get_moonlight()
        
        self.setup_figure()
        #HACK
        self.cid = self.figure.canvas.mpl_connect('draw_event', self._on_first_draw)
        
    
    def get_daylight(self, accuracy=1):
        midnight = self.midnight
        if accuracy < 1:
            #WARNING: very slow
            dusk = [obs.sun_set_time(midnight),
                    obs.twilight_evening_civil(midnight),
                    obs.twilight_evening_nautical(midnight),
                    obs.twilight_evening_astronomical(midnight),]
            dawn = [obs.sun_rise_time(midnight),
                    obs.twilight_morning_civil(midnight),
                    obs.twilight_morning_nautical(midnight),
                    obs.twilight_morning_astronomical(midnight),]
        else:
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
            
        return Time(dusk), Time(dawn)
    
    def get_moonlight(self):
        '''get moon rising and setting times'''
        malt = self.moon.alt.degree
        maltrate = malt - np.roll(malt, 1)
        rising = maltrate > 0
        setting = maltrate < 0

        riseint = h[rising & (malt < 0)][0], h[rising & (malt > 0)][-1]
        setint = h[setting & (malt > 0)][0], h[setting & (malt < 0)][-1]

        ip = interp1d(self.hours, malt)
        hrise = brentq(ip, *riseint)
        hset = brentq(ip, *setint)
        
        moonrise = self.midnight + hrise * u.h
        moonset = self.midnight + hset * u.h
        
        return moonrise, moonset
    
    
    def add_coordinate(self, name):
        try:
            #Get the target coordinates
            coo = SkyCoord.from_name(name) # uses Simbad to resolve object names and retrieve coordinates.
            self.targets[name] = coo
        except:
            warnings.warn('Could not resolve name ' + name)
    
    def add_target(self, name):
        self.add_coordinate(name)
        self.add_curve(name)
    
    def add_curve(self, name):
        coords = self.targets[name]
        altaz = coords.transform_to(self.frames)
        trg_pl, = self.ax.plot(self.tp, altaz.alt, label=name)
        self.plots.append(trg_pl)    #maybe use dict here???????? 

    def add_targets(self, names):
        for name in names:
            self.add_target(name)
    
    def add_coordinates(self, names):
        for name in names:
            self.add_coordinate(name)
    
    
    def setup_figure(self):
        fig, ax = plt.subplots(figsize=(18,10), 
                               gridspec_kw=dict(hspace=0,
                                                 top=0.95,
                                                 left=0.05,
                                                 right=0.88,
                                                 bottom=0.1))
        self.figure = fig
        self.ax = ax
        
        horizon = ax.axhline(0, 0, 1, color='0.85')
        
        #Shade twighlight
        for i, (dsk, dwn) in enumerate(zip(self.dusk, self.dawn)):
            ax.axvspan(dsk.plot_date, dwn.plot_date, color=str((3-i)*(1/4)),)
        self.twilight_txt(ax, 'sunset', self.sunset)
        self.twilight_txt(ax, 'dusk', self.dusk[-1], color='w')
        self.twilight_txt(ax, 'sunrise', self.sunrise)
        self.twilight_txt(ax, 'dawn', self.dawn[-1], color='w')
        
        #Indicate moonrise/set
        ax.axvline(self.moonrise.plot_date, c='y', ls='--')
        self.twilight_txt(ax, 'moonrise', self.moonrise, color='y')
        ax.axvline(self.moonset.plot_date, c='y', ls='--')
        self.twilight_txt(ax, 'moonset', self.moonset, color='y')
        
        
        #site / date info text
        ax.text(0, 1.01, date_info_txt(self.midnight), ha='left', transform=ax.transAxes)
        ax.text(1, 1.01, self.obs_info_txt(), ha='right', transform=ax.transAxes)


        #setup axes
        ax.xaxis.set_tick_params('major', pad=10)
        ax.yaxis.set_tick_params('minor', labelsize=6, pad=5)
        dloc = AutoDateLocator()
        ax.xaxis.set_major_locator(dloc)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_major_formatter(DegreeFormatter())
        ax.xaxis.set_major_formatter(AutoDateFormatter(dloc))
        ax.yaxis.set_minor_formatter(DegreeFormatter())
        
        xlims = (self.dusk[0]-.25*u.h).plot_date, (self.dawn[0]+.25*u.h).plot_date
        ax.set_xlim(xlims)
        ax.set_ylim(-10, 90)

        #NOTE: This works only for ps backend
        #from matplotlib import rc
        #rc('text',usetex=True)
        #rc('text.latex', preamble=r'\usepackage{color}')
        #ax.set_xlabel(r'$\textcolor{green}{}$ / UTC (h)')
        rainbow_text(ax, 0.5, -.08, ['SAST','/ UTC'], ['g', 'k'], fontweight='bold')
        ax.set_ylabel('Altitude', fontweight='bold')

        ax.grid()
        
    @staticmethod
    def twilight_txt(ax, s, t, **kw):
        ax.text(t.plot_date, 1, '{} {} SAST'.format(s, local_time_str(t)),
                rotation=90, ha='right', va='top',
                transform=btf(ax.transData, ax.transAxes),
                clip_on=True,
                **kw)
    
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
    
    @staticmethod
    def date_info_txt(t):
        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dayname = weekdays[t.datetime.weekday()]
        datestr = t.iso.split()[0]
        return ', '.join((dayname, datestr))
   
    def make_local_time_labels(self):
        print('CALLING', 'make_local_time_labels')
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

    def plot_vis(self):
        ax = self.ax
        self.plots = []
        for name in self.targets.keys():
            self.add_curve(name)
        
        #TODO: enable picking for sun / moon
        sun_pl, = ax.plot(self.tp, self.sun.alt,
                         'orangered', marker='o', ls='none',
                         label='sun')
        moon_pl, = ax.plot(self.tp, self.moon.alt,
                         'yellow', marker='o', ls='none',
                         label='moon')
        self.plots += [sun_pl, moon_pl]
        
        
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., frameon=True)
        leg.get_frame().set_edgecolor('k')
        
        self.pickable = dict()
        for legline, origline in zip(leg.get_lines(), self.plots):
            legline.set_picker(5)  # 5 pts tolerance
            self.pickable[legline] = origline
    
    
    def onpick(self, event):
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
        #HACK! The rotate_labels borks if called before the figure is rendered. 
        # this method does the rotation upon the first cal to draw
        print('FIRST DRAW')
        fig = self.figure
        canvas = fig.canvas
        
        self.make_local_time_labels()
        canvas.mpl_disconnect(self.cid)         #disconnect callback.
        
        canvas.draw()        
    
    def connect():
        #Setup legend picking
        self.figure.canvas.mpl_connect('pick_event', onpick)


if __name__ == '__main__':
    #TODO: argparse????
    targets = ['CTCV J1928-5001',
            'IGR J14536-5522', 
            'IGR J19552+0044',
            'QS Tel', 
            'V1432 Aql',
            'HU Aqr', 
            'CD Ind']

    viz = VisPlot(targets=targets)
    viz.plot_vis()
    viz.add_target('V895 Cen')
