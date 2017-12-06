# coding: utf-8
#

# TODO:
# BAYESIAN !!
# GUI - interactively show masks when hovering over stars.
#     - sliders for r_sigma_star, r_sigma_sky
#     - PSFFitInspector class that plots stellar profiles from posterior
# TODO: for release
# optional context manager with coloured progress bar

# TODO: automatically figure out which is the target if name given and can be resolved to coordinates

# profiling
from decor.profiler.timers import Chrono, timer, timer_extra

chrono = Chrono()
# NOTE do this first so we can profile import times


# import matplotlib as mpl
# mpl.use('Agg')

# standard library imports
# import os
import sys
import ctypes
# import inspect
import socket
# import pickle
import logging
import traceback
import functools
# import itertools as itt
# from collections import OrderedDict
# import threading
import multiprocessing as mp
from pathlib import Path
# from functools import partial
# from queue import Empty
chrono.mark('Imports: std lib')

# ===============================================================================
# Check input file
fitsfile = sys.argv[1]
fitspath = Path(fitsfile)
path = fitspath.parent
if not fitspath.exists:
    raise IOError('File does not exist: %s' % fitsfile)


# related third party imports
# numeric tools
import numpy as np
from scipy.spatial.distance import cdist
# astronomy / photometry libs
import astropy.units as u
from photutils.detection import detect_threshold
from photutils.segmentation import detect_sources
from photutils import EllipticalAperture, CircularAnnulus
# monitoring libs
# import psutil
# ploting
# import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
# plt.ion()

chrono.mark('Imports: 3rd party')

# ===============================================================================
# local application libs
import ansi
from recipes.misc import is_interactive
from recipes.array import ndgrid
# from recipes.iter import chunker
from recipes.dict import AttrDict
from recipes.parallel.synched import SyncedCounter, SyncedArray
from recipes.parallel.pool import ConservativePool
# from recipes.logging import catch_and_log       #, LoggingMixin
is_interactive = is_interactive()

from obstools.fastfits import FitsCube
from obstools.phot.find.trackers import StarTracker

# from obstools.modelling import ModelDb, ImageModeller
from obstools.phot.diagnostics import FrameDisplay
from obstools.phot.utils import progressFactory, table_coords, table_cdist

# profiler = profile()
from IPython import embed
chrono.mark('Import: local libs')


# Logging setup
# TODO: move to module __init__.py
# ===============================================================================
# Decide how to log based on where we're running
if socket.gethostname().startswith('mensa'):
    plot_diagnostics = False
    print_progress = False
    log_progress = True
else:
    plot_diagnostics = True  # True
    print_progress = True
    log_progress = False

if is_interactive:  # turn off logging when running interactively (debug)
    from recipes.interactive import exit_register

    log_progress = print_progress = False
    monitor_mem = False
    monitor_cpu = False
    # monitor_qs = False
else:
    from atexit import register as exit_register
    from recipes.io.tracewarn import warning_traceback_on

    # check_mem = True            # prevent excecution if not enough memory available
    monitor_mem = False#True
    monitor_cpu = False#True  # True
    # monitor_qs = True  # False#

    # setup warnings to print full traceback
    warning_traceback_on()
    logging.captureWarnings(True)

# print section timing report at the end
exit_register(chrono.report)

# ===============================================================================
# create directory for logging / monitoring data to be saved
logpath = fitspath.with_suffix('.log')
if not logpath.exists():
    logpath.mkdir()

# create directory for figures
if plot_diagnostics:
    figpath = fitspath.with_suffix('.figs')
    if not figpath.exists():
        figpath.mkdir()

# create logger with name 'phot'
logbase = 'phot'
lvl = logging.DEBUG
logger = logging.getLogger(logbase)
logger.setLevel(lvl)                    # set root logger's level
logger.propagate = False


# create file handler which logs event debug messages
logfile = str(logpath / 'phot.log')
fh = logging.FileHandler(logfile, mode='w')
fh.setLevel(lvl)

# create console handler with a higher log level
# NOTE: this will log to both file and console
logerr = str(logpath / 'phot.log.err')
# ch = logging.FileHandler(logerr, mode='w')
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# ch

# class MyFilter(object):
#     def __init__(self, level):
#         self.__level = level
#
#     def filter(self, logRecord):
#         return logRecord.levelno <= self.__level


# def fuckyou(logRecord):
#     return logRecord.levelno <= logging.ERROR
# ch.addFilter(MyFilter(logging.ERROR))



# create formatter and add it to the handlers
fmt = '{asctime:<23} - {name:<32} - {process:5d}|{processName:<17} - {levelname:<10} - {message}'
# space = (23, 32, 5, 17, 10)
# indent = sum(space) * ' '
formatter = logging.Formatter(fmt, style='{')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

#
ModelDb.basename = logbase

# ===============================================================================
if monitor_mem or monitor_cpu:
    mon_mem_file = logpath / 'phot.mem.dat'
    mon_cpu_file = logpath / 'phot.cpu.dat'
    mon_q_file = logpath / 'phot.q.dat'

# Start process monitors if needed
if monitor_cpu:  # TODO: monitor load???
    from recipes.parallel.utils import monCPU

    # start cpu performance monitor
    mon_cpu_alive = mp.Event()
    mon_cpu_alive.set()  # it's alive!!!!
    monproc_cpu = mp.Process(name='cpu monitor',
                             target=monCPU,
                             args=(mon_cpu_file, 0.1, mon_cpu_alive),
                             daemon=True  # will not block main from exiting
                             )
    monproc_cpu.start()

    @exit_register
    def stop_cpu_monitor():
        # stop cpu performance monitor
        mon_cpu_alive.clear()
        # monproc_cpu.join()
        # del monproc_cpu

        # atexit.register(stop_cpu_monitor)

if monitor_mem:
    from recipes.parallel.utils import monMEM

    # start memory monitor
    mon_mem_alive = mp.Event()
    mon_mem_alive.set()  # it's alive!!!!
    monproc_mem = mp.Process(name='memory monitor',
                             target=monMEM,
                             args=(mon_mem_file, 0.5, mon_mem_alive),
                             daemon=True  # will not block main from exiting
                             )
    monproc_mem.start()

    @exit_register
    def stop_mem_monitor():
        # stop memory monitor
        mon_mem_alive.clear()
        # monproc_mem.join()
        # del monproc_mem


        # atexit.register(stop_mem_monitor)
chrono.mark('Setup')

# ===============================================================================
# @chrono.timer
# def init_shared_memory(N, Nstars, Naps, Nfit):
#     global modlr, apData
#
#     # NOTE: You should check how efficient these memory structures are.
#     # We might be spending a lot of our time synching access??
#
#     # Initialize shared memory with nans...
#     SyncedArray.__new__.__defaults__ = (None, None, np.nan, ctypes.c_double)  # lazy HACK
#
#     apData = AttrDict()
#     apData.bg = SyncedArray(shape=(N, Nstars))
#     apData.flux = SyncedArray(shape=(N, Nstars, Naps))
#     # FIXME: efficiency - don't need all the modeldb stuff if you are saving these here
#     apData.sigma_xy = SyncedArray(shape=(N, 2))      #TODO: for Nstars (optionally) ???
#     # apData.rsky = SyncedArray(shape=(N, 2))
#     # apData.theta = SyncedArray(shape=(N,))
#
#     # cog_data = np.empty((N, Nstars, 2, window*window))
#
#     mdlr.init_mem(N, Nfit)

# ===============================================================================

# TODO: move to separate module
# TODO: make picklable


def trackerFactory(how='centroid 5 brightest'):

    return StarTracker

    # # if not track and how == 'fit':
    # #     # NOTE: this is SHIT, because the convergence with least squares depends most sensitively
    # #     # on positional params. should only be a problem if we expect large shifts
    # #     track = True
    #
    # # TODO: 'centroid 5 brightest stars in fixed 25 pix window over median bg'
    # # TODO: 'centroid 5 best stars in 25 pix window over median bg'
    # # TODO: 'daofind'
    # # TODO: 'marginal fit'
    # # TODO: 'ofilter'
    # # etc:
    # # if 'best' replace bad stars
    # # if 'brightest', remove and warn
    #
    # tracking = True
    # descript = set(how.lower().split())
    #
    # # resolve nrs
    # nrs = list(filter(str.isdigit, descript))
    # descript -= set(nrs)
    # if len(nrs) == 1:
    #     nrs = list(range(int(nrs[0])))  # n brightest stars
    # elif 'all' in descript:
    #     nrs = list(range(Nstars))
    #     descript -= set(('all'))
    #
    # descript -= set(['brightest', 'stars'])  # this is actually redundant, as brightest are default
    # # 'centroid 5' understood as 'centroid 5 brightest stars'
    #
    # # embed()
    #
    # # resolve how to track
    # clsDict = {'detect': SourceFinder,
    #            'fit': StarTrackerFit, }
    #
    # for kind in clsDict.keys():
    #     if kind in descript:
    #         descript -= {kind}
    #         cls = clsDict[kind]
    #         break
    # else:
    #     # say 'using default' else for-else pretentious
    #     cls = (StarTrackerFixedWindow, StarTracker)[tracking]
    #
    #     # resolve the position locator function
    #     fDict = {'centroid': CoM,
    #              'com': CoM}  # TODO: add other trackers
    #
    #     # resolve the tracking function.
    #     for name in fDict.keys():
    #         if name in descript:
    #             descript -= {name}
    #             cfunc = fDict[name]
    #             break
    #     else:
    #         # say 'using default' else for-else pretentious
    #         cfunc = CoM  # Default is track by centroid
    # bgfunc = np.median
    #
    # if len(descript):
    #     raise ValueError('Finder: %s not understood' % str(tuple(descript)))
    #
    # return cls, nrs, cfunc, bgfunc
    #
    # # if how == 'detect':
    # #     return SourceFinder, None
    #
    # # if how in [None, 'fit']:  # + modelDb.gaussians:
    # #     return StarTrackerFit
    #
    # # if how.startswith('cent'):
    # #     how = 'med'  # centroid with median bg subtraction
    # #
    # # if how.startswith('med'):
    # #     bgfunc = np.median
    # #
    # # elif how.startswith('mode'):  # return partial(fixed_window_finder, mode)
    # #     from scipy import stats
    # #
    # #     def mode(data):
    # #         return stats.mode(data, axis=None).mode
    # #
    # #     bgfunc = mode
    # # else:
    # #     raise ValueError('Unknown finder option %r' % how)
    #
    # # cls = (StarTrackerFixedWindow, StarTrackerMovingWindow)[track]
    # # #cls.bgfunc = bgfunc
    # # return cls, bgfunc







def plot_max_shift(image_max, seg_image, filename=None):
    outlines = seg_image.outline_segments(True)
    shiftplot = FrameDisplay(image_max, Rcoo)
    shiftplot.mark_found('rx')
    shiftplot.add_windows(tracker.window, preFind.sdist)
    shiftplot.add_detection_outlines(outlines)
    shiftplot.figure.canvas.set_window_title('Envelope')
    # shiftplot.ax.set_title()
    if filename:
        shiftplot.ax.figure.savefig(filename)



# ===============================================================================


def coo_within_window(p):
    w2 = np.divide(window, 2)
    return np.all(np.abs(p[:2] - w2) < w2)      # center coordinates outside of grid



def catch_and_log(func):
    """Decorator that catches and logs errors instead of raising"""

    @functools.wraps(func)
    def wrapper(*args, **kws):
        try:
            answer = func(*args, **kws)
            return answer
        except Exception as err:
            logger.exception('\n')  # logs full trace by default

    return wrapper





# @catch_and_log
def frame_proc(incoming):
    """Process frame i"""
    i, data = incoming
    data_std = np.ones_like(data)        # FIXME:



    # for k in range(Naps):

        # apData.bg[i, j] = flux_bg_pp

        #
        # for k in range(Naps):
        #     apData.flux[i, j, k] = do_phot(udata, photmasks[j], cxx[j], rxy[k], theta, flux_bg_pp)
    #
    show_progress()





def per_frame(t, N, Nstars, Naps):
    t0 = t / N
    t1 = t0 / Nstars
    t2 = t1 / Naps
    print('\t\t%2.4f sec per frame\n'
          '\t\t%2.4f sec per star\n'
          '\t\t%2.4f sec per aperture\n' % (t0, t1, t2))



def get_saturation(h):
    try:
        import pySHOC.readnoise as shocRNT
        return shocRNT.get_saturation(h)
    except ValueError as err:
        logging.info('NOT SHOC DATA')
        logging.debug(str(err))

def update_aps(aps, i):
    aps.coords = coords[i]



# ===============================================================================
chrono.mark('Class/func defs')



# ===============================================================================
# MAIN SETUP
# ===============================================================================
# TODO: functions / classes here...


import argparse
parser = argparse.ArgumentParser(description='Generic photometry routines')
parser.add_argument('fitsfile', nargs=1, type=str, help='filename')
parser.add_argument('-n', '--subset', type=int,
                    help='Data subset to load.  Useful for testing/debugging.')
parser.add_argument('--plot', action='store_true', default=True,
                    help='Do plots')
parser.add_argument('--no-plots', dest='plot', action='store_false',
                    help="Don't do plots")
parser.add_argument('--track', default='centroid', type=str,
                    help='How to track stars')
# parser.add_argument('--window', default=20, type=int,
#                     help='Tracking window size')
parser.add_argument('--ap', nargs='*', default=['circular fixed 0.5:10:0.5 pix'],
                    type=str, help='Aperture description')

args = parser.parse_args(sys.argv[1:])


if not args.plot:
    plot_diagnostics = False

# read data directly from disc using memory map
ff = FitsCube(fitsfile)

N = min((args.subset or len(ff)), len(ff)) 	#min(20000, len(ff))
N_max_stars = 20     # Only the `N_max_stars` brightest stars will be processed

# sub-framing params
# window = args.window
# create xy grid
Grid = np.indices(ff.ishape)  # grid for full frame



# Star Finding Setup
# =================================================================================================
# NOTE: trackhow can be one of ('median', 'mode', 'centroid', 'fit'), or one of the psf models
# eg. CircularGaussianPSF.  If 'fit', the best fitting model's coordinates will be used (as
# judged by ---- goodness-of-fit)
# TODO: check out photutils finders - marginal distribution fitting. 'ofilter'
# TODO: daofind??


# use location of reference star (first k frames mean) as starting point for search
# NOTE: using coordinates of the previous (i-1) frames as a starting point might be a better
# choice, especially if the stars move from one frame to the next.  This
# however will require *sequential* iteration over the frames which will
# require some magic to multiprocess # TODO
# TODO: One might be able to tell which option will be the preferred one
# by looking at the maximum frame-to-frame across the CCD array.

# select Nmean frames randomly from first 100
Nmean = 10
ix = np.random.randint(0, 100, Nmean)
preImage = np.median(ff[:Nmean], 0)

# Get tracker
TrackerClass, ix_loc_rqst, centreFunc, bgFunc = trackerFactory(args.track)

# init the tracker
tracker = TrackerClass.from_image(preImage, dilate=3, snr=3, npixels=7,
                                  deblend=True)
                                 # bad_pixel_mask=bad_pixel_mask)

# check which stars are good for centroid tracking
satlvl = get_saturation(ff.header)
ix_loc = preFind.best_for_tracking(window, saturation=satlvl)
tracker = TrackerClass(Rcoo, window, ix_loc, max_stars=N_max_stars,
                      cfunc=centreFunc, bgfunc=bgFunc)
ix_loc = tracker.ix_loc


Nstars = min(len(tracker.rcoo), N_max_stars)






# ix_scale = list(range(Nscale))  # brightest stars (since they are sorted)

# Pre-diagnostics to help choose algorithmic parameters
# Estimate max shift in pixels throughout run
mxshift, max_image, seg_image_max = estimate_max_shift(ff, 100)
large_shifts = np.any(mxshift > window)
#TODO: cosmic ray flagging in the envelope??

# =================================================================================================


# Phot Setup
# =================================================================================================
# NOTE: ix_fit: Indices of stars to fit model psf to for psf photometry
# NOTE: ix_scale: Indices of stars used to scale (stretch) the apertures for psf-guided aperture
#       photometry.
# NOTE: ix_loc: Indices of stars used to pinpoint the centre point of the apertures. All apertures
#       will be located based on their relative position to these stars

phot_opt = ['psf', 'aperture', 'core', 'cog', 'opt'] # TODO: Bayesian
phothow = ['aperture']  #['psf']
assert all(map(phot_opt.__contains__, phothow))
aphow = ' '.join(args.ap)
if 'psf' in phothow:
    ix_fit = list(range(Nstars))  # TODO: GUI optionally on indexed stars only
    # ix_loc = ix_scale = [ir]

if 'aperture' in phothow:
    # ix_fit = ix_loc[:1]     # TODO: GUI optionally on indexed stars only
    ix_scale = ix_fit = ix_loc

# build scaler
ApScalerCls, sizes, unit = scalerFactory(aphow)
Naps = len(sizes)
if unit in ('"', 'pix'):
    ix_fit, ix_scale = [], [] # HACK  ==> no fitting
    # TODO: convert units

# make sure we are fitting for the stars we are using for the scaling
assert all(map(ix_fit.__contains__, ix_scale))

Nfit = len(ix_fit)
if not Nfit:
    mdlrCls = FrameModellerBase
else:
    mdlrCls = FrameModeller

# aperture params
# =================================================================================================
apscaler = ApScalerCls(Grid,  # Rcoo,
                       r_sigma=sizes,
                       rsky_sigma=(7.5, 15.))
# Naps = 8
# aps_scale_min, aps_scale_max = 1, 6     # scaling radii for apertures in units of sigma
# aps_scale = np.linspace(aps_scale_min, aps_scale_max, Naps)
# aps_scale = np.arange(1, Naps+1) * .5     #in units of sigma
# Naps = len(aps_scale)

# skyin, skyout = 4.5, 8.5  # in units of sigma
# mask_within_sigma = 5.0  # mask other stars within this radius (in units of sigma) when doing photometry / fitting
# rsky0 = np.multiply((skyin, skyout), sigma0)


# Model Setup
#  =================================================================================================
modWindow = 20
track_residuals = True
metrics = ('aic', 'bic', 'redchi')
modnames = ('CircularGaussianPSF', 'EllipticalGaussianPSF', 'ConstantBG')
# TODO: models.psf.elliptical.gauss, models.psf.circular.gauss, models.bg.median

modelDb = ModelDb(modnames)
for model in modelDb.gaussians:
     model.add_validation(coo_within_window) # FIXME: maybe better to check post facto

# Pre fit to set default params # NOTE: pre-fitting even when psf phot not being done
ansi.banner('Running Pre-fit')
preModel = modelDb.db.EllipticalGaussianPSF()

mdlr = FrameModeller(modelDb.models, modWindow, ix_fit, tracker, metrics=metrics)
preParams = mdlr(preFind.found, window, preFind.image)
mdlr.fallback = preParams
#sigma_xy0, theta0 = apscaler.scale(*preParams[1:])

# Init apertures
# =================================================================================================
# NOTE: when fixed apertures, currently have to init this class after pre_fit




# print setup info
# =================================================================================================
# Print pre-calc information here
print('\nFile: %s' % str(fitsfile))
print('%i frames\nProcessing %s' % (len(ff), 'interactively' if is_interactive else N))

# =================================================================================================
# table: coordinates
cxtbl = table_coords(Rcoo, ix_fit, ix_scale, tracker.ix_loc)
# table: coordinate distances
cdtbl = table_cdist(preFind.sdist, window)  # TODO: remove ansi when logging to file
print('\n\nThe following stars have been found:')
print(cxtbl)
print(cdtbl)



# info: maximal shift
print('\nEstimated maximal shift = {}'
      '\n                 window = {}\n'.format(mxshift, window))
if large_shifts:
    # TODO: enabled in green / only log if green
    logger.warning('Estimated positional shift throughout run is %s > window %d. Tracking %sabled'
                    %(mxshift, window, ('dis', 'en')[tracking]))



#
# def plot_frame(fmean, found, sdist, outfile=None):
#     preplot = FrameDisplay(fmean, found)
#     preplot.mark_found('rx')
#     preplot.add_windows(window, sdist)
#     preplot.figure.canvas.set_window_title('PrePlot')
#     # preplot.ax.set_title()
#     if outfile:
#         preplot.ax.figure.savefig(outfile)
#     return preplot

def init_plot(filename, preFind, preModel, preParams, apscaler, outfile=None):
    prePlotPath = figpath / 'found.png'

    # p = mp.Process(target=plot_max_shift, args=(max_image, seg_image_max))
    # p.start()
    # if large_shifts:
    plot_max_shift(max_image, seg_image_max)

    xy = preFind.found[:, ::-1]

    preplot = FrameDisplay(filename)
    preplot.figure.canvas.set_window_title('PrePlot')
    preplot.mark_found(xy.T, 'rx')
    preplot.add_windows(xy, window, preFind.sdist)



    # preplot = plot_frame(preFind.data, Rcoo, preFind.sdist)
    preplot.ax.set_autoscale_on(False)
    xyfound = preFind.found
    preplot.add_aperture_from_model(preModel, preParams,
                                    apscaler.r[-1], apscaler.rsky)
    preplot.add_legend()

    # child process will block here
    plt.show()

    # save figure
    fig = preplot.ax.figure
    fig.savefig(str(prePlotPath))
    return fig


# plot prefit diagnostics
if plot_diagnostics:

    # TODO legend!!!!!!
    # TODO: gui features + display:
    #     : draggable apertures + annotated ito sigma / pixels
    #     : display params for current fits / multiple stars switcher
    #     : mask display
    #
    #
    job = mp.Process(target=do_preplot,
                     args=(preFind, preModel, preParams, apscaler))
    job.start()

    # do_preplot(preFind, preModel, preParams, apscaler)



print(Rcoo)
# raise SystemExit

# ===============================================================================
chrono.mark('Pre-calc')


# TODO: print info on cube
# TODO: Error estimation
# TODO: CoG!!!!

# TODO: how costly is pickling
# TODO: local variable lookups are much faster than global or built-in variable lookups:
# the Python "compiler" optimizes most function bodies so that for local variables,
# no dictionary lookup is necessary, but a simple array indexing operation is sufficient.
# TODO: profile this shit!

# ===============================================================================
def loadframe(i):
    # get i'th frame
    return i, ff[i]


# ===============================================================================
@timer_extra(per_frame, N, Nstars, Naps)
def MAIN():

    # Initialize shared memory
    init_shared_memory(N, Nstars, Naps, Nfit)

    if print_progress:
        bar.create(N)

    # for chunk in chunker(range(N), 5000):
    pool = ConservativePool(maxtasksperchild=100)  # maxtasksperchild=mxtpc
    res = pool.map(frame_proc,
                   map(loadframe, range(N)),
                   chunksize=1)
    pool.close()  # NOTE: this stops the worker handler!!
    pool.join()

    return res


# ===============================================================================
def run_profiled(n):  # SHOW Progress?
    # from line_profiler import HLineProfiler
    from decor.profiler import HLineProfiler
    profiler = HLineProfiler()

    # profiler.add_function(do_find)
    # profiler.add_function(do_fit)
    # profiler.add_function(do_bg_phot)
    profiler.enable_by_count()

    run_sequential_test(n)

    profiler.print_stats()


# ===============================================================================
def run_memory_profiled(n):  # SHOW Progress?

    from memory_profiler import show_results, LineProfiler as MLineProfiler
    profiler = MLineProfiler()

    # profiler.add_function(do_find)
    # profiler.add_function(do_fit)
    profiler.add_function(do_bg_phot)
    profiler.add_function(do_phot)
    profiler.enable_by_count()

    run_sequential_test(n)

    show_results(profiler)


# ===============================================================================
@timer_extra(per_frame, N, Nstars, Naps)
def run_sequential_test(n):
    if print_progress:
        bar.create(n * Nstars * Naps)

    res = []
    for i in range(n):
        res.append(frame_proc((i, ff[i])))

    return res


@timer_extra(per_frame, N, Nstars, Naps)
def run_pool_test(n):
    if print_progress:
        bar.create(n * Nstars * Naps)

    pool = mp.Pool()  # maxtasksperchild=mxtpc
    pool.map(frame_proc, map(loadframe, range(N)), chunksize=1)
    pool.close()
    pool.join()


# def null_func(*args):chrono
#     pass



def finalise():
    locData = AttrDict(
        # tracker=tracker,
                       window=tracker.window,
                       ir=tracker.ir,
                       rcoo=tracker.Rcoo,
                       rvec=tracker.Rvec,
                       coords=coords,      # FIXME: just let this live inside the tracker class instead
                       find_with_fit=isinstance(tracker, StarTrackerFit)
                       )


    # Normalise residual sum data
    for model in modelDb.models:
        modelDb.resData[model] /= N

    apData.scale = apscaler.r

    chrono.mark()
    if plot_diagnostics:
        try:
            from matplotlib import pyplot as plt
            from obstools.phot.diagnostics import diagnostic_figures

            diagnostic_figures(locData, apData, modelDb, fitspath)
            chrono.mark('Diagnostics')

            # NOTE: display with
            # allcoords = coords[None,:] + Rvec[:,None]
            # fig, ax = plt.subplots()
            # FitsCubeDisplay(ax, fitsfile, allcoords[...,::-1])

        except Exception as err:
            # exc_info = sys.exc_info()
            traceback.print_exc()

    # llcs = np.round(loc - window/2).astype(int)[:,None,:] + Rvecw
    # fitcoo = psf_par[..., 1::-1]
    # llcs + fitcoo      #fit coordinates for all stars

    # TODO: integrate with time data
    # t0 = time.time()
    from recipes.io import save_pickle

    savepath = fitspath.with_suffix('.phot')
    timer(save_pickle)(savepath, AttrDict(filename=fitsfile,
                                          locData=locData,
                                          apData=apData,
                                          modelDb=modelDb))

    # with savepath.open('wb') as sv:
    #     pickle.dump(,
    #                 sv)
    # took('Saving data', time.time() - t0)



ProgressPrinter, ProgressLogger = progressFactory(log_progress, print_progress)

progress = SyncedCounter(0)
bar = ProgressPrinter(symbol='=', properties={'bg': 'c', 'txt': 'r'})
barlog = ProgressLogger(symbol='=', width=50, sigfig=0, align=('<', '<'))
barlog.create(N)

def show_progress():
    progress.inc()
    state = progress.value()
    bar.progress(state)
    barlog.progress(state)

# stop here if in ipython
# if is_interactive:
    # init_shared_memory(N, Nstars, Naps, Nfit)
    #raise SystemExit

#TODO: estimate total excecution time
# Main work here
try:
    # MAIN()
    # init_shared_memory(N, Nstars, Naps, Nfit)
    a = run_sequential_test(N)

except Exception as err:
    print('Oh SHIT! Something went wrong...')
    traceback.print_exc()

    embed()
    raise

if is_interactive:
    logging.info('Interactive mode: SystemExit')
    raise SystemExit

finalise()

#


# run_pool_test(N)

# from IPython import embed
# embed()
# raise SystemExit

# try:
# run_sequential_test(10)
# run_profiled()
#run_memory_profiled()
#
#profiler.print_stats()
#raise SystemExit
# except Exception as err:
# traceback.print_exc()



# embed()

# if monitor_qs:
# try:
# from obstools.phot.diagnostics import plot_q_mon
# plot_q_mon(mon_q_file, save=False)
# except Exception as err:
# traceback.print_exc()

# if monitor_cpu and monitor_mem:
# from obstools.phot.diagnostics import plot_monitor_data
# plot_monitor_data(mon_cpu_file, mon_mem_file)
#