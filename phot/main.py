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
from decor.profile.timers import Chrono, timer, timer_extra

from phot.trackers import StarTrackerFixedWindow, StarTrackerFit, SourceFinder

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
from obstools.phot.masker import MaskMachine
from obstools.phot.modelling import ModelDb, FrameModellerBase, FrameModeller
from obstools.phot.diagnostics import FrameDisplay
from obstools.phot.utils import progressFactory, table_coords, table_cdist

import pySHOC.readnoise as shocRNT

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

    # check_mem = True            #prevent excecution if not enough memory available
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
@chrono.timer
def init_shared_memory(N, Nstars, Naps, Nfit):
    global modlr, apData

    # NOTE: You should check how efficient these memory structures are.
    # We might be spending a lot of our time synching access??

    # Initialize shared memory with nans...
    SyncedArray.__new__.__defaults__ = (None, None, np.nan, ctypes.c_double)  # lazy HACK

    apData = AttrDict()
    apData.bg = SyncedArray(shape=(N, Nstars))
    apData.flux = SyncedArray(shape=(N, Nstars, Naps))
    # FIXME: efficiency - don't need all the modeldb stuff if you are saving these here
    apData.sigma_xy = SyncedArray(shape=(N, 2))      #TODO: for Nstars (optionally) ???
    # apData.rsky = SyncedArray(shape=(N, 2))
    # apData.theta = SyncedArray(shape=(N,))

    # cog_data = np.empty((N, Nstars, 2, window*window))

    mdlr.init_mem(N, Nfit)

# ===============================================================================

# TODO: move to separate module
# TODO: make picklable


def trackerFactory(how='centroid 5 brightest'):
    # if not track and how == 'fit':
    #     # NOTE: this is SHIT, because the convergence with least squares depends most sensitively
    #     # on positional params. should only be a problem if we expect large shifts
    #     track = True

    # TODO: 'centroid 5 brightest stars in fixed 25 pix window over median bg'
    # TODO: 'centroid 5 best stars in 25 pix window over median bg'
    # TODO: 'daofind'
    # TODO: 'marginal fit'
    # TODO: 'ofilter'
    # etc:
    # if 'best' replace bad stars
    # if 'brightest', remove and warn

    tracking = True
    descript = set(how.lower().split())

    # resolve nrs
    nrs = list(filter(str.isdigit, descript))
    descript -= set(nrs)
    if len(nrs) == 1:
        nrs = list(range(int(nrs[0])))  # n brightest stars
    elif 'all' in descript:
        nrs = list(range(Nstars))
        descript -= set(('all'))

    descript -= set(['brightest', 'stars'])  # this is actually redundant, as brightest are default
    # 'centroid 5' understood as 'centroid 5 brightest stars'

    # embed()

    # resolve how to track
    clsDict = {'detect': SourceFinder,
               'fit': StarTrackerFit, }

    for kind in clsDict.keys():
        if kind in descript:
            descript -= {kind}
            cls = clsDict[kind]
            break
    else:
        # say 'using default' else for-else pretentious
        cls = (StarTrackerFixedWindow, StarTracker)[tracking]

        # resolve the position locator function
        fDict = {'centroid': CoM,
                 'com': CoM}  # TODO: add other trackers

        # resolve the tracking function.
        for name in fDict.keys():
            if name in descript:
                descript -= {name}
                cfunc = fDict[name]
                break
        else:
            # say 'using default' else for-else pretentious
            cfunc = CoM  # Default is track by centroid
    bgfunc = np.median

    if len(descript):
        raise ValueError('Finder: %s not understood' % str(tuple(descript)))

    return cls, nrs, cfunc, bgfunc

    # if how == 'detect':
    #     return SourceFinder, None

    # if how in [None, 'fit']:  # + modelDb.gaussians:
    #     return StarTrackerFit

    # if how.startswith('cent'):
    #     how = 'med'  # centroid with median bg subtraction
    #
    # if how.startswith('med'):
    #     bgfunc = np.median
    #
    # elif how.startswith('mode'):  # return partial(fixed_window_finder, mode)
    #     from scipy import stats
    #
    #     def mode(data):
    #         return stats.mode(data, axis=None).mode
    #
    #     bgfunc = mode
    # else:
    #     raise ValueError('Unknown finder option %r' % how)

    # cls = (StarTrackerFixedWindow, StarTrackerMovingWindow)[track]
    # #cls.bgfunc = bgfunc
    # return cls, bgfunc


#     #


# ===============================================================================


    #def loop(self):


def save_params(i, j, model, results):
    """save fitted paramers for this model"""
    p, punc, gof = results

    # Set shared memory
    psfData = modelDb.data[model]
    psfData.params[i, j] = p
    psfData.params_stddev[i, j] = punc
    # calculate psf flux
    if hasattr(model, 'integrate'):
        # FIXME: you can avoid this by having two classes - BGModel, PSFModel...
        psfData.flux[i, j] = model.integrate(p)
        psfData.flux_stddev[i, j] = model.int_err(p, punc)

    # Re-parameterize to more physically meaningful quantities
    if hasattr(model, 'reparameterize'):  # model in gaussianModels: #
        # FIXME: just work with converted parameters already !!!!!!!!!!!!!!!!!!!!!!!
        psfData.alt[i, j] = model.reparameterize(p)
        # FUCK!!

    # Goodness of fit statistics
    for metric in modelDb.metrics:
        modelDb.metricData[metric][i, j, modelDb._ix[model]] = gof[metric]



# ===============================================================================
# TODO: classes for different phot routines
# @memprof
def do_bg_phot(data, mask, cxx, rsky):
    # NOTE: now only working for uniform aperture sizes for all stars

    #TODO: clip outliers in bg  - this is more general than masking

    method = 'center'
    xypos = cxx[:, ::-1]
    rskyin, rskyout = rsky

    # nmasked = 0
    # if np.size(mask):
    #     nmasked = len(mask[0])  #WARNING: only for shrunken masks

    mask = apscaler.expand_mask(mask, data.shape)    #FIXME: inefficient?
    # NOTE:  do_photometry below will automatically exclude masked data from calculation

    ann = CircularAnnulus(xypos, r_in=rskyin, r_out=rskyout)
    flx_bg, flx_bg_err = ann.do_photometry(data,
                                # error,
                                mask=mask,
                                # effective_gain,#  must have same shape as data
                                # TODO: ERROR ESTIMATE
                                method=method)      # method='subpixel', subpixel=5)

    # TODO: photutils - better support for masked data!!
    # BUG: tuple masks are converted to arrays which broadcast differently. this effectively leads to the mask being completely ignored silently

    # calculate total area of aperture, excluding masked pixels (else underestimate the flux)
    # area = np.subtract(ann.mask_area(method), nmasked)
    # area = (m.data * m.cutout(~mask)).sum
    try:
        area = [(m.data * m.cutout(~mask)).sum() for m in ann.to_mask(method)]
    except:
        print('Oh fuck ' * 100)
        embed()
        raise
    # TODO: warn if area is below some threshold ?
    flux_bg_pp = flx_bg / area  # Background Flux per pixel

    return flux_bg_pp

    # WARNING:
    # With elliptical star profiles:  scaling aperture radii with fwhm as geometric mean
    # might be un-ideal if images tend to distrort in a particular direction preferentially
    # during a run (as is often the case)...


# ===============================================================================
# @memprof
def do_phot(data, masks, cxx, rxy, theta, flux_bg_pp):
    # THIS ONE WILL BE SLOWER I SUSPECT...
    method = 'exact'
    xypos = cxx[:Nstars, ::-1]

    mshape = data.shape + (Nstars,)
    masks = apscaler.expand_mask(masks, mshape)     #TODO: efficiency?

    Flux = np.empty((Nstars, Naps))
    for j in range(Nstars):
        mask = masks[..., j]
        for k in range(Naps):
            rx, ry = rxy[k]
            # get aperture class
            ap = EllipticalAperture(xypos[j], rx, ry, theta)
            flux, flux_err = ap.do_photometry(data,
                                              mask=mask,
                                     # error, #TODO: ERROR ESTIMATE
                                     # effective_gain,#  must have same shape as data
                                     method=method)
            # get the area of the aperture excluding masked pixels
            m = ap.to_mask(method)[0]
            area = (m.data * m.cutout(~mask)).sum()

            flux_res = flux - (flux_bg_pp[j] * area)

            Flux[j, k] = flux_res

    return Flux


def do_multi_phot(data, masks, cxx, rxy, theta, flux_bg_pp):
    # TODO: just loop ovr do_phot abve
    # NOTE: for now just ignoring the photmasks untill we can find a better way of dealing
    # TODO neighbourhood fill to preserve noise structure ?
    method = 'exact'
    xypos = cxx[:, ::-1]

    Flux = []
    for k in range(Naps):
        rx, ry = rxy[k]
        aps = EllipticalAperture(xypos, rx, ry, theta)
        flux, flux_err = aps.do_photometry(data,
                                 # error, #TODO: ERROR ESTIMATE
                                 # mask = masks[]
                                 # effective_gain,#  must have same shape as data
                                method=method)
        # TODO: take account of masked area
        aps.mask_area(method)
        flux_res = flux - (flux_bg_pp * aps.area())
        Flux.append(flux_res)
    return np.transpose(Flux)

# ===============================================================================
def estimate_max_shift(fitscube, nframes, snr=5, npixels=7):
    """Estimate the maximal positional shift for stars"""
    step = len(fitscube) // nframes  # take `nframes` frames evenly spaced across data set
    maxImage = fitscube[::step].max(0)      #
    threshold = detect_threshold(maxImage, snr)  # detection at snr of 5
    segImage = detect_sources(maxImage, threshold, npixels)
    mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                      for (xs, ys) in segImage.slices], 0)

    #TODO: check for cosmic rays inside sky apertures!

    return mxshift, maxImage, segImage


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



def check_aps_sky(i, rsky):
    rskyin, rskyout = rsky
    info = 'Frame {:d}, rin={:.1f}, rout={:.1f}'
    if np.isnan(rsky).any():
        logger.warning('Nans in sky apertures: ' + info.format(i, *rsky))
    if rskyin > rskyout:
        logger.warning('rskyin > rskyout: ' + info.format(i, *rsky))
    if rskyin > window:
        logger.warning('Large sky apertures: ' + info.format(i, *rsky))


# TODO:
def scalerFactory(how='circular nanmedian 3 sigma'):
    """*how* can be:
        eg: circular nanmean / elliptical nanmedian / circular fixed 2"
    """
    #TODO: circular nanmean 0.5:5:0.5 sigma

    clsDict = {'circular': ScalerCircular,
               'elliptical': ScalerElliptical,}
               # 'fixed': ScalerFixed, }
    # TODO: this as a classmethod
    # TODO: ScalableAperture / FixedAperture ??
    mxnDict = {'sigma': (),
               '"': (ScaleFixed,),
               'pix': (ScaleFixed,)}
    # mxnDict = {'scaled': (),
    #             'fixed': ScaleFixed}
    descript = set(how.lower().split())
    # get the str for the kind of aperture requested
    for kind in clsDict.keys():
        if kind in descript:
            descript -= {kind}
            break
    else:
        # say 'using default' else for-else pretentious
        kind = 'circular'

    cls = clsDict[kind]

    # resolve size(s)
    has_size = False
    for s in descript:
        if s[0].isdigit():
            descript -= {s}
            has_size = True
            break
    else:
        logger.warning('Using 5 sigma aperture size')
        # raise ValueError('Need size')
        s = '5'

    nrs = tuple(map(float, s.split(':')))
    if len(nrs) == 1:
        sizes = np.array(nrs)
    elif len(nrs) == 3:
        start, stop, step = nrs
        stop += step        #make the range top-inclusive (intuitive interpretation)
        sizes = np.mgrid[slice(start, stop, step)]
    else:
        raise ValueError('size %s not understood' %s)

    #embed()

    # resolve unit
    known_units = ('sigma', '"', 'pix')
    for unit in known_units: # arcsec
        if unit in descript:
            descript -= {unit}
            break
    else:
        unit = 'sigma'  # NOTE: put the default last in the loop to elliminate this code line
        if has_size:
            print('size without unit ' * 10)
            embed()
            raise ValueError('size without unit')

    s = {'sigma': 'scaled',
         '"': 'fixed',
         'pix': 'fixed'}[unit]
    descript -= {'fixed'}   # if both fixed and unit that implies fixed are given

    # resolve the combine function.
    # if aperture sizes fixed (i.e unit is '"' or  'pix'), cfunc will be used upon init only
    for name in descript:
        f = getattr(np, name, None)
        if f:
            descript -= {name}
            break
    else:
        f = np.nanmean  # Default is to scale by mean sigma / theta

    if len(descript):
        raise ValueError('%s not understood' %str(tuple(descript)))
    # Construct the class

    bases = mxnDict[unit] + (cls,)
    name = kind.title() + s.title() + 'Aperture'
    cls = type(name, bases, {'cfunc': staticmethod(f)})

    return cls, sizes, unit



# TODO: can use detect_sources + binary dilation to make the source mask
# TODO: checkout photutils.segmentation.detect.make_source_mask

class MaskHelper(MaskMachine):
    """
    Mixin class for helping produce masks for photometry / fitting from input radii in units of sigma
    """
    # def __init__(self, grid, coords=None, r_sigma=5.0, rsky_sigma=(4.5, 8.5)):
        # MaskMachine.__init__(self, grid, coords)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    @staticmethod
    def shrink(mask):
        # since we expect the masks to be sparse, pass down the indices to save RAM
        return np.where(mask)

    @staticmethod
    def expand_mask(mask, shape):
        # expand mask indices to boolean array of given shape
        z = np.zeros(shape, dtype=bool)
        z[mask] = True
        return z

    def skymask(self, rstar, rsky):
        """Mask all other stars within the sky annulus of frame star"""
        _, allmasked, _ = self.mask_all_others(rstar)

        skyannuli = self.mask_annuli(rsky)
        skymask = allmasked & skyannuli.any(-1)  #[..., None]
        # all stars inside all sky annuli masked in single frame
        return self.shrink(skymask)

    def get_masks(self, cxx, sigma, with_sky=False):
        """sigma shape: (), (1,) (N, 2) """
        # mask all pixels for fitting
        # cxx = coo + self.Rvec
        self.update(cxx)

        # sigma = np.zeros(len(self.Rcoo)) * sigma
        # sigma[:Nstars] = sigma0
        # sigma[Nstars:] = sigma0  # [ir]

        r_sigma = self.r * sigma
        rsky_sigma = self.rsky * np.atleast_2d(sigma) #
        # rsky_out = rsky_sigma[:, 1]

        # TODO: mask SATURATED pixels ??
        # NOTE: r_sigma may be array in case of multiaperture.  In this case mask will take largest radius

        # masking the other stars requires photometry on individual
        # apertures instead of groups of them with the same size. Is this slower?
        # Better to weight pixels?
        photmasks = self.mask_close_others(np.max(r_sigma), rsky_sigma[:, 1])
        if with_sky:
            # This would otherwise just be a waste of time since we don't need the skymasks for
            # fitting, only for photometry
            # NOTE: if r_sigma > rsky_sigma[0] there will be a ring of masked pixels in the sky mask
            skymasks = self.skymask(np.max(r_sigma), rsky_sigma.T)
            # if max(map(len, photmasks)) > 0:
            #     # logger.info('INTERFERANCE: frame {}'.format(i))
            return photmasks, skymasks
        return photmasks


# TODO:
# class SlotmodeMasks(MaskHelper):
#         #Edge mask
#         edgemask = np.ones(ff.ishape, bool)       #FIXME: NameError: name 'ff' is not defined
#         edgemask[6:-8,
#                  0:-15] = False
#
#         def skymask(self, rstar, rsky):
#             """Mask all other stars within the sky annulus of frame star"""
#             _, allstars, _ = self.mask_all_others(rstar)
#
#             skyannuli = self.mask_annuli(rsky)
#             skymask = self.edgemask & allstars & skyannuli.any(-1)
#             return self.shrink(skymask)
#
#         def get_masks(self, cxx, sigma, with_sky=False):
#             """sigma shape: (), (1,) (N, 2) """
#             # mask all pixels for fitting
#             # cxx = coo + self.Rvec
#             self.update(cxx)
#
#             r_sigma = self.r * sigma
#             rsky_sigma = self.rsky * np.atleast_2d(sigma) #
#             # rsky_out = rsky_sigma[:, 1]
#
#             photmasks = self.mask_close_others(r_sigma, rsky_sigma[:, 1])
#
#
#             if with_sky:
#                 # This would otherwise just be a waste of time since we don't need the skymasks for
#                 # fitting, only for photometry
#                 # NOTE: if r_sigma > rsky_sigma[0] there will be a ring of masked pixels in the sky mask
#                 skymasks = self.skymask(r_sigma, rsky_sigma.T)
#                 # if max(map(len, photmasks)) > 0:
#                 #     # logger.info('INTERFERANCE: frame {}'.format(i))
#                 return photmasks, skymasks
#             return photmasks


# TODO: inherit from photutils.aperture!?!
class SigmaScalerBase(MaskHelper):

    # axis along which to apply combine function
    axis = ()     # will return original array and therefore scale star apertures individually by default
    #NOTE this is ~4x slower than using a lambda to accomplish the same
    # cfunc = staticmethod(np.nanmedian)    # default combiner

    def __init__(self, grid, coords=None, r_sigma=5.0, rsky_sigma=(5.0, 8.5)):
        # print('init', 'SigmaScalerBase')
        self.r = np.atleast_1d(r_sigma)
        self.rsky = np.asarray(rsky_sigma)

        # Attach mask methods
        MaskHelper.__init__(self, grid, coords)

    def scale(self, sigma_xy, theta):
        return (self.cfunc(sigma_xy, axis=self.axis),
                self.cfunc(theta, axis=self.axis))

    def get_radii(self, sigma_xy, theta):

        sigma_xy, theta = self.scale(sigma_xy, theta)

        rxy = (self.r * np.array([sigma_xy]).T).T           # aperture radii xy
        rsky = np.multiply(self.rsky, sigma_xy)             # sky radii (circular)
        return rxy, rsky, sigma_xy, theta

    def prepare_phot(self, cxx, sigma_xy, theta):
        """
        Scale aperture radii, and sky annulus for frame i by the mean / median sigma (std_dev)
        of the fit params in ix_scale.
        """
        # cxx = coo + tracker.Rvec
        rxy, rsky, sigma_xy, theta = self.get_radii(sigma_xy, theta)

        # update (recalculate) the star/bg masks based on coo and sigma_xy from fits
        # TODO: use photutils aperture methods to deal with masks?
        photmasks, skymasks = self.get_masks(cxx, np.mean(sigma_xy), with_sky=True)

        return photmasks, skymasks, cxx, rxy, rsky, theta


class ScalerCircular(SigmaScalerBase):
    # will combine across stars and (sigma_x and sigma_y) of all stars to scale
    axis = None

    def scale(self, sigma_xy, theta=None):
        sxy = self.cfunc(sigma_xy)
        return np.array([sxy, sxy]), 0


class ScalerElliptical(SigmaScalerBase):
    # combine across all modelled stars to scale
    axis = 0


class ScaleFixed():
    """Mixin class that returns fixed aperture radii"""
    # axis = None     # only used on init

    def __init__(self, grid, coords=None, r_sigma=5.0, rsky_sigma=(5.0, 8.5)):
        # print('init', 'ScaleFixed')
        super().__init__(grid, coords, r_sigma, rsky_sigma)

        # embed()
        sigma_xy, theta = super().scale(sigma_xy0, theta0)
        rxy = (self.r * np.array([sigma_xy]).T).T           # aperture radii xy
        rsky = np.multiply(self.rsky, sigma_xy)             # sky radii (circular)

        self._fixed_results = rxy, rsky, sigma_xy, theta #= super().get_radii(sigma_xy0, theta0)
        self._sigma_xy_fixed = sigma_xy
        self._theta_fixed = theta

    def scale(self, sigma_xy, theta=None):
        return self._sigma_xy_fixed, self._theta_fixed

    def get_radii(self, sigma_xy, theta):
        return self._fixed_results


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
    data_stddev = np.ones_like(data)        # FIXME:

    # track stars
    coo = mdlr.tracker(i, data)
    cxx = coo + mdlr.tracker.Rvec # now relative to frame
    # save coordinates in shared data array.
    # NOTE: do this here for multiprocessing. since the fitting below is potentially slow, and we
    #       want *coo_tr* of this frame to be available to other processes. if the tracker is
    #       StarTrackerFit, the call below effectively does nothing
    tracker.save_coords(i, coo)

    # First guess params from previous frames if available
    _, sigma_xy0, theta0 = mdlr.average_best_params(i)

    # mask
    fitmasks = apscaler.get_masks(cxx, sigma_xy0.mean())

    # PSF photometry
    coo_fit, sigma_xy, theta = mdlr.fit(cxx, data, data_stddev, fitmasks, i=i)

    # save coordinates in shared data array.
    mdlr.save_coords(i, coo_fit)
    # only overwrites coordinates if mdlr.tracker is None

    # Aperture photometry
    cxx = coords[i] + tracker.Rvec       # coordinates via requested method
    photmasks, skymask, cxx, rxy, rsky, theta = apscaler.prepare_phot(cxx, sigma_xy, theta)
    check_aps_sky(i, rsky)

    # a quantity is needed for photutils
    udata = u.Quantity(data, copy=False)

    # BG phot
    flux_bg_pp = do_bg_phot(udata, skymask, cxx[:Nstars], rsky)  #for Nstars
    apData.bg[i] = flux_bg_pp

    # Do aperture photometry
    if np.size(photmasks):
        # same stars are close together
        apData.flux[i] = do_phot(data, photmasks, cxx, rxy, theta, flux_bg_pp)
    else:
        apData.flux[i] = do_multi_phot(data, photmasks, cxx, rxy, theta, flux_bg_pp)

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
        return shocRNT.get_saturation(h)
    except Exception as err:
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
parser.add_argument('--window', default=20, type=int,
                    help='Tracking window size')
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
window = args.window
# create xy grid
Grid = ndgrid.like(ff[0])  # grid for full frame



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


Nmean = 3
preImage = np.median(ff[:Nmean], 0)
preFind = SourceFinder(preImage, snr=3, npixels=7, deblend=True)






#TODO: Gui here. add stars? resize apertures / sky / window interactively
# print('ADD COO ' * 100)
# embed()
# HACK
Rcoo = np.vstack([preFind.found, (19.5, 90.5)])# HACK
# HACK

Nstars = min(len(Rcoo), N_max_stars)

# Get tracker
TrackerClass, ix_loc_rqst, centreFunc, bgFunc = trackerFactory(args.track)
tracking = TrackerClass is not StarTrackerFixedWindow

# ix_scale = list(range(Nscale))  # brightest stars (since they are sorted)

# Pre-diagnostics to help choose algorithmic parameters
# Estimate max shift in pixels throughout run
mxshift, max_image, seg_image_max = estimate_max_shift(ff, 100)
large_shifts = np.any(mxshift > window)
#TODO: cosmic ray flagging in the envelope??


# def setup():
# initialize tracker class
# =================================================================================================
# check which stars are good for centroid tracking
satlvl = get_saturation(ff.header)
# window_scale = 12.5
# window = int(np.ceil(sigma0 * window_scale))
ix_loc = preFind.best_for_tracking(window, saturation=satlvl)
tracker = TrackerClass(Rcoo, window, ix_loc, max_stars=N_max_stars,
                      cfunc=centreFunc, bgfunc=bgFunc)
ix_loc = tracker.ix_loc

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
    from decor.profile import HLineProfiler
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
    init_shared_memory(N, Nstars, Naps, Nfit)
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