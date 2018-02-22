# standard library
import sys
import os
import time
import ctypes
import shutil
import socket
import inspect
import logging
import logging.config
import logging.handlers
import tempfile
import itertools as itt
from pathlib import Path
from collections import defaultdict
import multiprocessing as mp
from multiprocessing.managers import SyncManager  # , NamespaceProxy

# ===============================================================================
# Check input file | doing this before all the slow imports
fitsfile = sys.argv[1]
fitspath = Path(fitsfile)
path = fitspath.parent
if not fitspath.exists():
    raise IOError('File does not exist: %s' % fitsfile)

# execution time stamps
from motley.profiler.timers import Chrono, timer_extra

chrono = Chrono()
chrono.mark('start')

# ===============================================================================
# related third party libs
import numpy as np
import astropy.units as u
# from joblib import Parallel, delayed
from joblib.pool import MemmappingPool
import more_itertools as mit

chrono.mark('Imports: 3rd party')

# local application libs
from recipes.interactive import is_interactive
from recipes.parallel.synced import SyncedCounter
from recipes.logging import ProgressLogger

import slotmode
from slotmode.vignette import Vignette2DCross
# from slotmode import get_bad_pixel_mask
# * #StarTracker, SegmentationHelper, GraphicalStarTracker
#
from obstools.fastfits import FitsCube
from obstools.phot.trackers import SlotModeTracker, estimate_max_shift
from obstools.modelling import ImageModeller, AperturesFromModel, make_shared_mem
from obstools.modelling.bg import Poly2D
from obstools.modelling.psf.models_lm import EllipticalGaussianPSF
from obstools.phot import log  # listener_process, worker_init, logging_config
from obstools.phot.proc import FrameProcessor, TaskExecutor

# from motley.printers import func2str

from IPython import embed

chrono.mark('Import: local libs')
# ===============================================================================

# class SyncedProgressLogger(ProgressLogger):
#     def __init__(self, precision=2, width=None, symbol='=', align='^', sides='|',
#                  logname='progress'):
#         ProgressLogger.__init__(self, precision, width, symbol, align, sides, logname)
#         self.counter = SyncedCounter()


# TODO: colourful logs - like daquiry / technicolor
# TODO: python style syntax highlighting for exceptions in logs ??

__version__ = 3.14519

# class MyManager(Manager):
#     pass

SyncManager.register('Counter', SyncedCounter)


# MyManager.register('ProgressBar', SyncedProgressLogger)

def Manager():
    m = SyncManager()
    m.start()
    return m


def task(size, maxfail=None):
    # a little task factory
    counter = manager.Counter()
    fail_counter = manager.Counter()
    return TaskExecutor(size, counter, fail_counter, maxfail)


# def lm_extract_values_stderr(pars):
#     return np.transpose([(p.value, p.stderr) for p in pars.values()])


if __name__ == '__main__':
    # def main():
    chrono.mark('Main start')

    import argparse

    ncpus = os.cpu_count()

    # parse command line args
    parser = argparse.ArgumentParser('phot',  # fromfile_prefix_chars='@',
                                     description='Parallelized generic time-series photometry routines')
    parser.add_argument('fitsfile', type=FitsCube,
                        help='filename of fits data cube to process.')
    # TODO: process many files at once
    parser.add_argument(
            '-n', '--subset', nargs='*', type=int,
            help='Data subset to load. Useful for testing/debugging. If not given, '
                 'entire cube will be processed. If a single integer, first `n` '
                 'frames will be processed. If 2 integers `n`, `m`, all frames '
                 'starting at `n` and ending at `m-1` will be processed.')
    parser.add_argument(
            '-j', '--nprocesses', type=int, default=ncpus,
            help='Number of worker processes running concurrently in the pool.'
                 'Default is the value returned by `os.cpu_count()`: %i.'
                 % ncpus)
    parser.add_argument(
            '-k', '--clobber', action='store_true',
            help='Whether to resume computation, or start afresh. Note that the '
                 'frames specified by the `-n` argument will be recomputed if '
                 'overlapping with previous computation irrespective of the '
                 'value of `--clobber`.')

    parser.add_argument(
            '-a', '--apertures', default='circular',
            choices=['c', 'cir', 'circle', 'circular',
                     'e', 'ell', 'ellipse', 'elliptical'],
            help='Aperture specification')
    # TODO: option for opt


    # plotting
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--plot', action='store_true', default=True,
                        help='Do plots')
    group.add_argument('--no-plots', dest='plot', action='store_false',
                        help="Don't do plots")

    parser.add_argument('--verbose', '-v', action='count')
    parser.add_argument('--version', action='version',
                        version='%(prog)s %s')
    args = parser.parse_args(sys.argv[1:])


    # data
    cube = args.fitsfile

    # check / resolve options
    clobber = args.clobber
    # subset of frames for compute
    if args.subset is None:
        subset = (0, len(cube))
    elif len(args.subset) == 1:
        subset = (0, min(args.subset[0], len(cube)))
    elif len(args.subset) == 2:
        subset = args.subset
    else:
        raise ValueError('Invalid subset: %s' % args.subset)
    # number of frames to process
    subsize = np.ptp(subset)

    # ===============================================================================
    # Decide how to log based on where we're running

    if socket.gethostname().startswith('mensa'):
        plot_lightcurves = plot_diagnostics = False
        print_progress = False
        log_progress = True
    else:
        plot_lightcurves = plot_diagnostics = args.plot
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
        from recipes.io.utils import WarningTraceback

        # check_mem = True            # prevent execution if not enough memory available
        monitor_mem = False  # True
        monitor_cpu = False  # True  # True
        # monitor_qs = True  # False#

        # setup warnings to print full traceback
        wtb = WarningTraceback()

    logging.captureWarnings(True)

    # print section timing report at the end
    exit_register(chrono.report)

    # ===============================================================================
    # create folder output data to be saved
    resultsPath = fitspath.with_suffix('.phot')
    # create logging directory
    logPath = resultsPath / 'logs'
    if not logPath.exists():
        logPath.mkdir(parents=True)

    # create directory for plots
    if plot_diagnostics:
        figPath = resultsPath / 'plots'
        if not figPath.exists():
            figPath.mkdir()

    # setup logging processes
    logQ = mp.Queue()  # The logging queue for workers
    # TODO: open logs in append mode if resume
    config_main, config_listener, config_worker = log.config(logPath, logQ)
    logging.config.dictConfig(config_main)
    # create log listner process
    logger = logging.getLogger('setup')
    logger.info('Creating log listener')
    stop_logging_event = mp.Event()
    logListner = mp.Process(target=log.listener_process, name='logListener',
                            args=(logQ, stop_logging_event, config_listener))
    logListner.start()
    logger.info('Log listener active')
    #
    chrono.mark('Logging setup')

    # ===============================================================================
    # Image Processing setup

    # FIXME: any Exception that happens below will stall the log listner indefinitely

    n = len(cube)
    frame0 = cube[0]

    # create flat field from sky data
    # flat = slotmode.make_flat(cube[2136:])
    # image /= flat
    # TODO: save this somewhere so it's quicker to resume
    # TODO: same for all classes here

    # create the global background model
    # mdlBG = Poly2D(1, 1)
    # p0bg = np.ones(mdlBG.npar)  # [np.ones(m.nfree) for m in mdlBG.models]
    # edge_cutoffs = slotmode.get_edge_cutoffs(image)
    # border = make_border_mask(image, edge_cutoffs)
    # mdlBG.set_mask(border)

    # for SLOTMODE: fit median cross sections with piecewise polynomial
    # orders_x, orders_y = (1, 5), (4, 1, 5)
    # bpx = np.multiply((0, 0.94, 1), image.shape[1])
    # bpy = np.multiply((0, 0.24, 0.84, 1), image.shape[0])
    # #orders_y, bpy = (3, 1, 5), (0, 3.5, 17, image.shape[0]) # 0.15, 0.7
    # mdlBG = Vignette2DCross(orders_x, bpx, orders_y, bpy)

    # TODO: try fit for the breakpoints ??
    orders_x, orders_y = (5, 1), (4, 1, 5)
    # bpx = np.multiply((0, 0.94, 1), image.shape[1])
    sy, sx = cube.ishape
    bpx = (0, 10, sx)
    # bpy = np.multiply((0, 0.24, 0.84, 1), image.shape[0])
    bpy = (0, 10, 37.5, sy)
    mdlBG = Vignette2DCross(orders_x, bpx, orders_y, bpy)
    mdlBG.set_grid((sy, sx))

    # bad pixels
    bad_cols = [59, 295]
    bad_pix = [(22, 262), (0, 96), (1, 96), (2, 96), (3, 96)]
    #  Some SLOTMODE cubes have all 0s in first column for some obscure reason
    if np.median(frame0[:, 0]) / np.median(cube[0]) < 0.2:
        bad_cols.append(0)
    #
    bad_pixel_mask = slotmode.make_bad_pixel_mask(frame0, bad_pix, bad_cols)

    # init the tracker
    # TODO: SlotModeTracker.from_cube(cube, ncomb, *args)
    tracker, image, p0bg = SlotModeTracker.from_cube_segment(cube, 25, 100,
                                                             mask=bad_pixel_mask,
                                                             bgmodel=mdlBG,
                                                             snr=(10, 3),
                                                             dilate=(3,))

    # estimate maximal positional shift of stars by running detection loop on
    # maximal image of 1000 frames evenly distributed across the cube
    mxshift, maxImage, segImx = estimate_max_shift(cube, 1000, bad_pixel_mask,
                                                   snr=3)

    # TODO: plot some initial diagnostics
    # image, detections, rcoo, bad_pixels


    # TODO: for slotmode: if stars on other amplifiers, can use these to get sky
    # variability and decorellate TS

    # TODO: set p0ap from image
    sky_width, sky_buf = 5, 0.5
    if args.apertures.startswith('c'):
        p0ap = (3,)
    else:
        p0ap = (3, 3, 0)


    # create psf models
    # models = EllipticalGaussianPSF(),
    models = ()
    # create image modeller
    mdlr = ImageModeller(tracker.segm, models, mdlBG,
                         use_labels=tracker.use_labels)

    # create object that generates the apertures from modelling results
    cmb = AperturesFromModel(3, (8, 12))

    chrono.mark('Pre-compute')

    # ===============================================================================
    # create shared memory
    # aperture positions / radii / angles
    nstars = tracker.nsegs
    ngroups = tracker.ngroups
    naps = 1  # number of apertures per star

    # create frame processor
    proc = FrameProcessor()
    # TODO: folder structure for multiple aperture types circular / elliptical
    proc.init_mem(n, nstars, ngroups, naps, resultsPath, clobber=clobber)

    # modelling results
    mdlr.init_mem(n, nstars, resultsPath / 'modelling', clobber=clobber)
    # BG residuals
    bgResiduPath = resultsPath / 'cube-bg'
    bgshape = (n,) + cube.ishape
    residu = make_shared_mem(bgResiduPath, bgshape, np.nan, clobber=clobber)

    # aperture parameters
    # cmb.init_mem(n, resultsPath / 'aps.par', clobber=clobber)
    optStatPath = resultsPath / 'opt.stat'
    opt_stat = make_shared_mem(optStatPath, (n, tracker.ngroups), np.nan, clobber=clobber)

    # tracking data
    cooFile = resultsPath / 'coords'
    coords = make_shared_mem(cooFile, (n, 2), np.nan, clobber=clobber)

    chrono.mark('Memory alloc')

    # ===============================================================================
    # main compute
    # synced counter to keep track of how many jobs have been completed
    manager = Manager()

    # task executor  # TODO: there is probably a better one in joblib
    Task = task(subsize)
    worker = Task(proc.process)

    chunks = mit.divide(args.nprocesses, range(*subset))
    chunks2 = chunks.copy()
    rngs = mit.pairwise(next(zip(*chunks2)) + (subset[1],))


    def proc_(irng, data, residu, coords, opt_stat,
              tracker, mdlr,
              p0bg, p0ap, sky_width, sky_buf):

        # use detections from max image to compute CoM and offset from initial
        # reference image
        i0, i1 = irng
        coo = segImx.com_bg(data[i0])
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)
        lbad2 = ~segImx.inside_detection_region(coo)
        weights = (~(lbad | lbad2)).astype(int)
        tracker.update_offset(coo, weights)

        for i in range(*irng):
            worker(i, data, residu, coords, opt_stat, tracker, mdlr,
                   p0bg, p0ap, sky_width, sky_buf)

    try:
        # Fork the worker processes to perform computation concurrently
        logger.info('About to fork into %i processes', args.nprocesses)

        # NOTE: This is for testing!!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.map(Task(test), range(*subset))

        # NOTE: This is for tracking!!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.starmap(bg_sub,
        #         ((chunk, cube.data, residu, coords, tracker, mdlr)
        #             for chunk in chunks))

        # NOTE: This is for photometry!
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker,)) as pool:
        #     results = pool.starmap(
        #             Task(proc.proc1),
        #             ((i, cube.data, residu, coords, tracker, optstat,
        #               p0ap, sky_width, sky_buf)
        #              for i in range(*subset)))


        # from IPython import embed
        # embed()
        # raise SystemExit

        # NOTE: chunked sequential mapping (doesn't work if there are frame shifts)
        # chunks = mit.divide(args.nprocesses, range(*subset))
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker,)) as pool:
        #     results = pool.starmap(proc_,
        #             ((chunk, cube.data, residu, coords, opt_stat,
        #               tracker, mdlr, p0bg, p0ap,
        #               sky_width, sky_buf)
        #                 for chunk in chunks))

        #
        with MemmappingPool(args.nprocesses, initializer=log.worker_init,
                            initargs=(config_worker,)) as pool:
            results = pool.starmap(proc_,
                ((rng, cube.data, residu, coords, opt_stat,
                  tracker, mdlr, p0bg, p0ap, sky_width, sky_buf)
                        for rng in rngs))




        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker, )) as pool:
        #     results = pool.starmap(
        #         proc, ((i, cube.data, coords,
        #                 tracker, mdlr, cmb,
        #                 successes, failures,
        #                 counter, prgLog)
        #                 for i in range(*subset)))

        # get frame numbers of successes and failures

    except Exception as err:
        # catch errors so we can safely shut down the listeners
        logger.exception('Exception during parallel loop.')
        plot_diagnostics = False
        plot_lightcurves = False
    else:
        # put code here that that must be executed if the try clause does
        # not raise an exception
        # The use of the else clause is better than adding additional code to
        # the try clause because it avoids accidentally catching an exception
        # that wasn’t raised by the code being protected by the try … except
        # statement.

        # Hang around for the workers to finish their work.
        pool.join()
        logger.info('Workers done')  # Logging in the parent still works normally.
        chrono.mark('Main compute')
    finally:
        # A finally clause is always executed before leaving the try statement,
        # whether an exception has occurred or not.
        # any unhandled exceptions will be raised after finally clause,
        # basically just KeyboardInterrupt for now.

        # check task status
        failures = Task.report()
        # TODO: print opt failures

        # Workers all done, listening can now stop.
        logger.info('Telling listener to stop ...')
        stop_logging_event.set()
        logListner.join()

        chrono.mark('Process shutdown')

        # diagnostics
        if plot_diagnostics:

            # TODO: GUI
            # TODO: if interactive dock figs together
            # dock for figures
            # connect ts plots with frame display

            from obstools.phot.diagnostics import new_diagnostics, save_figures

            figs = new_diagnostics(coords, tracker.rcoo[tracker.ir],
                                   proc.Appars, opt_stat)
            save_figures(figs, figPath)

        if plot_lightcurves:
            from obstools.phot.diagnostics import plot_aperture_flux

            figs = plot_aperture_flux(fitspath, proc, tracker)
            save_figures(figs, figPath)

        chrono.mark('Diagnostics')
        chrono.report()  # TODO: improve report formatting

        # try:
        # from _qtconsole import qtshell  # FIXME
        # qtshell(vars())
        # except Exception as err:
        embed()

    # with mp.Pool(10, worker_logging_init, (config_worker, )) as pool:   # , worker_logging_init, (q, logmix)
    #     results = pool.starmap(
    #         work, ((i, counter, prgLog)
    #                for i in range(n)))

    # #
    # import sys
    # from recipes.io.utils import TracePrints
    # sys.stdout = TracePrints()

    # n = 50
    # with Parallel(n_jobs=8, verbose=0, initializer=worker_logging_init,
    #               initargs=(counter, config_worker)) as parallel: #)) as parallel:#
    #     results = parallel(
    #         delayed(work)(i)#, cube.data, tracker, mdlr, counter, residu)
    #         for i in range(n))

    # sys.stdout = sys.stdout.stdout

# if __name__ == '__main__':
#     main()
