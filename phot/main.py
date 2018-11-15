# standard library
import multiprocessing as mp
import os
import socket
import sys
from multiprocessing.managers import SyncManager  # , NamespaceProxy
from pathlib import Path

import itertools as itt

import logging.handlers

# ===============================================================================
# Check input file | doing this before all the slow imports

fitsfile = sys.argv[1]
fitspath = Path(fitsfile)
path = fitspath.parent
if not fitspath.exists():
    raise IOError('File does not exist: %s' % fitsfile)

# execution time stamps
from motley.profiler.timers import Chrono

chrono = Chrono()
# TODO: option for active reporting; inherit from LoggingMixin, make Singleton
chrono.mark('start')

# ===============================================================================
# related third party libs
import numpy as np
from scipy import ndimage
from joblib.pool import MemmapingPool as MemmappingPool  # Parallel, delayed
from addict.addict import Dict
import more_itertools as mit

chrono.mark('Imports: 3rd party')

# local application libs
from obstools.phot.utils import ImageSampler  # , rand_median
from obstools.phot.proc import FrameProcessor
from obstools.modelling.utils import make_shared_mem_nans

from recipes.interactive import is_interactive
from recipes.parallel.synced import SyncedCounter

import slotmode
from slotmode.modelling.image import SlotBackground, SlotModeBackground
from obstools.modelling import nd_sampler
from slotmode.modelling.image import FrameTransferBleed

# from slotmode import get_bad_pixel_mask
# * #StarTracker, SegmentationHelper, GraphicalStarTracker
#
# from obstools.modelling.core import *
# from obstools.fastfits import FitsCube
# from obstools.modelling.bg import Poly2D
# from obstools.modelling.psf.models_lm import EllipticalGaussianPSF
from obstools.phot import log  # listener_process, worker_init, logging_config
from obstools.phot.proc import TaskExecutor
from obstools.phot.utils import id_stars_kmeans, merge_segmentations, \
    shift_combine
from obstools.phot.tracking import StarTracker
from obstools.phot.segmentation import SegmentationHelper

# SlotModeTracker, check_image_drift


# from motley.printers import func2str
from graphical.imagine import ImageDisplay
from graphical.multitab import MplMultiTab

chrono.mark('Import: local libs')

# version
__version__ = 3.14519


# TODO: colourful logs - like daquiry / technicolor
# TODO: ipython style syntax highlighting for exceptions in logs ??

# TODO: for slotmode: if stars on other amplifiers, can use these to get sky
# variability and decorellate TS

# todo: these samples can be done with Mixin class for FitsCube
# think `cube.sample(100).max()` # :))

# ===============================================================================
# def create_sample_image(interval, ncomb):
#     image = sampler.median(ncomb, interval)
#     scale = nd_sampler(image, np.median, 100)
#     mimage = np.ma.MaskedArray(image, BAD_PIXEL_MASK, copy=True)
#     # copy here prevents bad_pixel_mask to be altered (only important if
#     # processing is sequential)
#     return mimage / scale, scale

def create_sample_image(interval, ncomb):
    image = sampler.median(ncomb, interval)
    # scale = nd_sampler(image, np.median, 100)
    # image_NORM = image / scale
    mimage = np.ma.MaskedArray(image, BAD_PIXEL_MASK)
    return mimage  # , scale


def init_model(interval, ncomb, snr=3, npixels=5, deblend=False, dilate=2):
    image = create_sample_image(interval, ncomb)
    mdl, seg = SlotModeBackground.from_image(image, SPLINE_ORDERS, snr,
                                             npixels, deblend, dilate)
    return mdl, seg, image  # , scale


def display(image, title=None, ui=None, **kws):
    if isinstance(image, SegmentationHelper):
        im = image.display(**kws)
    else:
        im = ImageDisplay(image, **kws)

    if title:
        im.ax.set_title(title)

    if args.live:
        idisplay(im.figure)
    return im


# class MyManager(Manager):
#     pass

SyncManager.register('Counter', SyncedCounter)


# MyManager.register('ProgressBar', SyncedProgressLogger)

def Manager():
    m = SyncManager()
    m.start()
    return m


def task(size, max_fail=None):
    # a little task factory
    counter = manager.Counter()
    fail_counter = manager.Counter()
    return TaskExecutor(size, counter, fail_counter, max_fail)


if __name__ == '__main__':
    #
    chrono.mark('Main start')
    from motley import banner
    import argparse

    # say hello
    header = banner('⚡ ObsTools Photometry ⚡', align='^')
    header = header + '\nv1.3\n'  # __version__
    print(header)

    # how many cores?!?
    ncpus = os.cpu_count()

    # parse command line args
    parser = argparse.ArgumentParser(
            'phot',  # fromfile_prefix_chars='@',
            description='Parallelized generic time-series photometry routines')

    # FIXME: you aren't loading fits files anymore!!
    parser.add_argument('fitsfile', type=str,  # type=FitsCube,
                        help='filename of fits data cube to process.')
    parser.add_argument('-ch', '--channel', type=int,  # type=FitsCube,
                        help='amplifier channel',
                        required=True)  # TODO: if not given, do all channels!!
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
            help='Whether to resume computation, or start afresh. Note that the'
                 ' frames specified by the `-n` argument will be recomputed if '
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

    # add some args manually
    args.live = is_interactive()

    # load image data (memmap shape (n, 4, r, c))
    cube4 = np.lib.format.open_memmap(args.fitsfile)  # TODO: paths.input
    ch = args.channel
    cube = cube4[:, ch]

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

    if args.live:  # turn off logging when running interactively (debug)
        from recipes.interactive import exit_register
        from IPython.display import display as idisplay

        log_progress = print_progress = False
        monitor_mem = False
        monitor_cpu = False
        # monitor_qs = False
    else:
        from atexit import register as exit_register
        from recipes.io.utils import WarningTraceback

        # check_mem = True    # prevent execution if not enough memory available
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
    resultsPath = fitspath.with_suffix('.ch%i.phot' % ch)
    # create logging directory
    logPath = resultsPath / 'logs'  # TODO: paths.log
    if not logPath.exists():
        logPath.mkdir(parents=True)

    # create directory for plots
    if plot_diagnostics:
        figPath = resultsPath / 'plots'  # TODO: paths.figures
        if not figPath.exists():
            figPath.mkdir()

    # setup logging processes
    logQ = mp.Queue()  # The logging queue for workers
    # TODO: open logs in append mode if resume
    config_main, config_listener, config_worker = log.config(logPath, logQ)
    logging.config.dictConfig(config_main)
    # create log listner process
    logger = logging.getLogger()
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
    from slotmode.imaging import image_full_slot

    # plot quick-look image for all 4 amplifier channels
    idisplay(image_full_slot(cube4[10], ch))

    # FIXME: any Exception that happens below will stall the log listner indefinitely

    n = len(cube)
    ishape = cube.shape[-2:]
    frame0 = cube[0]

    # bad pixels
    # ----------
    BAD_PIXEL_MASK = slotmode.get_bad_pixel_mask(frame0, args.channel + 1)
    # TODO: handle bad_pixels as negative label in SegmentationHelper ???
    # these can actually now just be label 0. however special treatment desired
    # since we don't really need slices of these
    # TODO: option to try flat field out the bad pixels
    # TODO: ALTERNATIVELY: try to flat field out bad pixels??

    # flat fielding
    # -------------
    # Some of the bad pixels can be flat fielded out
    flatpath = fitspath.with_suffix('.flat.npz')  # TODO: paths.calib.flat
    if flatpath.exists():
        logger.info('Loading flat field image from %r', flatpath.name)
        flat = np.load(flatpath)['flat']
        flat_flat = flat[flat != 1]
    else:
        'still experimental'
        # construct flat field for known bad pixels by computing the ratio
        # between the median pixel value and the median of it's neighbours
        # across multiple (default 1000) frames
        # from slotmode import neighbourhood_median
        #
        # flat_cols = [100]
        # flat_ranges = [[14, (130, 136)],
        #                [15, (136, 140)]]
        # flat_mask = slotmode._mask_from_indices(ishape, None, flat_cols,
        #                                         None, flat_ranges)
        # flat = neighbourhood_median(cube, flat_mask)  #
        # flat_flat = flat[flat_mask]

    # calib = (None, flat)
    calib = (None, None)

    # create sample image
    sampler = ImageSampler(cube)  # prefer nd_sampler??

    # extract flat field from sky
    # flat = slotmode.make_flat(cube[2136:])
    # image /= flat
    # TODO: save this somewhere so it's quicker to resume
    # TODO: same for all classes here

    # create the global background model

    # This for Amp3 (channel 2) binning 3x3
    sy, sx = ishape
    SPLINE_ORDERS = YORD, XORD = (5, 1, 5), (5, 1)
    # TODO: print some info about the model: dof etc

    # estimate maximal positional shift of stars by running detection loop on
    # maximal image of 1000 frames evenly distributed across the cube
    # mxshift, maxImage, segImx = check_image_drift(cube, 1000, bad_pixel_mask,
    #                                               snr=5)

    # TODO: set p0ap from image
    sky_width, sky_buf = 5, 0.5
    if args.apertures.startswith('c'):
        p0ap = (3,)
    else:
        p0ap = (3, 3, 0)

    # create psf models
    # models = EllipticalGaussianPSF(),
    # models = ()
    # create image modeller
    # mdlr = ImageModeller(tracker.segm, models, mdlBG,
    #                      use_labels=tracker.use_labels)

    # create object that generates the apertures from modelling results
    # cmb = AperturesFromModel(3, (8, 12))

    chrono.mark('Pre-compute')
    # input('Musical interlude')

    # ===============================================================================
    # create shared memory
    # aperture positions / radii / angles
    # nstars = tracker.nsegs
    # # ngroups = tracker.ngroups
    # naps = 1  # number of apertures per star
    #
    # # TODO: single structured file may be better??
    # # TODO: paths.modelling etc..
    #
    # # create frame processor
    # proc = FrameProcessor()
    # # TODO: folder structure for multiple aperture types circular /
    # # elliptical / optimal
    # proc.init_mem(n, nstars, ngroups, naps, resultsPath, clobber=clobber)
    #
    # # modelling results
    # mdlr.init_mem(n, resultsPath / 'modelling/bg.par', clobber=clobber)
    # # BG residuals
    # bgResiduPath = resultsPath / 'residual'
    # bgshape = (n,) + ishape
    # residu = make_shared_mem_nans(bgResiduPath, bgshape, clobber)
    #
    # # aperture parameters
    # # cmb.init_mem(n, resultsPath / 'aps.par', clobber=clobber)
    # optStatPath = resultsPath / 'opt.stat'
    # opt_stat = make_shared_mem_nans(optStatPath, (n, tracker.ngroups), clobber)
    #
    # # tracking data
    # cooFile = resultsPath / 'coords'
    # coords = make_shared_mem_nans(cooFile, (n, 2), clobber)
    #
    # chrono.mark('Memory alloc')

    # ===============================================================================
    # main compute
    # synced counter to keep track of how many jobs have been completed
    manager = Manager()

    # task executor  # there might be a better one in joblib ??
    Task = task(subsize)  # PhotometryTask


    # worker = Task(proc.process)

    def _pre_fit(i, mdl, image):
        # normalize image
        # scale = nd_sampler(image, np.ma.median, 100)
        # image_NORM = image / scale
        # mimage = np.ma.MaskedArray(image_NORM, bad_pixel_mask, copy=True)
        # optimize knot positions

        # note: leastsq is ~4 times faster than other minimizers
        r_best = mdl.optimize_knots(image, method='leastsq')
        results = mdl.fit(image, method='leastsq')

        # rescale result
        # results[...] = tuple(r * scale for r in results.tolist())
        #

        # residuals
        residuals = mdl.residuals(results, image)

        # since pickled clones are optimized, we need to either set the
        # knots in the original models, or pass the model back to the main
        # process.  Avoid this by having models live entirely in forked process.
        # (todo)

        return mdl, results, residuals


    def proc_(interval, data, calib,
              model, residu,
              tracker, offset, coords,
              opt_stat, p0bg, p0ap, sky_width, sky_buf):
        # main routine for image processing for frames from data in interval

        logger.info('Starting frame processing for interval %s', interval)

        i0, i1 = interval

        for i in range(*interval):
            worker(i, data, calib, residu, coords, opt_stat, tracker, mdlr,
                   p0bg, p0ap, sky_width, sky_buf)


    # setup
    PreFitTask = task(args.nprocesses, '30%')
    pre_fit = PreFitTask(_pre_fit)

    chunks = mit.divide(args.nprocesses, range(*subset))
    chunks2 = chunks.copy()
    pairs = list(mit.pairwise(next(zip(*chunks2)) + (subset[1],)))

    # mdlr.logger.info('hello world')

    # raise SystemExit

    try:
        # Fork the worker processes to perform computation concurrently
        logger.info('About to fork into %i processes', args.nprocesses)

        # global parameters for object detection
        NCOMB = 10
        SNR = 3
        NPIX = 5
        DILATE = 2
        variters = tuple(map(itt.repeat, (NCOMB, SNR, NPIX, DILATE)))
        vargen = zip(pairs, *variters)
        #
        logger.info('Detecting stars & initializing models')
        # initialize worker pool
        pool = MemmappingPool(args.nprocesses, initializer=log.worker_init,
                              initargs=(config_worker,))
        # initialize models (concurrently)
        # TODO: do all this stuff in parallel to minimize serialization
        # overheads
        models, segmentations, sample_images = zip(
                *pool.starmap(init_model, vargen)
        )

        # aggregate results
        logger.info('Identifying stars')
        cx, ccom, shifts = id_stars_kmeans(sample_images, segmentations)
        ishifts = shifts.round()

        # global image segmentation
        logger.info('Combining sample image segmentations')
        segm_a = merge_segmentations(segmentations, ishifts)
        sh = SegmentationHelper(segm_a)
        sh.dilate(4, 2)
        tracking_labels = sh.labels

        # update segmentation for models
        for i, (mdl, shift) in enumerate(zip(models, ishifts)):
            # add global segmentation to model
            segm_stars = ndimage.shift(sh.data, shift)
            _, star_labels0 = mdl.segm.add_segments(segm_stars)
            mdl.groups['stars0'] = star_labels0

        # fit
        logger.info('Fitting sample images')
        counter = range(args.nprocesses)
        models, results, sample_residuals = zip(
                *pool.starmap(pre_fit, zip(counter, models, sample_images))
        )

        # residuals
        sample_residuals = np.ma.array(sample_residuals)

        # combined image
        mdl = models[0]
        mean_residuals = shift_combine(mdl.segm.grid, sample_residuals, shifts)
        # mask frame transfer bleeding
        mean_flux = sh.flux(mean_residuals)
        l = sh.labels[mean_flux > 75]
        sh_ft, new_labels = FrameTransferBleed._adapt_segments(sh, l, copy=True)
        # second round detection to pick up faint stars for forced photometry
        sh_ft.add_segments(sh)
        sky = sh_ft.mask_segments(mean_residuals)
        sh2 = sh.detect(sky, edge_cutoff=5)
        sh2.dilate(4, 2)
        # update global segmentation
        sh.add_segments(sh2)

        # update coordinates
        rcoo = sh.com(mean_residuals)

        # update segmentation for models (again)
        for i, (mdl, shift) in enumerate(zip(models, ishifts)):
            # add global segmentation to model
            segm_stars = ndimage.shift(sh2.data, shift)
            _, star_labels1 = mdl.segm.add_segments(segm_stars)
            mdl.groups['stars1'] = star_labels1
        #
        star_labels = np.r_[star_labels0, star_labels1]

        # plot results of sample fits
        if args.plot:
            # initial diagnostic images (for the modelled sample image)
            logger.info('Plotting results')

            # embed plots in multi tab window
            ui = MplMultiTab()
            for i, (image, mdl, params) in enumerate(
                    zip(sample_images, models, results)):
                #
                mimage = mdl.segm.mask_segments(image, star_labels)
                fig = mdl.plot_fit_results(mimage, params)
                ui.add_tab(fig, '%i:%s' % (i, pairs[i]))

            ui.show()

        # global background subtraction
        





        # setup camera tracker
        tracker = StarTracker(rcoo, sh, None, tracking_labels, BAD_PIXEL_MASK)

        # THIS IS FOR DEBUGGING PICKLING ERRORS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # import pickle
        # clone = pickle.loads(pickle.dumps(mdlr))
        #
        # for i in range(1000):
        #     if i % 10:
        #         print(i)
        #     proc.process(i, cube, calib, residu, coords, opt_stat,
        #                  tracker, clone, p0bg, p0ap, sky_width, sky_buf)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # rng = next(pairs)
        # proc_(rng, cube, residu, coords, opt_stat,tracker, mdlr, p0bg, p0ap, sky_width, sky_buf)

        # raise SystemExit

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
        #             ((chunk, cube, residu, coords, opt_stat,
        #               tracker, mdlr, p0bg, p0ap,
        #               sky_width, sky_buf)
        #                 for chunk in chunks))

        # from IPython import embed
        # embed()
        #
        # raise SystemExit
        #
        # with MemmappingPool(args.nprocesses, initializer=log.worker_init,
        #                     initargs=(config_worker,)) as pool:
        #     results = pool.starmap(proc_,
        #                            ((rng, cube, calib, residu, coords, opt_stat,
        #                              tracker, mdlr, p0bg, p0ap, sky_width,
        #                              sky_buf)
        #                             for rng in pairs))

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
        pool.close()
        pool.join()
        logger.info('Workers done')  # Logging in the parent still works
        chrono.mark('Main compute')

        # Workers all done, listening can now stop.
        logger.info('Telling listener to stop ...')
        stop_logging_event.set()
        logListner.join()
    finally:
        # A finally clause is always executed before leaving the try statement,
        # whether an exception has occurred or not.
        # any unhandled exceptions will be raised after finally clause,
        # basically only KeyboardInterrupt for now.

        # check task status
        # failures = Task.report()  # FIXME:  we sometimes get stuck here
        # TODO: print opt failures

        chrono.mark('Process shutdown')

        # diagnostics
        # if plot_diagnostics:
        #     # TODO: GUI
        #     # TODO: if interactive dock figs together
        #     # dock for figures
        #     # connect ts plots with frame display
        #
        #     from obstools.phot.diagnostics import new_diagnostics, save_figures
        #
        #     figs = new_diagnostics(coords, tracker.rcoo[tracker.ir],
        #                            proc.Appars, opt_stat)
        #     if args.live:
        #         for fig, name in figs.items():
        #             idisplay(fig)
        #
        #     save_figures(figs, figPath)
        #
        #     # GUI
        #     from obstools.phot.gui_dev import FrameProcessorGUI
        #
        #     gui = FrameProcessorGUI(cube, coords, tracker, mdlr, proc.Appars,
        #                             residu, clim_every=1e6)
        #
        # if plot_lightcurves:
        #     from obstools.phot.diagnostics import plot_aperture_flux
        #
        #     figs = plot_aperture_flux(fitspath, proc, tracker)
        #     save_figures(figs, figPath)

        chrono.mark('Diagnostics')
        chrono.report()  # TODO: improve report formatting

        if not args.live:
            # try:
            # from _qtconsole import qtshell  # FIXME
            # qtshell(vars())
            # except Exception as err:
            from IPython import embed

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
