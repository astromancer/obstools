# std libs
import time
import logging
import tempfile
import itertools as itt

# third-party libs
import numpy as np
import astropy.units as u
from scipy.optimize import minimize
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                EllipticalAperture,
                                EllipticalAnnulus
                                )

# local libs
from recipes import pprint
from recipes.string import resolve_percentage
from recipes.logging import LoggingMixin, ProgressLogger

# relative libs
from ..modelling.utils import load_memmap

OPT_SNR_THRESH = 10


class AbortCompute(Exception):
    pass


# TODO: Flux upper limits for faint stars merging into bg


# def lm_extract_values_stderr(pars):
#     return np.transpose([(p.value, p.stderr) for p in pars.values()])


# def weighted_avg_and_std(values, weights):
#     """
#     Return the weighted average and standard deviation.
#
#     values, weights -- Numpy ndarrays with the same shape.
#     """
#     average = np.average(values, weights=weights)
#     variance = np.average((values-average)**2, weights=weights)
#     return average, np.sqrt(variance)


def phot(ap, image, mask, method='exact'):
    """
    Calculate aperture sum, stddev and area while ignoring masked pixels

    Parameters
    ----------
    ap
    image
    mask
    method

    Returns
    -------

    """
    # get pixel values
    apMask = ap.to_boolean(method)[0]
    valid = apMask.cutout(~mask)
    pixWeights = apMask.data * apMask.cutout(~mask)
    pixVals = apMask.cutout(image)

    # weighted pixel sum
    apsum = (pixVals * pixWeights).sum()

    # weighted pixel deviation (for sky)
    area = pixWeights.sum()
    av = apsum / area
    apstd = np.sqrt((pixWeights * (pixVals - av) ** 2).sum())

    return apsum, apstd, area


def ap_stats(ap, image, masks, method='exact'):
    """
    Calculate aperture sum, stddev and area while ignoring masked pixels

    Parameters
    ----------
    ap
    image
    mask
    method

    Returns
    -------
    apsums
        weighted aperture sum
    stddevs
        weighted deviation from mean
     areas

    """

    # get pixel values
    m = len(ap.positions)

    if masks.ndim == 2:
        masks = masks[None]

    masks = np.atleast_3d(masks)
    assert len(masks) <= m
    apMasks = ap.to_mask(method)

    apsums = np.empty(m)
    areas = np.empty(m)
    stddevs = np.empty(m)

    for i, apMask, mask in itt.zip_longest(range(m), apMasks, masks,
                                           fillvalue=masks[0]):
        pixWeights = apMask.data * apMask.cutout(~mask)
        pixVals = apMask.cutout(image)

        # weighted pixel sum
        apsums[i] = apsum = (pixVals * pixWeights).sum()
        areas[i] = area = pixWeights.sum()

        # weighted pixel deviation (for sky)
        av = apsum / area
        stddevs[i] = np.sqrt((pixWeights * (pixVals - av) ** 2).sum())

    return apsums, stddevs, areas


def ap_sums(ap, image, masks, method='exact'):
    """
    Calculate aperture sum, stddev and area while ignoring masked pixels

    Parameters
    ----------
    ap
    image
    mask
    method

    Returns
    -------
    apsums
        weighted aperture sum
    stddevs
        weighted deviation from mean
     areas

    """

    # get pixel values
    m = len(ap.positions)

    if masks.ndim == 2:
        masks = masks[None]

    masks = np.atleast_3d(masks)
    assert len(masks) <= m
    apMasks = ap.to_mask(method)

    apsums = np.empty(m)
    areas = np.empty(m)

    for i, apMask, mask in itt.zip_longest(range(m), apMasks, masks,
                                           fillvalue=masks[0]):
        pixWeights = apMask.data * apMask.cutout(~mask)
        pixVals = apMask.cutout(image)

        # weighted pixel sum
        apsums[i] = (pixVals * pixWeights).sum()
        areas[i] = pixWeights.sum()

    return apsums, areas


def flux_estimate(ap, image, masks, ap_sky, im_sky, mask_sky, method='exact'):
    """
    Total background subtracted source counts and uncertainty in
    (ADU / electrons / photons) as well as
    background counts per pixel

    Parameters
    ----------
    ap
    image
    masks
    ap_sky
    im_sky
    mask_sky
    method

    Returns
    -------

    """
    counts, npix = ap_sums(ap, image, masks, method)
    counts_bg, std_bg, npixbg = ap_stats(ap_sky, im_sky, mask_sky, method)
    counts_std = std_ccd(counts, npix, counts_bg, npixbg)

    flx_bg_pp = counts_bg / npixbg
    flx_bg_std = std_bg / npixbg
    return counts, counts_std, flx_bg_pp, flx_bg_std


def snr_star(counts, npix, counts_bg, npixbg):
    # merline & howell 1995: revised CCD equation
    return counts / std_ccd(counts, npix, counts_bg, npixbg)


def std_ccd(counts, npix, counts_bg, npixbg):
    # howell & merlin 1995: revised CCD equation
    return np.sqrt(counts + npix * (1 + npix / npixbg) * (counts_bg / npixbg))


# def simul_objective(p0, cooxy, ops, im, masks_phot, im_sky, mask_sky,
#                     sky_width, sky_buf, r_sky_min):
#     r = np.empty(len(ops))
#     for j, (op, mask, cxy) in enumerate(zip(ops, masks_phot, cooxy)):
#         r[j] = op.objective(p0, cxy, im, mask, im_sky, mask_sky,
#                             sky_width, sky_buf, r_sky_min)
#     return r


def opt_factory(p):
    if len(p) == 1:
        cls = CircleOptimizer
    else:
        cls = EllipseOptimizer
    # initialize
    return cls()


class ApertureOptimizer:
    def __init__(self, ap, ap_sky, method='exact', rmin=1, rmax=10):
        self.ap = ap
        self.ap_sky = ap_sky
        self.method = method
        self.rmin = rmin
        self.rmax = rmax

    def __str__(self):
        return '%s\n%s\n%s' % (self.__class__.__name__, self.ap, self.ap_sky)

    def __iter__(self):
        yield from (self.ap, self.ap_sky)

    def snr(self, image, masks, image_sky, mask_sky):
        counts, npix = ap_sums(self.ap, image, masks, self.method)
        counts_bg, npixbg = ap_sums(self.ap_sky, image_sky, mask_sky,
                                    self.method)
        return snr_star(counts, npix, counts_bg, npixbg)

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def update_snr(self, p0, cxy, im, mask, im_sky, mask_sky, sky_width,
                   sky_buf, r_sky_min):
        self.update(cxy, *p0, sky_width, sky_buf, r_sky_min)
        return self.snr(im, mask, im_sky, mask_sky)

    def objective(self, p0, cxy, im, mask, im_sky, mask_sky,
                  sky_width, sky_buf, r_sky_min):
        """Inverted SNR for minimization"""
        return -self.update_snr(p0, cxy, im, mask,
                                im_sky, mask_sky,
                                sky_width, sky_buf, r_sky_min).sum()

    def fit(self, p0, *args):
        return minimize(self.objective, p0, args, bounds=self.bounds)

    def flux_estimate(self, image, masks, im_sky, mask_sky):
        return flux_estimate(self.ap, image, masks,
                             self.ap_sky, im_sky, mask_sky)


class CircleOptimizer(ApertureOptimizer):

    def __init__(self, ap=None, ap_sky=None, method='exact', rmin=1, rmax=10):
        if ap is None:
            ap = CircularAperture((0., 0.), 0)
        if ap_sky is None:
            ap_sky = CircularAnnulus((0., 0.), 0, 1)

        ApertureOptimizer.__init__(self, ap, ap_sky, method, rmin, rmax)
        self.bounds = [(self.rmin, self.rmax)]

    def update(self, cxy, r, sky_width, sky_buf, r_sky_min):
        # rescale the aperture
        ap, ap_sky = self.ap, self.ap_sky
        ap.positions = ap_sky.positions = cxy
        ap.r = r

        ap_sky.r_in = max(r_sky_min, r + sky_buf)
        ap_sky.r_out = ap_sky.r_in + sky_width


class EllipseOptimizer(ApertureOptimizer):

    def __init__(self, ap=None, ap_sky=None, method='exact', rmin=1, rmax=10,
                 scale_sky=1):
        if ap is None:
            ap = EllipticalAperture((0., 0.), 0, 0, 0)
        if ap_sky is None:
            ap_sky = EllipticalAnnulus((0., 0.), 0, 1, 0, 0)

        ApertureOptimizer.__init__(self, ap, ap_sky, method, rmin, rmax)
        self.bounds = [(self.rmin, self.rmax),
                       (self.rmin, self.rmax),
                       (-np.pi / 2, np.pi / 2)]

        self.scale_sky = scale_sky

    def update(self, cxy, a, b, theta, sky_width, sky_buf, r_sky_min):
        # rescale the aperture
        ap, ap_sky = self.ap, self.ap_sky
        ap.positions = ap_sky.positions = cxy
        ap.a, ap.b, ap.theta = a, b, theta

        if self.scale_sky == 1:
            ap_sky.a_in = ap_sky.b_in = r_sky_min
            ap_sky.a_out = ap_sky.b_out = r_sky_min + sky_width
            ap_sky.theta = 0

        if self.scale_sky == 2:
            ap_sky.a_in = max(r_sky_min, a + sky_buf)
            ap_sky.b_in = max(r_sky_min * (b / a), b + sky_buf)
            # would be nice if this were set automatically
            ap_sky.a_out = ap_sky.a_in + sky_width
            ap_sky.b_out = ap_sky.b_in + sky_buf + sky_width
            ap_sky.theta = theta


class TaskExecutor:
    """
    Decorator that catches and logs exceptions instead of actively raising.

    Intended use is for data-parallel loops in which the same function will be
    called many times with different parameters. For this class to work
    properly, it requires the decorated function/method to have a call signature
    in which the first parameter is an integer count corresponding to the
    in-sequence number of the task.
    """
    SUCCESS = 1
    FAIL = -1

    def __init__(self, compute_size, counter, fail_counter, max_fail=None,
                 time=False):
        """


        Parameters
        ----------
        compute_size
        counter
        fail_counter
        max_fail:
            percentage string eg: '1%' or an integer

        """
        # TODO: timer
        # TODO: make progressbar optional

        self.compute_size = n = int(compute_size)
        self.loc = tempfile.mktemp()
        self.status = load_memmap(self.loc, n, 'i', 0)
        self.counter = counter
        self.fail_counter = fail_counter
        self.time = bool(time)
        self.timings = None
        if self.time:
            self.loct = tempfile.mktemp()
            self.timings = load_memmap(self.loct, n, 'f', 0)

        # resolve `max_fail`
        if max_fail is None:
            # default is 1% or 50, whichever is smaller
            max_fail = resolve_percentage('1%', n)
            max_fail = min(max_fail, 50)
        else:
            max_fail = resolve_percentage(max_fail, n)
        self.max_fail = max_fail

        # progress "bar"
        self.progLog = ProgressLogger(width=10, symbol='', align='<')
        self.progLog.create(n, None)

    def __call__(self, func):

        self.func = func
        self.name = pprint.method(func, show_class=True, submodule_depth=1)
        self.progLog.name = self.name

        # optional timer
        self.run = self._run_timed if self.time else self._run

        # log
        # if np.isfinite(max_fail):
        n = self.compute_size
        msg = 'Exception threshold is %.2f%% (%i/%i)' % (
            (self.max_fail / n) * 100, self.max_fail, n)
        # else:
        #     msg = 'All exceptions will be ignored'
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(msg)

        return self.catch

    # @property  # making this a property avoids pickling errors for the logger
    # def logger(self):
    #     logger = logging.getLogger(self.name)
    #     return logger

    def reset(self):
        self.counter.set_value(0)
        self.fail_counter.set_value(0)

    def _run(self, *args, **kws):
        return self.func(*args, **kws)

    def _run_timed(self, *args, **kws):
        ts = time.time()
        result = self.func(*args, **kws)
        self.timings[args[0]] = time.time() - ts
        return result

    def catch(self, *args, **kws):
        """
        This is the decorated function

        Parameters
        ----------
        args
        kws

        Returns
        -------

        """
        # exceptions like moths to the flame
        abort = self.fail_counter.get_value() >= self.max_fail
        if not abort:
            try:
                result = self.run(*args, **kws)
            except Exception as err:
                # logs full trace by default
                i = args[0]
                self.status[i] = self.FAIL
                nfail = self.fail_counter.inc()
                logger = logging.getLogger(self.name)
                logger.exception('Processing failed at frame %i. (%i/%i)',
                                 i, nfail, self.max_fail)

                # check if we are beyond exception threshold
                if nfail >= self.max_fail:
                    logger.critical('Exception threshold reached!')
                    # self.logger.critical('Exception threshold reached!')
            else:
                i = args[0]
                self.status[i] = self.SUCCESS
                return result  # finally clause executes before this returns

            finally:
                # log progress
                counter = self.counter
                if counter:
                    n = counter.inc()
                    if self.progLog:
                        self.progLog.update(n)

            # if there was a KeyboardInterrupt, it will be raised at this point
        else:
            # doing this here (instead of inside the except clause) avoids
            # duplication by chained exception traceback when logging
            raise AbortCompute(
                    'Number of exceptions larger than threshold of %i'
                    % self.max_fail)

    def report(self):
        # not_done, = np.where(self.status == 0)
        failures, = np.where(self.status == -1)
        n_done = self.counter.get_value()
        n_fail = self.fail_counter.get_value()

        # TODO: one multi-line message better when multiprocessing

        logger = logging.getLogger(self.name)
        logger.info('Processed %i/%i frames. %i successful; %i failed',
                    n_done, self.compute_size, n_done - n_fail, n_fail)
        if len(failures):
            logger.info('The following frames failed: %s', list(failures))
        elif n_done > 0:
            logger.info('No failures in main compute!')

        if self.time:
            #  print timing info
            logger.info('Timing results for %s: %.3f ± .3f s',
                        self.name, self.timings.mean(), self.timings.std())

        return failures


def _make_named_dtype(names, base_dtype=float):
    return np.dtype(list(zip(names, itt.repeat(base_dtype))))


class FrameProcessor(LoggingMixin):
    # OptimalApertureProcessor

    def __init__(self):
        self._results = self._appars = self.status = None

        self._dtype_appars = _make_named_dtype(('stars', 'sky'))
        self._dtype_results = _make_named_dtype([
            # Four columns are
            'counts',  # total counts source only
            'counts_std',  # propagated uncertainty on above
            'bg',  # average per pixel background count
            'bg_std',  # (weighted) deviation on above
        ])

    @property
    def results(self):
        return self._results.view(self._dtype_results, np.recarray)

    @property
    def appars(self):
        return self._appars.view(self._dtype_appars, np.recarray)

    def init_mem(self, n, nstars, ngroups, loc, clobber=False):
        """

        Parameters
        ----------
        n:
            number of frames
        nstars:
            number of stars
        ngroups:
            number of star groups
        naps:
            number of apertures per star
        loc
        clobber

        Returns
        -------

        """

        comm = float, np.nan, clobber
        path_results = loc / 'aps.dat'

        #
        self._results = load_memmap(path_results, (n, nstars, 4), *comm)

        # aperture parameters
        path_params = loc / 'aps.par'
        self._appars = load_memmap(path_params, (n, ngroups, 3, 2), *comm)

        # house keeping (optimization status)
        self.status = load_memmap(loc / 'opt.stat', (n, ngroups), int, np.nan,
                                  clobber=clobber)

    def process(self, i, data, calib, residu, coords, tracker, p0bg, p0aps,
                sky_width=12, sky_buf=1):
        self.optimal_aperture_photometry(i, data, residu, coords, tracker,
                                         p0aps,
                                         sky_width, sky_buf)

    # def track(self, i, data, coords):
    #     tracker
    #

    def pre_process(self, i, data, flat, output, region=...):

        image = data[i]

        # bias, flat = calib
        # if bias is not None:
        #     image = image - bias

        if flat is not None:
            image = image / flat

        output[i] = image

        # prep background image
        # imbg = tracker.mask_image(image)

        # fit and subtract background
        # p_bg, resi = model.fit(image)
        # model.data[i] = p_bg
        # residu[i] = resi

        # p, pu = lm_extract_values_stderr(p_bg)
        # mdlr.data[-1].params[i] = np.hstack(p_bg)
        # mdlr.data[-1].params_std[i] = pu
        # mdlr._save_params(mdlr.bg, i, 0, (p, pu, None))

        # track stars
        # com = tracker(resi)
        # save coordinates in shared data array.
        # coords[i] = com[tracker.ir]

    def optimal_aperture_photometry(self, i, data, residu, coords, tracker,
                                    p0, sky_width=12, sky_buf=0.5):
        """
        Optimization step to determine best aperture size and shape.

        first try for bright stars, then for faint.  if faint opt failed fall
        back to results for bright. if both fail, fall back to opt init values

        Parameters
        ----------
        i
        data
        residu
        coords
        tracker
        p0
        sky_width
        sky_buf

        Returns
        -------

        """

        # TODO: optimize on sum of apertures for faint (all) star groups???

        # star coordinates frame i
        coords = coords[:, ::-1]
        # print(coords)

        # check valid coordinates
        if np.isnan(coords).any():
            self.logger.warning(
                    'Invalid coords: frame %s. Skipping photometry.', i)
            return

        # masks
        # NOTE: using rmax here and passing a subset of the full array to do
        # photometry may improve memory and cpu usage
        start = tracker.xy_offsets[i].round().astype(int)
        phot_masks, sky_mask = tracker.get_masks(start, data.shape[-2:])

        # results = []
        skip_opt = False
        prevr = None
        count = 0
        last_group = min(len(tracker.groups), 2)
        for g, (name, labels) in enumerate(tracker.groups.items()):
            if 0 in labels:
                continue  # this is the sky image
            count += 1

            self.logger.debug('Attempting optimized aperture photometry for '
                              'group %i (%s): %s', g, name, tuple(labels))
            # print(g, labels, ix, photmasks.shape)

            # indices corresponding to labels (labels may not be sequential)
            ix = labels - 1
            masks = phot_masks[ix]

            if skip_opt:
                flag = None
            else:
                # estimate minimal sky radius from detection segments
                areas = tracker.segm.areas[labels - 1]
                r_sky_min = np.ceil(np.sqrt(areas.max() / np.pi))

                # run optimization
                r, opt, flag = self.optimize_apertures(
                        i, p0, coords[ix],
                        residu[i], masks, data[i],
                        sky_mask, r_sky_min, sky_width, sky_buf,
                        labels)

                # print(labels)
                # print(r, opt, flag)
                # print('-' * 20)

                # save status
                self.status[i, g] = flag

                if flag == 1:
                    # success
                    prevr = r
                    p = r.x

            if flag != 1:  # there was an error or no convergence
                if prevr is not None and prevr.success:
                    # use bright star appars for faint stars (if available) if
                    # optimization failed for this group
                    p = prevr.x
                else:
                    # no convergence for this opt or previous. fall back to p0
                    p = p0

                # update to fallback values
                opt.update(coords[ix], *p, sky_width, sky_buf, r_sky_min)

                skip_opt = True
                # if fit didn't converge for bright stars, it won't for the
                # fainter ones. save some time by skipping opt

            # get apertures
            aps, aps_sky = opt

            if flag is not None:  # ie. optimization was at least attempted
                # save appars
                if len(p) == 1:  # circle
                    a, = b, = p
                    theta = 0
                    a_sky_in = opt.ap_sky.r_in
                    a_sky_out = b_sky_out = opt.ap_sky.r_out

                else:  # ellipse
                    a, b, theta = p
                    a_sky_in = opt.ap_sky.a_in
                    a_sky_out = opt.ap_sky.a_out
                    b_sky_out = opt.ap_sky.b_out

            else:
                # no optimization attempted
                # use the radii, angle of the previous group for photometry on
                # remaining groups
                aps.positions = coords[ix]
                aps_sky.positions = coords[ix]

            # save appars
            self._appars[i, g] = list(zip(
                    (a, b, theta),
                    (a_sky_in, a_sky_out, b_sky_out)))

            # do photometry with optimized apertures
            self.do_phot(i, ix, data, residu, aps, masks, aps_sky, sky_mask)

            if count == last_group:
                # only try do the optimization for the first 2 label groups
                skip_opt = True
                # This means the aperture parameters of the last group will
                # be used for photometry on all subsequent groups

    def optimize_apertures(self, i, p0, coo_xy, im, photmasks, im_sky, skymask,
                           r_sky_min, sky_width, sky_buf, labels):

        """

        Parameters
        ----------
        i
        p0
        coo_xy
        im
        photmasks
        im_sky
        skymask
        r_sky_min
        sky_width
        sky_buf
        labels

        Returns
        -------
        flags -
         1 : success
         0 : Optimization converged on boundary
        -1 : no convergence
        -2 : low SNR, skip
        -3 : minimize subroutine exception
        """

        # optimized aperture photometry - search for highest snr aperture
        # create optimizer
        opt = opt_factory(p0)

        # optimization only really makes sense if we have respectable snr
        # to start with. We skip the optimization step for faint stars if
        # the snr is too low based on the p0 params
        opt.update(coo_xy, *p0, sky_width, sky_buf, r_sky_min)

        snr = opt.snr(im, photmasks, im_sky, skymask)
        low_snr = snr < OPT_SNR_THRESH
        self.logger.debug('SNR: %s', snr)
        if low_snr.all():
            # skip opt
            self.logger.debug('Skipping optimization: frame %s. low SNR for '
                              'stars %s', i, labels)
            return None, opt, -2

        # remove low snr stars
        # coo_xy = coo_xy[~low_snr]
        # photmasks = photmasks[~low_snr]

        # from IPython import embed
        # embed()

        try:
            # maximize snr
            r = minimize(opt.objective, p0,
                         (coo_xy[~low_snr],
                          im, photmasks[~low_snr],
                          im_sky, skymask,
                          sky_width, sky_buf, r_sky_min),
                         bounds=opt.bounds)

        except Exception as err:
            self.logger.exception('Optimization error: frame %s, labels %s',
                                  i, labels)
            return None, opt, -3

        if not r.success:
            self.logger.warning('Optimization failed: frame %s, labels %s\n%s',
                                i, labels, r.message)
            flag = -1
        elif np.any(r.x == opt.bounds):
            self.logger.warning('Optimization converged on boundary:'
                                ' frame %s, labels %s', i, labels)
            flag = 0
        else:
            flag = 1

        if low_snr.any():
            # put back the low snr coordinates we removed
            opt.update(coo_xy, r.x, sky_width, sky_buf, r_sky_min)

        return r, opt, flag  # .ap, opt.ap_sky

    def do_phot(self, i, js, data, residue, aps, masks, aps_sky, sky_mask):

        # photometry for optimized apertures
        self._results[i, js] = np.transpose(flux_estimate(
                aps, residue[i], masks, aps_sky, data[i], sky_mask))

        # residual sky image noise (read-, dark-, sky-  noise)

    # def display_video(self, data):
    # TODO

    def multi_aperture_photometry(self, data, aps, skyaps, tracker):

        method = 'exact'

        # a quantity is needed for photutils
        udata = u.Quantity(data, copy=False)

        m3d = tracker.segm.to_boolean_3d()
        masks = m3d.any(0, keepdims=True) & ~m3d
        masks |= tracker.bad_pixel_mask

        FluxBG = np.empty(np.shape(skyaps))
        # FluxBGu = np.empty(np.shape(skyaps))
        Flux = np.empty(np.shape(aps))
        # Fluxu = np.empty(np.shape(aps))
        if Flux.ndim == 1:
            Flux = Flux[:, None]

        for j, (ap, ann) in enumerate(zip(aps, skyaps)):
            mask = masks[j]

            # sky
            flxBG, flxBGu = ann.do_photometry(udata,
                                              # error,
                                              mask=mask,
                                              # effective_gain,#  must have same shape as data
                                              # TODO: ERROR ESTIMATE
                                              method=method)  # method='subpixel', subpixel=5)
            try:
                m = ap.to_mask(method)[0]
                area = (m.data * m.cutout(~mask)).sum()
            except Exception as err:
                print(err)
                print(m.cutout(~mask))
                print(m.data)
                raise

            # per pixel fluxes
            fluxBGpp = FluxBG[j] = flxBG / area  # Background Flux per pixel
            # FluxBGu[j] = flxBGu / area # FIXME since flxBGu is []

            # multi apertures ??
            for k, app in enumerate(np.atleast_1d(ap)):
                flux, flux_err = app.do_photometry(udata,
                                                   mask=mask,
                                                   # error, #TODO: ERROR ESTIMATE
                                                   # effective_gain,#  must have same shape as data
                                                   method=method)
                # get the area of the aperture excluding masked pixels
                m = ap.to_mask(method)[0]
                area = (m.data * m.cutout(~mask)).sum()

                Flux[j, k] = flux - (fluxBGpp * area)
                # Fluxu[j, k] = flux_err

        return Flux, FluxBG
        # return (Flux, Fluxu), (FluxBG, FluxBGu)

    def model_selection(self, i, mdlr, results):
        """
        Do model selection (per star) based on goodness of fit metric(s)
        """

        pars, paru, gof = results
        bestIx, bestModels, params, pstd = [], [], [], []
        # loop over stars
        for j, g in enumerate(gof.swapaxes(0, 1)):
            # TODO: kill this for loop
            ix, mdl, msg = mdlr.model_selection(g)
            if msg:
                self.logger.warning('%s (Frame %i, Star %i)', msg, i, j)

            if ix == -99:
                p = pu = None
            else:
                p = pars[ix][j]
                pu = paru[ix][j]
                if mdlr.nmodels > 1:
                    self.logger.info('Best model: %s (Frame %i, Star %i)', mdl,
                                     i, j)

            # TODO: if best_model is self.db.bg:
            #     "logging.warning('Best model is BG')"
            #     "flux is upper limit?"

            # yield mdl, p
            bestModels.append(mdl)
            bestIx.append(ix)
            params.append(p)
            pstd.append(pu)
        return bestIx, bestModels, params, pstd

    def write_lightcurve_ascii(self, filename, t, mask=None, meta={},
                               obj_name='<unknown>', column_indices=...):

        from obstools.lc.misc import write_ascii

        # TODO: Include info about photometry routine
        #  and source file

        # write light curves to ascii
        results = self.results
        write_ascii(filename,
                    t, results.counts.squeeze().T[column_indices],
                    results.counts_std.squeeze().T[column_indices],
                    mask, meta, obj_name)

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
#         #nfit = len(self.modeller.use_labels)
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
#             m = ap.to_mask(method)[0]
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
#                 m = ap.to_mask(method)[0]
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
