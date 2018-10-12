"""
Methods for tracking camera movements in astronomical time-series CCD photometry
"""

# import inspect
import functools
import itertools as itt
import warnings

import more_itertools as mit
import numpy as np
from astropy.stats import median_absolute_deviation as mad
from astropy.stats.sigma_clipping import sigma_clip
from astropy.utils import lazyproperty


from photutils.detection.core import detect_threshold
from photutils.segmentation import detect_sources
from recipes.dict import ListLike, AttrReadItem
from recipes.logging import LoggingMixin
from recipes.string import get_module_name
from scipy import ndimage
from scipy.spatial.distance import cdist

import logging

from obstools.phot.segmentation import SegmentationHelper

# from IPython import embed

# TODO: watershed segmentation on the negative image ?
#

# TODO: split off segmentation stuff?

# module level logger
logger = logging.getLogger(get_module_name(__file__))


def _make_border_mask(data, xlow=0, xhi=None, ylow=0, yhi=None):
    """Edge mask"""
    mask = np.zeros(data.shape, bool)

    mask[:ylow] = True
    if yhi is not None:
        mask[yhi:] = True

    mask[:, :xlow] = True
    if xhi is not None:
        mask[:, xhi:] = True
    return mask


def make_border_mask(data, edge_cutoffs):
    if isinstance(edge_cutoffs, int):
        return _make_border_mask(data,
                                 edge_cutoffs, -edge_cutoffs,
                                 edge_cutoffs, -edge_cutoffs)
    edge_cutoffs = tuple(edge_cutoffs)
    if len(edge_cutoffs) == 4:
        return _make_border_mask(data, *edge_cutoffs)

    raise ValueError('Invalid edge_cutoffs %s' % edge_cutoffs)


def check_image_drift(cube, nframes, mask=None, snr=5, npixels=10):
    """Estimate the maximal positional drift for stars"""

    #
    logger.info('Estimating maximal image drift for %i frames.', nframes)

    # take `nframes` frames evenly spaced across data set
    step = len(cube) // nframes
    maxImage = cube[::step].max(0)  #

    segImx = SegmentationHelper.detect(maxImage, mask, snr=snr, npixels=npixels,
                                       dilate=3)

    mxshift = np.max([(xs.stop - xs.start, ys.stop - ys.start)
                      for (xs, ys) in segImx.slices], 0)
    return mxshift, maxImage, segImx


def inside_segment(coords, sub, grid):
    b = []
    ogrid = grid[0, :, 0], grid[1, 0, :]
    for j, (g, f) in enumerate(zip(ogrid, coords)):
        bi = np.digitize(f, g - 0.5)
        b.append(bi)

    mask = (sub == 0)
    if np.equal(grid.shape[1:], b).any() or np.equal(0, b).any():
        inside = False
    else:
        inside = not mask[b[0], b[1]]

    return inside


def iter_repeat_last(it):
    it, it1 = itt.tee(mit.always_iterable(it))
    return mit.padded(it, next(mit.tail(1, it1)))


def detect(image, mask=False, background=None, snr=3., npixels=7,
           edge_cutoff=None, deblend=False):
    """
    Image detection that returns a SegmentationHelper instance

    Parameters
    ----------
    image
    mask
    background
    snr
    npixels
    edge_cutoff
    deblend
    dilate

    Returns
    -------

    """

    logger.info('Running detect(snr=%.1f, npixels=%i)', snr, npixels)

    # calculate threshold without edges so that std accurately measured for
    # region of interest
    if edge_cutoff:
        border = make_border_mask(image, edge_cutoff)
        mask = mask | border

    # separate pixel mask for threshold calculation (else the mask gets
    # duplicated to threshold array, which will skew the detection stats)
    if np.ma.isMA(image):
        mask = mask | image.mask
        image = image.data

    # detection
    threshold = detect_threshold(image, snr, background, mask=mask)
    if mask is False:
        mask = None  # annoying photutils #HACK
    obj = detect_sources(image, threshold, npixels, mask=mask)

    #
    if np.all(obj.data == 0):
        logger.info('No objects detected')
        return obj

    # if border is not False:
    #     obj.remove_masked_labels(border, partial_overlap=False)

    if deblend:
        from photutils import deblend_sources
        obj = deblend_sources(image, obj, npixels)
        # returns altered copy of instance

    logger.info('Detected %i objects across %i pixels.', obj.nlabels,
                obj.data.astype(bool).sum())
    return obj


def detect_loop(image, mask=None, bgmodel=None, snr=3., npixels=7,
                edge_cutoff=None, deblend=False, dilate=1):
    """
    Multi-threshold image blob detection, segmentation and grouping.

    `snr`, `npixels`, `dilate` can be sequences in which case the next value
    from each will be used in the next detection loop. If scalar, same value
    during each iteration while mask in updated

    Parameters
    ----------
    image
    mask
    bgmodel
    snr
    npixels
    dilate
    edge_cutoff
    deblend

    Returns
    -------

    """

    # TODO: can you do better with GMM ???

    # make iterables
    _snr, _npix, _dil = (iter_repeat_last(o) for o in (snr, npixels, dilate))

    data = np.zeros(image.shape, int)
    if mask is None:
        mask = np.zeros(image.shape, bool)

    background = results = None  # first round detection without bgmodel
    groups = []
    counter = itt.count(0)
    j = 0
    go = True
    while go:
        # keep track of iteration number
        count = next(counter)

        # detect
        sh = SegmentationHelper.detect(image, mask, background, next(_snr),
                                       next(_npix), edge_cutoff, deblend,
                                       next(_dil))

        # update mask, get labels
        labelled = sh.data
        new_mask = labelled.astype(bool)
        new_data = new_mask & np.logical_not(mask)
        new_labels = np.unique(labelled[new_data])

        # since we dilated the detection masks, we may now be overlapping with
        # previous detections. Remove overlapping pixels here
        # if dilate:
        #     overlap = mask & new_data
        #     new_data[overlap] = 0

        if not new_data.any():
            # done
            # final group of detections are determined by sigma clipping
            if bgmodel:
                resi = bgmodel.residuals(results, image)
            else:
                resi = image
            resi_masked = np.ma.MaskedArray(resi, mask)
            resi_clip = sigma_clip(resi_masked)  # TODO: make this optional
            new_data = resi_clip.mask & ~resi_masked.mask
            new_data = ndimage.binary_dilation(new_data) & ~resi_masked.mask
            labelled, nlabels = ndimage.label(new_data)
            new_labels = np.unique(labelled[labelled != 0])
            go = False

        # aggregate
        data[new_data] += labelled[new_data] + j
        mask = mask | new_mask
        groups.append(new_labels + j)
        j += new_labels.max()

        logger.info('detect_loop: round nr %i: %i new detections: %s',
                    count, len(new_labels), tuple(new_labels))
        # print('detect_loop: round nr %i: %i new detections: %s' %
        #       (count, len(new_labels), tuple(new_labels)))

        # slotmode HACK
        if count == 0:
            # BackgroundModel(sh, None, )
            # init bg model here
            segmbg = sh.copy()

        if bgmodel:
            # fit background
            imm = np.ma.MaskedArray(image, mask)
            results, residu = bgmodel.minimized_residual(imm)

            # background model
            background = bgmodel(results)

        break

    # TODO: log what you found

    return data, groups, results


# class NullSlice():
#     """Null object pattern for getitem"""
#
#     def __getitem__(self, item):
#         return None

# constructor used for pickling
# def _construct_SegmentedArray(*args):
#     return SegmentedArray(*args)


class LabelUser(object):
    """
    Mixin class for inheritors that use `SegmentationHelper`.
    Adds the `use_label` and `ignore_label` properties.  Whenever a function
    that takes the `label` parameter is called with the default value `None`,
    the labels in `use_labels` will be used instead. Helps with dynamically
    including / excluding labels and with grouping labelled segments together

    If initialized without any arguments, all labels in the segmentation image
    will be in use.

    """

    def __init__(self, use_labels=None, ignore_labels=None):
        if use_labels is None:
            use_labels = self.segm.labels
        if ignore_labels is None:
            ignore_labels = []

        self._use_labels = np.setdiff1d(use_labels, ignore_labels)
        self._ignore_labels = np.asarray(ignore_labels)

    @property
    def ignore_labels(self):  # ignore_labels / use_labels
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, labels):
        self._ignore_labels = np.asarray(labels, int)
        self._use_labels = np.setdiff1d(self.segm.labels, labels)

    @property
    def use_labels(self):  # ignore_labels / use_labels
        return self._use_labels

    @use_labels.setter
    def use_labels(self, labels):
        self._use_labels = np.asarray(labels, int)
        self._ignore_labels = np.setdiff1d(self.segm.labels, labels)

    def resolve_labels(self, labels=None):
        if labels is None:
            return self.use_labels
        return self.segm.check_labels(labels)

    @property
    def nlabels(self):
        return len(self.use_labels)

    @property
    def sizes(self):
        return [len(labels) for labels in self.values()]


class Record(AttrReadItem, ListLike):
    pass


class LabelGroups(Record):
    """
    Ordered dict with key access via attribute lookup. Also has some
    list-like functionality: indexing by int and appending new data.
    Best of both worlds.  Also makes sure labels are always arrays.
    """
    _auto_name_fmt = 'group%i'

    def _allow_item(self, item):
        return bool(len(item))

    def _convert_item(self, item):
        return np.array(item, int)

    @property
    def sizes(self):
        return list(map(len, self.values()))
        # return [len(item) for item in self.values()]

    # def inverse(self):
    #     return {lbl: gid for gid, labels in self.items() for lbl in labels}


class LabelGroupsMixin(object):
    """Mixin class for grouping and labelling image segments"""

    def __init__(self, groups=None):
        # if groups is None:
        #     groups = zip(('bg', 'group0'),  # FIXME: seems bad
        #                  ([0], [self.segm.labels_nonzero]))

        self._groups = LabelGroups(groups)


    @property
    def groups(self):
        return self._groups

    # todo
    # def remove_group()




# TODO:
# optionally keep track of distribution of CoM for each star
# keep track of changes in relative positions, and switch off the update step when below threshold
# once stability is reached, start photometry


class StarTracker(LabelUser, LabelGroupsMixin, LoggingMixin):
    """
    A class to track stars in a CCD image frame to aid in doing time series photometry.

    Stellar positions in each new frame are calculated as the background subtracted centroids
    of the chosen image segments. Centroid positions of chosen stars are filtered for bad
    values (or outliers) and then combined by a weighted average. The weights are taken
    as the SNR of the star brightness. Also handles arbitrary pixel masks.

    Constellation of stars in image assumed unchanging. The coordinate positions of the stars
    with respect to the brightest star are updated upon each iteration.


    """

    # TODO: groups.bad_pixels

    # TODO: bayesian version
    # TODO: make multiprocessing safe

    segmClass = SegmentationHelper

    # TODO: manage as properties ?
    snr_weighting = True
    snr_cut = 3
    _distance_cut = 10
    _saturation_cut = 95  # %

    # this allows tracking stars with low snr, but not computing their
    # centroids which will have larger scatter

    # TODO:
    # def from_fits(cls, filename, snr=3., npixels=7, edge_cutoff=3, deblend=False,
    #                flux_sort=True, dilate=1, bad_pixel_mask=None, edge_mask=None)

    @classmethod
    def from_image(cls, image, mask=None, bgmodel=None, snr=3., npixels=7,
                   edge_cutoffs=None, deblend=False, dilate=1, flux_sort=True):
        """Initialize the tracker from an image"""

        # detect and group
        segdata, groups, p0bg = detect_loop(image, mask, bgmodel,
                                            snr, npixels, edge_cutoffs,
                                            deblend, dilate)
        # init helper
        sh = SegmentationHelper(segdata)
        # last group of detections are those that have been sigma clipped.
        # Don't include these in found stars

        use_labels = np.hstack(groups[:-1])
        # ignore_labels = groups[-1]

        # use_labels = groups[0]
        # ignore_labels = None

        # Center of mass
        found = sh.com_bg(image, use_labels, mask)

        # cls.best_for_tracking(image)
        obj = cls(found, sh, groups, use_labels, mask, edge_cutoffs)
        # log nice table with what's been found.
        obj.logger.info('Found the following stars:\n%s\n', obj.pprint())

        return obj, p0bg

    @classmethod
    def from_cube_segment(cls, cube, ncomb, subset, nrand=None, mask=None,
                          bgmodel=None, snr=(10., 3.), npixels=7,
                          edge_cutoffs=None, deblend=False, dilate=1,
                          flux_sort=True):

        # select `ncomb`` frames randomly from `nrand` in the interval `subset`
        if isinstance(subset, int):
            subset = (0, subset)  # treat like a slice

        i0, i1 = subset
        if nrand is None:  # if not given, select from entire subset
            nrand = i1 - i0
        nfirst = min(nrand, i1 - i0)
        ix = np.random.randint(i0, i0 + nfirst, ncomb)
        # create ref image for init
        image = np.median(cube[ix], 0)
        cls.logger.info('Combined %i frames from amongst frames (%i->%i) for '
                        'reference image.', ncomb, i0, i0 + nfirst)

        # init the tracker
        tracker, p0bg = SlotModeTracker.from_image(
                image, mask, bgmodel, snr, npixels, edge_cutoffs, deblend,
                dilate, flux_sort)
        return tracker, image, p0bg

    def __init__(self, coords, segm, label_groups=None, use_labels=None,
                 bad_pixel_mask=None, edge_cutoffs=None):
        """
        Use to track the location of stars on CCD frame

        Parameters
        ----------
        coords : array-like
            reference coordinates of the stars
        segm : SegmentationHelper instance
            (or inherited from)


        ignore_labels : array-like
            indices of stars to ignore when calculating frame shift. These stars
            will still be tracked by their relative position wrt the others.
            Their relative positions will also be updated upon each call.
        """
        # counter for incremental update
        self.counter = itt.count(1)  # TODO: multiprocess!!!
        # store zero and relative positions separately so we can update them
        # independently
        self.ir = 0  # TODO: optionally pass in

        # pixel coordinates of reference position from which the shift will
        # be measured
        self.yx0 = coords[self.ir]
        self.rvec = coords - coords[self.ir]
        self.segm = GriddedSegments(segm)
        self._original_data = self.segm.data

        # init groups
        LabelGroupsMixin.__init__(self, label_groups)

        # TODO: once you figure out how to use negative labels, these things can
        # live inside the segmentation image.  also adapt_segments??
        self.bad_pixel_mask = bad_pixel_mask
        if edge_cutoffs is not None:
            self.edge_mask = make_border_mask(self.segm.data, edge_cutoffs)
        else:
            self.edge_mask = None
        #
        self.offset = [0, 0]

        # label logic
        # if use_labels is None:
        #     # we want only to use the stars with high snr fro CoM tracking


        LabelUser.__init__(self, use_labels)

    def __call__(self, image, mask=None):
        """
        Track the shift of the image frame from initial coordinates

        Parameters
        ----------
        image
        mask

        Returns
        -------

        """

        # more robust that individual CoM track.
        # eg: cosmic rays:
        # '/media/Oceanus/UCT/Observing/data/Feb_2015/MASTER_J0614-2725/20150225.001.fits' : 236

        if not ((mask is None) or (self.bad_pixel_mask is None)):
            mask |= self.bad_pixel_mask
        else:
            mask = self.bad_pixel_mask  # might be None

        # calculate CoM
        com = self.segm.com_bg(image, self.use_labels, mask)
        # weights
        weights = self.get_weights(com)

        # update relative positions from CoM measures
        self.update_rvec(com)

        # calculate median com shift using current best relative positions
        # optionally using snr weighting scheme
        if self.snr_weighting or self.snr_cut:
            weights = self.set_snr_weights(weights, image)

        off = self.update_offset(com, weights)

        # finally return the new coordinates
        return self.rcoo + off

    # @property
    # def ngroups(self):
    #     return len(self.groups)

    @property
    def nsegs(self):
        return self.segm.nlabels

    @property
    def rcoo(self):
        """
        Reference coordinates (yx). Computed as initial coordinates + relative coordinates to
        allow updating the relative positions upon call
        """
        return self.yx0 + self.rvec

    @property
    def rcoo_xy(self):
        """Reference coordinates (xy)"""
        return self.rcoo[:, ::-1]

    def pprint(self):
        from motley.table import Table

        # TODO: say what is being ignored

        # coo = [:, ::-1]  # , dtype='O')
        tbl = Table(self.rcoo_xy,
                    col_headers=list('xy'),
                    col_head_props=dict(bg='g'),
                    row_headers=self.use_labels,
                    # number_rows=True,
                    align='>',  # easier to read when right aligned
                    )

        return tbl

    def sdist(self):
        coo = self.rcoo
        return cdist(coo, coo)

    def mask_segments(self, image, mask=None):
        """
        Prepare a background image by masking stars, bad pixels and whatever
        else
        """
        # mask stars
        imbg = self.segm.mask_segments(image)
        if mask is not None:
            imbg.mask |= mask

        if self.bad_pixel_mask is not None:
            imbg.mask |= self.bad_pixel_mask

        return imbg

    def prep_masks_phot(self, labels=None):
        # NOTE: this is the only method that really uses the edge_mask...

        segm = self.segm
        m3d = segm.to_bool_3d()
        sky_mask = m3d.any(0)
        if not self.bad_pixel_mask is None:
            sky_mask |= self.bad_pixel_mask
        if self.edge_mask is not None:
            sky_mask |= self.edge_mask

        labels = self.resolve_labels(labels)
        indices = self.segm.index(labels)

        phot_masks = sky_mask & ~m3d[indices]
        return phot_masks, sky_mask

    def get_weights(self, com):
        # use = self.use_labels - 1
        # ign = self.ignore_labels - 1
        weights = np.ones(len(self.use_labels))

        # flag outliers
        bad = self.is_bad(com)
        self.logger.debug('bad: %s', np.where(bad)[0])
        weights[bad] = 0
        # weights[use[bad]] = 0
        return weights

    def set_snr_weights(self, weights, image):
        # snr weighting scheme (for robustness)

        # if weights is None:
        # use = self.use_labels - 1
        ix = np.where(weights != 0)[0]
        lbl = ix + 1

        weights[ix] = snr = self.segm.snr(image, lbl)
        # self.logger.debug('snr: %s', snr)
        # self.logger.debug('weights: %s', weights)

        # ignore stars with low snr (their positions will still be tracked,
        # but not used to compute new positions)
        low_snr = snr < self.snr_cut
        if low_snr.sum() == len(lbl):
            self.logger.warning('SNR for all stars below cut: %s < %.1f',
                                snr, self.snr_cut)
            low_snr = snr < snr.max()  # else we end up with nans

        weights[ix[low_snr]] = 0

        self.logger.debug('weights: %s', weights)

        if np.all(weights == 0):
            self.logger.warning('All weights are 0!')
            weights += 1

        return weights

    def is_bad(self, coo):
        """
        improve the robustness of the algorithm by removing centroids that are
        bad.  Bad is inf / nan / outside segment.
        """

        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter COM positions that are outside of detection regions
        lbad2 = ~self.segm.inside_segment(coo, self.use_labels)

        return lbad | lbad2

    def is_outlier(self, coo, mad_thresh=5, jump_thresh=5):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """
        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter centroid measurements that are obvious errors (large jumps)
        r = np.sqrt(np.square(self.rcoo - coo).sum(1))
        lj = r > jump_thresh

        if len(coo) < 5:  # scatter large for small sample sizes
            lm = np.zeros(len(coo), bool)
        else:
            lm = r - np.median(r) > mad_thresh * mad(r)

        return lj | lm | lbad

    def update_offset(self, coo, weights):
        """
        Calculate the offset from the previous frame

        Parameters
        ----------
        coo
        weights

        Returns
        -------

        """
        # shift calculated as snr weighted mean of individual CoM shifts

        offset = np.ma.average(coo - self.rcoo, 0, weights)
        self.logger.debug('offset: (%.2f, %.2f)', *offset)

        # shift from previous frame
        shift = offset - self.offset
        ishift = np.round(shift).astype(int).data  # array not ma

        # if frame has shifted by more than one pixel, update the segmentationImage
        if np.abs(ishift).sum():  # any shift greater or equal one pixel
            # shift the segmentation data
            ioff = np.round(offset).astype(int).data
            self.logger.debug('shifting to offset: (%i, %i)', *ioff)
            # TODO: lock here
            self.segm.data = ndimage.shift(self._original_data, ioff)

            # update pixel offset from original
            # self.offset = np.round(offset).astype(int).data # array not ma
            self.offset = offset.data

        return offset

    def update_rvec(self, coo, weights=None):
        """"
        Incremental average of relative positions of stars (since they are
        considered static)
        """
        # see: https://math.stackexchange.com/questions/106700/incremental-averageing
        vec = coo - coo[self.ir]
        n = next(self.counter)
        if weights is not None:
            weights = (weights / weights.max() / n)[:, None]
        else:
            weights = 1. / n

        inc = (vec - self.rvec) * weights
        self.logger.debug('rvec increment:\n%s', str(inc))
        self.rvec += inc

    # def get_shift(self, image):
    #     com = self.segm.com_bg(image)
    #     l = ~self.is_outlier(com)
    #     shift = np.median(self.rcoo[l] - com[l], 0)
    #     return shift

    def best_for_tracking(self, image, close_cut=None, snr_cut=snr_cut,
                          saturation=None):
        """
        Find stars that are best suited for centroid tracking based on the
        following criteria:
        """
        too_bright, too_close, too_faint = [], [], []
        msg = 'Stars: %s too %s for tracking'
        if saturation:
            too_bright = self.too_bright(image, saturation)
            if len(too_bright):
                self.logger.debug(msg, str(too_bright), 'bright')
        if close_cut:
            too_close = self.too_close(close_cut)
            if len(too_close):
                self.logger.debug(msg, str(too_close), 'close')
        if snr_cut:
            too_faint = self.too_faint(snr_cut)
            if len(too_faint):
                self.logger.debug(msg, str(too_faint), 'faint')

        ignore = functools.reduce(np.union1d,
                                  (too_bright, too_close, too_faint))
        ix = np.setdiff1d(np.arange(len(self.coords)), ignore)
        if len(ix) == 0:
            self.logger.warning('No suitable stars found for tracking!')
        return ix

    # def auto_window(self):
    #     sdist_b = self.sdist[snr > self._snr_thresh]

    def too_faint(self, image, threshold=snr_cut):
        crude_snr = self.segm.snr(image)
        return np.where(crude_snr < threshold)[0]

    def too_close(self, threshold=_distance_cut):
        # Check for potential interference problems from stars that are close together
        return np.unique(np.ma.where(self.sdist < threshold))

    def too_bright(self, data, saturation, threshold=_saturation_cut):
        # Check for saturated stars by flagging pixels withing 1% of saturation level
        # TODO: make exact
        lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        # TODO check if you can improve speed here - dont have to check entire array?

        satpix = np.where((lower < data) & (data < upper))
        b = np.any(np.abs(np.array(satpix)[:, None].T - self.coords) < 3, 0)
        w, = np.where(np.all(b, 1))
        return w


# def get_streakmasks(self, image, flx_thresh=2.5e3, width=6):
#     # Mask streaks
#     flx = self.segm.flux(image)
#     strkCoo = self.rcoo[flx > flx_thresh]
#     w = np.multiply(width / 2, [-1, 1])
#     strkRng = strkCoo[:, None, 1] + w
#     strkSlc = map(slice, *strkRng.round(0).astype(int).T)
#
#     strkMask = np.zeros(image.shape, bool)
#     for sl in strkSlc:
#         strkMask[:, sl] = True
#
#     return strkMask


# ModelContainer = namedtuple('ModelContainer', ('psf', 'bg'))

# class BetterTracker(StarTracker):
#
#     @classmethod
#     def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
#                    flux_sort=True, dilate=True, bad_pixel_mask=None):
#         # create segmentationHelper
#         sh = cls.segmClass.from_image(
#             image, snr, npixels, edge_cutoff, deblend, flux_sort, dilate)
#
#         # Center of mass
#         found = sh.com_bg(image, None, bad_pixel_mask)
#
#         return cls(found, sh, None, bad_pixel_mask)


# def __call__(self, image, mask=None):
#
#     if not ((mask is None) or (self.bad_pixel_mask is None)):
#         mask |= self.bad_pixel_mask
#     else:
#         mask = self.bad_pixel_mask  # might be None
#
#     #image, results = self.background_subtract(image, mask)
#     # track stars (on bg subtracted image)
#     # print('returning')
#     return StarTracker.__call__(self, image)


class SlotModeTracker(StarTracker):
    snr_cut = 1.5

    @classmethod
    def from_image(cls, image, bgmodel=None, snr=3., npixels=7,
                   edge_cutoffs=None,
                   deblend=False, dilate=1, flux_sort=True, bad_pixel_mask=None,
                   streak_threshold=1e3):
        #
        import slotmode

        # get the bad pixel mask
        if bad_pixel_mask is None:
            bad_pixel_mask = slotmode.get_bad_pixel_mask(image)

        # create edge mask for sky photometry
        if edge_cutoffs is None:
            edge_cutoffs = slotmode.get_edge_cutoffs(image)

        # init parent class
        tracker, p0bg = super(SlotModeTracker, cls).from_image(
                image, bgmodel, snr, npixels, edge_cutoffs, deblend, dilate,
                flux_sort, bad_pixel_mask)

        # mask streaks
        # tracker.bright = tracker.bright_star_labels(image)
        # tracker.streaks = False
        # tracker.streaks = tracker.get_streakmasks(
        #         tracker.rcoo[tracker.bright - 1])
        return tracker, p0bg

    # def __call__(self, image, mask=None):
    #
    #     # mask = self.bad_pixel_mask | self.streaks
    #     com = StarTracker.__call__(self, image, mask)
    #     # return com
    #
    #     # update streak mask
    #     # self.streaks = self.get_streakmasks(com[:len(self.bright)])
    #     return com

    def bright_star_labels(self, image, flx_thresh=1e3):
        flx = self.segm.flux(image)
        w, = np.where(flx > flx_thresh)
        w = np.setdiff1d(w, self.ignore_labels)
        coo = self.rcoo[w]
        bsl = self.segm.data[tuple(coo.round(0).astype(int).T)]
        return bsl

    # def mask_segments(self, image, mask=None):
    #     imbg = StarTracker.mask_segments(self, image, mask)
    #     # imbg.mask |= self.streaks
    #     return imbg

    # def get_streakmasks(self, coo, width=6):
    #     # Mask streaks
    #
    #     w = np.multiply(width / 2, [-1, 1])
    #     strkRng = coo[:, None, 1] + w
    #     strkSlc = map(slice, *strkRng.round(0).astype(int).T)
    #
    #     strkMask = np.zeros(self.segm.data.shape, bool)
    #     for sl in strkSlc:
    #         strkMask[:, sl] = True
    #
    #     return strkMask

    # return grid, data arrays
    # return [(np.where(m), image[m]) for m in reg]

    # def get_edgemask(self, xlow=0, xhi=None, ylow=0, yhi=None):
    #     """Edge mask"""
    #     # can create by checking derivative of vignette model
    #     # return

    #     def add_streakmasks(self, image, flx_thresh=2.5e3, width=6):

    #         strkMask = self.get_streakmasks(image, flx_thresh, width)
    #         labelled, nobj = ndimage.label(strkMask)
    #         novr = strkMask & (self.segm.data != 0) # non-overlapping streaks
    #         labelled[novr] = 0

    #         streak_labels = self.segm.add_segments(labelled)
    #         self.ignore_labels = np.union1d(self.ignore_labels, streak_labels)
    #         return streak_labels

    # def refine(self, image, bgmodel=None, use_edge_mask=True, snr=3., npixels=6,
    #            edge_cutoff=0, deblend=False, flux_sort=False, dilate=1):
    #     # mask bad pixels / stars for bg fit
    #     # TODO: combine use_edge_mask with edge_cutoff
    #     # TODO: incorporate in from_image method??
    #
    #     # mask stars, streaks, bad pix
    #     imm = self.segm.mask_segments(image)  # stars masked
    #     # starmask = imm.mask.copy()
    #     use_edge_mask = use_edge_mask and self.edge_mask is not None
    #     if self.bad_pixel_mask is not None:
    #         imm.mask |= self.bad_pixel_mask
    #     if use_edge_mask:
    #         imm.mask |= self.edge_mask
    #     # if self.streaks is not None:
    #     #     imm.mask |= self.streaks
    #
    #     if bgmodel:
    #         # fit background
    #         results = bgmodel.fit(np.ma.array(imm, mask=self.streaks))  # = (ry, rx)
    #         # input image with background model subtracted
    #         im_bgs = bgmodel.residuals(results, image)
    #
    #         # do detection run without streak masks since there may be faint but
    #         # detectable stars partly in the streaks
    #         resi_sm = np.ma.array(im_bgs, mask=imm.mask)
    #         # prepare frame: fill bright stars
    #         resif = np.ma.filled(resi_sm, np.ma.median(resi_sm))
    #     else:
    #         resif = np.ma.filled(imm, np.ma.median(imm))
    #         im_bgs = image
    #
    #     # detect faint stars
    #     new_segm = SegmentationHelper.from_image(resif, snr, npixels,
    #                                              edge_cutoff, deblend,
    #                                              flux_sort, dilate)
    #     # since we dilated the detection masks, we may now be overlapping with
    #     # previous detections. Remove overlapping pixels here
    #     if dilate:
    #         overlap = new_segm.data.astype(bool) & self.segm.data.astype(bool)
    #         new_segm.data[overlap] = 0
    #
    #     # remove detections with small areas:
    #     # l = tracker.segm.areas[1:] < npixels
    #
    #     # add new detections
    #     faint_labels = self.segm.add_segments(new_segm)
    #     # calculate centroids of new detections
    #     if len(faint_labels):
    #         coords = self.segm.com_bg(im_bgs, mask=self.bad_pixel_mask)
    #         # ignore the faint stars for subsequent frame shift calculations
    #         self.ignore_labels = np.union1d(self.ignore_labels, faint_labels)
    #
    #         # update coords
    #         self.yx0 = coords[self.ir]
    #         self.rvec = coords - self.yx0
    #
    #         # set bg model
    #         self.bgmodel = bgmodel
    #
    #         return faint_labels, results, im_bgs
    #     else:
    #         return faint_labels, None, image

# from graphical.imagine import FitsCubeDisplay
# from matplotlib.widgets import CheckButtons
#
#
# class GraphicalStarTracker(
#         FitsCubeDisplay):  # TODO: Inherit from mpl Animation??
#
#     trackerClass = StarTracker
#     marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)
#
#     def __init__(self, filename, **kws):
#         FitsCubeDisplay.__init__(self, filename, ap_prop_dict={},
#                                  ap_updater=None, **kws)
#
#         self.outlines = None
#
#         # toDO
#         self.playing = False
#
#         # bbox = self.ax.get_position()
#         # rect = bbox.x0, bbox.y1, 0.2, 0.2
#         rect = 0.05, 0.825, 0.2, 0.2
#         axb = self.figure.add_axes(rect, xticks=[], yticks=[])
#         self.image_buttons = CheckButtons(axb, ('Tracking Regions', 'Data'),
#                                           (False, True))
#         self.image_buttons.on_clicked(self.image_button_action)
#
#         # window slices
#         # rect = 0.25, 0.825, 0.2, 0.2
#         # axb = self.figure.add_axes(rect, xticks=[], yticks=[])
#         # self.window_buttons = CheckButtons(axb, ('Windows', ), (False, ))
#         # self.window_buttons.on_clicked(self.toggle_windows)
#
#         self.tracker = None
#         self._xmarks = None
#         # self._windows = None
#
#     # def toggle_windows(self, label):
#
#     def image_button_action(self, label):
#         image = self.get_image_data(self.frame)
#         self.imagePlot.set_data(image)
#         self.figure.canvas.draw()
#
#     def init_tracker(self, first, **findkws):
#         if isinstance(first, int):
#             first = slice(first)
#         img = np.median(self.data[first], 0)
#         self.tracker = self.trackerClass.from_image(img, **findkws)
#
#         return self.tracker, img
#
#     @property
#     def xmarks(self):
#         # dynamically create track marks when first getting this property
#         if self._xmarks is None:
#             self._xmarks, = self.ax.plot(*self.tracker.rcoo[:, ::-1].T,
#                                          **self.marker_properties)
#         return self._xmarks
#
#     def get_image_data(self, i):
#         tr, d = self.image_buttons.get_status()
#         image = self.data[i]
#         trk = self.tracker
#         if (tr and d):
#             mask = trk.segm.to_bool(trk.use_labels)
#             data = np.ma.masked_array(image, mask=mask)
#             return data
#         elif tr:  # and not d
#             return trk.segm.data
#         else:
#             return image
#
#     def set_frame(self, i, draw=True):
#         self.logger.debug('set_frame: %s', i)
#
#         i = FitsCubeDisplay.set_frame(self, i, False)
#         # needs_drawing = self._needs_drawing()
#
#         # data = self.get_image_data(i)
#
#         centroids = self.tracker(self.data[i])  # unmask and track
#         self.xmarks.set_data(centroids[:, ::-1].T)
#
#         if self.outlines is not None:
#             segments = []
#             off = self.tracker.offset[::-1]
#             for seg in self.outlineData:
#                 segments.append(seg + off)
#             self.outlines.set_segments(segments)
#
#         # if self.use_blit:
#         #     self.draw_blit(needs_drawing)
#
#     def show_outlines(self, **kws):
#         from matplotlib.collections import LineCollection
#         from matplotlib._contour import QuadContourGenerator
#
#         segm = self.tracker.segm
#         data = segm.data
#         outlines = []
#         for s in segm.slices:
#             sy, sx = s
#             e = np.array([[sx.start - 1, sx.stop + 1],
#                           [sy.start - 1, sy.stop + 1]])
#             im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]
#             f = lambda x, y: im[int(y), int(x)]
#             g = np.vectorize(f)
#
#             yd, xd = im.shape
#             x = np.linspace(0, xd, xd * 25)
#             y = np.linspace(0, yd, yd * 25)
#             X, Y = np.meshgrid(x[:-1], y[:-1])
#             Z = g(X[:-1], Y[:-1])
#
#             gen = QuadContourGenerator(X[:-1], Y[:-1], Z, None, False, 0)
#             c, = gen.create_contour(0)
#             outlines.append(c + e[:, 0] - 0.5)
#
#         col = LineCollection(outlines, **kws)
#         self.ax.add_collection(col)
#
#         self.outlineData = outlines
#         self.outlines = col
#         return col
#
#     def play(self):
#         if self.playing:
#             return
#
#     def pause(self):
#         'todo'
