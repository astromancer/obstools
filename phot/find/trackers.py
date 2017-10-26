import logging
import itertools as itt

import numpy as np
from scipy import ndimage
# from astropy.utils import lazyproperty
from photutils.detection import detect_threshold
from photutils.segmentation import detect_sources, SegmentationImage

from obstools.phot.utils import mad


from IPython import embed

class SegmentationHelper(SegmentationImage):
    """
    Additions to the SegmentationImage class.

    Adds basic support for iterating over segments
    Adds methods that operate on images for calculating counts / flux in segments.
    Useful for preparing masks for photometry
    Creating cutouts of segments for modelling
    """
    # This class is essentially a mapping layer that lives on top of images
    @classmethod
    def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
                   flux_sort=True, dilate=1):
        """
        Detect stars in an image and return a SegmentationHelper object

        Parameters
        ----------
        image
        snr
        npixels
        edge_cutoff
        deblend
        flux_sort
        dilate

        Returns
        -------

        """
        # detect
        threshold = detect_threshold(image, snr)
        segm = detect_sources(image, threshold, npixels)
        # NOTE: making this a detect classmethod might be useful and avoid double initialization
        # initialize
        obj = cls(segm.data)

        if edge_cutoff:
            obj.remove_border_labels(edge_cutoff, partial_overlap=False)
            # obj.relabel_sequential()

        if deblend:
            from photutils import deblend_sources
            obj = deblend_sources(image, obj, npixels)

        if dilate:
            obj.dilate(iterations=dilate)

        if flux_sort:
            obj.flux_sort(image)

        return obj

    # Mask methods
    # --------------------------------------------------------------------------------------
    @classmethod
    def from_masks(cls, masks):
        data = np.zeros(masks.shape[1:], int)
        # print(masks.shape)
        for i, mask in enumerate(masks):
            # print(data.shape, mask.shape)
            data[mask] = (i + 1)
        return cls(data)

    def to_mask(self, labels=None):
        """Extract the objects with labels and return as a masked array"""
        return self.masks3D(labels).any(0)

    def masks3D(self, labels=None):  # labels=None
        # expand segments into 3D sequence of masks, one segment per image
        if labels is None:
            labels = self.labels
        return self.data[None] == labels[:, None, None]

    def masked3D(self, image, labels):
        """
        Expand image to 3D masked array, indexed on segment label with
        background masked
        """
        m3D = np.ma.masked_equal(self.masks3D(labels), 0)
        return m3D * image

    def background(self, image, labels=None):
        """Mask foreground objects"""
        return np.ma.array(image, mask=self.to_mask(labels))
        #return np.ma.masked_where(self.data != 0, image)

    def foreground(self, image):
        """Mask background objects"""
        return np.ma.masked_where(self.data == 0, image)

    # --------------------------------------------------------------------------------------
    def iter_segments(self, data, labels=None, mask_overlapping=False, with_slices=False):
        """
        Yields rectangular sub-regions of the image

        Parameters
        ----------
        data
        labels
        mask_overlapping
        with_slices

        Returns
        -------

        """
        if labels is None:
            labels = self.labels
            indices = np.arange(len(labels))
        else:
            indices = self.indices(labels)
            # TODO: label_to_slice map as property

        slices = self.slices
        for i, lbl in zip(indices, labels):
            sl = slices[i]
            if sl is not None:              # NOTE filter None slices
                d = data[sl]
                q = self.data[sl]
                if mask_overlapping:
                    # mask other segments that overlap this slice
                    m = (q != lbl) & (q != 0)
                    d = np.ma.array(d, mask=m)
                if with_slices:
                    yield d, sl
                else:
                    yield d

    def check_labels(self, labels):

        labels = np.asarray(labels)
        invalid = np.setdiff1d(labels, self.labels)
        if len(invalid):
            raise ValueError('Invalid label(s): %s' % str(tuple(invalid)))

    def indices(self, labels):
        # FIXME: you can eliminate this method by handeling the indices like labels as
        #  lazyproperty
        self.check_labels(labels)

        indices = np.digitize(labels, self.labels) - 1
        return indices

    def thumbnails(self, image=None, labels=None):
        """
        Thumbnail cutout images based on segmentation

        Parameters
        ----------
        image

        Returns
        -------
        list of arrays
        """
        data = self.data if image is None else image
        # TODO: masked
        return list(self.iter_segments(data, labels))


    @property #lazyproperty
    def llc(self):
        """lower left corners of segment slices"""
        return [(y.start, x.start) for (y, x) in self.slices]

    @property #lazyproperty
    def urc(self):
        """upper right corners of segment slices"""
        return [(y.stop, x.stop) for (y, x) in self.slices]

    # TODO: ulc, lrc

    def box_sizes(self, labels=None):
        if labels is None:
            labels = self.labels
        ix = self.indices(labels)
        slices = np.take(self.slices, ix, 0)
        sizes = np.zeros((len(ix), 2))

        for i, sl in enumerate(slices):
            if sl is not None:
                sizes[i] = [np.subtract(*s.indices(sh)[1::-1])
                             for s, sh in zip(sl, self.shape)]
        return sizes


    def counts(self, image, labels=None):
        if labels is None:
            labels = self.labels
        return ndimage.sum(image, self.data, labels)

    def flux(self, image, labels=None): # bgfunc=np.median
        if labels is None:
            labels = self.labels

        counts = self.counts(image, labels)
        areas = self.area(labels)

        # bg = bgfunc(image[self.data == 0])
        return (counts / areas)# - bg

    def flux_sort(self, image):
        flx = self.flux(image)
        flx_srt, lbl_srt = zip(*sorted(zip(flx, self.labels), reverse=1))

        # re-order segmented image labels
        data = np.zeros_like(self.data)
        for new, old in enumerate(lbl_srt):
            data[self.data == old] = (new + 1)
        self.data = data

    def snr(self, image, labels=None):
        if labels is None:
            labels = self.labels

        flx = self.flux(image, labels)
        bgflx, bgstd = np.empty((2,) + flx.shape)
        for i, m in enumerate(self.to_annuli(1, 5, labels)):
            bgflx[i] = np.median(image[m])
            bgstd[i] = image[m].std()

        return (flx - bgflx) / bgstd


    def dilate(self, connectivity=4, iterations=1, mask=None,):
        # expand masks to 3D sequence
        masks = self.masks3D()

        if connectivity == 4:
            struct = ndimage.generate_binary_structure(2, 1)
        elif connectivity == 8:
            struct = ndimage.generate_binary_structure(2, 2)
        else:
            raise ValueError('Invalid connectivity={0}.  '
                             'Options are 4 or 8'.format(connectivity))
        # structure array needs to have same rank as masks
        struct = struct[None]
        masks = ndimage.binary_dilation(masks, struct, iterations, mask)
        data = np.zeros_like(self.data)
        for i, mask in enumerate(masks):
            data[mask] = (i + 1)

        self.data = data

    # @property #lazy
    def subgrids(self, labels=None):
        return list(self.gen_grids(labels))

    def gen_grids(self, labels):

        ix = self.indices(labels)
        slices = np.take(self.slices, ix, 0)
        grid = np.indices(self.shape)  # TODO: move to init??

        for s in slices:
            if s is None:
                yield None
            else:
                sly, slx = s
                yield grid[:, sly, slx]

    def com(self, image, labels=None):
        if labels is None:
            labels = self.labels
        return np.array(ndimage.center_of_mass(image, self.data, labels))

    def com_bg(self, image, labels=None, bad_pixel_mask=None, bg_func=np.median):
        if labels is None:
            labels = self.labels

        counter = itt.count()
        com = np.empty((len(labels), 2))
        gg = self.gen_grids(labels)
        for i, (s, g) in enumerate(zip(self.slices, gg)):
            lbl = i + 1
            if lbl in labels:
                j = next(counter)
                sub = image[s]
                sub = sub - bg_func(sub)
                sub[self.data[s] != lbl] = 0  # HMMMMM ... ?

                if bad_pixel_mask is not None:
                    sub = np.ma.array(sub, mask=bad_pixel_mask[s])

                com[j] = (sub * g).sum((1, 2)) / sub.sum() # may sum to zero

        return com

    def shift(self, offset):
        self.data = ndimage.shift(self.data, offset)
        # todo: optimization: update slices instead of re-running find_objects

    def to_annuli(self, buffer=0, width=5, labels=None):
        """
        Create 3D boolean array containing annular masks for sky regions

        Parameters
        ----------
        buffer
        width
        labels

        Returns
        -------

        """
        # sky annulli
        if labels is None:
            labels = self.labels

        masks = self.masks3D()
        # structure array needs to have same rank as masks
        struct = ndimage.generate_binary_structure(2, 1)[None]
        # dilate
        m0 = ndimage.binary_dilation(masks[labels], struct, iterations=buffer)
        m1 = ndimage.binary_dilation(m0, struct, iterations=width)
        msky = (m1 & ~masks.any(0))
        return msky
        #annuli = SegmentationHelper.from_masks(msky)
        #return annuli

    def add_segments(self, data):
        """NOTE: Assumes non-overlapping segments"""
        if isinstance(data, SegmentationImage):
            new_labels = data.labels
            data = data.data
        else:
            new_labels = np.unique(data)[1:]

        # new labels are indexed after existing ones
        off = self.labels.max()
        new_labels += off
        data[data != 0] += off
        self.data += data

        return new_labels

    def inside_detection_region(self, coords, labels=None):
        # filter COM positions that are outside of detection regions

        if labels is None:
            labels = self.labels

        assert len(coords) == len(labels)

        keep = np.zeros(len(coords), bool)
        count = 0
        for i, lbl in enumerate(labels):
            sl = self.slices[lbl-1]
            grid = np.mgrid[sl]  # use subgrids property
            keep[i] = inside_detection_region(coords[i], self.data[sl], grid)

            if not keep[i]:
                count += 1
                logging.debug('Discarding %i of %i at coords (%3.1f, %3.1f)',
                              count, len(coords), *coords[i])

        return keep


def inside_detection_region(coords, sub, grid):

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


class LabelLogicMixin():

    @property
    def ignore_labels(self): # ignore_labels / use_labels
        return self._ignore_labels

    @ignore_labels.setter
    def ignore_labels(self, labels):
        self._ignore_labels = np.asarray(labels)
        self._use_labels = np.setdiff1d(self.segm.labels, labels)

    @property
    def use_labels(self): # ignore_labels / use_labels
        return self._use_labels

    @use_labels.setter
    def use_labels(self, labels):
        self._use_labels = np.asarray(labels)
        self._ignore_labels = np.setdiff1d(self.segm.labels, labels)


class StarTracker(LabelLogicMixin):
    """
    A class to track stars in a CCD image frame to aid in doing time series photometry.

    Stellar positions in each new frame are calculated as the background subtracted centroids
    of the chosen image segments. Centroid positions of chosen stars are filtered for bad
    values (or outliers) and then combined by a weighted average. The weights are taken
    as the SNR of the star brightness.

    Handles bad pixels masks.

    Constellation of stars in image assumed unchanging. The coordinate positions of the stars
    with respect to the brightest star are updated upon each iteration.


    """
    segmClass = SegmentationHelper
    snr_weighting = True # TODO: manage as properties ?
    snr_cut = 3
    # this allows tracking stars with low snr, but not computing their
    # centroids which will have larger scatter

    @classmethod
    def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
                   flux_sort=True, dilate=True, bad_pixel_mask=None):
        """Initialize the tracker from an image"""
        # create segmentationHelper
        sh = cls.segmClass.from_image(
            image, snr, npixels, edge_cutoff, deblend, flux_sort, dilate)

        # Center of mass
        found = sh.com_bg(image, None, bad_pixel_mask)

        return cls(found, sh, None, bad_pixel_mask)


    def __init__(self, coords, segm, ignore_labels=None, bad_pixel_mask=None):
        """
        Use to track the location of stars on CCD frame

        Parameters
        ----------
        coords : array-like
            refrence coordinates of the stars
        segm : SegmentationHelper instance
            (or inherited from)
        ignore_labels : array-like
            indices of stars to ignore when calculating shift
        """
        # counter for incremental update
        self.counter = itt.count(1)
        # store zero and relative positions separately so we can update them independently
        self.ir = 0

        # pixel coordinates of reference position from which the shift will be measured
        self.yx0 = coords[self.ir]
        self.rvec = coords - coords[self.ir]
        self.segm = segm
        self._original_data = self.segm.data

        if ignore_labels is None:
            ignore_labels = []
        self.ignore_labels = ignore_labels

        #
        self.bad_pixel_mask = bad_pixel_mask

        #
        self.offset = [0, 0]


    def __call__(self, image, mask=None):
        """
        Track the shift of the image frame from

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

        # flag outliers
        bad = self.is_bad(com)
        logging.debug('bad: %s', bad)

        # snr weighting scheme (robustness)
        if self.snr_weighting or self.snr_cut:
            snr = self.segm.snr(image, self.use_labels)
            logging.debug('snr: %s', snr)
            weights = snr * ~bad
            logging.debug('weights: %s', weights)
        else:
            weights = np.ones(self.segm.nlabels)

        if self.snr_cut:
            # ignore stars with low snr (their positions will still be tracked,
            # but not used to compute new positions)
            low_snr = snr < self.snr_cut
            if low_snr.sum() == len(self.use_labels):
                logging.warning('SNR for all stars below cut: %s < %.1f', snr, self.snr_cut)
                low_snr = snr < snr.max() # else we end up with nans

            weights[low_snr] = 0

        logging.debug('weights: %s', weights)

        if np.all(weights == 0):
            # from IPython import embed
            # embed()
            # raise ValueError
            weights += 1

        # update relative positions
        self.update_rvec(com, weights)
        # calculate median com shift using current best relative positions
        off = self.update_offset(com, weights)

        # finally return the new coordinates
        return self.rcoo + off

    @property
    def rcoo(self):
        """
        Reference coordinates. Computed as initial coordinates + relative coordinates to
        allow updating the relative positions upon call

        """
        return self.yx0 + self.rvec


    def background(self, image, mask=None):
        """
        Prepare a background image by masking stars, badpixels and whatever else
        """
        # mask stars
        imbg = self.segm.background(image)
        if mask is not None:
            imbg.mask |= mask

        if self.bad_pixel_mask is not None:
            imbg.mask |= self.bad_pixel_mask

        return imbg


    def is_bad(self, coo):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """

        # flag inf / nan values
        lbad = (np.isinf(coo) | np.isnan(coo)).any(1)

        # filter COM positions that are outside of detection regions
        lbad2 = ~self.segm.inside_detection_region(coo, self.use_labels)

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
        # shift calculated as snr weighted mean of individual com shifts
        offset = np.ma.average(coo - self.rcoo, 0, weights)
        logging.debug('offset: (%.2f, %.2f)', *offset)

        # shift from previous frame
        shift = offset - self.offset
        ishift = np.round(shift).astype(int).data # array not ma

        # if frame has shifted by more than one pixel, update the segmentationImage
        if np.abs(ishift).sum():  # any shift greater or equal one pixel
            # shift the segmentation data
            ioff = np.round(offset).astype(int).data
            logging.debug('shifting to offset: (%i, %i)', *ioff)
            self.segm.data = ndimage.shift(self._original_data, ioff)

            # update pixel offset from original
            # self.offset = np.round(offset).astype(int).data # array not ma
            self.offset = offset.data

        return offset


    def update_rvec(self, coo, weights=None):
        """"
        Incremental average of relative positions of stars (since they are considered static)
        """
        # see: https://math.stackexchange.com/questions/106700/incremental-averageing
        vec = coo - coo[self.ir]
        n = next(self.counter)
        weights = (weights / weights.max() / n)[:, None]
        inc = (vec - self.rvec) * weights
        logging.debug('inc: %s', str(inc))
        self.rvec += inc

    # def get_shift(self, image):
    #     com = self.segm.com_bg(image)
    #     l = ~self.is_outlier(com)
    #     shift = np.median(self.rcoo[l] - com[l], 0)
    #     return shift


    # def mask_outliers(self):




        # def best_for_tracking(self, close_cut=None, snr_cut=_snr_cut, saturation=None):
        #     """
        #     Find stars that are best suited for centroid tracking based on the
        #     following criteria:
        #     """
        #     too_bright, too_close, too_faint = [], [], []
        #     msg = 'Stars: %s too %s for tracking'
        #     if saturation:
        #         too_bright = self.too_bright(self.image, saturation)
        #         if len(too_bright):
        #             logging.debug(msg, str(too_bright), 'bright')
        #     if close_cut:
        #         too_close = self.too_close(close_cut)
        #         if len(too_close):
        #             logging.debug(msg, str(too_close), 'close')
        #     if snr_cut:
        #         too_faint = self.too_faint(snr_cut)
        #         if len(too_faint):
        #             logging.debug(msg, str(too_faint), 'faint')
        #
        #     ignore = functools.reduce(np.union1d, (too_bright, too_close, too_faint))
        #     ix = np.setdiff1d(np.arange(len(self.found)), ignore)
        #     if len(ix) == 0:
        #         logging.warning('No suitable stars found for tracking!')
        #     return ix
        #
        # # def auto_window(self):
        # #     sdist_b = self.sdist[snr > self._snr_thresh]
        #
        # def too_faint(self, threshold=_snr_cut):
        #     crude_snr = self.flux / self.image[self.segm.data_masked.mask].std()
        #     return np.where(crude_snr < threshold)[0]
        #
        # def too_close(self, threshold=_distance_cut):
        #     # Check for potential interference problems from stars that are close together
        #     # threshold = threshold or self.window
        #     return np.unique(np.ma.where(self.sdist < threshold))
        #     # return np.intersect1d(self.ix_loc, too_close)
        #
        # def too_bright(self, data, saturation, threshold=_saturation_cut):
        #     # Check for saturated stars by flagging pixels withing 1% of saturation level
        #     # TODO: make exact
        #     lower, upper = saturation * (threshold + np.array([-1, 1]) / 100)
        #     # TODO check if you can improve speed here - dont have to check entire array?
        #     satpix = np.where((lower < data) & (data < upper))
        #     b = np.any(np.abs(np.array(satpix)[:, None].T - self.found) < 3, 0)
        #     w, = np.where(np.all(b, 1))
        #     return w




def get_streakmasks(self, image, flx_thresh=2.5e3, width=6):
    # Mask streaks
    flx = self.segm.flux(image)
    strkCoo = self.rcoo[flx > flx_thresh]
    w = np.multiply(width / 2, [-1, 1])
    strkRng = strkCoo[:, None, 1] + w
    strkSlc = map(slice, *strkRng.round(0).astype(int).T)

    strkMask = np.zeros(image.shape, bool)
    for sl in strkSlc:
        strkMask[:, sl] = True

    return strkMask


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
    def __call__(self, image, mask=None):

        # mask = self.bad_pixel_mask | self.streaks
        com = StarTracker.__call__(self, image, mask)
        # return com

        # update streak mask
        self.streaks = self.get_streakmasks(com[:len(self.bright)])

        return com

    @classmethod
    def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
                   flux_sort=True, dilate=True, bad_pixel_mask=None):

        #
        tracker = super(SlotModeTracker, cls).from_image(
            image, snr, npixels, edge_cutoff, deblend, flux_sort, dilate, bad_pixel_mask)

        tracker.bright = tracker.bright_star_labels(image)
        tracker.streaks = tracker.get_streakmasks(tracker.rcoo[tracker.bright - 1])
        return tracker

    def bright_star_labels(self, image, flx_thresh=2.5e3):
        flx = self.segm.flux(image)
        w, = np.where(flx > flx_thresh)
        w = np.setdiff1d(w, self.ignore_labels)
        coo = self.rcoo[w]
        bsl = self.segm.data[tuple(coo.round(0).astype(int).T)]
        return bsl

    def get_streakmasks(self, coo, width=6):
        # Mask streaks

        w = np.multiply(width / 2, [-1, 1])
        strkRng = coo[:, None, 1] + w
        strkSlc = map(slice, *strkRng.round(0).astype(int).T)

        strkMask = np.zeros(self.segm.data.shape, bool)
        for sl in strkSlc:
            strkMask[:, sl] = True

        return strkMask

    #     def add_streakmasks(self, image, flx_thresh=2.5e3, width=6):

    #         strkMask = self.get_streakmasks(image, flx_thresh, width)
    #         labelled, nobj = ndimage.label(strkMask)
    #         novr = strkMask & (self.segm.data != 0) # non-overlapping streaks
    #         labelled[novr] = 0

    #         streak_labels = self.segm.add_segments(labelled)
    #         self.ignore_labels = np.union1d(self.ignore_labels, streak_labels)
    #         return streak_labels


    def background(self, image, mask=None):
        imbg = StarTracker.background(self, image, mask)
        imbg.mask |= self.streaks
        return imbg


    def refine(self, image, bgmodel):
        # mask bad pixels / stars for bg fit
        # TODO: full frame model please!!!
        # TODO: incorporate in from_image method??

        # mask stars, streaks, bad pix

        # self.background(image)

        imm = self.segm.background(image)  # stars masked
        starmask = imm.mask.copy()
        if self.bad_pixel_mask is not None:
            imm.mask |= self.bad_pixel_mask
        if self.streaks is not None:
            imm.mask |= self.streaks

        # fit background
        results = (ry, rx) = bgmodel.fit(imm, report=False)
        p = bgmodel.make_params_from_result(results)

        # remove the streak masks for detection run since there may be faint but detectable stars partly in the streaks
        # prep frame: fill bright stars
        ism = np.ma.array(image, mask=starmask)
        resi = bgmodel.residuals(p, ism)
        resif = np.ma.filled(resi, np.ma.median(resi))  # fill bright stars

        # detect faint stars
        new_segm = SegmentationHelper.from_image(resif, npixels=6, edge_cutoff=0, dilate=1)
        new_segm.data[self.streaks] = 0
        faint_labels = self.segm.add_segments(new_segm)
        self.ignore_labels = np.union1d(self.ignore_labels, faint_labels)
        # flux_sort ??

        ibs = bgmodel.residuals(p, image)
        coords = self.segm.com_bg(ibs, self.use_labels, self.bad_pixel_mask)

        # TODO: optionally add faint stars for photometry
        # self.segm.areas[1:] > 10


        self.yx0 = coords[self.ir]
        self.rvec = coords - coords[self.ir]

        return faint_labels


from grafico.imagine import FitsCubeDisplay
from matplotlib.widgets import CheckButtons


class GraphicalStarTracker(FitsCubeDisplay):  # TODO: Inherit from mpl Animation??

    trackerClass = StarTracker
    marker_properties = dict(c='r', marker='x', alpha=1, ls='none', ms=5)

    def __init__(self, filename, **kws):
        FitsCubeDisplay.__init__(self, filename, ap_prop_dict={}, ap_updater=None, **kws)

        self.outlines = None

        # toDO
        self.playing = False

        # bbox = self.ax.get_position()
        # rect = bbox.x0, bbox.y1, 0.2, 0.2
        rect = 0.05, 0.825, 0.2, 0.2
        axb = self.figure.add_axes(rect, xticks=[], yticks=[])
        self.image_buttons = CheckButtons(axb, ('Tracking Regions', 'Data'), (False, True))
        self.image_buttons.on_clicked(self.image_button_action)

        # window slices
        # rect = 0.25, 0.825, 0.2, 0.2
        # axb = self.figure.add_axes(rect, xticks=[], yticks=[])
        # self.window_buttons = CheckButtons(axb, ('Windows', ), (False, ))
        # self.window_buttons.on_clicked(self.toggle_windows)

        self.tracker = None
        self._xmarks = None
        # self._windows = None

    # def toggle_windows(self, label):


    def image_button_action(self, label):
        image = self.get_image_data(self.frame)
        self.imagePlot.set_data(image)
        self.figure.canvas.draw()

    def init_tracker(self, first, **findkws):
        if isinstance(first, int):
            first = slice(first)
        img = np.median(self.data[first], 0)
        self.tracker = self.trackerClass.from_image(img, **findkws)

        return self.tracker, img

    @property
    def xmarks(self):
        # dynamically create track marks when first getting this property
        if self._xmarks is None:
            self._xmarks, = self.ax.plot(*self.tracker.rcoo[:, ::-1].T,
                                   **self.marker_properties)
        return self._xmarks

    def get_image_data(self, i):
        tr, d = self.image_buttons.get_status()
        image = self.data[i]
        trk = self.tracker
        if (tr and d):
            mask = trk.segm.to_mask(trk.use_labels)
            data = np.ma.masked_array(image, mask=mask)
            return data
        elif tr:  # and not d
            return trk.segm.data
        else:
            return image

    def set_frame(self, i, draw=True):
        logging.debug('set_frame: %s', i)

        i = FitsCubeDisplay.set_frame(self, i, False)
        # needs_drawing = self._needs_drawing()

        #data = self.get_image_data(i)


        centroids = self.tracker(self.data[i])  # unmask and track
        self.xmarks.set_data(centroids[:, ::-1].T)

        if self.outlines is not None:
            segments = []
            off = self.tracker.offset[::-1]
            for seg in self.outlineData:
                segments.append(seg + off)
            self.outlines.set_segments(segments)

        # if self.use_blit:
        #     self.draw_blit(needs_drawing)

    def show_outlines(self, **kws):
        from matplotlib.collections import LineCollection
        from matplotlib._contour import QuadContourGenerator

        segm = self.tracker.segm
        data = segm.data
        outlines = []
        for s in segm.slices:
            sy, sx = s
            e = np.array([[sx.start - 1, sx.stop + 1],
                          [sy.start - 1, sy.stop + 1]])
            im = data[e[1, 0]:e[1, 1], e[0, 0]:e[0, 1]]
            f = lambda x, y: im[int(y), int(x)]
            g = np.vectorize(f)

            yd, xd = im.shape
            x = np.linspace(0, xd, xd * 25)
            y = np.linspace(0, yd, yd * 25)
            X, Y = np.meshgrid(x[:-1], y[:-1])
            Z = g(X[:-1], Y[:-1])

            gen = QuadContourGenerator(X[:-1], Y[:-1], Z, None, False, 0)
            c, = gen.create_contour(0)
            outlines.append(c + e[:, 0] - 0.5)

        col = LineCollection(outlines, **kws)
        self.ax.add_collection(col)

        self.outlineData = outlines
        self.outlines = col
        return col

    def play(self):
        if self.playing:
            return

    def pause(self):
        'todo'
