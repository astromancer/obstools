# import logging

import numpy as np
from scipy import ndimage
from photutils.segmentation import SegmentationImage

from obstools.phot.utils import mad


class SegmentationHelper(SegmentationImage):
    @classmethod
    def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
                   flux_sort=True, dilate=True):

        from photutils.detection import detect_threshold
        from photutils.segmentation import detect_sources

        # detect
        threshold = detect_threshold(image, snr)
        segm = detect_sources(image, threshold, npixels)
        # NOTE: making this a detect classmethod might be useful and avoid double
        # initialization
        # initialize
        ins = cls(segm.data)

        if edge_cutoff:
            ins.remove_border_labels(edge_cutoff, partial_overlap=False)
            # ins.relabel_sequential()

        if deblend:
            from photutils import deblend_sources
            ins = deblend_sources(image, ins, npixels)

        if flux_sort:
            ins.flux_sort(image)

        if dilate:
            ins.dilate()

        return ins

    @classmethod
    def from_masks(cls, masks):
        data = np.zeros_like(masks.shape[1:])
        for i, mask in enumerate(masks):
            data[mask] = (i + 1)
        return cls(data)

    def counts(self, image, labels=None):
        if labels is None:
            labels = self.labels
        return ndimage.sum(image, self.data, labels)

    def flux(self, image, labels=None, bgfunc=np.median):
        if labels is None:
            labels = self.labels

        counts = self.counts(image, labels)
        areas = self.area(labels)
        bg = bgfunc(image[self.data == 0])
        return (counts / areas) - bg

    def flux_sort(self, image):
        flx = self.flux(image)
        flx_srt, lbl_srt = zip(*sorted(zip(flx, self.labels), reverse=1))

        # re-order segmented image labels
        data = np.zeros_like(self.data)
        for new, old in enumerate(lbl_srt):
            data[self.data == old] = (new + 1)
        self.data = data

    # @lazyproperty
    def masks3D(self): #labels=None
        # expand segments into 3D sequence of masks, one segment per image
        # if labels is None:
        #     labels = self.labels
        return self.data[None] == self.labels[:, None, None]

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

    def com(self, image, labels=None):
        if labels is None:
            labels = self.labels

        return np.array(ndimage.center_of_mass(image, self.data, labels))

    def snr(self, image, labels=None):
        if labels is None:
            labels = self.labels

        flx = self.flux(image, labels)
        bgstd = image[self.data == 0].std()
        return flx / bgstd

    def shift(self, offset):
        self.data = ndimage.shift(self.data, offset)

    def to_annuli(self, buffer=3, width=5):
        # bg regions
        masks = sh.masks3D()
        struct = ndimage.generate_binary_structure(2, 1)
        # structure array needs to have same rank as masks
        struct = struct[None]
        m0 = ndimage.binary_dilation(masks, struct, iterations=buffer)
        m1 = ndimage.binary_dilation(m0, struct, iterations=width)
        msky = (m1 & ~(m0 | ~m0.any(0)))
        sky_segm = SegmentationHelper.from_masks(msky)

class StarTracker():
    @classmethod
    def from_image(cls, image, snr=3., npixels=7, edge_cutoff=3, deblend=False,
                   flux_sort=True, dilate=True):
        # create segmentationHelper
        sh = SegmentationHelper.from_image(
            image, snr, npixels, edge_cutoff, deblend, flux_sort, dilate)

        # Center of mass
        found = sh.com(image)



        return cls(found, sh)

    def __init__(self, rcoo, segm):
        self.rcoo = rcoo
        self.segm = segm

    def __call__(self, image):
        shift = self.get_shift(image)
        if np.any(shift > 1):
            # shift the segmentation data
            shift = np.round(shift).astype(int)
            self.segm.shift(shift)
        return self.rcoo + shift

    def get_shift(self, image):

        bg = np.median(image[self.segm.data == 0])

        com = self.segm.com(image - bg)



        l = ~self.is_outlier(com)
        shift = np.mean(self.rcoo[l] - com[l], 0)
        return shift

    def is_outlier(self, coo, mad_thresh=5):
        """
        improve the robustness of the algorithm by removing centroids that are
        outliers.  Here an outlier is any point further that 5 median absolute
        deviations away. This helps track stars in low snr conditions.
        """
        if len(coo) < 6:  # scatter large for small sample sizes
            return np.zeros(len(coo), bool)

        r = np.sqrt(np.square(self.rcoo - coo).sum(1))
        return r - np.median(r) > mad_thresh * mad(r)


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

# class GraphicStarTracker(ImageCubeDisplayX):
