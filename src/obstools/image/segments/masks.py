
# third-party
import numpy as np
from astropy.utils import lazyproperty

# local
from recipes.dicts import AttrReadItem

# relative
from .core import LabelGroupsMixin, SegmentedImage, SegmentsModelHelper


# class MaskUser:
#     """Mixin class giving masks property"""
#
#     def __init__(self, groups=None):
#         self._groups = None
#         self.set_groups(groups)
#


class MaskContainer(AttrReadItem):
    def __init__(self, seg, groups, **persistent):
        assert isinstance(groups, dict)  # else __getitem__ will fail
        super().__init__(persistent)
        self.persistent = np.array(list(persistent.values()))
        self.seg = seg
        self.groups = groups

    def __getitem__(self, name):
        if name in self:
            return super().__getitem__(name)

        if name in self.groups:
            # compute the mask of this group on demand
            mask = self[name] = self.of_group(name)
            return mask

        raise KeyError(name)

    @lazyproperty
    def all(self):
        return self.seg.to_binary()

    def for_phot(self, labels=None, ignore_labels=None):
        """
        Select sub-regions of the image that will be used for photometry.
        """
        labels = self.seg.resolve_labels(labels)
        # np.setdiff1d(labels, ignore_labels)
        # indices = np.digitize(labels, self.seg.labels) - 1
        kept_labels = np.setdiff1d(labels, ignore_labels)
        indices = np.digitize(labels, kept_labels) - 1

        m3d = self.seg.to_binary_3d(None, ignore_labels)
        all_masked = m3d.any(0)
        return all_masked & ~m3d[indices]

    def prepare(self, labels=None, ignore_labels=None, sky_buffer=2,
                sky_width=10, edge_cutoffs=None):
        """
        Select sub-regions of the image that will be used for photometry.
        """

        # sky_regions = self.prepare_sky(labels, sky_buffer, sky_width,
        #                                edge_cutoffs)
        self['phot'] = masks_phot = self.for_phot(labels, ignore_labels)
        self['sky'] = masks_phot.any(0)

    def prepare_sky(self, labels, sky_buffer=2, sky_width=10,
                    edge_cutoffs=None):
        sky_regions = self.seg.to_annuli(sky_buffer, sky_width, labels)
        if edge_cutoffs is not None:
            edge_mask = make_border_mask(self.seg.data, edge_cutoffs)
            sky_regions &= ~edge_mask

        return sky_regions

    def of_group(self, g):
        return self.seg.to_binary(self.groups[g])


class SegmentsMasksHelper(SegmentsModelHelper, LabelGroupsMixin):

    def __init__(self, data, grid=None, domains=None, groups=None,
                 **persistent_masks):
        #
        SegmentsModelHelper.__init__(self, data, grid, domains)
        LabelGroupsMixin.__init__(self, groups)
        self._masks = None
        self._persistent_masks = persistent_masks

    def set_groups(self, groups):
        LabelGroupsMixin.set_groups(self, groups)
        del self.group_masks  #

    @lazyproperty
    # making group_masks a lazyproperty means it will reset auto-magically when
    # the segmentation data changes
    def group_masks(self):
        return MaskContainer(self, self.groups, **self._persistent_masks)


class GlobalSegmentation(SegmentsMasksHelper):
    @classmethod
    def merge(cls, segmentations, delta_xy, extend=True, f_accept=0.5,
              post_dilate=1):
        # zero point correspond to minimum offset from initializer and serves as
        # coordinate reference
        # make int type to avoid `Cannot cast with casting rule 'same_kind'
        # downstream
        return cls(merge_segmentations(segmentations, delta_xy,
                                       extend, f_accept, post_dilate),
                   zero_point=np.floor(delta_xy.min(0)).astype(int))

    def __init__(self, data, zero_point):
        super().__init__(data)
        self.zero_point = zero_point

    def get_start_indices(self, delta_xy):
        if np.ma.is_masked(delta_xy):
            raise ValueError('Cannot get start indices for masked offsets.')

        return np.abs((delta_xy + self.zero_point).round().astype(int))

    def select_overlap(self, start, shape, type_=SegmentedImage):
        return super().select_overlap(start, shape, type_)

    def for_offset(self, delta_xy, shape, type_=SegmentedImage):
        if np.ma.is_masked(delta_xy):
            raise ValueError('Cannot get segmented image for masked offsets.')

        return self.select_overlap(
            self.get_start_indices(delta_xy), shape, type_)

    def flux(self, image, origin, labels=None, labels_bg=(0,), bg_stat='median'):
        sub = self.select_overlap(origin, image.shape)
        return sub.flux(image, labels, labels_bg, bg_stat)

    def sort(self, measure, descend=False):
        if (n := len(measure)) != self.nlabels:
            raise ValueError('Cannot reorder labels for measure with '
                             f'incompatable size {n}. {describe(self)} has '
                             f'{self.nlabels} labels.')

        o = slice(None, None, -1) if descend else ...
        order = np.ma.argsort(measure, endwith=False)[o]
        self.relabel_many(order + 1, self.labels)
        return order
