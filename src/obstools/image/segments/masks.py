
# std
from collections import defaultdict

# third-party
import numpy as np
from astropy.utils import lazyproperty

# local
from recipes.dicts import AttrReadItem

# relative
from ..utils import make_border_mask


# class MaskUser:
#     """Mixin class giving masks property"""
#
#     def __init__(self, groups=None):
#         self._groups = None
#         self.set_groups(groups)
#

class SegmentMasks(defaultdict):
    """
    Container for segment masks
    """

    def __init__(self, seg):
        self.seg = seg
        defaultdict.__init__(self, None)

    def __missing__(self, label):
        # the mask is computed at lookup time here and inserted into the dict
        # automatically after this func executes

        # allow zero!
        if label != 0:
            self.seg.check_label(label)
        return self.seg.sliced(label) != label


class SegmentMasksMixin:
    @lazyproperty
    def masks(self):
        """
        A dictionary. For each label, a boolean array of the cutout segment
        with `False` wherever pixels have different labels. The dict will
        initially be empty - the masks are computed only when lookup happens.

        Returns
        -------
        dict of arrays (keyed on labels)
        """
        return SegmentMasks(self)


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
