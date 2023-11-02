

# third-party
import numpy as np

# local
from pyxides.vectorize import vdict
from recipes.utils import duplicate_if_scalar

from collections import namedtuple

# ---------------------------------------------------------------------------- #
# simple container for 2-component objects
yxTuple = namedtuple('yxTuple', ['y', 'x'])

# ---------------------------------------------------------------------------- #

class SliceDict(vdict):
    """
    Dict-like container for tuples of slices. Aids selecting rectangular
    sub-regions of images more easily.
    """

    # maps semantic corner positions to slice attributes
    _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}

    def __init__(self, mapping, **kws):
        super().__init__()
        kws.update(mapping)
        for k, v in kws.items():
            self[k] = v  # ensures item conversion in `__setitem__`

    def __setitem__(self, key, value):
        # convert items to namedtuple so we can later do things like
        # >>> image[:, seg.slices[7].x]
        super().__setitem__(key, yxTuple(*value))

    @property
    def x(self):
        _, x = zip(*self.values())
        return vdict(zip(self.keys(), x))

    @property
    def y(self):
        y, _ = zip(*self.values())
        return vdict(zip(self.keys(), y))

    def _get_corners(self, vh, slices):
        # vh - vertical horizontal positions as two character string
        yss, xss = (self._corner_slice_mapping[_] for _ in vh)
        unpack = list
        if isinstance(slices, yxTuple):
            slices = [slices]
            unpack = next

        return unpack(((getattr(y, yss), getattr(x, xss))
                       for (y, x) in slices))

    def lower_left_corners(self, labels=...):
        """lower left corners of segment slices"""
        return self._get_corners('ll', self[labels])

    def lower_right_corners(self, labels=...):
        """lower right corners of segment slices"""
        return self._get_corners('lr', self[labels])

    def upper_right_corners(self, labels=...):
        """upper right corners of segment slices"""
        return self._get_corners('ur', self[labels])

    def upper_left_corners(self, labels=...):
        """upper left corners of segment slices"""
        return self._get_corners('ul', self[labels])

    # aliases
    llc = lower_left_corners
    lrc = lower_right_corners
    urc = upper_right_corners
    ulc = upper_left_corners

    def sizes(self, labels=...):
        return np.subtract(self.urc(labels), self.llc(labels))

    # alias
    shapes = sizes

    # def extents(self, labels=None):
    #     """xy sizes"""
    #     slices = self[labels]
    #     sizes = np.zeros((len(slices), 2))
    #     for i, sl in enumerate(slices):
    #         if sl is not None:
    #             sizes[i] = [np.subtract(*s.indices(sz)[1::-1])
    #                         for s, sz in zip(sl, self.seg.shape)]
    #     return sizes

    def extend(self, labels, increment=1, clip=False):
        """
        Increase the size of each slice in either / both  directions by an
        increment.
        """
        # FIXME: carry image size in slice 0 so we can default clip=True

        # z = np.array([slices.llc(labels), slices.urc(labels)])
        # z + np.array([-1, 1], ndmin=3).T
        urc = np.add(self.urc(labels), increment).astype(int)
        llc = np.add(self.llc(labels), -increment).astype(int)
        if clip:
            urc = urc.clip(None, duplicate_if_scalar(clip))  # TODO:.parent.shape
            llc = llc.clip(0)

        return [tuple(slice(*i) for i in _)
                for _ in zip(*np.swapaxes([llc, urc], -1, 0))]

    # def around_centroids(self, image, size, labels=None):
    #     com = self.seg.centroid(image, labels)
    #     slices = self.around_points(com, size)
    #     return com, slices
    #
    # def around_points(self, points, size):
    #
    #     yxhw = duplicate_if_scalar(size) / 2
    #     yxdelta = yxhw[:, None, None] * [-1, 1]
    #     yxp = np.atleast_2d(points).T
    #     yxss = np.round(yxp[..., None] + yxdelta).astype(int)
    #     # clip negative slice indices since they yield empty slices
    #     return list(zip(*(map(slice, *np.clip(ss, 0, sz).T)
    #                       for sz, ss in zip(self.seg.shape, yxss))))

    def plot(self, ax, **kws):
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection
        from matplotlib import colormaps

        kws.setdefault('facecolor', 'None')

        rectangles = []
        slices = list(filter(None, self))
        ec = kws.get('edgecolor')
        ecs = kws.get('edgecolors')
        if ec is None and ecs is None:
            cmap = colormaps[kws.get('cmap', 'gist_ncar')]
            ecs = cmap(np.linspace(0, 1, len(slices)))
            kws['edgecolors'] = ecs

        # make the patches
        for y, x in slices:
            xy = np.subtract((x.start, y.start), 0.5)  # pixel centres at 0.5
            w = x.stop - x.start
            h = y.stop - y.start
            r = Rectangle(xy, w, h)
            rectangles.append(r)

        # collect
        boxes = PatchCollection(rectangles, **kws)
        # plot
        ax.add_collection(boxes)
        return boxes


# class Slices:
#     # FIXME: remove this now superceded by SliceDict
#     """
#     Container emulation for tuples of slices. Aids selecting rectangular
#     sub-regions of images more easil
#     """
#     # maps semantic corner positions to slice attributes
#     _corner_slice_mapping = {'l': 'start', 'u': 'stop', 'r': 'stop'}
#
#     def __init__(self, seg_or_slices):
#         """
#         Create container of slices from SegmentationImage, Slice instance, or
#         from list of 2-tuples of slices.
#
#         Slices are stored in a numpy object array. The first item in the
#         array is a slice that will return the entire object to which it is
#         passed as item getter. This represents the "background" slice.
#
#         Parameters
#         ----------
#         slices
#         seg
#         """
#
#         # self awareness
#         if isinstance(seg_or_slices, Slices):
#             slices = seg_or_slices
#             seg = slices.seg
#
#         elif isinstance(seg_or_slices, SegmentationImage):
#             # get slices from SegmentationImage
#             seg = seg_or_slices
#             slices = SegmentationImage.slices.fget(seg_or_slices)
#         else:
#             raise TypeError('%r should be initialized from '
#                             '`SegmentationImage` or `Slices` object' %
#                             self.__class__.__name__)
#
#         # use object array as container so we can get items by indexing with
#         # list or array which is really nice and convenient
#         # secondly, we include index 0 as the background ==> full array slice!
#         # this means we can index this object directly with an array of
#         # labels, or integer label, instead of needing the -1 every time you
#         # want a slice
#         self.slices = np.empty(len(slices) + 1, 'O')
#         self.slices[0] = (slice(None),) * seg.data.ndim
#         self.slices[1:] = slices
#
#         # add SegmentedImage instance as attribute
#         self.seg = seg
