
# third-party
import numpy as np
from scipy import ndimage
from astropy.utils import lazyproperty
from photutils.segmentation import SegmentationImage

# relative
from ..utils import shift_combine


# ---------------------------------------------------------------------------- #
def is_lazy(_):
    return isinstance(_, lazyproperty)


# def is_sequence(obj):
#     """Check if obj is non-string sequence"""
#     return isinstance(obj, (tuple, list, np.ndarray))


def radial_source_profile(image, seg, labels=None):
    com = seg.com(image, labels)
    grid = np.indices(image.shape)
    profiles = []
    for i, (sub, g) in enumerate(seg.cutouts(image, grid, flatten=True,
                                             labels=labels)):
        r = np.sqrt(np.square(g - com[i, None].T).sum(0))
        profiles.append((r, sub))
    return profiles


def merge_segmentations(segmentations, xy_offsets, extend=True, f_accept=0.2,
                        post_merge_dilate=1):
    """

    Parameters
    ----------
    segmentations
    xy_offsets
    extend
    f_accept
    post_merge_dilate

    Returns
    -------

    """
    # merge detections masks by align, summation, threshold
    if isinstance(segmentations, (list, tuple)) and \
            isinstance(segmentations[0], SegmentationImage):
        segmentations = np.array([seg.data for seg in segmentations])
    else:
        segmentations = np.asarray(segmentations)

    n_images = len(segmentations)
    n_accept = max(f_accept * n_images, 1)

    eim = shift_combine(segmentations.astype(bool), xy_offsets, 'sum',
                        extend=extend)
    seg_image_extended, n_sources = ndimage.label(eim >= n_accept,
                                                  structure=np.ones((3, 3)))

    # edge case: it may happen that when creating the boolean array above
    # with the threshold `n_accept`, that pixels on the edges of sources
    # become separated from the main source. eg:
    #                     ________________
    #                     |              |
    #                     |    ████      |
    #                     |  ████████    |
    #                     |    ██████    |
    #                     |  ██  ██      |
    #                     |              |
    #                     |              |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # These pixels will receive a different label if we do not provide a
    # block structure element.
    # Furthermore, it also sometimes happens for faint sources that the
    # thresholding splits the source in two, like this:
    #                     ________________
    #                     |              |
    #                     |    ████      |
    #                     |  ██          |
    #                     |      ████    |
    #                     |      ██      |
    #                     |              |
    #                     |              |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # After dilating post merger, we end up with two labels for a single source
    # We fix these by running the "blend" routine. Note that this will
    #  actually blend blend sources that are touching but are actually
    #  distinct sources eg:
    #                     __________________
    #                     |    ██          |
    #                     |  ██████        |
    #                     |██████████      |
    #                     |  ██████        |
    #                     |    ████  ██    |
    #                     |      ████████  |
    #                     |        ████████|
    #                     |        ██████  |
    #                     |          ██    |
    #                     ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
    # If you have crowded fields you may need to run "deblend" again
    # afterwards to separate them again
    seg_extended = SegmentedImage(seg_image_extended)
    seg_extended.dilate(post_merge_dilate)
    seg_extended.blend()
    return seg_extended


def inside_segment(coords, sub, grid):
    b = []
    ogrid = grid[0, :, 0], grid[1, 0, :]
    for g, f in zip(ogrid, coords):
        bi = np.digitize(f, g - 0.5)
        b.append(bi)

    mask = (sub == 0)
    if np.equal(grid.shape[1:], b).any() or np.equal(0, b).any():
        return False
    return not mask[b[0], b[1]]


def boundary_proximity(seg, points, labels=None):
    labels = seg.resolve_labels(labels)
    return np.array([np.sqrt(np.square(xy - seg.traced[l][0][0]).sum(1).min())
                     for l, xy in zip(labels, points)])
    # return np.array([np.sqrt(np.square(xy - boundary).sum(1).min())
    #                  for ((boundary,), _), xy in zip(seg.traced.values(), points)])
