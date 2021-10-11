"""
Display image cutouts in the console.
"""

# third-party
import numpy as np

# local
import motley


def source_thumbnails_terminal(image, seg, top,
               cmap='cmr.voltage_r', contour_color='r',
               title=None,
               label_fmt='{{label:d|B_}: ^{width}}'):
    """
    Cutout image thumbnails displayed as a grid in terminal

    Parameters
    ----------
    image : np.ndarray
        Image array with sources to display.
    seg : obstools.image.segmentation.SegmentedImage
        The segmented image of detected sources
    top : int
        Number of brightest sources to display images for.
    image_cmap : str, optional
        Colour map, by default 'cmr.voltage_r'.
    contour_color : str, optional
        Colour for the overlaid contour, by default 'r'.
    label_fmt : str, optional
        Format string for the image titles, by default
        '{{label:d|Bu}: ^{width}}'. This will produce centre justified lables in
        bold, underlined text above each image.

    """
    #    contour_cmap='hot'):
    # contours_cmap = seg.get_cmap(contour_cmap)
    #line_colours  = cmap(np.linspace(0, 1, top))

    labels = seg.labels[:top]
    image_stack = np.ma.array(seg.thumbnails(image, labels, True, True))
    return motley.image.thumbnails(image_stack.data, image_stack.mask,
                                   cmap, contour_color,
                                   title, labels, label_fmt)
