"""
Display image cutouts in the console.
"""

# std
import functools as ftl

# third-party
import numpy as np
from loguru import logger

# local
import motley
from recipes.lists import split
from recipes.functionals import echo0

# relative
from .trace import trace_boundary


RIGHT_BORDER = '\N{RIGHT ONE EIGHTH BLOCK}'
LEFT_BORDER = '\N{LEFT ONE EIGHTH BLOCK}'

# '⎹' RIGHT VERTICAL BOX LINE
# '⎸' LEFT VERTICAL BOX LINE
# '⎺' HORIZONTAL SCAN LINE-1
# '⎽' HORIZONTAL SCAN LINE-9
# '⎥' RIGHT SQUARE BRACKET EXTENSION
# '▏' LEFT ONE EIGHTH BLOCK
# '▕' RIGHT ONE EIGHTH BLOCK
# '▔' UPPER ONE EIGHTH BLOCK
# '▁' LOWER ONE EIGHTH BLOCK

# '⎢' Left square bracket extension 023A2
# '⎥' Right square bracket extension 023A5
# '⎜' Left parenthesis extension 0239C
# '⎟' Right parenthesis extension  0239F

# '＿' Fullwidth Low Line (U+FF3F)
# '￣' Fullwidth Macron U+FFE3
# '｜' Fullwidth Vertical Line (U+FF5C)
# '［' Fullwidth Left Square Bracket(U+FF3B)
# '］' Fullwidth Right Square Bracket (U+FF3D)
# '⎴' Top square bracket 023B4
# '⎵' Bottom square bracket 023B5


def _add_edge(pixel, left, char, color):
    csi, nrs, fin, text, end = next(motley.ansi.parse(pixel))
    idx = int(left or -1)
    text = ''.join(split(text, idx)[idx])
    fg = motley.codes.get_code_str(color)
    # Apply color if this pixel doesn't already has an edge on the opposite side
    # with a fg color!
    if not nrs.startswith(fg):
        nrs = fg + nrs
    return ''.join((csi, nrs, fin, *(text, char)[::-idx], end))


def add_right_edge(pixel, color=None):
    return _add_edge(pixel, 0, RIGHT_BORDER, color)


def add_left_edge(pixel, color):
    return _add_edge(pixel, 1, LEFT_BORDER, color)


def _get_edge_drawing_funcs(color):
    left = ftl.partial(add_left_edge, color=color)
    right = ftl.partial(add_right_edge, color=color)
    top = bottom = ftl.partial(motley.apply, fg=('_', color))
    return [(left,   echo0, right),
            (bottom, echo0, top)]


def overlay(mask, pixels, color=None):
    """
    Overlay the contours from `mask` on the image `pixels`.

    Parameters
    ----------
    mask : array-like
        Boolean array of region to overlay the outline of.
    pixels : array-like
        Image pixels (strings).
    color : str or None, optional
        Colour of the contours, by default None.


    Returns
    -------
    np.ndarray(dtype=str)
        Pixels with edge characters added.
    """
    mask = np.asarray(mask)
    assert mask.shape == pixels.shape

    # edge drawing functions
    edge_funcs = _get_edge_drawing_funcs(color)

    # get boundary line
    (current, *_), boundary, _ = trace_boundary(mask)

    # underline top edge pixels requires extending the image size
    out = np.full(np.add(pixels.shape, (1, 0)), '  ', 'O')
    out[:-1] = pixels
    # out = pixels.astype('O')
    # i = 0
    for step in np.diff(boundary, axis=0):
        axis, = np.where(step)[0]
        add_vertex = edge_funcs[axis][step[axis] + 1]

        # logger.debug(step, axis, step[axis])
        offset = (0, 0)
        if step[axis] < 0:
            offset = step
        elif axis == 0:
            offset = (0, -1)
        ix = tuple(current + offset)

        # add edge character
        out[ix] = add_vertex(out[ix])
        logger.opt(lazy=True).debug('\n{}', lambda: motley.image.stack(out[::-1]))
        # update current pixel position
        current += step

    return out.astype(str)


class AnsiImage(motley.image.AnsiImage):
    # frame = True
    _top_row_added = False

    def overlay(self, mask, color=None):
        self.pixels = overlay(mask, self.pixels[::-1], color)[::-1].astype(str)
        self._top_row_added = True

    def format(self, frame=None):
        pixels = self.pixels
        if frame is None:
            frame = self.frame

        if frame:
            pixels = motley.image.framed(self.pixels, not self._top_row_added)

        return motley.image.stack(pixels)


def thumbnails(image, seg, top, image_cmap='cmr.voltage_r', contour_color='r',
               title_fmt='{{label:d|B_}: ^{width}}'):
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
    title_fmt : str, optional
        Format string for the image titles, by default
        '{{label:d|Bu}: ^{width}}'. This will produce centre justified lables in
        bold, underlined text above each image.

    """
    #    contour_cmap='hot'):
    # contours_cmap = seg.get_cmap(contour_cmap)
    #line_colours  = cmap(np.linspace(0, 1, top))

    labels = seg.labels[:top]
    sizes = seg.slices.sizes(labels)
    biggest = sizes.max(0)
    slices = seg.slices.grow(labels, (biggest - sizes) / 2, seg.shape)

    images = []
    for i, lbl in enumerate(seg.labels[:top]):
        sec = slices[i]
        title = f'{lbl: ^{2 * biggest[1]}}'
        title = motley.format(title_fmt, label=lbl, width=2 * biggest[1])
        img = AnsiImage(image[sec], image_cmap)
        img.overlay(seg.data[sec], contour_color)
        # contours_cmap(i / top)[:-1]
        images.append('\n'.join((title, str(img))))

    return motley.hstack(images, 2)
