# std
import functools as ftl

# third-party
import numpy as np

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


def _add_vert(pixel, left, char, color):
    csi, bg, fin, text, end = next(motley.ansi.parse(pixel))
    idx = int(left or -1)
    parts = split(text, idx)
    text = ''.join(parts[idx])
    fg = motley.codes.get_code_str(color)
    return ''.join((csi, fg, bg, fin, *(text, char)[::-idx], end))


def right_vert(pixel, color=None):
    return _add_vert(pixel, 0, RIGHT_BORDER, color)


def left_vert(pixel, color):
    return _add_vert(pixel, 1, LEFT_BORDER, color)


def overlay(mask, pixels, color=None):
    assert mask.shape == pixels.shape

    # edge drawing functions
    left = ftl.partial(left_vert, color=color)
    right = ftl.partial(right_vert, color=color)
    top = bottom = ftl.partial(motley.apply, fg=('_', color))
    edge_funcs = [(left,   echo0, right),
                  (bottom, echo0, top)]

    # get boundary line
    (current, *_), boundary, _ = trace_boundary(mask)

    # underline top edge pixels requires extending the image size
    out = np.full(np.add(pixels.shape, (1, 0)), '  ', 'O')
    out[:-1] = pixels
    # out = pixels.astype('O')
    # i = 0
    for s in np.diff(boundary, axis=0):
        w, = np.where(s)[0]
        f = edge_funcs[w][s[w] + 1]

        # logger.debug(s, w, s[w])
        offset = (0, 0)
        if s[w] < 0:
            offset = s
        elif w == 0:
            offset = (0, -1)
        ix = tuple(current + offset)

        # add edge character
        out[ix] = f(out[ix])
        # update current pixel position
        current += s

    return out.astype(str)


class AnsiImage(motley.image.AnsiImage):
    frame = True
    _top_row_added = False

    def overlay(self, mask, color=None):
        self.pixels = overlay(mask, self.pixels[::-1], color)[::-1].astype(str)
        self._top_row_added = True

    def format(self, frame=frame):
        pixels = self.pixels
        if frame:
            pixels = motley.image.framed(self.pixels, not self._top_row_added)

        return motley.image.stack(pixels)
