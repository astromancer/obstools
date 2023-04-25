"""
Radial pixel sweep / Moore's neighbour trace.
"""

# std
import itertools as itt

# third-party
import numpy as np


# ---------------------------------------------------------------------------- #
# pixel edge vectors for neighbour positions
EDGES = {(-1,  0): ( 0,  1),
         ( 0,  1): ( 1,  0),
         ( 1,  0): ( 0, -1),
         ( 0, -1): (-1,  0)}

# Moore neighbour positions # autopep8: off
STEPS = np.array(
    [[-1,  0 ],
     [-1,  1 ],
     [ 0,  1 ],
     [ 1,  1 ],
     [ 1,  0 ],
     [ 1, -1 ],
     [ 0, -1 ],
     [-1, -1 ]]
)
STEP_SIZES = np.sqrt(abs(STEPS).sum(1))  # autopep8: on
# NOTE: These pixels trace out the neighbours in direction
#    "anti-clockwise"
#         from bottom right for origin of the array at the bottom left
#              [ 3,  2,  1 ]]
#              [ 4,  x,  0 ],
#             [[ 5,  6,  7 ]
#     "clockwise"
#         from top right for origin at the top middle (the way numpy prints)
#            [[ 7,  0,  1 ]
#             [ 6,  x,  2 ],
#             [ 5,  4,  3 ]]


# map from relative position index to neighbouring element for unravelled array
UNRAVEL_STEP = np.array([[0, 2, 6],
                         [4, 3, 5],
                         [0, 1, 7]])

# ---------------------------------------------------------------------------- #


def trace_boundary(b, stop=int(1e4)):
    """
    Radial pixel sweep / Moore's neighbour trace

    Parameters
    ----------
    b: array
        Binary image array.
    stop: int
        Emergency stop criterion. Algorithm will terminate after this many
        pixels have been searched and the starting pixel has not yet been
        reached.  This is used mostly for debugging / instruction purposes.
        You can safely ignore this argument unless your segments are very
        very large.

    Returns
    -------
    pixels: array (n, 2)
        Indices of pixels that make up the object boundary.
    boundary: array (m, 2)
        The boundary outline of the object composed of pixel edges.
    perimeter: float
        measurement of the object perimeter distance (circumference).
    """

    # find first pixel
    start = None
    for start in np.ndindex(b.shape):
        if b[start]:
            break

    if start is None:
        # image is blank
        raise ValueError('No sources in image.')

    # first pixel was entered from the left. we know it's bottom and left
    # neighbours are black. Since we are moving right, we can add the bottom
    # edge for the first pixel
    start = current = np.array(start)
    boundary = [start]
    pixels = [start]
    perimeter = 0       # circumference

    # trace
    ss = 0
    done = False
    counter = itt.count()
    while not done:
        count = next(counter)
        for i in range(ss, ss + 8):
            i %= 8  # loop neighbours
            mv = STEPS[i]
            step_size = STEP_SIZES[i]
            new = current + mv  # new pixel position

            # check if new position is outside image, or is a black pixel
            # out_of_bounds = ((-1 in new) or (new == b.shape).any())
            if ((-1 in new) or (new == b.shape).any()) or not b[tuple(new)]:
                # add edge if not diagonal neighbour
                if step_size == 1:
                    boundary.append(boundary[-1] + EDGES[tuple(mv)])

                # check if we are done. Jacob's stopping criterion
                if done := (current == start).all() and (mv == (0, -1)).all():
                    # close the perimeter
                    perimeter += step_size
                    break
            else:
                # reached a white pixel
                current = new
                pixels.append(new)
                perimeter += step_size

                # new position to start neighbour search
                ss = UNRAVEL_STEP[tuple(STEPS[i - 1] - mv)]
                break

        if count >= stop:
            break

    return np.array(pixels), np.array(boundary), perimeter
