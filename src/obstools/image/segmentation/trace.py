"""
Radial pixel sweep / Moore's neighbour trace.
"""

import itertools as itt

import numpy as np

# pixel edge vectors for neighbour positions
EDGES = {(-1, 0): (0, 1),
         (0, 1): (1, 0),
         (1, 0): (0, -1),
         (0, -1): (-1, 0)}

# get Moore neighbour positions, anti-clockwise from bottom middle.
# Note: "anti-clockwise" for origin of the image at the bottom left
n = np.ones((3, 3))
n[1, 1] = 0
moves = np.subtract(np.where(n), 1).T
a = np.arctan2(*moves.T) + np.pi / 2
a[a < 0] += 2 * np.pi
# clockwise
o = a.argsort()
moves = moves[o]
step_sizes = np.sqrt(abs(moves).sum(1))

# index array for sequence position from relative neighbour positions
nix = np.zeros((3, 3), int)
nix[tuple(moves.T)] = range(8)

# cleanup module namespace
del n, a, o

# TODO: test:
# b = np.array([[ 1,  1,  1,  1,  1,  1,  1],
#               [ 1,  0,  1,  0,  1,  0,  1],
#               [ 1,  1,  1,  1,  1,  0,  0],
#               [ 0,  1,  0,  1,  0,  0,  0],
#               [ 1,  1,  0,  0,  0,  0,  0],
#               [ 1,  0,  1,  0,  1,  1,  1],
#               [ 0,  1,  1,  0,  1,  0,  1],
#               [ 1,  0,  0,  0,  1,  1,  1]])

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
        raise Exception
        # return

    # first pixel was entered from the left. we know it's bottom and left
    # neighbours are black. Since we are moving right, we can add the bottom
    # edge for the first pixel
    start = current = np.array(start)
    boundary = [start]
    pixels = [start]
    perimeter = 0       # circumference

    # trace
    counter = itt.count()
    ss = 0
    done = False
    while not done:
        count = next(counter)
        for i in range(ss, ss + 8):
            i %= 8  # loop neighbours
            mv = moves[i]
            step_size = step_sizes[i]
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
                ss = nix[tuple(moves[i - 1] - mv)]
                break

        if count >= stop:
            break

    return np.array(pixels), np.array(boundary), perimeter
