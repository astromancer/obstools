
# third-party
import numpy as np

# local
from recipes.functionals import echo


# ---------------------------------------------------------------------------- #
_neighbours = {
    4: np.array([(-1, 0), (0, -1), (0, 1), (1, 0)]).T,
    8: np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]).T
}


# ---------------------------------------------------------------------------- #
def get_neighbour_index(pos, size=1, connectivity=8):
    return np.round(pos).astype(int)[..., np.newaxis] + _neighbours[connectivity]


def get_neighbours(a, pos, size=1, connectivity=8, func=echo):
    indices = get_neighbour_index(pos, size, connectivity)
    bad = (-1 > indices) | indices >= np.atleast_3d(a.shape)
    for i, x in zip(indices, bad):
        yield func(a[tuple(i[:, np.logical_not(x.any(0))])])
        # return a[tuple(np.moveaxis(indices, 1, 0))]
