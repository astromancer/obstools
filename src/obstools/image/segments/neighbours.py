
import numpy as np


_neighbours = {
    4: np.array([(-1, 0), (0, -1), (0, 1), (1, 0)]).T,
    8: np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]).T
}


def get_neighbour_index(pos, size=1, connectivity=8):
    return np.round(pos).astype(int)[..., np.newaxis] + _neighbours[connectivity]


def get_neighbours(a, pos, size=1, connectivity=8):
    n = get_neighbour_index(pos, size, connectivity)
    return a[tuple(np.moveaxis(n, 1, 0))]
