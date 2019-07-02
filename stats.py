import numpy as np
from scipy.spatial.distance import cdist, euclidean


def mad(data, data_median=None, axis=None):
    """
    Median absolute deviation

    Parameters
    ----------
    data
    data_median

    Returns
    -------

    """
    nans = np.isnan(data)
    if nans.any():
        data = np.ma.MaskedArray(data, nans)

    if data_median is None:
        data_median = np.ma.median(data, axis, keepdims=True)
    # else:
    # make sure we can broadcast them together
    return np.ma.median(np.abs(data - data_median), axis)


def median_scaled_median(data, axis):
    """

    Parameters
    ----------
    data
    axis

    Returns
    -------

    """
    frame_med = np.median(data, (1, 2))[:, None, None]
    scaled = data / frame_med
    return np.median(scaled, axis)


def geometric_median(x, eps=1e-5):
    """
    The geometric median of a discrete set of sample points in a Euclidean space
    is the point minimizing the sum of distances to the sample points. This
    generalizes the median, which has the property of minimizing the sum of
    distances for one-dimensional data, and provides a central tendency in
    higher dimensions. It is also known as the 1-median, spatial median,
    Euclidean minisum point, or Torricelli point.

    Parameters
    ----------
    x
    eps: float
        accuracy

    Returns
    -------

    """
    # about: https://en.wikipedia.org/wiki/Geometric_median
    # adapted from: https://stackoverflow.com/a/30305181/1098683

    # ensure we have an array
    x = np.asanyarray(x)
    ndim = x.ndim

    # make sure we have some data
    assert x.size

    # remove any masked points along trailing dimensions
    if np.ma.is_masked(x):
        x = x[~x.mask.any(tuple(range(1, ndim)))].data
        # if all points are masked, return masked array
        if not x.size:
            return np.ma.masked_all(ndim - 1, x.dtype)

    # starting point: sample mean
    y = x.mean(0)

    while True:
        dist = cdist(x, [y])    # distance to mean
        non_zeros = (dist != 0)[:, 0]

        dist_inv = 1 / dist[non_zeros]
        dist_inv_sum = np.sum(dist_inv)
        w = dist_inv / dist_inv_sum
        T = np.sum(w * x[non_zeros], 0)

        num_zeros = len(x) - np.sum(non_zeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(x):
            return y
        else:
            R = (T - y) * dist_inv_sum
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
