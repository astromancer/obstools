"""
Transformation helper functions for cartesian coordinate arrays.
"""

# TODO: move up

import numpy as np
from recipes.transforms.rotation import rotation_matrix


def rotate(xy, theta):
    """
    Rotate cartesian coordinates `xy` be `theta` radians

    Parameters
    ----------
    xy: np.ndarray
        shape (n_samples, 2)
    theta : float
        angle of rotation in radians

    Returns
    -------
    np.ndarray
        transformed coordinate points
    """
    # Xnew = (rot @ np.transpose(X)).T
    # xy2 = np.matmul(rot, np.transpose(xy)).T
    # einsum is about 2x faster than using @ and transposing twice, and about
    # 1.5x faster than np.matmul with 2 transposes
    xy = np.atleast_2d(xy)
    return np.einsum('ij,...hj', rotation_matrix(theta), xy)


def rigid(xy, p):
    """
    A rigid transformation of the 2 dimensional cartesian coordinates `X`. A
    rigid transformation represents a rotatation and/or translation of a set of
    coordinate points, and is also known as a roto-translation, Euclidean
    transformation or Euclidean isometry depending on the context.

    See: https://en.wikipedia.org/wiki/Rigid_transformation

    Parameters
    ----------
    xy: np.ndarray
         shape (n_samples, 2)
    p: np.ndarray
         δx, δy, θ

    Returns
    -------
    np.ndarray
        transformed coordinate points shape (n_samples, 2)

    """
    xy = np.atleast_2d(xy)

    if (np.ndim(xy) < 2) or (np.shape(xy)[-1] != 2):
        raise ValueError('Invalid dimensions for coordinate array `xy`')

    if len(p) != 3:
        raise ValueError('Invalid parameter array for rigid transform `xy`')

    return rotate(xy, p[-1]) + p[:2]


def affine(xy, p, scale=1):
    """
    An affine transform

    Parameters
    ----------
    xy : [type]
        [description]
    p : [type]
        [description]
    scale : int, optional
        [description], by default 1

    Returns
    -------
    np.ndarray
        transformed coordinate points shape (n_samples, 2)
    """
    return rigid(xy * scale, p)
