import numpy as np
from recipes.transforms.rotation import rotation_matrix


def rotate(xy, theta):
    # Xnew = (rot @ np.transpose(X)).T
    # xy2 = np.matmul(rot, np.transpose(xy)).T
    # einsum is about 2x faster than using @ and transposing twice, and about
    # 1.5x faster than np.matmul with 2 transposes
    return np.einsum('ij,...hj', rotation_matrix(theta), np.atleast_2d(xy))


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

    """
    return rotate(xy, p[-1]) + p[:2]


def affine(xy, p, scale=1):
    return rigid(xy * scale, p)
