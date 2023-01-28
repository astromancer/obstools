# third-party
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.stats import binned_statistic_2d

# local
import motley
from motley.table import Table


def non_masked(xy):
    xy = np.asanyarray(xy)
    if np.ma.is_masked(xy):
        return xy[~xy.mask.any(-1)].data
    return np.array(xy)


def table_coords(coo, ix_fit, ix_scale, ix_loc):
    # TODO: maybe add flux estimate

    # create table: coordinates
    ocoo = np.array(coo[:, ::-1], dtype='O')
    cootbl = Table(ocoo,
                   col_headers=list('xy'),
                   col_head_props=dict(bg='g'),
                   row_headers=range(len(coo)),  # starts numbering from 0
                   # row_nrs=True,
                   align='>',  # easier to read when right aligned
                   )

    # add colour indicators for tracking / fitting / scaling info
    ms = 2
    m = np.zeros((len(coo) + 1, 3), int)
    # m[0] = 1, 2, 3

    for i, ix in enumerate((ix_fit, ix_scale, ix_loc)):
        m[ix, i] = i + 1

    # flag stars
    cols = 'gbm'
    labels = ('fit|', 'scale|', 'loc|')
    tags = np.empty(m.shape, dtype='U1')
    tags[m != 0] = 'x'
    # tags[:] = 'x' * ms

    col_headers = motley.rainbow(labels, bg=cols)
    tt = Table(tags, title='\n',  # title,   # title_props=None,
               col_headers=col_headers,
               frame=False, align='^',
               col_borders='', cell_whitespace=0)
    tt.colourise(m, fg=cols)
    # ts = tt.add_colourbar(str(tt), ('fit|', 'scale|', 'loc|'))

    # join tables
    tbl = Table([[str(cootbl), str(tt)]], frame=False, col_borders='')
    return tbl


def table_cdist(sdist, window, _print=False):
    # from scipy.spatial.distance import cdist
    n = len(sdist)
    # check for stars that are close together
    # sdist = cdist(coo, coo)  # pixel distance between stars
    # sdist[np.tril_indices(n)] = np.inf
    #  since the distance matrix is symmetric, ignore lower half
    # mask = (sdist == np.inf)

    # create distance matrix as table, highlighting stars that are potentially
    # too close together and may cause problems
    bg = 'light green'
    # tbldat = np.ma.array(sdist, mask=mask, copy=True)
    tbl = Table(sdist,  # tbldat,
                title='Distance matrix',
                col_headers=range(n),
                row_headers=range(n),
                col_head_props=dict(bg=bg),
                row_head_props=dict(bg=bg),
                align='>')

    if sdist.size > 1:
        # Add colour as distance warnings
        c = np.zeros_like(sdist)
        c += (sdist < window / 2)
        c += (sdist < window)
        tbl.colourise(c, *' yr')
        tbl.show_colourbar = False
        tbl.flag_headers(c, bg=[bg] * 3, fg='wyr')

    if _print and n > 1:
        print(tbl)

    return tbl  # , c


def rand_median(cube, ncomb, subset, nchoose=None):
    """
    median combine `ncomb`` frames randomly from amongst `nchoose` in the interval
    `subset`

    Parameters
    ----------
    cube
    ncomb
    subset
    nchoose

    Returns
    -------

    """
    if isinstance(subset, int):
        subset = (0, subset)  # treat like a slice

    i0, i1 = subset
    if nchoose is None:  # if not given, select from entire subset
        nchoose = i1 - i0

    # get frame indices
    nfirst = min(nchoose, i1 - i0)
    ix = np.random.randint(i0, i0 + nfirst, ncomb)
    # create median image for init
    logger.info('Combining {:d} frames from amongst frames ({:d}->{:d}) for '
                'reference image.', ncomb, i0, i0 + nfirst)
    return np.median(cube[ix], 0)


def shift_combine(images, offsets, stat='mean', extend=False):
    """
    Statistics on image stack each being offset by some xy-distance

    Parameters
    ----------
    images
    offsets
    stat
    extend

    Returns
    -------

    """
    # convert to (masked) array
    images = np.asanyarray(images)
    offsets = np.asanyarray(offsets)

    # it can happen that `offsets` is masked (no stars in that frame)
    if np.ma.is_masked(offsets):
        # ignore images for which xy offsets are masked
        bad = offsets.mask.any(1)
        good = ~bad

        logger.info(f'Removing {bad.sum()} images from stack due to null '
                    f'detection')
        images = images[good]
        offsets = offsets[good]

    # get pixel grid ignoring masked elements
    shape = sy, sx = images.shape[1:]
    grid = np.indices(shape)
    gg = grid[:, None] - offsets[None, None].T
    if np.ma.is_masked(images):
        y, x = gg[:, ~images.mask]
        sample = images.compressed()
    else:
        y, x = gg.reshape(2, -1)
        sample = images.ravel()

    # use maximal area coverage. returned image may be larger than input images
    if extend:
        y0, x0 = np.floor(offsets.min(0))
        y1, x1 = np.ceil(offsets.max(0)) + shape + 1
    else:
        # returned image same size as original
        y0 = x0 = 0
        y1, x1 = np.add(shape, 1)

    # compute statistic
    yb, xb = np.ogrid[y0:y1, x0:x1]
    bin_edges = (yb.ravel() - 0.5, xb.ravel() - 0.5)
    results = binned_statistic_2d(y, x, sample, stat, bin_edges)
    image = results.statistic
    # mask nans (empty bins (pixels))

    # note: avoid downstream warnings by replacing np.nan with zeros and masking
    # todo: make optional
    nans = np.isnan(image)
    image[nans] = 0
    return np.ma.MaskedArray(image, nans)


def scale_combine(images, stat='mean'):
    """
    Statistics on image stack each being scaled to equal size

    Parameters
    ----------
    images
    stat


    Returns
    -------

    """
    # convert to (masked) array
    # images = np.asanyarray(images)

    # get pixel grid ignoring masked elements
    gy, gx = [], []
    values = []
    for image in images:
        sy, sx = images.shape[1:]
        yx = (y, x) = np.mgrid[0:1:complex(sy), 0:1:complex(sx)]
        if np.ma.is_masked(image):
            y, x = yx[:, ~image.mask]
            values.extend(image.compressed())
        else:
            values.extend(images.ravel())
        gy.extend(y)
        gx.extend(x)

    # compute bins
    yb, xb = np.ogrid[0:1:complex(sy), 0:1:complex(sx)]
    bin_edges = (yb.ravel() - 0.5, xb.ravel() - 0.5)
    # compute statistic
    results = binned_statistic_2d(gy, gx, values, stat, bin_edges)
    image = results.statistic
    # mask nans (empty bins (pixels))

    # note: avoid downstream warnings by replacing np.nan with zeros and masking
    nans = np.isnan(image)
    image[nans] = 0
    return np.ma.MaskedArray(image, nans)


def deep_sky(images, fovs, params, resolution=None, statistic='mean',
             masked=True):
    # todo rename
    from obstools.image.registration import roto_translate_yx

    data = []
    gy, gx = [], []
    yrng, xrng = [np.inf, -np.inf], [np.inf, -np.inf]

    def update(x, rng, g):
        rng[0] = min(rng[0], x.min())
        rng[1] = max(rng[1], x.max())
        g.extend(x.ravel())

    for image, fov, p in zip(images, fovs, params):
        sy, sx = map(complex, image.shape)
        yx = np.mgrid[:1:sy, :1:sx] * fov[:, None, None]
        y, x = roto_translate_yx(yx.reshape(2, -1).T, p).T
        update(x, xrng, gx)
        update(y, yrng, gy)
        data.extend(image.ravel())

    # get bins
    y0, y1 = yrng
    x0, x1 = xrng
    if resolution is None:
        resolution = np.max(list(map(np.shape, images)), 0).astype(int)

    sy, sx = map(complex, resolution)
    yb, xb = map(np.ravel, np.ogrid[y0:y1:sy, x0:x1:sx])
    δy, δx = map(np.diff, (yb[:2], xb[:2]))
    yb = np.hstack([yb.ravel() - 0.5 * δy, yb[-1] + δy])
    xb = np.hstack([xb.ravel() - 0.5 * δx, xb[-1] + δx])

    #
    results = binned_statistic_2d(gy, gx, data, statistic, (yb, xb))
    image = results.statistic
    # mask nans (empty bins (pixels))

    if masked:
        # replace nans with zeros and mask
        nans = np.isnan(image)
        image[nans] = 0
        image = np.ma.MaskedArray(image, nans)
    return image


def view_neighbours(array, neighbourhood=7):
    """
    Return a view of the neighbourhood surrounding each pixel in the image.
    The returned image will have shape (r, c, n, n) where (r, c) is the
    original image shape and n is the size of the neighbourhood. Note that
    the array returned by this function uses numpy's stride tricks to avoid
    copying data and therefore has multiple elements that refer to the same
    unique element in memory.

    Examples
    --------
    >>> z = np.arange(4 * 5).reshape(4, 5)
    >>> q = view_neighbours(z, 3)
    >>> z[2, 4]  # 14
    >>> q[2, 4]
    The neighbourhood of element in (2, 4), out of bounds items are masked
        [[8 9 --]
         [13 14 --]
         [18 19 --]]

    Parameters
    ----------
    array
    neighbourhood

    Returns
    -------
    masked array
    """
    n = int(neighbourhood)  # neighborhood size
    assert n % 2, '`neighbourhood` should be an odd integer'

    array = np.asanyarray(array)
    view = _view_neighbours(array, n, 0)
    if np.ma.isMA(array):
        mask = np.ma.getmaskarray(array)
        # ignore the edge items by padding masked elements.
        # can't use np.pad since MaskedArray will silently be converted to
        # array. This is a known issue:
        #  https://github.com/numpy/numpy/issues/8881
        return np.ma.MaskedArray(view, _view_neighbours(mask, n, True))
    #
    return view


def _view_neighbours(array, n, pad_value=0):
    # worker for
    pad_width = (n - 1) // 2
    padding = [(0, 0)] * (array.ndim - 2) + [(pad_width, pad_width)] * 2
    padded = np.pad(array, padding, mode='constant', constant_values=pad_value)
    *d, h, w = padded.shape
    new_shape = tuple(d) + (h - n + 1, w - n + 1, n, n)
    new_strides = padded.strides * 2
    return as_strided(padded, new_shape, new_strides, writeable=False)
