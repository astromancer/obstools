"""
Sampling and statistics of images from a stack.
"""

# std
from collections.abc import Collection

# third-party
import numpy as np
from astropy.utils import lazyproperty

# local
from recipes.caching import Cached as cached
from recipes.logging import LoggingMixin

# relative
from .. import _hdu_hasher, cachePaths


class BootstrapResample(LoggingMixin):
    """
    Sampling with replacement
    """

    def __init__(self, data, sample_size=None, subset=..., axis=0):
        """
        Draw a sample of `n` arrays randomly from the index interval `subset`
        along the given axis.

        Parameters
        ----------
        data
        sample_size: int
            Size of the sample to use in creating sample image.
        subset: int or tuple
            sample_interval
        axis
        """

        #
        self.data = data
        self.axis = axis
        self.sample_size = sample_size
        self.subset = subset

    def draw(self, n=None, subset=...):
        """
        Select a sample of `n` arrays randomly from the interval `subset`
        along the axis.

        Parameters
        ----------
        n : int
        subset : int or tuple of int or Ellipsis or None
            note if indices of subset are beyond array size along axis 0,
            the entire array will be used.

        Returns
        -------

        """

        if (n is None) and (self.sample_size is None):
            raise ValueError('Please give sample size (or initialize this '
                             'class with a sample size)')
        # m = len(self.data)

        # make a slice
        if isinstance(subset, Collection):
            subset = slice(*subset)
        elif subset is not ...:
            subset = slice(subset)

        # get subset array
        sub = self.data[subset]
        if sub.size == 0:
            raise ValueError('Cannot draw sample from an empty subset of the '
                             f'data: {subset}.')

        if (n is ...) or (len(sub) <= n):
            return sub

        # get frame indices (subspace sampled with replacement)
        self.logger.debug('Selecting {:d} frames from amongst frames '
                          '({:d}->{:d}) for sample image.', n, *subset)
        return self.data[np.random.randint(0, len(sub), n)]

    def max(self, n=None, subset=...):
        return self.draw(n, subset).max(self.axis)

    def mean(self, n=None, subset=...):
        return self.draw(n, subset).mean(self.axis)

    def std(self, n=None, subset=...):
        return self.draw(n, subset).std(self.axis)

    def median(self, n=None, subset=...):
        return np.ma.median(self.draw(n, subset), self.axis)

# class ImageSampler(BootstrapResample):
#     def __init__(self, stat='median', sample_size=None, subset=..., axis=0):
#         #
#         BootstrapResample.__init__(self, None, sample_size, subset, axis)
#
#         self.func = getattr(self, stat, None)
#         if self.func is None:
#             raise ValueError('Invalid statistic')
#
#     def __call__(self, data):
#         self.func(data)
#
#
# class ImageSamplerHDU(ImageSampler):
#     def __init__(self, sample_size=None, subset=..., axis=0):
#         # delay data
#         BootstrapResample.__init__(self, None, sample_size, subset, axis)


class ImageSamplerMixin:
    """
    A mixin class that can draw sample images from the HDU data.
    """

    @lazyproperty  # lazyproperty ??
    def sampler(self):
        """
        Use this property to get calibrated sample images and image statistics
        from the stack.

        >>> stack.sampler.median(10, 100)

        """
        #  allow higher dimensional data (multi channel etc), but not lower
        #  than 2d
        if self.ndim < 2:
            raise ValueError('Cannot create image sampler for data with '
                             f'{self.ndim} dimensions.')

        # ensure NE orientation
        data = self.calibrated

        # make sure we pass 3d data to sampler. This is a hack so we can use
        # the sampler to get thumbnails from data that is a 2d image,
        # eg. master flats.  The 'sample' will just be the image itself.

        if self.ndim == 2:
            # insert axis in front
            data = self.data[None]

        return BootstrapResample(data)

    # since this function deals with random statistics, caching is disabled by
    # default. We enable this cache in the pipeline to reduce unnecessary repeat
    # computation
    @cached(cachePaths.samples, typed={'self': _hdu_hasher}, enabled=False)
    def get_sample_image(self, stat='median', min_depth=5, subset=...):
        """
        Get sample image to a certain minimum simulated exposure depth by
        averaging data.

        Parameters
        ----------
        stat : str, optional
            The statistic to use when computing the sample image, by default
            'median'.
        min_depth : int, optional
            Minimum simulated exposure depth in seconds, by default 5.

        Returns
        -------
        np.ndarray
            An image of the requested statistic across a sample of images drawn
            from the stack. 
        """
        # FIXME: get this to work for SALTICAM

        n = int(np.ceil(min_depth // self.timing.exp)) or 1

        self.logger.info('Computing {stat} of {n} images (exposure depth of '
                         '{min_depth:.1g} seconds) for sample image from '
                         '{name!r} for the data interval {subset}.',
                         **locals(), name=self.file.name)

        sampler = getattr(self.sampler, stat)
        return sampler(n, subset)
