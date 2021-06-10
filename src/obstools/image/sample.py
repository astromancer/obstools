import numbers

import numpy as np

from recipes.logging import LoggingMixin


class BootstrapResample(LoggingMixin):
    # Sampling with replacement

    # Draw a sample of `n` arrays randomly from the index interval `subset`
    # along the given axis
    def __init__(self, data, sample_size=None, subset=None, axis=0):
        """
        Draw a sample of `n` arrays randomly from the index interval `subset`
        along the given axis

        Parameters
        ----------
        data
        sample_size: int
            Size of the sample to use in creating sample image
        subset: int or tuple
            sample_interval
        axis
        """



        #
        self.data = data
        self.axis = axis
        self.sample_size = sample_size
        self.subset = subset

    def draw(self, n=None, subset=None):
        """
        Select a sample of `n` arrays randomly from the interval `subset`
        along the axis

        Parameters
        ----------
        n
        subset:
            note if indices of subset are beyond array size along axis 0,
            the entire array will be used

        Returns
        -------

        """

        if (n is None) and (self.sample_size is None):
            raise ValueError('Please give sample size (or initialize this '
                             'class with a sample size)')
        # m = len(self.data)
        if subset is None:
            subset = (0, None)
        elif isinstance(subset, numbers.Integral):
            subset = (0, subset)  # treat like a slice

        # get subset array
        sub = self.data[slice(*subset)]
        if sub.size == 0:
            raise ValueError('Cannot draw sample from an empty subset of the '
                             'data')

        if (n is ...) or (len(sub) <= n):
            return sub

        # get frame indices (subspace sampled with replacement)
        self.logger.debug('Selecting %i frames from amongst frames (%i->%i) '
                          'for sample image.', n, *subset)
        ix = np.random.randint(0, len(sub), n)
        return self.data[ix]

    def max(self, n=None, subset=None):
        return self.draw(n, subset).max(self.axis)

    def mean(self, n=None, subset=None):
        return self.draw(n, subset).mean(self.axis)

    def std(self, n=None, subset=None):
        return self.draw(n, subset).std(self.axis)

    def median(self, n=None, subset=None):
        return np.ma.median(self.draw(n, subset), self.axis)



# class ImageSampler(BootstrapResample):
#     def __init__(self, stat='median', sample_size=None, subset=None, axis=0):
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
#     def __init__(self, sample_size=None, subset=None, axis=0):
#         # delay data
#         BootstrapResample.__init__(self, None, sample_size, subset, axis)