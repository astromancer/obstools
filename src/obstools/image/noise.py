
# third-party
import numpy as np
from astropy import units

# local
from recipes.dicts import pformat
from recipes.oo.property import cached_property

# relative
from ..unit_helpers import default_units


class StdDev:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, _):
        return self.sigma


class AddativeNoiseModel:

    def __init__(self, **stddevs):
        self.__dict__.update(**stddevs)

    def __setattr__(self, name, value):
        if name != 'total_var':
            del self.total_var
        super().__setattr__(name, value)

    def __str__(self):
        return pformat(self.__dict__, self.__class__.__name__,
                       brackets='()', rhs='{:g}'.format)  # ignore=('total_var')

    __repr__ = __str__

    def __call__(self, data):
        """Standard deviation of CCD pixel data in `unit_out` ** 2."""
        return self.std(data)

    def copy(self):
        return type(self)(**self.__dict__)

    def std(self, data):
        """Variance of CCD pixel data in `unit_out`."""
        return np.sqrt(self.var(data))

    def var(self, data):
        """Variance of CCD pixel data in `unit_out` ** 2."""
        return self.total_var

    def _vars(self, ignore=('total_var',)):
        for source, sigma in self.__dict__.items():
            if source not in ignore:
                yield sigma * sigma

    @cached_property
    def total_var(self):
        return sum(self._vars())


e_per_adu = units.electron / units.adu


class CCDNoiseModel(AddativeNoiseModel):

    @default_units(readout=units.electron,
                   other_noise_sources_stddev=units.electron)
    def __init__(self, readout=0, unit_out=units.electron, **other_noise_sources_stddev):
        """
        Model of the uncertainty associated with data measured with a
        charge-coupled device.

        Parameters
        ----------
        readout : float
            Amplifier readout noise level 1 sigma standard deviation in
            electrons or adu.

        """
        other_noise_sources_stddev.update(readout=readout)
        super().__init__(**other_noise_sources_stddev)
        self.unit_out = unit_out

    def var(self, data):
        """
        Variance of CCD pixel data in `unit_out` ** 2. Poisson noise plus
        additional noise terms.
        """
        return data + self.total_var

    def var_of_mean(self, data, axis=0):
        """Variance of mean combined CCD data in `unit_out` ** 2."""
        return self.var(np.mean(data, axis))

    def std_of_mean(self, data, axis=0):
        """Standard deviation of sample mean in `unit_out`."""
        return np.sqrt(self.var_of_mean(data, axis))

    def var_of_median(self, data, axis=0):
        """Unbiased variance of sample median in `unit_out` ** 2."""

        # The efficiency of the median, measured as the ratio of the variance of
        # the mean to the variance of the median, depends on the sample size
        # N=2n+1 as (4n)/(pi(2n+1)) which tends to the value 2/pi approx 0.637
        # as N becomes large (Kenney and Keeping 1962, p. 211).

        # https://mathworld.wolfram.com/StatisticalMedian.html
        n = len(data)
        return n * np.pi / (2 * (n - 1)) * self.var_of_mean(data, axis)

    def std_of_median(self, data, axis=0):
        """Standard deviation of sample median in `unit_out`."""
        return np.sqrt(self.var_of_median(data, axis))

    variance_of_mean = var_of_mean
    stddev_of_mean = std_of_mean
    variance_of_median = var_of_median
    stddev_of_median = std_of_median
