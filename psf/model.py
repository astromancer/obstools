import numpy as np
from numpy.polynomial.polynomial import polyval2d

class Model():

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def residuals(self, p, data, grid):
        '''Difference between data and model'''
        return data - self(p, grid)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rs(self, p, data, grid):
        '''squared residuals'''
        return np.square(self.residuals(p, data, grid))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def frs(self, p, data, grid):
        '''squared residuals flattened'''
        return self.rs(p, data, grid).flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rss(self, p, data, grid):
        '''residual sum of squares'''
        return self.rs(p, data, grid).sum()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wrs(self, p, data, grid, data_stddev=None):
        '''weighted squared residuals'''
        if data_stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / data_stddev

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def fwrs(self, p, data, grid, data_stddev=None):
        '''weighted squared residuals flattened'''
        return self.wrs(p, data, grid, data_stddev).flatten()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def wrss(self, p, data, grid, data_stddev=None):
        '''weighted residual sum of squars'''
        return self.wrs(p, data, grid, data_stddev).sum()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def validate(self, p, *args):
        '''validate parameter values.  To be overwritten by sub-class'''
        return all([vf(p) for vf in self.validations])


class PolyBG(Model):
    def __call__(self, p, grid):
        shape = int(np.size(p) // 2), -1
        p = np.reshape(p, shape)
        return polyval2d(*grid, p)