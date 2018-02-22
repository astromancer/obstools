import numpy as np


class Model(object):
    """Mixin class for models"""
    def __call__(self, p, grid):
        raise NotImplementedError

    def residuals(self, p, data, grid):
        """Difference between data and model"""
        return data - self(p, grid)

    def rs(self, p, data, grid):
        """squared residuals"""
        return np.square(self.residuals(p, data, grid))

    def frs(self, p, data, grid):
        """squared residuals flattened"""
        return self.rs(p, data, grid).flatten()

    def rss(self, p, data, grid):
        """residual sum of squares"""
        return self.rs(p, data, grid).sum()

    def wrs(self, p, data, grid, data_stddev=None):
        """weighted squared residuals"""
        if data_stddev is None:
            return self.rs(p, data, grid)
        return self.rs(p, data, grid) / data_stddev

    def fwrs(self, p, data, grid, data_stddev=None):
        """weighted squared residuals flattened"""
        return self.wrs(p, data, grid, data_stddev).flatten()

    def wrss(self, p, data, grid, stddev=None):
        """weighted residual sum of squars"""
        return self.wrs(p, data, grid, stddev).sum()

    def validate(self, p, *args):
        """validate parameter values.  To be overwritten by sub-class"""
        return all([vf(p) for vf in self.validations])



class StaticGridMixin():
    """
    static grid mixin classfor performance gain when fitting the same model
    repeatedly on the same grid for different data.
    """
    grid = None

    def set_grid(self, data):
        raise NotImplementedError('Derived class should implement this method.')

    def residuals(self, p, data, grid=None):
        # grid argument ignored
        if grid is None:
            if self.grid is None:
                self.set_grid(data)
            grid = self.grid

        return super().residuals(p, data, grid)