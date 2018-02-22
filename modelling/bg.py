import numpy as np
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import minimize

from .model import Model, StaticGridMixin


def grid_from_data(image):
    gxy = np.indices(image.shape)[::-1]
    if np.ma.is_masked(image):
        gxy = gxy[:, ~image.mask]
    return gxy




class Poly2D(StaticGridMixin, Model):

    name = 'poly2d'

    def __init__(self, nx, ny):
        self.nxy = nx, ny
        self.ncoeff = np.add(self.nxy, 1)
        self.npar = np.prod(self.ncoeff)

        self.grid = None
        self.mask = False

    def set_mask(self, mask):
        # static_mask
        self.mask = np.array(mask, bool)
        #self.grid = np.indices(self.mask.shape)[::-1]

    def set_grid(self, image):
        self.grid = np.indices(image.shape)[::-1]

        # self.grid = grid_from_data(image)
        # if np.ma.is_masked(image):
        #     self.mask = image.mask
        # else:
        #     self.mask = None

    def __call__(self, p, grid):
        # grid is xy-coords
        return polyval2d(*grid, np.reshape(p, self.ncoeff))

    # def residuals(self, p, image, grid=None):
    #     # grid argument ignored
    #     if grid is None:
    #         if self.grid is None:
    #             self.set_grid(image)
    #         grid = self.grid
    #
    #     return Model.residuals(self, p, image, grid)

    def fit(self, image, p0=None, **kws):

        if self.grid is None:
            self.set_grid(image)

        grid = self.grid
        mask = self.mask
        if np.ma.isMA(image):
            mask |= image.mask
            grid = grid[:, ~mask]

        # only fit unmasked pixels
        image = image[~mask]

        if p0 is None:
            p0 = np.ones(self.npar)

        r = minimize(self.rss, p0, (image, grid), **kws)

        return r.x



# class Poly2DMasked(Poly2D):
#     name = 'poly2dMasked'
#
#     def __init__(self, nx, ny, mask):
