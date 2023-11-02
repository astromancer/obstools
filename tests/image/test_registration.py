
# third-party
import pytest
import numpy as np

# local
from obstools.image import ImageRegister
from conftest import *

# @pytest.mark.incremental

# ---------------------------------------------------------------------------- #


@pytest.mark.skip
class TestImageRegister:

    def test_init(self):
        # construct without data
        reg = ImageRegister(snr=5, npixels=7)
        assert reg.find_kws['snr'] == 5
        assert reg.find_kws['npixels'] == 7

        # HACK to reuse this later
        self.__class__.reg = reg

    def test_init_with_data(self, skyimage0):
        # init without params raises
        with pytest.raises(ValueError):
            ImageRegister([skyimage0])

        # init with params OK
        ImageRegister([skyimage0], params=[(0, 0, 0)])

    def test_attr_lookup(self):
        # construct with data
        reg = ImageRegister(images, fovs, np.zeros(len(fovs, 3)))

        # test vectorized item attribute lookup
        assert reg.fovs == pytest.approx(fovs)
        assert reg.scales == pytest.approx(fovs / reg.attrs('data.shapes'))

    @pytest.mark.parametrize('data, fov', zip(images, fovs), indirect=True)
    def test_no_rotation(self, skyimage):
        # this will build the register, and will register as individual tests...
        self.reg(skyimage)

    def test_refine(self):
        self.reg.refine()
        self.reg.recentre()

    # TODO: test switch idx, mosaic should be identical with text removed


# *images, slot_image = load_test_data('images.npz')
# *fovs, slot_fov = load_test_data('fovs.npz')
# slot_pa = 0.464128

# reg = ImageRegister.from_images(images, fovs)

# #mos = reg.mosaic(alpha=0.3)

# # dss = ImageRegisterDSS('CTCV J1928-5001', (4.25, 4.25))

# p = reg(slot_image, slot_fov, slot_pa)

# test_construct()

# #import operator
# from pathlib import Path
# # from atexit import register as exit_register

# from matplotlib import rc

# # from grafico.ts import TSplotter
# #from tsa.spectral import FFT
# #from outliers import WindowOutlierDetection, generalizedESD
# from recipes.io import parse

# from pyshoc.image.registration import *
# from pyshoc.core import shocRun

# # exit_register(embed)

# root = logging.getLogger()
# root.setLevel(logging.DEBUG)

# root = Path('/media/Oceanus/UCT/Observing/data')
# dir2015 = root / 'Feb_2015/MASTER_J0614-2725'
# dir2017 = root / 'Feb_2017/J0614-2725/'

# rc('savefig', directory=str(root))
# fits2015 = parse.to_list('2015*[0-9].fits',
#                         path=dir2015,
#                         exclude=['20150228.008.fits',     #no magnitude db
#                                   '20150301.002.fits',     #this one causes problems (len 1)
#                                   '20150301.003.fits',     #borks (len 45)
#                                   '20150301.004.fits',     #also short not that useful
#                                   '20150301.005.fits'],    #TODO: COMBINE THESE!!!
#                         convert=Path)
# fits2017 = parse.to_list('*bff.fits', path=dir2017, convert=Path)

# newlist = []
# for fits in fits2015:
#     cald = fits.with_suffix('.bff.fits')
#     if cald.exists:
#         newlist.append(cald)
#     else:
#         newlist.append(fits)

# obs = shocRun.load(filenames=newlist + fits2017)
# sr = obs.group_by('date.year')

# self = obs
# first = 10
# flip = True
# align_on = -2

# npar = 3
# n = len(self)
# P = np.zeros((n, npar))
# FoV = np.empty((n, 2))
# scales = np.empty((n, 2))
# I = []
# for i, cube in enumerate(self):
#     image = cube.data[:first].mean(0)
#     if flip:
#         image = np.fliplr(image)
#     I.append(image)
#     FoV[i] = fov = cube.get_FoV()
#     scales[i] = fov / image.shape

# # align on highest res image if not specified
# a = align_on
# if align_on is None:
#     a = scales.argmin(0)[0]
# others = set(range(n)) - {a}

# matcher = ImageRegisterDSS(I[a], FoV[a])
# self = matcher
# for i in others:
#     print(i)
#     # p = matcher.match_image(I[i], FoV[i])
#     # P[i] = p
#     # break

#     try:
#         image = I[i]
#         fov = FoV[i]

#         coo, flxr, segmr = sourceFinder(image, **self._findkws)
#         coo = (coo / image.shape) * fov

#         # do grid search
#         o = np.ones((2, 2))
#         o[:, 1] = -1

#         (ys, ye), (xs, xe) = self.fov * self.searchFrac * o
#         xres = int((xs - xe) / self.gridStep)
#         yres = int((ys - ye) / self.gridStep)
#         grid = np.mgrid[ys:ye:complex(yres),
#                         xs:xe:complex(xres)]
#         # add 0s for angle grid
#         z = np.zeros(grid.shape[1:])[None]
#         grid = np.r_[grid, z]
#         logging.info("Doing search on (%.1f' x %.1f') (%d x %d) sky grid",
#                      *fov, yres, xres)
#         r, ix, pGs = gridsearch_mp(objective1, (self.coords, coo), grid)
#         logging.debug('Grid search optimum: %s', pGs)

#         # try:
#         # match patterns
#         cooGs = roto_translate_yx(coo, pGs)
#         ir, ic = match_constellation(self.coords, cooGs)
#         logging.info('Matched %d stars in constellation.', len(ir))

#         # final alignment
#         # pick a star to re-center coordinates on
#         cooDSSsub = self.coords[ir]
#         cooGsub = cooGs[ic]
#         distDSS = cdist(cooDSSsub, cooDSSsub)
#         ix = distDSS.sum(0).argmax()
#         # translate to origin at star `ix`
#         yx = y, x = (cooGsub - cooGsub[ix]).T
#         vu = v, u = (cooDSSsub - cooDSSsub[ix]).T  # transform destination
#         # calculate rotation angle
#         thetas = np.arctan2(v * x - u * y, u * x + v * y)
#         theta = np.median(np.delete(thetas, ix))
#         # calc final offset in DSS coordinates
#         rotm = rotation_matrix_2d(-theta)
#         yxoff = yo, xo = np.median(cooDSSsub - (rotm @ coo[ic].T).T, 0)

#         p = (yo, xo, theta)
#         P[i] = p
#     except Exception as err:
#         from IPython import embed
#         embed()
#         raise err
