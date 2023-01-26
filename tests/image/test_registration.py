# std
import textwrap
from pathlib import Path

# third-party
import numpy as np
import pytest
from pytest import approx

# local
from recipes.transforms import rotate
from obstools.image.registration import (CoherentPointDrift, ImageRegister,
                                         MultivariateGaussians, SkyImage, rigid)


# from pytest_steps import test_steps as steps


def load_test_data(filename):
    """load test data npz"""
    here = Path(__file__).parent
    # here = Path('/home/hannes/work/obstools/tests/')
    filename = (here / 'data' / filename).resolve()
    return list(np.load(filename).values())


def make_id(name, n):
    for i in range(n):
        yield f'{name}{i}'


# dynamically generate some simple fixtures for combinatorial tests
# sourcery skip: remove-dict-items
for name, params in dict(
        xy=[((5, 10), (5, 6), (8, 2))],
        sigmas=[1,
                (1, 0.5),
                (1, 0.5, 0.6),
                ((0.5, 0.1),
                 (0.5, 0.6),
                 (0.8, 0.2))
                ],
        amplitudes=[10,
                    (1, 2, 3)
                    ]
).items():
    exec(textwrap.dedent(
        f"""
        @pytest.fixture(params=params, ids=make_id(name, len(params)))
        def {name}(request):
            return request.param
        """))


@pytest.fixture
def model(xy, sigmas, amplitudes):
    # this actually already tests the initialization, but it's basically
    # impossible to re-use the model created in a test :(
    return MultivariateGaussians(xy, sigmas, amplitudes)


# class TestTransforms:
#     def test_rotate(self):
#         ''

#     def test_rigid(self):
#         ''

#     def test_affine(self):
#         ''


# @pytest.mark.skip       # TODO autouse
class TestMultivariateGaussians:
    """test MultivariateGaussians"""

    xy_test = (5, 5)
    xy_test2 = [(3, 3), (6, 6)]
    expected = iter([6.066578,
                     1.213435,
                     1.353353,
                     0.270671,
                     1.353390,
                     0.270674,
                     2.493522,
                     0.498704,
                     ])
    expected2 = iter([[0.015057, 6.06779528],
                      [0.00301366, 1.21340099],
                      [5.06408816e-06, 6.06530660e+00],
                      [1.51716529e-06, 1.21306132e+00],
                      [8.20811573e-11, 1.35538752e+00],
                      [1.33176199e-11, 2.70874035e-01],
                      [1.25016514e-08, 1.35335283e+00],
                      [2.50034256e-09, 2.70670566e-01]]
                     )

    def test_init(self, model):
        assert model.xy.shape == model.sigmas.shape

    def test_set_attr(self, model):
        # test setting hyper parameters
        amplitudes = (3, 4, 5)
        model.amplitudes = amplitudes

        # test bork for negative amplitudes
        with pytest.raises(ValueError):
            model.amplitudes = np.negative(amplitudes)

        # test bork for bad shape
        with pytest.raises(ValueError):
            model.amplitudes = [1, 2]

        # test bork bad sigma shape
        with pytest.raises(ValueError):
            model.sigmas = [1, 2, 3, 4]

        # TODO: test borks for non-finite / masked amp / sigma

    # TODO: def test_flux(self, model): check integral

    # @pytest.mark.parametrize('expected', (1,2))

    def test_call(self, model):
        # model call
        # use the next 2 lines to generate expected answers
        # with open('foo.txt', 'a') as fp:
        #     fp.write('%f\n' % model((), self.xy_test))

        pytest.approx(model((), self.xy_test), next(self.expected))

    def test_call2(self, model):
        # model call
        pytest.approx(model((), self.xy_test2), next(self.expected2))

    @pytest.mark.skip
    @pytest.mark.mpl_image_compare(baseline_dir='images',
                                   remove_text=True)
    def test_plot(self, model):
        # print('running test_plot')
        im = model.plot(sliders=False, hist=False)
        # im.ax.plot(*self.xy_test, 'mx')
        # im.figure.savefig(f'{next(counter)}.png')
        return im.figure

# TODO: testGMM - check always integrates to 1


@pytest.fixture
def cpd(xy, sigmas, amplitudes):
    # this actually already tests the initialization, but it's basically
    # impossible to re-use the model created in a test :(
    weights = amplitudes / np.sum(amplitudes)
    return CoherentPointDrift(xy, sigmas, weights)

# @pytest.mark.skip


class TestCoherentPointDrift:
    @classmethod
    def setup_class(cls):
        cls.model = CoherentPointDrift(((5, 10), (5, 6), (8, 2)),
                                       2.5)

    # @pytest.mark.skip
    def test_transform(self):
        xy = np.random.rand(5, 2)
        assert xy == pytest.approx(self.model.transform(xy, (0, 0, 0)))

        self.model.fit_rotation = False
        assert xy + 1 == pytest.approx(self.model.transform(xy, (1, 1)))

    # @pytest.mark.skip
    def test_fit_translate(self, cpd):
        cpd.fit_rotation = False
        off = (0.5, 0.5)
        r = cpd.fit(cpd.gmm.xy + off)
        assert pytest.approx(off, 0.01) == -r

    def test_fit(self, cpd):
        cpd.fit_rotation = True
        θ = np.pi / 12
        off = (0.5, 0.5)

        xy = rigid(cpd.gmm.xy, np.hstack([off, θ]))
        r = cpd.fit(xy)

        assert pytest.approx(r, 0.01) == np.hstack([-rotate(off, -θ), -θ])


# @pytest.fixture
# def random_skyimage():
#     return SkyImage(np.random.randn(10, 10), (1, 1))


# @pytest.fixture()
# def images():
#     return load_test_data('images.npz')


# @pytest.fixture
# def fovs():
#     return load_test_data('fovs.npz')


images = load_test_data('images.npz')
fovs = load_test_data('fovs.npz')


@pytest.fixture
def skyimage(data, fov):
    return SkyImage(data, fov)


@pytest.fixture
def skyimage0():
    return SkyImage(images[0], fovs[0])


@pytest.mark.skip
class TestSkyImage():
    def test_init(self):
        img = np.random.randn(10, 10)
        SkyImage(img, (1, 1))

        with pytest.raises(ValueError):
            SkyImage(img)

    def test_detect(self, skyimage0):
        skyimage0.detect()
        assert len(skyimage0.xy)
        assert len(skyimage0.counts)

    @pytest.mark.mpl_image_compare(baseline_dir='images',
                                   remove_text=True)
    def test_plot(self, skyimage0):
        art, frame = skyimage0.plot(p=(0, 0, np.pi/12))
        return art.figure


# @pytest.mark.incremental


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
        assert reg.fovs == approx(fovs)
        assert reg.scales == approx(fovs / reg.attrs('data.shapes'))

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

# # dss = ImageRegistrationDSS('CTCV J1928-5001', (4.25, 4.25))

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

# matcher = ImageRegistrationDSS(I[a], FoV[a])
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
