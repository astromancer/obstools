
# std
import pickle
from collections import OrderedDict

# third-party
import pytest
import numpy as np
from photutils.datasets import (make_gaussian_sources_image,
                                make_random_gaussians_table)

# local
from obstools.image.segments import (MaskedStatsMixin, SegmentedImage,
                                     trace_boundary)


# ---------------------------------------------------------------------------- #
# DetectionBase.__call__.__cache__.disable()

np.random.seed(1234)

# fixtures
# ------------------------------------------------------------------------------


def make_test_image(n_sources=50, sigma_psf=1.0, shape=(128, 128), bg_rate=100,
                    n_masked_pixels=0):
    # use an OrderedDict to ensure reproducibility
    params = OrderedDict([('flux', [500, 5000]),
                          ('x_mean', [0, shape[0]]),
                          ('y_mean', [0, shape[1]]),
                          ('x_stddev', [sigma_psf, sigma_psf]),
                          ('y_stddev', [sigma_psf, sigma_psf]),
                          ('theta', [0, np.pi])])

    star_list = make_random_gaussians_table(n_sources, params)
    noise = np.random.poisson(bg_rate, shape)
    image = make_gaussian_sources_image(shape, star_list) + noise

    if n_masked_pixels:
        image = np.ma.MaskedArray(image)
        idx = np.random.randint(0, image.size, n_masked_pixels)
        image[np.unravel_index(idx, image.shape)] = np.ma.masked

    return image


@pytest.fixture(params=[0, 10])  # n_masked_pixels
def sim_image(request):
    # make a small test image with noise, with and without masked pixels
    return make_test_image(10, 0.5, (28, 28), 500, request.param)


@pytest.fixture(params=[0, 1, 3])  # size of stack
def sim_data(sim_image, request):
    # simulate image stack with noise, with and without masked pixels
    n = request.param
    if n == 0:
        return sim_image
    return np.asanyarray([sim_image] * n)


@pytest.fixture
def seg(sim_image):
    # detect objects in test image
    return SegmentedImage.detect(sim_image)

# ------------------------------------------------------------------------------

# @pytest.mark.skip
# @pytest.mark.incremental


class TestSegmentation:
    def test_detect(self, sim_image):
        seg = SegmentedImage.detect(sim_image)

    def test_console_image(self, seg):
        s = seg.show.console()
        # do some test with s

# @pytest.mark.skip


def test_self_awareness():
    seg = SegmentedImage(np.zeros((10, 10)))
    seg2 = SegmentedImage(seg)
    # todo check that lazy properties don't recompute
    # todo check that groups are preserved


# @pytest.mark.skip
def test_pickle():
    shape = (25, 100)
    z = np.zeros(shape, int)
    z[10:15, 30:40] = 1
    image = np.random.rand(*shape)
    image[10:15, 30:40] += 5
    seg = SegmentedImage(z)
    clone = pickle.loads(pickle.dumps(seg))
    
    # todo some equality tests ...
    assert np.all(seg.detect(image).data == clone.detect(image).data)
    assert np.all(seg.com(image) == clone.com(image))

# # @pytest.mark.skip
# def test_slices():
#     data = np.zeros(10, 10)
#     data[:2, :2] = 1
#     data[-3:, -3:] = 2
#     data[:3, -3:] = 3

#     seg = SegmentedImage(data)
#     slices = Slices(seg)
#     slices = Slices(slices)  # self aware init

#     slices[[0, 2]]  # should be length 2 array of 2-tuples


# @pytest.mark.skip
def test_add_segments():
    # test all combination of arguments
    z = np.zeros((10, 10), int)
    z[3, 3] = 1
    g = {'one': 1}
    s = SegmentedImage(z, g)

    z = np.zeros((10, 10), int)
    z[5, 5] = 1
    g = {'two': 1}
    ss = SegmentedImage(z, g)

    r, nl = s.add_segments(ss)
    assert r.data[3, 3] == 1
    assert r.data[5, 5] == 2

    return r, nl


# @pytest.mark.skip
def test_trace_contours():
    from scipy import ndimage

    d = ndimage.distance_transform_edt(np.ones((15, 15)))
    cases = [d > 5]
    z = np.square(np.indices((15, 15)) - 7.5).sum(0) < 7.5
    z[9, 4] = 1
    cases.append(z)

    z = np.zeros((5, 5), bool)
    z[3:5, 2] = 1
    cases.append(z)

    for t in cases:
        pixels, boundary, perimeter = trace_boundary(t)

        seg = SegmentedImage(t)
        im = seg.display(extent=np.c_[(0, 0), t.shape].ravel())

        im.ax.plot(*boundary.T[::-1], 'r-')

    # TODO: test tracing for segments with multiple seperate sections.
    # TODO: test:
    # b = np.array([[ 1,  1,  1,  1,  1,  1,  1],
    #               [ 1,  0,  1,  0,  1,  0,  1],
    #               [ 1,  1,  1,  1,  1,  0,  0],
    #               [ 0,  1,  0,  1,  0,  0,  0],
    #               [ 1,  1,  0,  0,  0,  0,  0],
    #               [ 1,  0,  1,  0,  1,  1,  1],
    #               [ 0,  1,  1,  0,  1,  0,  1],
    #               [ 1,  0,  0,  0,  1,  1,  1]])

@pytest.mark.parametrize('stat', MaskedStatsMixin._supported)
def test_stats(seg, sim_data, stat):
    # check that stats works for label zero as well as for images containing
    # masked pixels

    # seg.mean
    # seg.median
    # seg.minimum
    # maximum_position
    # extrema
    # sum
    # variance
    # standard_deviation
    # data = request.getfixturevalue(data)
    # seg = request.getfixturevalue('seg')

    sd = seg.data.copy()
    result = getattr(seg, stat)(sim_data)

    # check shape of result
    is2d = (sim_data.ndim == 2)

    assert result.shape[:2 - is2d] == (len(sim_data), seg.nlabels)[is2d:]

    # make sure we return masked arrays for input masked arrays
    if np.ma.isMA(sim_data):
        assert np.ma.isMA(result)

    # make sure segmentation data has not been changed
    assert np.all(sd == seg.data)
