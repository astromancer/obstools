# third-party libs
import numpy as np

# local libs
from obstools.image.segmentation import SegmentedImage, Slices, \
    MaskedStatsMixin, trace_boundary

from collections import OrderedDict
from photutils.datasets import (make_random_gaussians_table,
                                make_gaussian_sources_image)





def test_self_awareness():
    seg = SegmentedImage(np.zeros(10, 10))
    seg2 = SegmentedImage(seg)
    # todo check that lazy properties don't recompute
    # todo check that groups are preserved


def test_pickle():
    # pickling test
    import pickle

    z = np.zeros((25, 100), int)
    z[10:15, 30:40] = 1
    segm = SegmentedImage(z)
    clone = pickle.loads(pickle.dumps(segm))
    # todo some equality tests ...


def test_slices():
    data = np.zeros(10, 10)
    data[:2, :2] = 1
    data[-3:, -3:] = 2
    data[:3, -3:] = 3

    seg = SegmentedImage(data)
    slices = Slices(seg)
    slices = Slices(slices)  # self aware init

    slices[[0, 2]]  # should be length 2 array of 2-tuples


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


def test_trace_contours():
    from scipy import ndimage

    tests = []
    d = ndimage.distance_transform_edt(np.ones((15, 15)))
    tests.append(d > 5)

    z = np.square(np.indices((15, 15)) - 7.5).sum(0) < 7.5
    z[9, 4] = 1
    tests.append(z)

    z = np.zeros((5, 5), bool)
    z[3:5, 2] = 1
    tests.append(z)

    for i, t in enumerate(tests):
        boundary = trace_boundary(t)

        seg = SegmentedImage(t)
        im = seg.display(extent=np.c_[(0, 0), t.shape].ravel())

        im.ax.plot(*boundary.T[::-1], 'r-')

    # TODO: test tracing for segments with multiple seperate sections.


import pytest


def make_test_image(n_sources=50, sigma_psf=1.0, shape=(128, 128), bg_rate=100):
    # use an OrderedDict to ensure reproducibility
    params = OrderedDict([('flux', [500, 5000]),
                          ('x_mean', [0, shape[0]]),
                          ('y_mean', [0, shape[1]]),
                          ('x_stddev', [sigma_psf, sigma_psf]),
                          ('y_stddev', [sigma_psf, sigma_psf]),
                          ('theta', [0, np.pi])])

    np.random.set_state(1234)
    star_list = make_random_gaussians_table(n_sources, params)
    noise = np.random.poisson(bg_rate, shape)
    return make_gaussian_sources_image(shape, star_list) + noise

# https://pytest.org/en/latest/fixture.html#fixtures
@pytest.fixture(scope="module")
def sim_image():
    # make a small test image with noise
    return make_test_image(30, 0.5, (28, 28), 500)


@pytest.mark.incremental
class TestSegmentation:
    def test_detect(self):
        seg = SegmentedImage.detect(sim_image)

    def test_display_term(self):
        s = seg.display_term()
        # do some test with s


@pytest.fixture(scope="module")
def seg():
    # detect objects in test image
    return SegmentedImage.detect(sim_image)

    #


def test_stats():
    # check that stats works for label zero as well as for images containing
    # masked pixels

    for stat in MaskedStatsMixin._supported:
        getattr(seg, stat)(image)

    # seg.mean
    # seg.median
    # seg.minimum
    # maximum_position
    # extrema
    # sum
    # variance
    # standard_deviation


import pickle

def test_pickle():
    clone = pickle.loads(pickle.dumps(gph.seg))


def test_display():

    z = np.zeros((3,3))
    for i in range(3):
        z[i] = i

    SegmentedImage(z).display_term()