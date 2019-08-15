
# third-party libs
import numpy as np

# local libs
from obstools.phot.segmentation import SegmentationHelper, Slices

def test_self_awareness():
    seg = SegmentationHelper(np.zeros(10, 10))
    seg2 = SegmentationHelper(seg)


def test_slices():
    data = np.zeros(10, 10)
    data[:2, :2] = 1
    data[-3:, -3:] = 2
    data[:3, -3:] = 3

    seg = SegmentationHelper(data)
    slices = Slices(seg)
    slices = Slices(slices)  # self aware init

    slices[[0, 2]] # should be length 2 array of 2-tuples


def test_stats():
    ''
    # check that stats works for label zero as well as for images containing
    # masked pixels

    # seg.mean
    # seg.median
    # seg.minimum
    # maximum_position
    # extrema
    # sum,
    # variance,
    # standard_deviation